"""LLM repair tuỳ chọn cho generated question object.

Module này chỉ sửa generated question object khi service được bật `auto_repair`.
Nó không dùng question_plan, raw question, raw answer và không ghi file output.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..infra.config import AppConfig
from ..infra.debug import debug_llm_messages
from ..infra.llm_client import LLMClient
from ..shared.utils import parse_json_output
from .generated_question_schema import generated_question_id, validate_generated_question_object


KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"
CRITERIA_PATH = KNOWLEDGE_DIR / "generated_question_quality_criteria.md"
TYPE_RULES_PATH = KNOWLEDGE_DIR / "generated_question_type_rules.md"
IMPORTANT_GENERATED_QUESTION_FIELDS = {
    "schemaVersion",
    "concepts",
    "difficulty",
    "bloom",
    "source",
    "createdAt",
    "updatedAt",
    "createdBy",
    "updatedBy",
}


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def compact_repair_payload(generated_question: dict[str, Any], check_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": generated_question.get("id") or generated_question.get("_id") or "",
        "_id": generated_question.get("_id"),
        "aiId": generated_question.get("aiId"),
        "difficulty": generated_question.get("difficulty"),
        "bloom": generated_question.get("bloom"),
        "interactionTypes": generated_question.get("interactionTypes"),
        "instruction": generated_question.get("instruction"),
        "questionItems": generated_question.get("questionItems"),
        "solutions": generated_question.get("solutions"),
        "check_result": {
            "id": check_result.get("id"),
            "is_good": check_result.get("is_good"),
            "failed_reason": check_result.get("failed_reason") or [],
            "suggestions": check_result.get("suggestions") or [],
            "issues": check_result.get("issues") or [],
        },
    }


def build_generated_question_repair_messages(
    generated_question: dict[str, Any],
    check_result: dict[str, Any],
    criteria_text: str,
    type_rules_text: str,
) -> list[dict[str, str]]:
    payload = compact_repair_payload(generated_question, check_result)
    return [
        {
            "role": "system",
            "content": (
                "Bạn là LLM repair cho generated question object. "
                "Chỉ sửa chính generated question object được cung cấp, không dùng question_plan, "
                "raw question, raw answer, source images hoặc answer images. "
                "Nếu không thể sửa an toàn thành một full object hợp lệ, hãy trả new_generated_question=null. "
                "Toàn bộ reason/suggestion viết bằng tiếng Việt."
            ),
        },
        {
            "role": "user",
            "content": (
                "Hãy sửa generated question object dựa trên các issue đã được checker phát hiện.\n\n"
                "Quy tắc bắt buộc:\n"
                "- Chỉ sửa các lỗi nằm trong check_result. Không tự phát minh lỗi mới.\n"
                "- Trả về toàn bộ generated question object sau sửa trong new_generated_question nếu đủ an toàn.\n"
                "- Giữ nguyên id/_id gốc nếu có.\n"
                "- Giữ các field quan trọng nếu có: schemaVersion, concepts, difficulty, bloom, source, "
                "instruction, questionItems, solutions, createdAt, updatedAt, createdBy, updatedBy.\n"
                "- Không thêm metadata report/issues vào new_generated_question.\n"
                "- Không trả patch, before/after hoặc giải thích ngoài JSON.\n"
                "- Nếu repair không đủ chắc, đặt repair_status='failed' và new_generated_question=null.\n\n"
                "Output JSON bắt buộc:\n"
                "{\n"
                '  "repair_status": "repaired|failed|skipped",\n'
                '  "failed_reason": [],\n'
                '  "suggestions": [],\n'
                '  "new_generated_question": null\n'
                "}\n\n"
                "QUALITY CRITERIA:\n"
                f"{criteria_text}\n\n"
                "GENERATED QUESTION TYPE RULES:\n"
                f"{type_rules_text}\n\n"
                "PAYLOAD:\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def repair_error_result(reason: str, *, generated_question: dict[str, Any], index: int = 0) -> dict[str, Any]:
    return {
        "id": generated_question_id(generated_question, index),
        "repair_status": "failed",
        "failed_reason": [reason],
        "suggestions": ["Cần review thủ công generated question hoặc chạy lại repair."],
        "new_generated_question": None,
    }


def preserve_original_identity(
    original: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    repaired = deepcopy(candidate)
    for key in ("id", "_id"):
        if key in original and original.get(key):
            repaired[key] = original[key]
    for key in IMPORTANT_GENERATED_QUESTION_FIELDS:
        if key in original and key not in repaired:
            repaired[key] = deepcopy(original[key])
    return repaired


def normalize_generated_question_repair_result(
    parsed: Any,
    *,
    generated_question: dict[str, Any],
    index: int = 0,
) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return repair_error_result("Repair LLM không trả JSON object hợp lệ.", generated_question=generated_question, index=index)

    status = str(parsed.get("repair_status") or "").strip()
    if status not in {"repaired", "failed", "skipped"}:
        status = "failed"

    failed_reason = [str(item).strip() for item in parsed.get("failed_reason") or [] if str(item).strip()]
    suggestions = [str(item).strip() for item in parsed.get("suggestions") or [] if str(item).strip()]
    candidate = parsed.get("new_generated_question")

    if status == "skipped":
        return {
            "id": generated_question_id(generated_question, index),
            "repair_status": "skipped",
            "failed_reason": failed_reason,
            "suggestions": suggestions,
            "new_generated_question": None,
        }

    if not isinstance(candidate, dict):
        if not failed_reason:
            failed_reason.append("Repair không tạo được new_generated_question an toàn.")
        return {
            "id": generated_question_id(generated_question, index),
            "repair_status": "failed",
            "failed_reason": failed_reason,
            "suggestions": suggestions or ["Review thủ công generated question."],
            "new_generated_question": None,
        }

    candidate = preserve_original_identity(generated_question, candidate)
    validation = validate_generated_question_object(candidate, index)
    bad_reasons = [
        str(issue.get("reason") or "").strip()
        for issue in validation.get("issues") or []
        if issue.get("severity") == "bad"
    ]
    if bad_reasons:
        return {
            "id": generated_question_id(generated_question, index),
            "repair_status": "failed",
            "failed_reason": failed_reason + [f"new_generated_question sau repair vẫn lỗi schema: {bad_reasons[0]}"],
            "suggestions": suggestions or ["Review thủ công generated question sau repair."],
            "new_generated_question": None,
        }

    return {
        "id": generated_question_id(generated_question, index),
        "repair_status": "repaired",
        "failed_reason": failed_reason,
        "suggestions": suggestions,
        "new_generated_question": candidate,
    }


def repair_generated_question_object(
    generated_question: dict[str, Any],
    check_result: dict[str, Any],
    config: AppConfig,
    client: LLMClient,
    *,
    index: int = 0,
    debug: bool = False,
) -> dict[str, Any]:
    criteria_text = load_text(CRITERIA_PATH)
    type_rules_text = load_text(TYPE_RULES_PATH)
    messages = build_generated_question_repair_messages(
        generated_question,
        check_result,
        criteria_text,
        type_rules_text,
    )
    start = time.perf_counter()
    try:
        debug_llm_messages(
            step="generated_question_repair",
            model=config.primary_judge_model,
            messages=messages,
            debug=debug,
        )
        response = client.chat_completion(
            model=config.primary_judge_model,
            messages=messages,
            temperature=0,
        )
        parsed, ok, parse_error = parse_json_output(str(response.get("content") or ""))
        if not ok:
            return repair_error_result(
                parse_error or "Không parse được JSON repair.",
                generated_question=generated_question,
                index=index,
            )
        result = normalize_generated_question_repair_result(parsed, generated_question=generated_question, index=index)
        result["repair_latency_seconds"] = response.get("latency_seconds", time.perf_counter() - start)
        return result
    except Exception as exc:
        return repair_error_result(str(exc), generated_question=generated_question, index=index)

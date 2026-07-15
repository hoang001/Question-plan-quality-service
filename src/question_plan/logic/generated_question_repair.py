"""Scoped-only repair for generated question objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..infra.config import AppConfig
from ..infra.debug import debug_llm_messages
from ..infra.llm_client import LLMClient
from ..shared.utils import parse_json_output
from ..utils.json_pointer import JsonPointerError, apply_json_patch, get_by_json_pointer
from .generated_question_schema import (
    default_context_paths,
    generated_question_id,
    location_to_json_pointer,
    validate_generated_question_object,
)


KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"
REPAIR_RULES_PATH = KNOWLEDGE_DIR / "generated_question_repair_rules.md"
SCOPED_REPAIR_OUTPUT_SCHEMA_PATH = KNOWLEDGE_DIR / "generated_question_scoped_repair_output_schema.md"


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def compact_issue_for_repair(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "severity": issue.get("severity"),
        "category": issue.get("category"),
        "location": issue.get("location"),
        "reason": issue.get("reason"),
        "suggestion": issue.get("suggestion"),
        "repair_intent": issue.get("repair_intent"),
    }


def compact_check_result_for_repair(check_result: dict[str, Any]) -> dict[str, Any]:
    anchor = check_result.get("solution_anchor_result")
    return {
        "id": check_result.get("id"),
        "issues": [
            compact_issue_for_repair(issue)
            for issue in check_result.get("issues") or []
            if isinstance(issue, dict)
        ],
        "solution_anchor_result": anchor if isinstance(anchor, dict) else None,
    }


def is_safe_generated_question_patch_path(path: str) -> bool:
    pointer = location_to_json_pointer(path)
    if not pointer.startswith("/"):
        return False
    disallowed = {
        "/question",
        "/answer",
        "/question_plan",
        "/questionPlan",
        "/images",
        "/answer_images",
        "/answerImages",
        "/start_page",
        "/end_page",
        "/difficulty",
        "/bloom",
    }
    return not any(pointer == item or pointer.startswith(f"{item}/") for item in disallowed)


def unique_paths(paths: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in paths:
        pointer = location_to_json_pointer(str(value or "").strip())
        if pointer.startswith("/") and pointer not in normalized:
            normalized.append(pointer)
    return [
        pointer
        for pointer in normalized
        if not any(
            other != pointer and other.startswith(f"{pointer}/")
            for other in normalized
        )
    ]


def build_scoped_repair_context(
    generated_question: dict[str, Any],
    issue: dict[str, Any],
    check_result: dict[str, Any],
    *,
    max_context_chars: int = 12000,
) -> dict[str, Any]:
    location = location_to_json_pointer(str(issue.get("location") or ""))
    if not location.startswith("/"):
        return {"context_ok": False, "reason": "Issue thiếu location JSON Pointer hợp lệ."}

    paths = unique_paths(
        [
            *default_context_paths(str(issue.get("category") or ""), location),
            *(issue.get("required_context_paths") or []),
            location,
        ]
    )
    context: dict[str, Any] = {}
    for path in paths:
        try:
            context[path] = get_by_json_pointer(generated_question, path)
        except JsonPointerError as exc:
            return {"context_ok": False, "reason": f"Đường dẫn `{path}` không tồn tại: {exc}"}

    normalized_issue = compact_issue_for_repair(issue)
    normalized_issue["location"] = location
    payload = {
        "generated_question_id": generated_question_id(generated_question),
        "issue": normalized_issue,
        "check_result": compact_check_result_for_repair(check_result),
        "extracted_context": context,
    }
    if len(json.dumps(payload, ensure_ascii=False)) > max_context_chars:
        return {"context_ok": False, "reason": "Scoped repair context vượt giới hạn an toàn."}
    return {"context_ok": True, "payload": payload}


def build_generated_question_scoped_repair_messages(
    scoped_payload: dict[str, Any],
    repair_rules_text: str,
    output_schema_text: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Bạn sửa một issue cụ thể bằng JSON Patch trên generated question. Chỉ scoped repair, không full rewrite, "
                "không tự giải bài và không tạo đáp án mới. Solution Resolver final_answer là mốc phải giữ nguyên. "
                "Chỉ trả JSON hợp lệ; reason/suggestion phải là tiếng Việt có dấu."
            ),
        },
        {
            "role": "user",
            "content": (
                "Áp dụng đúng repair_intent:\n"
                "- align_hint_to_solution: chỉ sửa hint mâu thuẫn với solution resolved.\n"
                "- clean_solution_reasoning: bỏ thử-sai/tự vấn/đoạn nháp nhưng giữ nguyên final answer; không tự giải lại.\n"
                "- fix_schema: chỉ patch cấu trúc/render nhỏ và an toàn.\n"
                "Nếu không đủ context hoặc không có patch an toàn, trả needs_manual_review.\n\n"
                f"REPAIR RULES:\n{repair_rules_text}\n\n"
                f"OUTPUT SCHEMA:\n{output_schema_text}\n\n"
                f"SCOPED PAYLOAD:\n{json.dumps(scoped_payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def manual_review_result(reason: str, generated_question: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": generated_question_id(generated_question, index),
        "repair_status": "needs_manual_review",
        "failed_reason": [reason] if reason else [],
        "suggestions": ["Review thủ công issue; không dùng full repair."],
        "new_generated_question": None,
        "patches": [],
    }


def normalize_scoped_repair_result(
    parsed: Any,
    *,
    generated_question: dict[str, Any],
    index: int = 0,
    repair_intent: str = "",
) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return manual_review_result("Scoped repair không trả JSON object hợp lệ.", generated_question, index)
    status = str(parsed.get("repair_status") or "")
    if status == "needs_manual_review":
        reasons = [str(value).strip() for value in parsed.get("failed_reason") or [] if str(value).strip()]
        return manual_review_result(reasons[0] if reasons else "Không có patch an toàn.", generated_question, index)
    if status == "failed":
        return {
            "id": generated_question_id(generated_question, index),
            "repair_status": "failed",
            "failed_reason": [str(value).strip() for value in parsed.get("failed_reason") or [] if str(value).strip()],
            "suggestions": [str(value).strip() for value in parsed.get("suggestions") or [] if str(value).strip()],
            "new_generated_question": None,
            "patches": [],
        }
    if status != "repaired":
        return manual_review_result("repair_status không hợp lệ.", generated_question, index)

    patches = parsed.get("patches")
    if not isinstance(patches, list) or not patches:
        return manual_review_result("Scoped repair không trả patch.", generated_question, index)
    for patch in patches:
        if not isinstance(patch, dict):
            return manual_review_result("Patch không phải JSON object.", generated_question, index)
        if patch.get("op") not in {"replace", "add", "remove"}:
            return manual_review_result("Patch operation không hợp lệ.", generated_question, index)
        if not is_safe_generated_question_patch_path(str(patch.get("path") or "")):
            return manual_review_result("Patch path không an toàn.", generated_question, index)
        pointer = location_to_json_pointer(str(patch.get("path") or ""))
        if repair_intent == "align_hint_to_solution" and "/hints/" not in f"{pointer}/":
            return manual_review_result("Patch align_hint_to_solution nằm ngoài hints.", generated_question, index)
        if repair_intent == "clean_solution_reasoning" and not pointer.startswith("/solutions/"):
            return manual_review_result("Patch clean_solution_reasoning nằm ngoài solutions.", generated_question, index)

    try:
        candidate = apply_json_patch(generated_question, patches)
    except JsonPointerError as exc:
        return manual_review_result(str(exc), generated_question, index)
    if candidate == generated_question:
        return manual_review_result("Patch không làm thay đổi generated question.", generated_question, index)

    validation = validate_generated_question_object(candidate, index)
    bad_issue = next(
        (issue for issue in validation.get("issues") or [] if issue.get("severity") == "bad"),
        None,
    )
    if bad_issue:
        return manual_review_result(
            f"Object sau patch không hợp lệ: {bad_issue.get('reason')}",
            generated_question,
            index,
        )
    return {
        "id": generated_question_id(generated_question, index),
        "repair_status": "repaired",
        "failed_reason": [],
        "suggestions": [],
        "new_generated_question": candidate,
        "patches": patches,
    }


def repair_generated_question_scoped(
    generated_question: dict[str, Any],
    check_result: dict[str, Any],
    issue: dict[str, Any],
    config: AppConfig,
    client: LLMClient,
    *,
    index: int = 0,
    debug: bool = False,
) -> dict[str, Any]:
    context = build_scoped_repair_context(generated_question, issue, check_result)
    if not context.get("context_ok"):
        return manual_review_result(str(context.get("reason") or "Scoped context không an toàn."), generated_question, index)
    messages = build_generated_question_scoped_repair_messages(
        context["payload"],
        load_text(REPAIR_RULES_PATH),
        load_text(SCOPED_REPAIR_OUTPUT_SCHEMA_PATH),
    )
    try:
        debug_llm_messages(
            step="generated_question_scoped_repair",
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
            return manual_review_result(parse_error or "Không parse được scoped repair output.", generated_question, index)
        return normalize_scoped_repair_result(
            parsed,
            generated_question=generated_question,
            index=index,
            repair_intent=str(issue.get("repair_intent") or ""),
        )
    except Exception as exc:
        return manual_review_result(str(exc), generated_question, index)

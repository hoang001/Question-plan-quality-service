"""Scoped-only repair for generated question objects."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from ..infra.config import AppConfig, generated_question_reasoning_model
from ..infra.debug import debug_llm_messages
from ..infra.llm_client import LLMClient
from ..shared.utils import parse_json_output
from ..schemas.generated_question_contracts import (
    RepairPatchOutput,
    ScopedRepairOutput,
    contract_schema_text,
    validation_error_text,
)
from ..utils.json_pointer import JsonPointerError, apply_json_patch, get_by_json_pointer
from .generated_question_schema import (
    default_context_paths,
    generated_question_id,
    location_to_json_pointer,
    validate_generated_question_object,
)

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
    allowed_paths: list[str] | None = None,
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
        "allowed_paths": allowed_paths or [],
    }
    if len(json.dumps(payload, ensure_ascii=False)) > max_context_chars:
        return {"context_ok": False, "reason": "Scoped repair context vượt giới hạn an toàn."}
    return {"context_ok": True, "payload": payload}


GENERATED_QUESTION_REPAIR_RULES = """# Quy tắc Repair

- Chỉ trả JSON Patch, không trả toàn bộ generated question và không trả verdict/status.
- Chỉ dùng `op=replace`; mỗi patch phải có `path`, `value`, `reason`.
- `path` bắt buộc thuộc `allowed_paths`. Không tạo, xóa hoặc đổi thứ tự field/block.
- `fix_solution_correctness`: sửa lỗi toán học trong solution theo issue đã được Review xác nhận.
- `fix_solution_process`: chỉ bổ sung hoặc làm rõ solution, không đổi kết quả toán.
- `align_hint_to_solution`: chỉ sửa hint được chỉ định.
- Không sửa ID, instruction, stem, interaction, option, metadata hoặc field ngoài phạm vi.
- Không bịa dữ kiện, hình ảnh, bảng hoặc giả thiết còn thiếu.
- `reason` phải ngắn gọn, bằng tiếng Việt và giải thích đúng thay đổi của patch."""

def build_generated_question_scoped_repair_messages(
    scoped_payload: dict[str, Any],
) -> list[dict[str, str]]:
    output_schema_text = contract_schema_text(RepairPatchOutput)
    return [
        {
            "role": "system",
            "content": (
                "Bạn sửa đúng một issue đã được Review xác nhận bằng JSON Patch. "
                "Chỉ trả JSON hợp lệ theo schema, không viết nội dung ngoài JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"QUY TẮC SỬA:\n{GENERATED_QUESTION_REPAIR_RULES}\n\n"
                f"LƯỢC ĐỒ ĐẦU RA:\n{output_schema_text}\n\n"
                f"DỮ LIỆU TRONG PHẠM VI:\n{json.dumps(scoped_payload, ensure_ascii=False, indent=2)}"
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
    allowed_paths: list[str] | None = None,
) -> dict[str, Any]:
    try:
        parsed = ScopedRepairOutput.model_validate(parsed).model_dump()
    except ValidationError as exc:
        return manual_review_result(
            "Bộ sửa chữa có giới hạn trả kết quả không đúng cấu trúc: " + validation_error_text(exc),
            generated_question,
            index,
        )
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
        if patch.get("op") != "replace":
            return manual_review_result("Patch operation không hợp lệ.", generated_question, index)
        pointer = location_to_json_pointer(str(patch.get("path") or ""))
        if allowed_paths is not None and pointer not in allowed_paths:
            return manual_review_result("Patch path nằm ngoài allowed_paths.", generated_question, index)
        if allowed_paths is None and not is_safe_generated_question_patch_path(pointer):
            return manual_review_result("Patch path không an toàn.", generated_question, index)
        if repair_intent == "align_hint_to_solution" and "/hints/" not in f"{pointer}/":
            return manual_review_result("Patch align_hint_to_solution nằm ngoài hints.", generated_question, index)
        if repair_intent in {"fix_solution_correctness", "fix_solution_process"} and not pointer.startswith("/solutions/"):
            return manual_review_result("Patch sửa solution nằm ngoài solutions.", generated_question, index)
        try:
            current = get_by_json_pointer(generated_question, pointer)
        except JsonPointerError as exc:
            return manual_review_result(str(exc), generated_question, index)
        replacement = patch.get("value")
        if isinstance(current, bool):
            same_type = isinstance(replacement, bool)
        elif isinstance(current, (int, float)):
            same_type = isinstance(replacement, (int, float)) and not isinstance(replacement, bool)
        else:
            same_type = isinstance(replacement, type(current))
        if not same_type:
            return manual_review_result("Patch value không cùng kiểu với giá trị hiện tại.", generated_question, index)

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
    allowed_paths: list[str],
    index: int = 0,
    debug: bool = False,
) -> dict[str, Any]:
    if not allowed_paths:
        return manual_review_result("Issue không có allowed_paths an toàn.", generated_question, index)
    context = build_scoped_repair_context(
        generated_question,
        issue,
        check_result,
        allowed_paths=allowed_paths,
    )
    if not context.get("context_ok"):
        return manual_review_result(str(context.get("reason") or "Scoped context không an toàn."), generated_question, index)
    messages = build_generated_question_scoped_repair_messages(
        context["payload"],
    )
    model = generated_question_reasoning_model(config)
    try:
        debug_llm_messages(
            step="generated_question_scoped_repair",
            model=model,
            messages=messages,
            debug=debug,
        )
        response = client.chat_completion(
            model=model,
            messages=messages,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "generated_question_repair_patch",
                    "strict": True,
                    "schema": RepairPatchOutput.model_json_schema(),
                },
            },
            max_tokens=2048,
        )
        parsed, ok, parse_error = parse_json_output(str(response.get("content") or ""))
        if not ok:
            return manual_review_result(parse_error or "Không parse được scoped repair output.", generated_question, index)
        try:
            patch_output = RepairPatchOutput.model_validate(parsed).model_dump()
        except ValidationError as exc:
            return manual_review_result(
                "Repair patch không đúng contract: " + validation_error_text(exc),
                generated_question,
                index,
            )
        return normalize_scoped_repair_result(
            {
                "repair_status": "repaired",
                "patches": patch_output["patches"],
                "failed_reason": [],
                "suggestions": [],
            },
            generated_question=generated_question,
            index=index,
            repair_intent=str(issue.get("repair_intent") or ""),
            allowed_paths=allowed_paths,
        )
    except Exception as exc:
        return manual_review_result(str(exc), generated_question, index)

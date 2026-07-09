"""LLM repair suggestions for question_plan quality issues.

This module only creates reviewer-facing suggestions and optional rewritten
question_plan previews. It never mutates input records.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .config import AppConfig
from .interaction_type_knowledge import compact_interaction_knowledge
from .llm_client import LLMClient
from .question_plan_eval_knowledge import QUESTION_PLAN_EVAL_KNOWLEDGE
from .real_schema import VALID_REAL_INTERACTION_TYPES, content_to_text
from .utils import parse_json_output


REPAIR_TRIGGER_STATUSES = {"warning", "needs_review", "bad", "structural_error"}
SKIP_REPAIR_STATUSES = {"ok", "skipped_due_to_source_issue"}
REPAIR_CONFIDENCES = {"high", "medium", "low"}
PREVIEW_STRATEGIES = {"full_object", "not_safe"}
AMBIGUOUS_PREVIEW_PHRASES = (
    "hoặc",
    "có thể",
    "cân nhắc",
    "nếu muốn",
    "tùy",
    "nên xem xét",
)
DISALLOWED_PREVIEW_FIELDS = {"suggested_change", "before", "after", "patch_preview"}
DISALLOWED_REPAIR_SUGGESTION_FIELDS = {"before", "after", "patch_preview", "suggested_change"}
REPAIR_ACTION_CODES = {
    "add_missing_item",
    "add_missing_interaction",
    "revise_question_statement",
    "revise_item_requirement",
    "revise_interaction_requirement",
    "change_interaction_type",
    "split_item",
    "split_interaction",
    "adapt_unsupported_format",
    "reorder_items",
    "mark_for_human_review",
    "no_auto_fix",
}
REQUIRED_REPAIR_RESULT_FIELDS = {
    "repair_suggestions",
    "rewritten_question_plan_preview",
    "preview_strategy",
    "repair_confidence",
    "manual_review_required",
}
REQUIRED_REPAIR_SUGGESTION_FIELDS = {
    "issue_id",
    "action_code",
    "location",
    "problem_summary",
    "why_it_matters",
    "specific_change",
    "primary_decision",
    "reasoning",
    "affects_generation",
    "requires_human_review",
}


def truncate_text(value: Any, limit: int = 5000) -> str:
    text = content_to_text(value) if not isinstance(value, str) else value
    text = str(text or "").strip()
    return text[: limit - 3] + "..." if len(text) > limit else text


def compact_plan_eval_result(plan_eval_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": plan_eval_result.get("record_id"),
        "record_name": plan_eval_result.get("record_name"),
        "overall_status": plan_eval_result.get("overall_status"),
        "coverage_status": plan_eval_result.get("coverage_status"),
        "source_to_plan_mapping": plan_eval_result.get("source_to_plan_mapping") or [],
        "source_subparts": plan_eval_result.get("source_subparts") or [],
        "covered_subparts": plan_eval_result.get("covered_subparts") or [],
        "missing_subparts": plan_eval_result.get("missing_subparts") or [],
        "selected_scope_summary": plan_eval_result.get("selected_scope_summary"),
        "plan_quality_summary": plan_eval_result.get("plan_quality_summary"),
        "issues": plan_eval_result.get("issues") or [],
        "recommended_actions": plan_eval_result.get("recommended_actions") or [],
        "confidence": plan_eval_result.get("confidence"),
        "structural_result": plan_eval_result.get("structural_result") or {},
    }


def build_repair_payload(record: dict[str, Any], plan_eval_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record.get("_id") or record.get("record_id") or record.get("id") or "",
        "record_name": record.get("name") or record.get("record_name") or "",
        "raw_question": truncate_text(record.get("question"), 6000),
        "raw_answer": truncate_text(record.get("answer"), 5000),
        "question_plan": record.get("question_plan"),
        "plan_eval_result": compact_plan_eval_result(plan_eval_result),
        "issues": plan_eval_result.get("issues") or [],
        "interaction_type_knowledge": compact_interaction_knowledge(),
        "question_plan_eval_knowledge": QUESTION_PLAN_EVAL_KNOWLEDGE,
    }


def build_question_plan_repair_messages(
    record: dict[str, Any],
    plan_eval_result: dict[str, Any],
) -> list[dict[str, str]]:
    payload = build_repair_payload(record, plan_eval_result)
    action_values = " | ".join(sorted(REPAIR_ACTION_CODES))
    confidence_values = " | ".join(sorted(REPAIR_CONFIDENCES))
    return [
        {
            "role": "system",
            "content": (
                "Bạn là hệ thống gợi ý sửa question_plan cho reviewer. "
                "Không được sửa dữ liệu gốc, không apply patch, và không thêm field ngoài schema question_plan. "
                "Hãy tạo repair_suggestions và chỉ khi đủ an toàn mới tạo preview là toàn bộ object question_plan sau sửa. "
                "Tất cả giải thích/gợi ý cho người đọc phải viết bằng tiếng Việt có dấu. "
                "Giữ nguyên enum/code value bằng tiếng Anh vì đó là contract dữ liệu."
            ),
        },
        {
            "role": "user",
            "content": (
                "Nhiệm vụ:\n"
                "1. Đọc raw_question, raw_answer, question_plan hiện tại và các issue đánh giá.\n"
                "2. Xác định một quyết định sửa chính cho từng issue.\n"
                "3. Nếu có nhiều cách sửa hợp lý, hãy chọn đúng một quyết định sửa chính phù hợp nhất với raw_question/raw_answer và schema hệ thống.\n"
                "4. Nếu đủ an toàn, trả về toàn bộ question_plan sau sửa trong rewritten_question_plan_preview.\n"
                "5. Preview phải giữ nguyên schema và cấu trúc question_plan ban đầu.\n"
                "6. Preview không được là patch, không được là danh sách before/after, và không được là ghi chú text.\n"
                "7. Preview không được chứa lựa chọn mơ hồ như: hoặc, có thể, cân nhắc, nếu muốn, tùy, nên xem xét.\n"
                "8. Nếu không thể tạo full object an toàn, đặt rewritten_question_plan_preview=null và preview_strategy='not_safe'.\n\n"
                "Quy tắc cho preview full object:\n"
                "- Nếu preview_strategy='full_object', rewritten_question_plan_preview phải là object question_plan đầy đủ.\n"
                "- Preview object phải có type='advanced_question_plan' và plan là list.\n"
                "- Chỉ dùng các field schema hiện có: type, plan, questionOrder, questionStatement, questionItems, itemOrder, requirement, interactions, interactionOrder, interactionType, interactionRequirement.\n"
                "- Không tạo field như suggested_change, before, after, patch_preview, table hoặc bất kỳ field nào ngoài schema.\n"
                "- Không tạo interactionType không hợp lệ.\n"
                "- Chỉ sửa phần thật sự cần thiết; không thêm yêu cầu nằm ngoài source.\n"
                "- Nếu source có bảng, hãy chuyển thành các questionItems/interactions nhỏ hơn.\n"
                "- Nếu hệ/phương trình vô nghiệm, không yêu cầu học sinh nhập giá trị x/y cụ thể; hãy yêu cầu kết luận số nghiệm/tính chất nghiệm bằng interaction phù hợp.\n"
                "- Nếu hệ/phương trình có vô số nghiệm hoặc nghiệm tổng quát, không yêu cầu nhập một cặp x/y cụ thể; hãy yêu cầu biểu diễn nghiệm tổng quát hoặc kết luận vô số nghiệm.\n"
                "- Nếu thiếu coverage, hãy bổ sung item/interaction còn thiếu vào đúng vị trí và giữ order hợp lý.\n\n"
                "Quy tắc dùng source_to_plan_mapping khi sửa:\n"
                "- Nếu source_to_plan_mapping cho thấy một source_subpart có coverage_state='present' hoặc quality_state='present_but_invalid', chỉ được sửa các location đã match trong matched_plan_locations.\n"
                "- Chỉ được thêm questionOrder/item/interaction mới khi coverage_state='absent'.\n"
                "- Không được thêm questionOrder mới cho một source_subpart đã có matched_plan_locations.\n"
                "- Nếu issue là hỏi sai/không chấm được cho phần đã có trong plan, hãy revise requirement/interactionRequirement/interactionType tại location đó thay vì thêm phần mới.\n\n"
                "Schema JSON bắt buộc:\n"
                "{\n"
                '  "repair_suggestions": [\n'
                "    {\n"
                '      "issue_id": "",\n'
                f'      "action_code": "{action_values}",\n'
                '      "location": {"questionOrder": null, "itemOrder": null, "interactionOrder": null},\n'
                '      "problem_summary": "",\n'
                '      "why_it_matters": "",\n'
                '      "specific_change": "",\n'
                '      "primary_decision": "",\n'
                '      "reasoning": "",\n'
                '      "affects_generation": "",\n'
                '      "requires_human_review": true\n'
                "    }\n"
                "  ],\n"
                '  "rewritten_question_plan_preview": null,\n'
                '  "preview_strategy": "full_object|not_safe",\n'
                f'  "repair_confidence": "{confidence_values}",\n'
                '  "manual_review_required": true\n'
                "}\n\n"
                "Quy tắc nội dung repair_suggestions:\n"
                "- problem_summary: mô tả vấn đề thật, không chỉ ghi tên issue_type.\n"
                "- why_it_matters: giải thích vì sao vấn đề ảnh hưởng đến generation/render/chấm điểm.\n"
                "- specific_change: nói rõ cần đổi gì.\n"
                "- primary_decision: chỉ một quyết định sửa cụ thể, không đưa các phương án mơ hồ.\n"
                "- reasoning: lý do toán học/sư phạm.\n"
                "- affects_generation: ảnh hưởng đến generation câu hỏi/options/answerSpec/chấm điểm.\n"
                "- Nếu action_code là no_auto_fix hoặc mark_for_human_review, vẫn phải giải thích lý do.\n\n"
                "PAYLOAD ĐẦU VÀO:\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]

def normalize_location(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    return {
        "questionOrder": value.get("questionOrder"),
        "itemOrder": value.get("itemOrder"),
        "interactionOrder": value.get("interactionOrder"),
    }


def normalize_repair_suggestion(value: Any, index: int) -> dict[str, Any]:
    suggestion = value if isinstance(value, dict) else {}
    action_code = suggestion.get("action_code")
    return {
        "issue_id": truncate_text(suggestion.get("issue_id"), 120) or f"issue_{index}",
        "action_code": action_code if action_code in REPAIR_ACTION_CODES else "mark_for_human_review",
        "location": normalize_location(suggestion.get("location")),
        "problem_summary": truncate_text(suggestion.get("problem_summary"), 1000),
        "why_it_matters": truncate_text(suggestion.get("why_it_matters"), 1200),
        "specific_change": truncate_text(suggestion.get("specific_change"), 1200),
        "primary_decision": truncate_text(suggestion.get("primary_decision"), 1500),
        "reasoning": truncate_text(suggestion.get("reasoning"), 1500),
        "affects_generation": truncate_text(suggestion.get("affects_generation"), 1200),
        "requires_human_review": suggestion.get("requires_human_review") if isinstance(suggestion.get("requires_human_review"), bool) else True,
    }


def contains_ambiguous_repair_phrase(value: Any) -> str:
    text = truncate_text(value, 2000).lower()
    for phrase in AMBIGUOUS_PREVIEW_PHRASES:
        if phrase in text:
            return phrase
    return ""


def validate_question_plan_preview(question_plan: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(question_plan, dict):
        return ["rewritten_question_plan_preview phải là object."]
    for key in question_plan:
        if key not in {"type", "plan"}:
            errors.append(f"Field lạ ở question_plan root: {key}.")
    if question_plan.get("type") != "advanced_question_plan":
        errors.append("question_plan.type phải là advanced_question_plan.")
    plan = question_plan.get("plan")
    if not isinstance(plan, list):
        errors.append("question_plan.plan phải là list.")
        return errors

    def scan_unknown_fields(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in DISALLOWED_PREVIEW_FIELDS:
                    errors.append(f"Field preview không hợp lệ tại {path or '$'}: {key}.")
                scan_unknown_fields(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                scan_unknown_fields(child, f"{path}[{index}]")
        elif isinstance(value, str):
            lowered = value.lower()
            for phrase in AMBIGUOUS_PREVIEW_PHRASES:
                if phrase in lowered:
                    errors.append(f"Cụm mơ hồ trong preview tại {path}: {phrase}.")

    scan_unknown_fields(question_plan)

    for question_index, question in enumerate(plan, start=1):
        if not isinstance(question, dict):
            errors.append(f"plan[{question_index}] phải là object.")
            continue
        for key in question:
            if key not in {"questionOrder", "questionStatement", "questionItems"}:
                errors.append(f"Field lạ ở plan[{question_index}]: {key}.")
        items = question.get("questionItems")
        if not isinstance(items, list):
            errors.append(f"plan[{question_index}].questionItems phải là list.")
            continue
        for item_index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                errors.append(f"plan[{question_index}].questionItems[{item_index}] phải là object.")
                continue
            for key in item:
                if key not in {"itemOrder", "requirement", "interactions"}:
                    errors.append(f"Field lạ ở questionItems[{item_index}]: {key}.")
            interactions = item.get("interactions")
            if not isinstance(interactions, list):
                errors.append(f"questionItems[{item_index}].interactions phải là list.")
                continue
            for interaction_index, interaction in enumerate(interactions, start=1):
                if not isinstance(interaction, dict):
                    errors.append(f"interactions[{interaction_index}] phải là object.")
                    continue
                for key in interaction:
                    if key not in {"interactionOrder", "interactionType", "interactionRequirement"}:
                        errors.append(f"Field lạ ở interactions[{interaction_index}]: {key}.")
                interaction_type = interaction.get("interactionType")
                if interaction_type not in VALID_REAL_INTERACTION_TYPES:
                    errors.append(f"interactionType không hợp lệ: {interaction_type}.")
    return errors


def normalize_repair_result(
    parsed: dict[str, Any],
    *,
    record: dict[str, Any],
    error: str = "",
) -> dict[str, Any]:
    if error:
        return {
            "repair_suggestions": [
                {
                    "issue_id": "repair_error",
                    "action_code": "mark_for_human_review",
                    "location": {"questionOrder": None, "itemOrder": None, "interactionOrder": None},
                    "problem_summary": "Không tạo được gợi ý sửa tự động.",
                    "why_it_matters": "Reviewer cần xem lại vì bước repair LLM lỗi hoặc trả JSON không hợp lệ.",
                    "specific_change": "Review thủ công question_plan và các issue đã phát hiện.",
                    "primary_decision": "Review thủ công, không tạo preview tự động.",
                    "reasoning": error,
                    "affects_generation": "Không thể kết luận tự động hướng sửa cụ thể.",
                    "requires_human_review": True,
                }
            ],
            "rewritten_question_plan_preview": None,
            "preview_strategy": "not_safe",
            "repair_confidence": "low",
            "manual_review_required": True,
            "repair_error": error,
            "preview_validation_error": "",
        }

    suggestions = [
        normalize_repair_suggestion(item, index)
        for index, item in enumerate(parsed.get("repair_suggestions") or [], start=1)
    ]
    raw_preview = parsed.get("rewritten_question_plan_preview")
    preview_strategy = parsed.get("preview_strategy")
    if preview_strategy not in PREVIEW_STRATEGIES:
        preview_strategy = "not_safe"
    preview_validation_errors: list[str] = []
    if preview_strategy == "full_object":
        preview_validation_errors = validate_question_plan_preview(raw_preview)
    elif raw_preview is not None:
        preview_validation_errors = ["preview_strategy là not_safe nhưng preview không phải null."]

    preview_valid = preview_strategy == "full_object" and not preview_validation_errors
    preview = deepcopy(raw_preview) if preview_valid else None
    if not preview_valid:
        preview_strategy = "not_safe"
    confidence = parsed.get("repair_confidence")
    if confidence not in REPAIR_CONFIDENCES:
        confidence = "medium"
    manual_review_required = parsed.get("manual_review_required")
    if not isinstance(manual_review_required, bool):
        manual_review_required = True
    if raw_preview is not None and not preview_valid:
        manual_review_required = True
    if preview is None and not suggestions:
        manual_review_required = True
    return {
        "repair_suggestions": suggestions,
        "rewritten_question_plan_preview": preview,
        "preview_strategy": preview_strategy,
        "repair_confidence": confidence,
        "manual_review_required": manual_review_required,
        "repair_error": "",
        "preview_validation_error": "; ".join(preview_validation_errors),
    }


def parse_and_normalize_repair_response(
    *,
    response: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    parsed, ok, parse_error = parse_json_output(str(response.get("content") or ""))
    errors: list[str] = []
    if not ok or not isinstance(parsed, dict):
        errors.append(parse_error or "Không parse được JSON repair.")
    else:
        missing = sorted(REQUIRED_REPAIR_RESULT_FIELDS - set(parsed))
        if missing:
            errors.append("Thiếu field repair: " + ", ".join(missing))
        if not isinstance(parsed.get("repair_suggestions"), list):
            errors.append("repair_suggestions phải là list.")
        if parsed.get("preview_strategy") not in PREVIEW_STRATEGIES:
            errors.append(f"preview_strategy không hợp lệ: {parsed.get('preview_strategy')}")
        for index, suggestion in enumerate(parsed.get("repair_suggestions") or [], start=1):
            if not isinstance(suggestion, dict):
                errors.append(f"repair_suggestions[{index}] phải là object.")
                continue
            disallowed_fields = sorted(DISALLOWED_REPAIR_SUGGESTION_FIELDS & set(suggestion))
            if disallowed_fields:
                errors.append(
                    f"repair_suggestions[{index}] không được chứa field dạng patch: {', '.join(disallowed_fields)}"
                )
            missing_suggestion = sorted(REQUIRED_REPAIR_SUGGESTION_FIELDS - set(suggestion))
            if missing_suggestion:
                errors.append(f"repair_suggestions[{index}] thiếu field: {', '.join(missing_suggestion)}")
            primary_decision = truncate_text(suggestion.get("primary_decision"), 1500)
            if not primary_decision:
                errors.append(f"repair_suggestions[{index}].primary_decision không được rỗng.")
            ambiguous_phrase = contains_ambiguous_repair_phrase(primary_decision)
            if ambiguous_phrase:
                errors.append(
                    f"repair_suggestions[{index}].primary_decision chứa cụm mơ hồ: {ambiguous_phrase}"
                )
        if parsed.get("repair_confidence") not in REPAIR_CONFIDENCES:
            errors.append(f"repair_confidence không hợp lệ: {parsed.get('repair_confidence')}")

    if errors:
        normalized = normalize_repair_result({}, record=record, error="; ".join(errors))
    else:
        normalized = normalize_repair_result(parsed, record=record)
    normalized.update(
        {
            "repair_json_parse_ok": bool(ok and not errors),
            "repair_raw_response": response.get("raw_response"),
            "repair_latency_seconds": response.get("latency_seconds", 0),
        }
    )
    return normalized


def suggest_question_plan_repair(
    record: dict[str, Any],
    plan_eval_result: dict[str, Any],
    config: AppConfig,
    client: LLMClient,
) -> dict[str, Any]:
    status = plan_eval_result.get("overall_status")
    if status in SKIP_REPAIR_STATUSES or status not in REPAIR_TRIGGER_STATUSES:
        return {
            "repair_suggestions": [],
            "rewritten_question_plan_preview": None,
            "preview_strategy": "not_safe",
            "repair_confidence": "high",
            "manual_review_required": False,
            "repair_error": "",
            "preview_validation_error": "",
            "repair_json_parse_ok": None,
            "repair_model": "",
            "repair_latency_seconds": 0,
        }

    model = config.primary_judge_model
    try:
        response = client.chat_completion(
            model=model,
            messages=build_question_plan_repair_messages(record, plan_eval_result),
            temperature=0,
        )
        result = parse_and_normalize_repair_response(response=response, record=record)
    except Exception as exc:
        result = normalize_repair_result({}, record=record, error=str(exc))
        result.update(
            {
                "repair_json_parse_ok": False,
                "repair_raw_response": None,
                "repair_latency_seconds": 0,
            }
        )
    result["repair_model"] = model
    return result

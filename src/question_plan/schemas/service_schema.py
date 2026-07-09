"""Service-facing schema helpers for question_plan evaluation."""

from __future__ import annotations

from typing import Any

from ...real_schema import VALID_REAL_INTERACTION_TYPES


VALID_INTERACTION_TYPES = VALID_REAL_INTERACTION_TYPES
AMBIGUOUS_PLAN_PHRASES = ("hoặc", "có thể", "cân nhắc", "nếu muốn", "tùy", "nên xem xét")
DISALLOWED_PLAN_FIELDS = {"suggested_change", "before", "after", "patch_preview"}


def validate_question_plan(question_plan: Any) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(question_plan, dict):
        return False, ["question_plan phải là object."]
    for key in question_plan:
        if key not in {"type", "plan"}:
            errors.append(f"Field lạ ở question_plan root: {key}.")
    if question_plan.get("type") != "advanced_question_plan":
        errors.append("question_plan.type phải là advanced_question_plan.")
    plan = question_plan.get("plan")
    if not isinstance(plan, list) or not plan:
        errors.append("question_plan.plan phải là list không rỗng.")
        return False, errors

    def scan_disallowed(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in DISALLOWED_PLAN_FIELDS:
                    errors.append(f"Field dạng patch không hợp lệ tại {path or '$'}: {key}.")
                scan_disallowed(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                scan_disallowed(child, f"{path}[{index}]")
        elif isinstance(value, str):
            lowered = value.lower()
            for phrase in AMBIGUOUS_PLAN_PHRASES:
                if phrase in lowered:
                    errors.append(f"Nội dung tại {path} chứa cụm mơ hồ: {phrase}.")

    scan_disallowed(question_plan)

    for question_index, question in enumerate(plan, start=1):
        if not isinstance(question, dict):
            errors.append(f"plan[{question_index}] phải là object.")
            continue
        for key in question:
            if key not in {"questionOrder", "questionStatement", "questionItems"}:
                errors.append(f"Field lạ ở plan[{question_index}]: {key}.")
        if not str(question.get("questionStatement") or "").strip():
            errors.append(f"plan[{question_index}].questionStatement bị thiếu hoặc rỗng.")
        items = question.get("questionItems")
        if not isinstance(items, list) or not items:
            errors.append(f"plan[{question_index}].questionItems phải là list không rỗng.")
            continue
        for item_index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                errors.append(f"questionItems[{item_index}] phải là object.")
                continue
            for key in item:
                if key not in {"itemOrder", "requirement", "interactions"}:
                    errors.append(f"Field lạ ở questionItems[{item_index}]: {key}.")
            if not str(item.get("requirement") or "").strip():
                errors.append(f"questionItems[{item_index}].requirement bị thiếu hoặc rỗng.")
            interactions = item.get("interactions")
            if not isinstance(interactions, list) or not interactions:
                errors.append(f"questionItems[{item_index}].interactions phải là list không rỗng.")
                continue
            for interaction_index, interaction in enumerate(interactions, start=1):
                if not isinstance(interaction, dict):
                    errors.append(f"interactions[{interaction_index}] phải là object.")
                    continue
                for key in interaction:
                    if key not in {"interactionOrder", "interactionType", "interactionRequirement"}:
                        errors.append(f"Field lạ ở interactions[{interaction_index}]: {key}.")
                interaction_type = interaction.get("interactionType")
                if interaction_type not in VALID_INTERACTION_TYPES:
                    errors.append(f"interactionType không hợp lệ: {interaction_type}.")
                if not str(interaction.get("interactionRequirement") or "").strip():
                    errors.append(f"interactions[{interaction_index}].interactionRequirement bị thiếu hoặc rỗng.")
    return not errors, errors


def normalize_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        value = [value]
    result = []
    for item in value:
        text = str(item or "").strip()
        if text:
            result.append(text)
    return result


def ensure_service_output_shape(result: dict[str, Any]) -> dict[str, Any]:
    is_good = bool(result.get("is_good"))
    if is_good:
        return {
            "is_good": True,
            "failed_reason": [],
            "suggestions": [],
            "new_question_plan": None,
        }

    failed_reason = normalize_text_list(result.get("failed_reason"))
    suggestions = normalize_text_list(result.get("suggestions"))
    new_question_plan = result.get("new_question_plan")
    valid_plan, _errors = validate_question_plan(new_question_plan)
    if not valid_plan:
        new_question_plan = None
    if not failed_reason:
        failed_reason = ["Question plan chưa đạt chất lượng nhưng judge không nêu lý do cụ thể."]
    return {
        "is_good": False,
        "failed_reason": failed_reason,
        "suggestions": suggestions,
        "new_question_plan": new_question_plan,
    }


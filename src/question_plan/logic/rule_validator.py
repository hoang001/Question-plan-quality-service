"""Validator cấu trúc cho kiểm tra chất lượng question_plan ở cấp source record.

Module này không đánh giá coverage, độ đúng toán học hoặc semantic của
interactionType. Những phần đó thuộc trách nhiệm của LLM judge.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ...real_schema import VALID_REAL_INTERACTION_TYPES, content_to_text


def has_text(value: Any) -> bool:
    return bool(content_to_text(value).strip())


def get_alias(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return default


def get_plan_entries(question_plan: Any) -> list[Any] | None:
    if isinstance(question_plan, list):
        return question_plan
    if isinstance(question_plan, dict):
        plan = question_plan.get("plan")
        return plan if isinstance(plan, list) else None
    return None


def path_join(*parts: Any) -> str:
    return ".".join(str(part) for part in parts if str(part) != "")


def add_issue(
    issues: list[dict[str, Any]],
    *,
    error_type: str,
    field: str,
    message: str,
) -> None:
    issues.append(
        {
            "error_type": error_type,
            "field": field,
            "message": message,
            "severity": "structural_error",
        }
    )


def valid_order(value: Any) -> bool:
    return value is None or (isinstance(value, int) and value >= 1)


def validate_unique_orders(
    issues: list[dict[str, Any]],
    values: list[Any],
    *,
    field: str,
) -> None:
    seen = Counter(value for value in values if value is not None)
    duplicates = [value for value, count in seen.items() if count > 1]
    if duplicates:
        add_issue(
            issues,
            error_type="duplicate_order",
            field=field,
            message=f"Order field bị trùng: {duplicates}",
        )


def validate_question_plan_structure(record: dict[str, Any], record_index: int = 0) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    record_id = record.get("_id") or record.get("id") or f"record_{record_index + 1}"

    if not isinstance(record, dict):
        return {
            "record_id": "",
            "record_name": "",
            "structural_valid": False,
            "rule_validation_status": "structural_error",
            "structural_error_type": "invalid_record_shape",
            "issues": [
                {
                    "error_type": "invalid_record_shape",
                    "field": "record",
                    "message": "Source record phải là object.",
                    "severity": "structural_error",
                }
            ],
            "reason": "Source record phải là object.",
            "evidence": "",
            "llm_judge_required": False,
        }

    if not has_text(record.get("_id") or record.get("id")):
        add_issue(
            issues,
            error_type="missing_record_id",
            field="_id",
            message="Thiếu `_id` hoặc `id` của source record.",
        )
    if not has_text(record.get("question")):
        add_issue(
            issues,
            error_type="missing_raw_question",
            field="question",
            message="Thiếu raw question để đối chiếu question_plan.",
        )

    question_plan = record.get("question_plan")
    if question_plan is None:
        add_issue(
            issues,
            error_type="missing_question_plan",
            field="question_plan",
            message="Thiếu question_plan.",
        )
    elif not isinstance(question_plan, (dict, list)):
        add_issue(
            issues,
            error_type="invalid_question_plan_shape",
            field="question_plan",
            message="question_plan phải là object wrapper hoặc list PlanQuestion.",
        )

    plans = get_plan_entries(question_plan)
    if question_plan is not None and plans is None:
        add_issue(
            issues,
            error_type="missing_or_invalid_plan_list",
            field="question_plan.plan",
            message="question_plan phải là list hoặc object có field `plan` là list.",
        )
    elif isinstance(plans, list) and not plans:
        add_issue(
            issues,
            error_type="empty_plan_list",
            field="question_plan.plan",
            message="question_plan.plan không được rỗng.",
        )

    if isinstance(plans, list):
        question_orders = []
        for plan_index, plan in enumerate(plans, start=1):
            plan_path = f"plan[{plan_index - 1}]"
            if not isinstance(plan, dict):
                add_issue(
                    issues,
                    error_type="invalid_plan_item_shape",
                    field=plan_path,
                    message="Mỗi phần tử trong plan phải là object.",
                )
                continue

            question_order = get_alias(plan, "questionOrder", "question_order")
            question_orders.append(question_order)
            if not valid_order(question_order):
                add_issue(
                    issues,
                    error_type="malformed_order",
                    field=path_join(plan_path, "questionOrder"),
                    message="questionOrder nếu có phải là số nguyên >= 1.",
                )
            if not has_text(get_alias(plan, "questionStatement", "question_statement")):
                add_issue(
                    issues,
                    error_type="missing_question_statement",
                    field=path_join(plan_path, "questionStatement"),
                    message="Mỗi PlanQuestion cần có questionStatement không rỗng.",
                )

            items = get_alias(plan, "questionItems", "question_items")
            if not isinstance(items, list):
                add_issue(
                    issues,
                    error_type="invalid_question_items_shape",
                    field=path_join(plan_path, "questionItems"),
                    message="questionItems phải là list.",
                )
                continue
            if not items:
                add_issue(
                    issues,
                    error_type="empty_question_items",
                    field=path_join(plan_path, "questionItems"),
                    message="questionItems không được rỗng.",
                )
                continue

            item_orders = []
            for item_index, item in enumerate(items, start=1):
                item_path = path_join(plan_path, f"questionItems[{item_index - 1}]")
                if not isinstance(item, dict):
                    add_issue(
                        issues,
                        error_type="invalid_question_item_shape",
                        field=item_path,
                        message="Mỗi questionItem phải là object.",
                    )
                    continue

                item_order = get_alias(item, "itemOrder", "item_order")
                item_orders.append(item_order)
                if not valid_order(item_order):
                    add_issue(
                        issues,
                        error_type="malformed_order",
                        field=path_join(item_path, "itemOrder"),
                        message="itemOrder nếu có phải là số nguyên >= 1.",
                    )
                if not has_text(item.get("requirement")):
                    add_issue(
                        issues,
                        error_type="missing_item_requirement",
                        field=path_join(item_path, "requirement"),
                        message="Mỗi questionItem cần có requirement không rỗng.",
                    )

                interactions = item.get("interactions")
                if not isinstance(interactions, list):
                    add_issue(
                        issues,
                        error_type="invalid_interactions_shape",
                        field=path_join(item_path, "interactions"),
                        message="interactions phải là list.",
                    )
                    continue
                if not interactions:
                    add_issue(
                        issues,
                        error_type="empty_interactions",
                        field=path_join(item_path, "interactions"),
                        message="interactions không được rỗng.",
                    )
                    continue

                interaction_orders = []
                for interaction_index, interaction in enumerate(interactions, start=1):
                    interaction_path = path_join(item_path, f"interactions[{interaction_index - 1}]")
                    if not isinstance(interaction, dict):
                        add_issue(
                            issues,
                            error_type="invalid_interaction_shape",
                            field=interaction_path,
                            message="Mỗi interaction phải là object.",
                        )
                        continue

                    interaction_order = get_alias(interaction, "interactionOrder", "interaction_order")
                    interaction_orders.append(interaction_order)
                    if not valid_order(interaction_order):
                        add_issue(
                            issues,
                            error_type="malformed_order",
                            field=path_join(interaction_path, "interactionOrder"),
                            message="interactionOrder nếu có phải là số nguyên >= 1.",
                        )

                    interaction_type = get_alias(interaction, "interactionType", "interaction_type")
                    if not has_text(interaction_type):
                        add_issue(
                            issues,
                            error_type="missing_interaction_type",
                            field=path_join(interaction_path, "interactionType"),
                            message="Mỗi interaction cần có interactionType.",
                        )
                    elif interaction_type not in VALID_REAL_INTERACTION_TYPES:
                        add_issue(
                            issues,
                            error_type="invalid_interaction_type",
                            field=path_join(interaction_path, "interactionType"),
                            message=f"interactionType không hợp lệ: {interaction_type}",
                        )

                    if not has_text(get_alias(interaction, "interactionRequirement", "interaction_requirement")):
                        add_issue(
                            issues,
                            error_type="missing_interaction_requirement",
                            field=path_join(interaction_path, "interactionRequirement"),
                            message="Mỗi interaction cần có interactionRequirement không rỗng.",
                        )

                validate_unique_orders(
                    issues,
                    interaction_orders,
                    field=path_join(item_path, "interactions.interactionOrder"),
                )

            validate_unique_orders(issues, item_orders, field=path_join(plan_path, "questionItems.itemOrder"))

        validate_unique_orders(issues, question_orders, field="question_plan.plan.questionOrder")

    structural_valid = not issues
    return {
        "record_id": record_id,
        "record_name": record.get("name", ""),
        "structural_valid": structural_valid,
        "rule_validation_status": "structural_ok" if structural_valid else "structural_error",
        "structural_error_type": (issues[0].get("error_type") if issues else ""),
        "issues": issues,
        "reason": "; ".join(issue.get("message", "") for issue in issues[:5]),
        "evidence": "; ".join(f"{issue.get('field')}: {issue.get('message')}" for issue in issues[:8]),
        "llm_judge_required": structural_valid,
    }


def inspect_question_plan_quality(records: list[dict[str, Any]]) -> dict[str, Any]:
    results = [validate_question_plan_structure(record, index) for index, record in enumerate(records)]
    status_counts = Counter(row.get("rule_validation_status") for row in results)
    error_counts = Counter(
        issue.get("error_type")
        for row in results
        for issue in row.get("issues") or []
    )
    return {
        "total_records": len(records),
        "structural_ok_count": status_counts.get("structural_ok", 0),
        "structural_error_count": status_counts.get("structural_error", 0),
        "count_by_rule_validation_status": dict(status_counts),
        "count_by_structural_error_type": dict(error_counts),
        "known_interaction_types": sorted(VALID_REAL_INTERACTION_TYPES),
        "results": results,
    }



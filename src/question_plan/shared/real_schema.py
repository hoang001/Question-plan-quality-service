"""Schema/helper chung cho data thật.

File này định nghĩa danh sách interaction type hợp lệ và các hàm đọc content,
choices, planned type để những pipeline khác dùng cùng một cách hiểu schema.
"""

from __future__ import annotations

from typing import Any


VALID_REAL_INTERACTION_TYPES = {
    "single_choice",
    "multiple_choice",
    "true_false",
    "true_false_multi_statement",
    "fill_blank",
    "short_answer",
    "matching",
    "ordering",
    "drag_drop",
    "number_line_range",
    "image_hotspot",
    "coloring_select",
    "column_arithmetic",
    "choice_blank_fill",
    "expression_transformation_step",
    "operation_chain",
    "chart_draw",
    "coordinate_input",
    "graph_draw",
}


CHOICE_KEYS = ("options", "choices", "items", "statements", "answers")


def content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(part for part in (content_to_text(item) for item in value) if part).strip()
    if isinstance(value, dict):
        parts = []
        for key in ("text", "label", "value", "name", "title"):
            if key in value:
                text = content_to_text(value.get(key))
                if text:
                    parts.append(text)
        if "content" in value:
            text = content_to_text(value.get("content"))
            if text:
                parts.append(text)
        return " ".join(parts).strip()
    return ""


def extract_choices(interaction: dict[str, Any]) -> list[Any]:
    config = interaction.get("config") if isinstance(interaction.get("config"), dict) else {}
    for container in (config, interaction):
        for key in CHOICE_KEYS:
            value = container.get(key)
            if isinstance(value, list):
                return value
    return []


def find_planned_interaction_type(
    question_plan: dict[str, Any] | None,
    item_index: int,
    interaction_index: int,
) -> str | None:
    if not isinstance(question_plan, dict):
        return None
    plan_items = []
    for plan in question_plan.get("plan") or []:
        if isinstance(plan, dict):
            plan_items.extend(item for item in plan.get("questionItems") or [] if isinstance(item, dict))
    if item_index >= len(plan_items):
        return None
    interactions = plan_items[item_index].get("interactions") or []
    if interaction_index >= len(interactions):
        return None
    interaction = interactions[interaction_index]
    if not isinstance(interaction, dict):
        return None
    value = interaction.get("interactionType")
    return str(value) if value else None


def compact_case_text(case: dict[str, Any]) -> str:
    return "\n".join(
        part
        for part in [
            content_to_text(case.get("instruction")),
            content_to_text(case.get("stem")),
        ]
        if part
    ).strip()

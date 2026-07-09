"""Service-facing question_plan repair wrapper.

This module returns only service-friendly repair data. It does not mutate input
records and does not create report/replacement files.
"""

from __future__ import annotations

from typing import Any

from .config import AppConfig
from .llm_client import LLMClient
from .question_plan_repair_suggester import suggest_question_plan_repair
from .question_plan_schema import validate_question_plan


def suggestion_texts(repair_result: dict[str, Any]) -> list[str]:
    suggestions: list[str] = []
    for item in repair_result.get("repair_suggestions") or []:
        if not isinstance(item, dict):
            continue
        for field in ("specific_change", "primary_decision", "problem_summary"):
            text = str(item.get(field) or "").strip()
            if text and text not in suggestions:
                suggestions.append(text)
    return suggestions


def repair_question_plan(
    record: dict[str, Any],
    plan_eval_result: dict[str, Any],
    config: AppConfig,
    client: LLMClient,
) -> dict[str, Any]:
    repair_result = suggest_question_plan_repair(record, plan_eval_result, config, client)
    new_question_plan = repair_result.get("rewritten_question_plan_preview")
    valid_plan, validation_errors = validate_question_plan(new_question_plan)
    suggestions = suggestion_texts(repair_result)
    if not valid_plan:
        new_question_plan = None
        if validation_errors:
            suggestions.append("Không tạo được question_plan sau sửa đủ an toàn, cần review thủ công.")
    return {
        "suggestions": suggestions,
        "new_question_plan": new_question_plan,
        "raw_repair_result": repair_result,
    }

"""API service gọn để đánh giá question_plan của source record."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..infra.config import AppConfig, load_config
from ..infra.llm_client import LLMClient
from ..schemas.eval_schema import structural_error_result
from ..logic.judge import judge_question_plan
from ..logic.repair import repair_question_plan
from ..logic.rule_validator import validate_question_plan_structure
from ..schemas.service_schema import ensure_service_output_shape


SERVICE_ROOT_DIR = Path(__file__).resolve().parents[2]


def default_config() -> AppConfig:
    return load_config(SERVICE_ROOT_DIR)


def issue_failed_reasons(plan_eval_result: dict[str, Any]) -> list[str]:
    reasons = []
    for issue in plan_eval_result.get("issues") or []:
        if not isinstance(issue, dict):
            continue
        text = str(issue.get("summary") or issue.get("evidence") or issue.get("issue_type") or "").strip()
        if text and text not in reasons:
            reasons.append(text)
    if not reasons:
        summary = str(plan_eval_result.get("plan_quality_summary") or "").strip()
        if summary:
            reasons.append(summary)
    return reasons


def issue_suggestions(plan_eval_result: dict[str, Any]) -> list[str]:
    suggestions = []
    for issue in plan_eval_result.get("issues") or []:
        if not isinstance(issue, dict):
            continue
        text = str(issue.get("suggested_fix") or issue.get("recommended_action") or "").strip()
        if text and text not in suggestions:
            suggestions.append(text)
    return suggestions


def evaluate_question_plan(
    record: dict[str, Any],
    *,
    config: AppConfig | None = None,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    config = config or default_config()
    client = client or LLMClient(config)

    structural = validate_question_plan_structure(record, 0)
    if not structural.get("llm_judge_required"):
        result = structural_error_result(record, structural)
        return ensure_service_output_shape(
            {
                "is_good": False,
                "failed_reason": issue_failed_reasons(result),
                "suggestions": issue_suggestions(result),
                "new_question_plan": None,
            }
        )

    plan_eval_result = judge_question_plan(record, config, client, structural)
    is_good = plan_eval_result.get("overall_status") == "ok"
    if is_good:
        return ensure_service_output_shape(
            {
                "is_good": True,
                "failed_reason": [],
                "suggestions": [],
                "new_question_plan": None,
            }
        )

    failed_reason = issue_failed_reasons(plan_eval_result)
    suggestions = issue_suggestions(plan_eval_result)
    repair_result = repair_question_plan(record, plan_eval_result, config, client)
    for suggestion in repair_result.get("suggestions") or []:
        if suggestion not in suggestions:
            suggestions.append(suggestion)

    return ensure_service_output_shape(
        {
            "is_good": False,
            "failed_reason": failed_reason,
            "suggestions": suggestions,
            "new_question_plan": repair_result.get("new_question_plan"),
        }
    )


def evaluate_question_plans(
    records: list[dict[str, Any]],
    *,
    config: AppConfig | None = None,
    client: LLMClient | None = None,
) -> list[dict[str, Any]]:
    config = config or default_config()
    client = client or LLMClient(config)
    return [evaluate_question_plan(record, config=config, client=client) for record in records]



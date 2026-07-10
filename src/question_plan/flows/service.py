"""API service gọn để đánh giá question_plan của source record."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..infra.config import AppConfig, load_config
from ..infra.debug import debug_loop_event
from ..infra.llm_client import LLMClient
from ..schemas.eval_schema import structural_error_result
from ..logic.judge import judge_question_plan
from ..logic.repair import repair_question_plan
from ..logic.rule_validator import validate_question_plan_structure
from ..schemas.service_schema import ensure_service_output_shape, validate_question_plan


SERVICE_ROOT_DIR = Path(__file__).resolve().parents[3]


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


def clamp_max_loop(max_loop: int | None) -> int:
    try:
        value = int(max_loop) if max_loop is not None else 3
    except (TypeError, ValueError):
        value = 3
    return max(1, min(value, 3))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def merge_unique_texts(*items: Any) -> list[str]:
    merged: list[str] = []
    for item in items:
        if item is None:
            continue
        values = item if isinstance(item, list) else [item]
        for value in values:
            text = str(value or "").strip()
            if text and text not in merged:
                merged.append(text)
    return merged


def valid_candidate_plan(value: Any) -> tuple[dict[str, Any] | None, list[str]]:
    valid, errors = validate_question_plan(value)
    if not valid or not isinstance(value, dict):
        return None, errors
    return deepcopy(value), errors


def fail_closed_result(
    *,
    latest_valid_candidate: dict[str, Any] | None,
    is_loop: bool = False,
    loop_count: int = 0,
    reason: str = "Không đánh giá hoặc refine được question_plan do lỗi runtime/LLM output không hợp lệ.",
    suggestion: str = "Cần kiểm tra thủ công hoặc chạy lại service.",
) -> dict[str, Any]:
    return ensure_service_output_shape(
        {
            "is_good": False,
            "failed_reason": [reason],
            "suggestions": [suggestion],
            "new_question_plan": latest_valid_candidate,
            "is_loop": is_loop,
            "loop_count": loop_count,
        }
    )


def _evaluate_question_plan_once(
    record: dict[str, Any],
    *,
    config: AppConfig,
    client: LLMClient,
    debug: bool = False,
) -> dict[str, Any]:
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

    plan_eval_result = judge_question_plan(record, config, client, structural, debug=debug)
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
    repair_result = repair_question_plan(record, plan_eval_result, config, client, debug=debug)
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


def evaluate_question_plan(
    record: dict[str, Any],
    *,
    config: AppConfig | None = None,
    client: LLMClient | None = None,
    debug: bool = False,
    is_loop: bool = False,
    max_loop: int = 3,
) -> dict[str, Any]:
    config = config or default_config()
    client = client or LLMClient(config)
    if not is_loop:
        return _evaluate_question_plan_once(record, config=config, client=client, debug=debug)
    return evaluate_question_plan_with_loop(
        record,
        config=config,
        client=client,
        debug=debug,
        max_loop=max_loop,
    )


def evaluate_question_plan_with_loop(
    record: dict[str, Any],
    *,
    config: AppConfig,
    client: LLMClient,
    debug: bool = False,
    max_loop: int = 3,
) -> dict[str, Any]:
    loop_limit = clamp_max_loop(max_loop)
    current_record = deepcopy(record)
    latest_valid_candidate: dict[str, Any] | None = None
    completed_loop_count = 0

    debug_loop_event(
        event="loop_start",
        debug=debug,
        is_loop=True,
        max_loop=loop_limit,
        loop_index=0,
        current_status="start",
        has_candidate=False,
        candidate_valid=False,
        stop_reason="",
    )

    try:
        original_result = _evaluate_question_plan_once(current_record, config=config, client=client, debug=debug)
    except Exception:
        return fail_closed_result(latest_valid_candidate=latest_valid_candidate, is_loop=True, loop_count=0)

    if original_result.get("is_good") is True:
        debug_loop_event(
            event="loop_stop",
            debug=debug,
            is_loop=True,
            max_loop=loop_limit,
            loop_index=0,
            current_status="good",
            has_candidate=False,
            candidate_valid=False,
            stop_reason="initial_plan_good",
        )
        return ensure_service_output_shape({**original_result, "is_loop": True, "loop_count": 0})

    current_result = original_result
    original_failed_reason = list(original_result.get("failed_reason") or [])
    original_suggestions = list(original_result.get("suggestions") or [])

    for loop_index in range(1, loop_limit + 1):
        candidate_raw = current_result.get("new_question_plan")
        candidate_plan, validation_errors = valid_candidate_plan(candidate_raw)
        has_candidate = candidate_raw is not None
        candidate_valid = candidate_plan is not None
        current_status = "good" if current_result.get("is_good") else "not_good"

        debug_loop_event(
            event="loop_iteration_candidate",
            debug=debug,
            is_loop=True,
            max_loop=loop_limit,
            loop_index=loop_index,
            current_status=current_status,
            has_candidate=has_candidate,
            candidate_valid=candidate_valid,
            stop_reason="",
        )

        if candidate_plan is None:
            suggestions = merge_unique_texts(
                current_result.get("suggestions"),
                "Không tạo được question_plan sau sửa đủ hợp lệ để tiếp tục loop.",
                "; ".join(validation_errors),
            )
            debug_loop_event(
                event="loop_stop",
                debug=debug,
                is_loop=True,
                max_loop=loop_limit,
                loop_index=loop_index,
                current_status=current_status,
                has_candidate=has_candidate,
                candidate_valid=False,
                stop_reason="candidate_invalid_or_null",
            )
            return ensure_service_output_shape(
                {
                    "is_good": False,
                    "failed_reason": current_result.get("failed_reason"),
                    "suggestions": suggestions,
                    "new_question_plan": latest_valid_candidate,
                    "is_loop": True,
                    "loop_count": completed_loop_count,
                }
            )

        current_plan = current_record.get("question_plan")
        if canonical_json(candidate_plan) == canonical_json(current_plan):
            debug_loop_event(
                event="loop_stop",
                debug=debug,
                is_loop=True,
                max_loop=loop_limit,
                loop_index=loop_index,
                current_status=current_status,
                has_candidate=True,
                candidate_valid=True,
                stop_reason="candidate_unchanged",
            )
            return ensure_service_output_shape(
                {
                    "is_good": False,
                    "failed_reason": current_result.get("failed_reason"),
                    "suggestions": merge_unique_texts(
                        current_result.get("suggestions"),
                        "Dừng loop vì candidate question_plan không thay đổi so với current plan.",
                    ),
                    "new_question_plan": latest_valid_candidate,
                    "is_loop": True,
                    "loop_count": completed_loop_count,
                }
            )

        latest_valid_candidate = deepcopy(candidate_plan)
        current_record["question_plan"] = deepcopy(candidate_plan)
        completed_loop_count = loop_index

        try:
            refined_result = _evaluate_question_plan_once(current_record, config=config, client=client, debug=debug)
        except Exception:
            return fail_closed_result(
                latest_valid_candidate=latest_valid_candidate,
                is_loop=True,
                loop_count=completed_loop_count,
            )

        if refined_result.get("is_good") is True:
            debug_loop_event(
                event="loop_stop",
                debug=debug,
                is_loop=True,
                max_loop=loop_limit,
                loop_index=loop_index,
                current_status="good",
                has_candidate=True,
                candidate_valid=True,
                stop_reason="candidate_verified_good",
            )
            return ensure_service_output_shape(
                {
                    "is_good": False,
                    "failed_reason": original_failed_reason,
                    "suggestions": merge_unique_texts(
                        original_suggestions,
                        "Đã tạo được new_question_plan và judge lại là đạt chất lượng.",
                    ),
                    "new_question_plan": latest_valid_candidate,
                    "is_loop": True,
                    "loop_count": completed_loop_count,
                }
            )

        current_result = refined_result
        next_candidate, _next_errors = valid_candidate_plan(current_result.get("new_question_plan"))
        if next_candidate is not None:
            latest_valid_candidate = deepcopy(next_candidate)

    debug_loop_event(
        event="loop_stop",
        debug=debug,
        is_loop=True,
        max_loop=loop_limit,
        loop_index=loop_limit,
        current_status="not_good",
        has_candidate=latest_valid_candidate is not None,
        candidate_valid=latest_valid_candidate is not None,
        stop_reason="max_loop_reached",
    )
    return ensure_service_output_shape(
        {
            "is_good": False,
            "failed_reason": current_result.get("failed_reason"),
            "suggestions": merge_unique_texts(
                current_result.get("suggestions"),
                f"Đã đạt giới hạn max_loop={loop_limit}; cần review thủ công nếu plan vẫn chưa đạt.",
            ),
            "new_question_plan": latest_valid_candidate,
            "is_loop": True,
            "loop_count": completed_loop_count,
        }
    )


def evaluate_question_plans(
    records: list[dict[str, Any]],
    *,
    config: AppConfig | None = None,
    client: LLMClient | None = None,
    debug: bool = False,
    is_loop: bool = False,
    max_loop: int = 3,
) -> list[dict[str, Any]]:
    config = config or default_config()
    client = client or LLMClient(config)
    return [
        evaluate_question_plan(
            record,
            config=config,
            client=client,
            debug=debug,
            is_loop=is_loop,
            max_loop=max_loop,
        )
        for record in records
    ]



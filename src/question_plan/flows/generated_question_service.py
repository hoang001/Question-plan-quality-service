"""Service đánh giá chất lượng generated question object.

Service này tách riêng khỏi luồng check/repair question_plan. Nó nhận trực tiếp
generated question object, list generated question object, hoặc wrapper cũ có
`generatedQuestions`; không yêu cầu `question_plan`, raw question hay raw answer.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..infra.config import AppConfig, load_config
from ..infra.debug import llm_prompt_debug_enabled
from ..infra.llm_client import LLMClient
from ..logic.generated_question_judge import judge_generated_question_object
from ..logic.generated_question_repair import repair_generated_question_object
from ..logic.generated_question_schema import (
    aggregate_generated_question_results,
    fail_closed_output,
    merge_generated_question_results,
    normalize_generated_question_input,
    normalize_generated_question_result,
    should_return_aggregate,
    validate_generated_question_object,
    validate_input_record,
)


SERVICE_ROOT_DIR = Path(__file__).resolve().parents[3]
GeneratedQuestionProgressCallback = Callable[[int, int, dict[str, Any]], None]


def default_config() -> AppConfig:
    return load_config(SERVICE_ROOT_DIR)


def has_bad_issue(issues: list[dict[str, Any]]) -> bool:
    return any(issue.get("severity") == "bad" for issue in issues)


def clamp_loop_count(value: int) -> int:
    return max(1, min(int(value or 1), 3))


def with_repair_defaults(result: dict[str, Any]) -> dict[str, Any]:
    result.setdefault("new_generated_question", None)
    result.setdefault("repair_status", "skipped")
    result.setdefault("repair_failed_reason", [])
    result.setdefault("repair_suggestions", [])
    result.setdefault("repair_loop_count", 0)
    return result


def merge_repair_result(base_result: dict[str, Any], repair_result: dict[str, Any], *, loop_count: int) -> dict[str, Any]:
    result = dict(base_result)
    repair_status = repair_result.get("repair_status") or "failed"
    result["repair_status"] = repair_status
    result["new_generated_question"] = repair_result.get("new_generated_question") if repair_status == "repaired" else None
    result["repair_failed_reason"] = repair_result.get("failed_reason") or []
    result["repair_suggestions"] = repair_result.get("suggestions") or []
    result["repair_loop_count"] = loop_count
    for suggestion in result["repair_suggestions"]:
        if suggestion and suggestion not in result.get("suggestions", []):
            result.setdefault("suggestions", []).append(suggestion)
    return result


def debug_generated_question_batch(
    generated_questions: list[dict[str, Any]],
    *,
    debug: bool = False,
) -> None:
    if not llm_prompt_debug_enabled(debug):
        return

    items: list[dict[str, Any]] = []
    for index, generated_question in enumerate(generated_questions):
        question_items = generated_question.get("questionItems")
        question_items = question_items if isinstance(question_items, list) else []
        interaction_count = 0
        for item in question_items:
            if isinstance(item, dict) and isinstance(item.get("interactions"), list):
                interaction_count += len(item["interactions"])
        items.append(
            {
                "index": index,
                "id": generated_question.get("id") or generated_question.get("_id") or f"item[{index}]",
                "question_item_count": len(question_items),
                "interaction_count": interaction_count,
            }
        )

    payload = {
        "step": "generated_questions_normalized",
        "generated_question_count": len(generated_questions),
        "items": items,
    }
    print("[DEBUG_GENERATED_QUESTIONS_SERVICE] " + json.dumps(payload, ensure_ascii=False), file=sys.stderr)


def evaluate_generated_question_object(
    generated_question: dict[str, Any],
    *,
    strict_mode: bool = True,
    config: AppConfig | None = None,
    client: LLMClient | None = None,
    debug: bool = False,
    index: int = 0,
    auto_repair: bool = False,
    is_loop: bool = False,
    max_loop: int = 3,
) -> dict[str, Any]:
    """Đánh giá một generated question object.

    Luồng xử lý:
    1. Chạy deterministic schema/internal-consistency checks.
    2. Nếu có lỗi `bad` chắc chắn thì trả kết quả fail-closed, không gọi LLM.
    3. Nếu schema đủ để đọc tiếp, gọi LLM judge để đánh giá semantic nội bộ.
    """

    try:
        schema_validation_result = validate_generated_question_object(generated_question, index)
        schema_issues = schema_validation_result.get("issues") or []

        if has_bad_issue(schema_issues):
            result = merge_generated_question_results(
                schema_issues=schema_issues,
                llm_result={"is_good": True, "failed_reason": [], "suggestions": [], "issues": []},
                strict_mode=strict_mode,
                generated_question=generated_question,
                index=index,
            )
            return maybe_repair_generated_question(
                generated_question,
                with_repair_defaults(result),
                strict_mode=strict_mode,
                config=config,
                client=client,
                debug=debug,
                index=index,
                auto_repair=auto_repair,
                is_loop=is_loop,
                max_loop=max_loop,
            )

        config = config or default_config()
        client = client or LLMClient(config)
        llm_result = judge_generated_question_object(
            generated_question,
            schema_validation_result,
            config,
            client,
            strict_mode=strict_mode,
            index=index,
            debug=debug,
        )
        result = merge_generated_question_results(
            schema_issues=schema_issues,
            llm_result=llm_result,
            strict_mode=strict_mode,
            generated_question=generated_question,
            index=index,
        )
        return maybe_repair_generated_question(
            generated_question,
            with_repair_defaults(result),
            strict_mode=strict_mode,
            config=config,
            client=client,
            debug=debug,
            index=index,
            auto_repair=auto_repair,
            is_loop=is_loop,
            max_loop=max_loop,
        )
    except Exception as exc:
        return with_repair_defaults(fail_closed_output(str(exc), generated_question=generated_question, index=index))


def maybe_repair_generated_question(
    generated_question: dict[str, Any],
    check_result: dict[str, Any],
    *,
    strict_mode: bool,
    config: AppConfig | None,
    client: LLMClient | None,
    debug: bool,
    index: int,
    auto_repair: bool,
    is_loop: bool,
    max_loop: int,
) -> dict[str, Any]:
    if not auto_repair or check_result.get("is_good"):
        return with_repair_defaults(check_result)

    config = config or default_config()
    client = client or LLMClient(config)
    current_question = generated_question
    current_result = check_result
    last_repair: dict[str, Any] | None = None
    loop_count = 0
    loop_limit = clamp_loop_count(max_loop) if is_loop else 1

    while loop_count < loop_limit and not current_result.get("is_good"):
        repair_result = repair_generated_question_object(
            current_question,
            current_result,
            config,
            client,
            index=index,
            debug=debug,
        )
        loop_count += 1
        last_repair = repair_result
        if repair_result.get("repair_status") != "repaired" or not isinstance(
            repair_result.get("new_generated_question"),
            dict,
        ):
            break

        if not is_loop:
            break

        current_question = repair_result["new_generated_question"]
        current_result = evaluate_generated_question_object(
            current_question,
            strict_mode=strict_mode,
            config=config,
            client=client,
            debug=debug,
            index=index,
            auto_repair=False,
        )
        if current_result.get("is_good"):
            break

    if not last_repair:
        return with_repair_defaults(check_result)
    return merge_repair_result(check_result, last_repair, loop_count=loop_count)


def evaluate_generated_questions(
    payload: dict[str, Any] | list[dict[str, Any]],
    *,
    strict_mode: bool = True,
    config: AppConfig | None = None,
    client: LLMClient | None = None,
    debug: bool = False,
    progress_callback: GeneratedQuestionProgressCallback | None = None,
    auto_repair: bool = False,
    is_loop: bool = False,
    max_loop: int = 3,
) -> dict[str, Any]:
    """Đánh giá generated question input dạng single/list/wrapper.

    Input hợp lệ:
    - một generated question object trực tiếp;
    - list generated question object;
    - wrapper cũ có field `generatedQuestions`.
    """

    input_issues = validate_input_record(payload)
    if input_issues:
        return normalize_generated_question_result(
            {
                "id": "input",
                "is_good": False,
                "failed_reason": [issue["reason"] for issue in input_issues if issue.get("severity") == "bad"],
                "suggestions": [issue["suggestion"] for issue in input_issues if issue.get("suggestion")],
                "issues": input_issues,
            },
            strict_mode=strict_mode,
        )

    try:
        generated_questions = normalize_generated_question_input(payload)
    except Exception as exc:
        return fail_closed_output(str(exc), generated_question=None, index=0)

    debug_generated_question_batch(generated_questions, debug=debug)

    results: list[dict[str, Any]] = []
    total = len(generated_questions)
    for index, generated_question in enumerate(generated_questions):
        if progress_callback:
            progress_callback(index + 1, total, generated_question)
        results.append(
            evaluate_generated_question_object(
                generated_question,
                strict_mode=strict_mode,
                config=config,
                client=client,
                debug=debug,
                index=index,
                auto_repair=auto_repair,
                is_loop=is_loop,
                max_loop=max_loop,
            )
        )

    if should_return_aggregate(payload, generated_questions):
        return aggregate_generated_question_results(results)
    if results:
        return results[0]
    return aggregate_generated_question_results(results)

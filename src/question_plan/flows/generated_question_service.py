"""Generated-question quality check and scoped repair service.

The flow accepts generated question objects only. Structural checks are
deterministic; interpretation of solution conclusions belongs exclusively to
the LLM solution-anchor resolver.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..infra.config import (
    AppConfig,
    generated_question_correctness_model,
    generated_question_gemma_concurrency,
    generated_question_presentation_model,
    load_config,
)
from ..infra.debug import llm_prompt_debug_enabled, trace_pipeline_event
from ..infra.llm_client import LLMClient
from ..logic.generated_question_judge import (
    build_no_prompt_messages,
    judge_generated_question_object,
)
from ..logic.generated_question_repair import normalize_scoped_repair_result, repair_generated_question_scoped
from ..logic.generated_question_schema import (
    aggregate_generated_question_results,
    fail_closed_output,
    make_issue,
    merge_generated_question_results,
    normalize_generated_question_input,
    normalize_generated_question_result,
    runtime_issue,
    should_return_aggregate,
    validate_generated_question_object,
    validate_input_record,
)
from ..logic.solution_anchor_resolver import compact_generated_question_for_solution_anchor, resolve_solution_anchor_consistency


SERVICE_ROOT_DIR = Path(__file__).resolve().parents[3]
GeneratedQuestionProgressCallback = Callable[[int, int, dict[str, Any]], None]
MAX_GENERATED_QUESTION_WORKERS = 2


def evaluate_generated_questions_raw(
    payload: dict[str, Any] | list[dict[str, Any]],
    *,
    config: AppConfig,
    client: LLMClient,
    progress_callback: GeneratedQuestionProgressCallback | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Call both solution models without schema, parsing, gates, or post-processing."""

    generated_questions = normalize_generated_question_input(payload)

    def evaluate_one(index: int, generated_question: dict[str, Any]) -> dict[str, Any]:
        if progress_callback:
            progress_callback(index + 1, len(generated_questions), generated_question)
        messages = build_no_prompt_messages(generated_question)
        with ThreadPoolExecutor(max_workers=2) as executor:
            correctness_future = executor.submit(
                client.chat_completion,
                model=generated_question_correctness_model(config),
                messages=messages,
                temperature=0,
            )
            process_future = executor.submit(
                client.chat_completion,
                model=generated_question_presentation_model(config),
                messages=messages,
                temperature=0,
            )
            correctness_response = correctness_future.result()
            process_response = process_future.result()
        return {
            "id": generated_question.get("id") or generated_question.get("_id") or f"item[{index}]",
            "correctness_output": str(correctness_response.get("content") or ""),
            "process_output": str(process_response.get("content") or ""),
        }

    results = [
        evaluate_one(index, generated_question)
        for index, generated_question in enumerate(generated_questions)
    ]
    return results if isinstance(payload, list) or len(results) != 1 else results[0]
JUDGE_ROUTING_FIELDS = (
    "correctness_comments",
    "process_comments",
    "judge_model",
    "correctness_primary_model",
    "process_presentation_model",
    "final_correctness_model",
    "judge_attempt_count",
    "judge_fallback_called",
    "correctness_fallback_called",
    "judge_gemma_run_count",
    "judge_fallback_reason",
    "correctness_primary_26b_calls",
    "correctness_contract_correction_26b_calls",
    "correctness_rejudge_26b_calls",
    "primary_12b_calls",
    "contract_correction_12b_calls",
    "fallback_26b_calls",
    "fallback_reason_contract",
    "fallback_reason_resolver_mismatch",
    "http_retry_count",
    "non_canonical_count",
    "runtime_count",
    "correctness_error_type_normalized_count",
)
PUBLIC_SUMMARY_FIELDS = (
    "total",
    "good",
    "bad",
    "needs_review",
    "warning",
    "repaired",
    "repair_failed",
    "judge_calls",
    "judge_fallbacks",
    "resolver_calls",
    "resolver_fallbacks",
    "judge_fallback_rate",
    "resolver_fallback_rate",
)


def clamp_generated_question_workers(workers: int) -> int:
    return max(1, min(int(workers), MAX_GENERATED_QUESTION_WORKERS))


def default_config() -> AppConfig:
    return load_config(SERVICE_ROOT_DIR)


def clamp_loop_count(value: int) -> int:
    return max(1, min(int(value or 1), 2))


def has_bad_issue(issues: list[dict[str, Any]]) -> bool:
    return any(issue.get("severity") == "bad" for issue in issues)


def issue_sort_key(issue: dict[str, Any]) -> tuple[int, int]:
    severity = {"bad": 0, "needs_review": 1, "warning": 2}
    category = {
        "interaction_schema": 0,
        "solution_anchor_consistency": 1,
        "answer_internal_consistency": 2,
        "solution_quality": 3,
        "choice_quality": 4,
        "hint_quality": 5,
        "render_schema": 6,
        "pedagogical_quality": 7,
        "runtime": 8,
    }
    return severity.get(str(issue.get("severity") or ""), 99), category.get(str(issue.get("category") or ""), 99)


def logical_issue_key(issue: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(issue.get("category") or ""),
        str(issue.get("location") or ""),
        str(issue.get("repair_intent") or ""),
        " ".join(str(issue.get("reason") or "").lower().split()),
    )


def canonical_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate logical issues, retaining the strongest and clearest form."""

    severity_rank = {"warning": 0, "needs_review": 1, "bad": 2}
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for issue in (item for item in issues if isinstance(item, dict)):
        key = logical_issue_key(issue)
        current = grouped.get(key)
        if current is None:
            grouped[key] = dict(issue)
            continue
        chosen = dict(current)
        if severity_rank.get(str(issue.get("severity") or ""), -1) > severity_rank.get(str(current.get("severity") or ""), -1):
            chosen["severity"] = issue.get("severity")
        for field in ("reason", "suggestion"):
            old_text = str(current.get(field) or "").strip()
            new_text = str(issue.get(field) or "").strip()
            if new_text and (not old_text or len(new_text) < len(old_text)):
                chosen[field] = new_text
        grouped[key] = chosen
    return sorted(grouped.values(), key=issue_sort_key)


def compact_issue(issue: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "severity": issue.get("severity"),
        "category": issue.get("category"),
        "location": issue.get("location"),
        "reason": issue.get("reason"),
        "suggestion": issue.get("suggestion"),
        "repair_intent": issue.get("repair_intent"),
    }
    if issue.get("disposition"):
        compact["disposition"] = issue.get("disposition")
    return compact


def compact_solution_anchor_result(anchor: dict[str, Any]) -> dict[str, Any]:
    answer = anchor.get("final_answer") if isinstance(anchor.get("final_answer"), dict) else {}
    compact_answer = {
        key: answer.get(key)
        for key in ("text", "matched_option_id", "correctOptionIds", "expected")
        if answer.get(key) not in (None, "", [])
    }
    compact_anchor: dict[str, Any] = {
        "resolver_status": anchor.get("resolver_status"),
        "answerSpec_matches_solution": anchor.get("answerSpec_matches_solution"),
        "resolver_model": anchor.get("resolver_model"),
        "fallback_called": bool(anchor.get("resolver_fallback_called", False)),
        "gemma_run_count": anchor.get("resolver_gemma_run_count"),
        "fallback_reason": anchor.get("resolver_fallback_reason"),
    }
    if compact_answer:
        compact_anchor["final_answer"] = compact_answer
    return compact_anchor


def public_generated_question_result(result: dict[str, Any], *, debug: bool) -> dict[str, Any]:
    issues = canonical_issues(result.get("issues") or [])
    repaired = result.get("new_generated_question") if isinstance(result.get("new_generated_question"), dict) else None
    if not debug:
        return {
            "id": result.get("id"),
            "is_good": bool(result.get("is_good")),
            "issues": [compact_issue(issue) for issue in issues],
            "correctness_comments": list(result.get("correctness_comments") or []),
            "process_comments": list(result.get("process_comments") or []),
            "new_generated_question": repaired,
        }

    has_runtime_issue = any(issue.get("category") == "runtime" for issue in issues)
    runtime_details = (
        [
            str(reason).strip()
            for reason in result.get("failed_reason") or []
            if str(reason).strip()
        ]
        if has_runtime_issue
        else []
    )
    if runtime_details:
        issues = [
            {
                **issue,
                "reason": " | ".join(runtime_details),
            }
            if issue.get("category") == "runtime"
            else issue
            for issue in issues
        ]
    debug_result: dict[str, Any] = {
        "id": result.get("id"),
        "is_good": bool(result.get("is_good")),
        "issues": [compact_issue(issue) for issue in issues],
        "correctness_comments": list(result.get("correctness_comments") or []),
        "process_comments": list(result.get("process_comments") or []),
        "new_generated_question": repaired,
    }
    if runtime_details:
        debug_result["failed_reason"] = runtime_details
    anchor = result.get("solution_anchor_result")
    if isinstance(anchor, dict):
        debug_result["solution_anchor_result"] = compact_solution_anchor_result(anchor)
    if result.get("repair_status") and result.get("repair_status") != "skipped":
        debug_result["repair_status"] = result["repair_status"]
    if result.get("patches"):
        debug_result["repair_patches"] = result["patches"]
    if result.get("repair_loop_count"):
        debug_result["loop_count"] = result["repair_loop_count"]
    if result.get("repair_stop_reason"):
        debug_result["stop_reason"] = result["repair_stop_reason"]
    for key in JUDGE_ROUTING_FIELDS:
        if key in result:
            debug_result[key] = result[key]
    return debug_result


def public_generated_question_output(result: dict[str, Any], *, debug: bool) -> dict[str, Any]:
    if "results" not in result:
        return public_generated_question_result(result, debug=debug)
    rows = [item for item in result.get("results") or [] if isinstance(item, dict)]
    internal_summary = result.get("summary") or {}
    public_summary = (
        internal_summary
        if debug
        else {
            key: internal_summary.get(key)
            for key in PUBLIC_SUMMARY_FIELDS
            if key in internal_summary
        }
    )
    return {
        "is_good": all(bool(item.get("is_good")) for item in rows),
        "summary": public_summary,
        "results": [public_generated_question_result(item, debug=debug) for item in rows],
    }


def with_internal_defaults(result: dict[str, Any]) -> dict[str, Any]:
    result.setdefault("new_generated_question", None)
    result.setdefault("repair_status", "skipped")
    result.setdefault("repair_loop_count", 0)
    result.setdefault("repair_stop_reason", "")
    result.setdefault("patches", [])
    result.setdefault("selected_issue", None)
    result.setdefault("solution_anchor_result", None)
    return result


def merge_anchor_result(
    base_result: dict[str, Any],
    anchor_result: Any,
    *,
    strict_mode: bool,
    generated_question: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    if not (
        isinstance(anchor_result, dict)
        and anchor_result.get("resolver_status") in {"resolved", "needs_manual_review"}
        and isinstance(anchor_result.get("final_answer"), dict)
        and isinstance(anchor_result.get("answerSpec_matches_solution"), bool)
        and anchor_result.get("answer_spec_alignment") in {"matched", "equivalent", "mismatched"}
        and isinstance(anchor_result.get("fields_to_fix"), list)
        and isinstance(anchor_result.get("issues"), list)
    ):
        return normalize_generated_question_result(
            {
                "id": base_result.get("id"),
                "is_good": False,
                "issues": canonical_issues([
                    *(base_result.get("issues") or []),
                    runtime_issue(
                        "Solution Resolver handoff không phải canonical result.",
                        location="/solutions",
                    ),
                ]),
            },
            strict_mode=strict_mode,
            generated_question=generated_question,
            index=index,
        )
    resolved_issues = [
        {
            **issue,
            "disposition": "blocking",
        }
        if anchor_result.get("resolver_status") == "resolved" and isinstance(issue, dict)
        else issue
        for issue in anchor_result.get("issues") or []
    ]
    merged = normalize_generated_question_result(
        {
            "id": base_result.get("id"),
            "is_good": base_result.get("is_good", True),
            "issues": canonical_issues([*(base_result.get("issues") or []), *resolved_issues]),
        },
        strict_mode=strict_mode,
        generated_question=generated_question,
        index=index,
    )
    for key in JUDGE_ROUTING_FIELDS:
        if key in base_result:
            merged[key] = base_result[key]
    merged["solution_anchor_result"] = {**anchor_result, "issues": resolved_issues}
    return merged


def solution_quality_gate_issues(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return solution issues that must be handled before downstream alignment."""

    return [
        issue
        for issue in result.get("issues") or []
        if isinstance(issue, dict)
        and issue.get("category") == "solution_quality"
        and issue.get("repair_intent") in {
            "fix_solution_correctness",
            "fix_solution_process",
            "needs_manual_review",
        }
    ]


def select_solution_quality_gate_issue(result: dict[str, Any]) -> dict[str, Any] | None:
    """Return the final reviewed solution blocker in aggregate order."""

    issues = solution_quality_gate_issues(result)
    return issues[0] if issues else None


def select_pre_resolver_blocker(result: dict[str, Any]) -> dict[str, Any] | None:
    """Block semantic alignment until solution quality was checked successfully."""

    solution_issue = select_solution_quality_gate_issue(result)
    if solution_issue:
        return solution_issue
    return next(
        (
            issue
            for issue in result.get("issues") or []
            if isinstance(issue, dict) and issue.get("category") == "runtime"
        ),
        None,
    )


def resolver_gate_open(
    judge_result: Any,
    checked_result: dict[str, Any],
) -> bool:
    """Chỉ mở Resolver sau một aggregate Judge canonical và good."""

    return (
        isinstance(judge_result, dict)
        and judge_result.get("is_good") is True
        and isinstance(judge_result.get("issues"), list)
        and isinstance(judge_result.get("failed_reason"), list)
        and isinstance(judge_result.get("suggestions"), list)
        and not judge_result["issues"]
        and select_pre_resolver_blocker(checked_result) is None
    )


REPAIRABLE_INTENTS = {
    "fix_solution_correctness",
    "fix_solution_process",
    "align_fields_to_solution",
    "align_hint_to_solution",
}


def issue_has_missing_context(issue: dict[str, Any]) -> bool:
    text = " ".join(
        str(issue.get(field) or "").lower()
        for field in ("reason", "suggestion")
    )
    context_terms = ("dữ liệu", "context", "hình", "ảnh", "bảng", "biểu đồ")
    missing_terms = (
        "thiếu",
        "không có",
        "không được cung cấp",
        "chưa cung cấp",
        "không truy cập được",
        "missing",
        "unavailable",
    )
    return any(term in text for term in context_terms) and any(term in text for term in missing_terms)


def is_repairable_final_issue(issue: Any) -> bool:
    return (
        isinstance(issue, dict)
        and issue.get("disposition") == "blocking"
        and issue.get("severity") == "bad"
        and issue.get("repair_intent") in REPAIRABLE_INTENTS
        and issue.get("category") != "runtime"
        and not issue_has_missing_context(issue)
    )


def _solution_index(issue: dict[str, Any]) -> int | None:
    match = re.match(r"^/solutions/(\d+)(?:/|$)", str(issue.get("location") or ""))
    return int(match.group(1)) if match else None


def allowed_paths_for_issue(
    generated_question: dict[str, Any],
    check_result: dict[str, Any],
    issue: dict[str, Any],
) -> list[str]:
    intent = str(issue.get("repair_intent") or "")
    if intent in {"fix_solution_correctness", "fix_solution_process"}:
        solution_index = _solution_index(issue)
        solutions = generated_question.get("solutions")
        if solution_index is None or not isinstance(solutions, list) or solution_index >= len(solutions):
            return []
        solution = solutions[solution_index]
        content = solution.get("solutionContent") if isinstance(solution, dict) else None
        if not isinstance(content, list):
            return []
        return [
            f"/solutions/{solution_index}/solutionContent/{content_index}/text"
            for content_index, block in enumerate(content)
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]

    anchor = check_result.get("solution_anchor_result")
    anchor = anchor if isinstance(anchor, dict) else {}
    if intent == "align_fields_to_solution":
        return [
            str(fix.get("path") or "")
            for fix in anchor.get("fields_to_fix") or []
            if isinstance(fix, dict) and str(fix.get("path") or "").startswith("/questionItems/")
        ]

    if intent == "align_hint_to_solution":
        location = str(issue.get("location") or "")
        paths: list[str] = []
        for item_index, item in enumerate(generated_question.get("questionItems") or []):
            if not isinstance(item, dict):
                continue
            for hint_index, hint in enumerate(item.get("hints") or []):
                if not isinstance(hint, dict):
                    continue
                hint_root = f"/questionItems/{item_index}/hints/{hint_index}"
                if location and not (location == hint_root or location.startswith(hint_root + "/")):
                    continue
                for content_index, block in enumerate(hint.get("content") or []):
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        paths.append(f"{hint_root}/content/{content_index}/text")
        return paths
    return []


def select_repair_issue(check_result: dict[str, Any]) -> dict[str, Any] | None:
    if int(check_result.get("non_canonical_count") or 0) or int(check_result.get("runtime_count") or 0):
        return None
    if any(
        isinstance(issue, dict) and issue.get("category") == "runtime"
        for issue in check_result.get("issues") or []
    ):
        return None
    first_blocking = next(
        (
            issue
            for issue in check_result.get("issues") or []
            if isinstance(issue, dict)
            and issue.get("disposition") == "blocking"
            and issue.get("severity") == "bad"
        ),
        None,
    )
    return first_blocking if is_repairable_final_issue(first_blocking) else None


def compact_selected_issue(issue: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(issue, dict):
        return None
    return {
        "severity": issue.get("severity"),
        "disposition": issue.get("disposition"),
        "category": issue.get("category"),
        "location": issue.get("location"),
        "repair_intent": issue.get("repair_intent"),
    }


def repair_once(
    generated_question: dict[str, Any],
    check_result: dict[str, Any],
    *,
    config: AppConfig,
    client: LLMClient,
    index: int,
    debug: bool,
) -> dict[str, Any]:
    issue = select_repair_issue(check_result)
    if not issue:
        first_issue = next(
            (item for item in check_result.get("issues") or [] if isinstance(item, dict)),
            None,
        )
        return {
            "repair_status": "needs_manual_review",
            "failed_reason": ["Không có blocking issue đủ điều kiện Repair tự động."],
            "suggestions": ["Review thủ công issue hoặc chạy lại stage bị non-canonical/runtime."],
            "new_generated_question": None,
            "patches": [],
            "selected_issue": compact_selected_issue(first_issue),
        }

    allowed_paths = allowed_paths_for_issue(generated_question, check_result, issue)
    if not allowed_paths:
        return {
            "repair_status": "needs_manual_review",
            "failed_reason": ["Issue thiếu context hoặc không có allowed_paths hợp lệ."],
            "suggestions": ["Review thủ công; không mở rộng phạm vi patch."],
            "new_generated_question": None,
            "patches": [],
            "selected_issue": compact_selected_issue(issue),
        }

    anchor = check_result.get("solution_anchor_result")
    anchor = anchor if isinstance(anchor, dict) else {}
    if issue.get("repair_intent") == "align_fields_to_solution":
        fixes = anchor.get("fields_to_fix") if isinstance(anchor.get("fields_to_fix"), list) else []
        if anchor.get("resolver_status") != "resolved" or not fixes:
            return {
                "repair_status": "needs_manual_review",
                "failed_reason": ["Resolver chưa xác nhận solution đúng hoặc không có field fix hợp lệ."],
                "suggestions": ["Không sửa answerSpec trước khi solution được xác minh."],
                "new_generated_question": None,
                "patches": [],
                "selected_issue": compact_selected_issue(issue),
            }
        patches = [
            {
                "op": "replace",
                "path": fix["path"],
                "value": fix["value"],
                "reason": str(fix.get("reason") or issue.get("reason") or "Căn chỉnh answerSpec với solution đã xác minh."),
            }
            for fix in fixes
            if isinstance(fix, dict) and fix.get("path") in allowed_paths and "value" in fix
        ]
        result = normalize_scoped_repair_result(
            {"repair_status": "repaired", "failed_reason": [], "suggestions": [], "patches": patches},
            generated_question=generated_question,
            index=index,
            repair_intent="align_fields_to_solution",
            allowed_paths=allowed_paths,
        )
        result["selected_issue"] = compact_selected_issue(issue)
        return result

    if issue.get("repair_intent") == "align_hint_to_solution" and anchor.get("resolver_status") != "resolved":
        return {
            "repair_status": "needs_manual_review",
            "failed_reason": ["Resolver chưa xác nhận solution đúng nên chưa được sửa hint."],
            "suggestions": ["Review solution trước khi căn chỉnh hint."],
            "new_generated_question": None,
            "patches": [],
            "selected_issue": compact_selected_issue(issue),
        }

    if llm_prompt_debug_enabled(debug):
        print(
            "[DEBUG_GENERATED_QUESTION_REPAIR] "
            + json.dumps(
                {
                    "id": generated_question.get("id") or generated_question.get("_id") or f"item[{index}]",
                    "selected_issue": compact_selected_issue(issue),
                    "policy": "scoped_only",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
    result = repair_generated_question_scoped(
        generated_question,
        check_result,
        issue,
        config,
        client,
        allowed_paths=allowed_paths,
        index=index,
        debug=debug,
    )
    result["selected_issue"] = compact_selected_issue(issue)
    return result


def merge_repair_result(base: dict[str, Any], repair: dict[str, Any], loop_count: int, stop_reason: str) -> dict[str, Any]:
    result = dict(base)
    result["repair_status"] = repair.get("repair_status") or "failed"
    result["new_generated_question"] = (
        repair.get("new_generated_question") if repair.get("repair_status") == "repaired" else None
    )
    result["patches"] = repair.get("patches") or []
    result["selected_issue"] = repair.get("selected_issue") if isinstance(repair.get("selected_issue"), dict) else None
    result["repair_loop_count"] = loop_count
    result["repair_stop_reason"] = stop_reason
    return result


def repair_issue_identity(issue: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(issue.get("category") or ""),
        str(issue.get("location") or ""),
        str(issue.get("repair_intent") or ""),
    )


def significant_issue_identities(result: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        repair_issue_identity(issue)
        for issue in result.get("issues") or []
        if isinstance(issue, dict) and issue.get("severity") in {"bad", "needs_review"}
    }


def manual_repair_stop(
    generated_question: dict[str, Any],
    issue: dict[str, Any] | None,
    reason: str,
    *,
    patches: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": generated_question.get("id") or generated_question.get("_id"),
        "repair_status": "needs_manual_review",
        "failed_reason": [reason],
        "suggestions": ["Giữ nguyên object gốc và review thủ công issue còn lại."],
        "new_generated_question": None,
        "patches": patches or [],
        "selected_issue": compact_selected_issue(issue),
    }


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
    max_loop: int,
) -> dict[str, Any]:
    if not auto_repair or check_result.get("is_good"):
        check_result["repair_stop_reason"] = "auto_repair_disabled" if not auto_repair else "already_good"
        return with_internal_defaults(check_result)

    config = config or default_config()
    client = client or LLMClient(config)
    current_question, current_result = generated_question, check_result
    accepted_patches: list[dict[str, Any]] = []
    last_result: dict[str, Any] | None = None
    stop_reason = "max_loop_reached"
    loop_count = 0

    while loop_count < clamp_loop_count(max_loop) and not current_result.get("is_good"):
        target_issue = select_repair_issue(current_result)
        if not target_issue:
            last_result = manual_repair_stop(
                generated_question,
                next((issue for issue in current_result.get("issues") or [] if isinstance(issue, dict)), None),
                "Issue đầu tiên không đủ điều kiện Repair tự động.",
                patches=accepted_patches,
            )
            stop_reason = "ineligible_issue"
            break
        target_identity = repair_issue_identity(target_issue)
        baseline_identities = significant_issue_identities(current_result) - {target_identity}
        last_result = repair_once(
            current_question,
            current_result,
            config=config,
            client=client,
            index=index,
            debug=debug,
        )
        loop_count += 1
        candidate = last_result.get("new_generated_question")
        if last_result.get("repair_status") != "repaired" or not isinstance(candidate, dict):
            stop_reason = str(last_result.get("repair_status") or "repair_failed")
            break

        structural = validate_generated_question_object(candidate, index)
        if has_bad_issue(structural.get("issues") or []):
            last_result = manual_repair_stop(
                generated_question,
                target_issue,
                "Patch tạo generated question không hợp lệ về cấu trúc.",
                patches=accepted_patches,
            )
            stop_reason = "structural_regression"
            break

        rechecked = evaluate_generated_question_object(
            candidate,
            strict_mode=strict_mode,
            config=config,
            client=client,
            debug=debug,
            index=index,
            auto_repair=False,
            max_loop=2,
        )
        rechecked_identities = significant_issue_identities(rechecked)
        if target_identity in rechecked_identities:
            last_result = manual_repair_stop(
                generated_question,
                target_issue,
                "Issue mục tiêu vẫn xuất hiện sau khi rejudge.",
                patches=accepted_patches,
            )
            stop_reason = "same_issue_reappeared"
            break
        new_identities = rechecked_identities - baseline_identities
        if new_identities:
            last_result = manual_repair_stop(
                generated_question,
                target_issue,
                "Patch làm phát sinh lỗi bad/needs_review mới sau khi rejudge.",
                patches=accepted_patches,
            )
            stop_reason = "new_issue_after_rejudge"
            break

        accepted_patches.extend(last_result.get("patches") or [])
        current_question = candidate
        current_result = rechecked
        if current_result.get("is_good"):
            last_result = {
                **last_result,
                "new_generated_question": current_question,
                "patches": accepted_patches,
            }
            stop_reason = "recheck_good"
            break

    if current_result.get("is_good") and last_result and last_result.get("repair_status") == "repaired":
        return merge_repair_result(check_result, last_result, loop_count, stop_reason)
    if last_result and last_result.get("repair_status") == "repaired":
        last_result = manual_repair_stop(
            generated_question,
            select_repair_issue(current_result),
            "Đã đạt giới hạn hai vòng nhưng object vẫn còn blocking issue.",
            patches=accepted_patches,
        )
        stop_reason = "max_loop_reached"
    return merge_repair_result(check_result, last_result or {}, loop_count, stop_reason)


def evaluate_generated_question_object(
    generated_question: dict[str, Any],
    *,
    strict_mode: bool = True,
    config: AppConfig | None = None,
    client: LLMClient | None = None,
    debug: bool = False,
    index: int = 0,
    auto_repair: bool = False,
    max_loop: int = 2,
) -> dict[str, Any]:
    try:
        schema_result = validate_generated_question_object(generated_question, index)
        trace_pipeline_event(
            stage="structural_validator",
            event="output",
            payload={
                "generated_question": generated_question,
                "validation_result": schema_result,
            },
            object_id=generated_question.get("id") or generated_question.get("_id"),
        )
        schema_issues = schema_result.get("issues") or []
        if has_bad_issue(schema_issues):
            checked = merge_generated_question_results(
                schema_issues=schema_issues,
                llm_result={"is_good": True, "issues": []},
                strict_mode=strict_mode,
                generated_question=generated_question,
                index=index,
            )
        else:
            config = config or default_config()
            client = client or LLMClient(config)
            judge_result = judge_generated_question_object(
                generated_question,
                schema_result,
                config,
                client,
                strict_mode=strict_mode,
                index=index,
                debug=debug,
            )
            checked = merge_generated_question_results(
                schema_issues=schema_issues,
                llm_result=judge_result,
                strict_mode=strict_mode,
                generated_question=generated_question,
                index=index,
            )
            if isinstance(judge_result, dict):
                for key in JUDGE_ROUTING_FIELDS:
                    checked[key] = judge_result.get(key)
            interaction_contexts = compact_generated_question_for_solution_anchor(
                generated_question
            )["interaction_contexts"]
            gate_open = resolver_gate_open(judge_result, checked)
            trace_pipeline_event(
                stage="quality_gate",
                event="output",
                payload={
                    "gate_open": gate_open,
                    "has_interaction_context": bool(interaction_contexts),
                    "judge_result": judge_result,
                    "checked_result": checked,
                },
                object_id=generated_question.get("id") or generated_question.get("_id"),
            )
            if gate_open and interaction_contexts:
                anchor = resolve_solution_anchor_consistency(
                    generated_question,
                    config=config,
                    client=client,
                    debug=debug,
                )
                checked = merge_anchor_result(
                    checked,
                    anchor,
                    strict_mode=strict_mode,
                    generated_question=generated_question,
                    index=index,
                )
            else:
                trace_pipeline_event(
                    stage="resolver",
                    event="skipped",
                    payload={
                        "reason": (
                            "quality_gate_closed"
                            if not gate_open
                            else "missing_interaction_context"
                        )
                    },
                    object_id=generated_question.get("id") or generated_question.get("_id"),
                )
        final_result = maybe_repair_generated_question(
            generated_question,
            with_internal_defaults(checked),
            strict_mode=strict_mode,
            config=config,
            client=client,
            debug=debug,
            index=index,
            auto_repair=auto_repair,
            max_loop=max_loop,
        )
        trace_pipeline_event(
            stage="public_result",
            event="internal_output_before_public_compaction",
            payload=final_result,
            object_id=generated_question.get("id") or generated_question.get("_id"),
        )
        return final_result
    except Exception as exc:
        return with_internal_defaults(fail_closed_output(str(exc), generated_question=generated_question, index=index))


def debug_generated_question_batch(generated_questions: list[dict[str, Any]], *, debug: bool) -> None:
    if not llm_prompt_debug_enabled(debug):
        return
    print(
        "[DEBUG_GENERATED_QUESTIONS_SERVICE] "
        + json.dumps(
            {
                "generated_question_count": len(generated_questions),
                "ids": [item.get("id") or item.get("_id") for item in generated_questions],
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )


def evaluate_generated_questions(
    payload: dict[str, Any] | list[dict[str, Any]],
    *,
    strict_mode: bool = True,
    config: AppConfig | None = None,
    client: LLMClient | None = None,
    debug: bool = False,
    progress_callback: GeneratedQuestionProgressCallback | None = None,
    auto_repair: bool = False,
    max_loop: int = 2,
    workers: int = 1,
) -> dict[str, Any]:
    input_issues = validate_input_record(payload)
    if input_issues:
        invalid = normalize_generated_question_result(
            {"id": "input", "is_good": False, "issues": input_issues},
            strict_mode=strict_mode,
        )
        return public_generated_question_output(with_internal_defaults(invalid), debug=debug)
    try:
        generated_questions = normalize_generated_question_input(payload)
    except Exception as exc:
        failed = fail_closed_output(str(exc), generated_question=None, index=0)
        return public_generated_question_output(with_internal_defaults(failed), debug=debug)

    debug_generated_question_batch(generated_questions, debug=debug)
    def evaluate_one(index: int, generated_question: dict[str, Any]) -> dict[str, Any]:
        if progress_callback:
            progress_callback(index + 1, len(generated_questions), generated_question)
        return evaluate_generated_question_object(
            generated_question,
            strict_mode=strict_mode,
            config=config,
            client=client,
            debug=debug,
            index=index,
            auto_repair=auto_repair,
            max_loop=max_loop,
        )

    gemma_object_limit = max(1, generated_question_gemma_concurrency(config) // 2)
    worker_count = min(
        clamp_generated_question_workers(workers),
        gemma_object_limit,
        max(1, len(generated_questions)),
    )
    if worker_count == 1:
        results = [evaluate_one(index, question) for index, question in enumerate(generated_questions)]
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="generated-question") as executor:
            results = list(executor.map(evaluate_one, range(len(generated_questions)), generated_questions))

    if should_return_aggregate(payload, generated_questions):
        return public_generated_question_output(aggregate_generated_question_results(results), debug=debug)
    if results:
        return public_generated_question_output(results[0], debug=debug)
    return public_generated_question_output(aggregate_generated_question_results(results), debug=debug)

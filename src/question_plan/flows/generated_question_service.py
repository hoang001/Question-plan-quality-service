"""Generated-question quality check and scoped repair service.

The flow accepts generated question objects only. Structural checks are
deterministic; interpretation of solution conclusions belongs exclusively to
the LLM solution-anchor resolver.
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
from ..logic.generated_question_repair import normalize_scoped_repair_result, repair_generated_question_scoped
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
from ..logic.generated_question_spelling import check_spelling_and_wording, extract_text_nodes_for_spelling
from ..logic.solution_anchor_resolver import resolve_solution_anchor_consistency


SERVICE_ROOT_DIR = Path(__file__).resolve().parents[3]
GeneratedQuestionProgressCallback = Callable[[int, int, dict[str, Any]], None]


def default_config() -> AppConfig:
    return load_config(SERVICE_ROOT_DIR)


def clamp_loop_count(value: int) -> int:
    return max(1, min(int(value or 1), 3))


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
    return {
        "severity": issue.get("severity"),
        "category": issue.get("category"),
        "location": issue.get("location"),
        "reason": issue.get("reason"),
        "suggestion": issue.get("suggestion"),
        "repair_intent": issue.get("repair_intent"),
    }


def public_generated_question_result(result: dict[str, Any], *, debug: bool) -> dict[str, Any]:
    issues = canonical_issues(result.get("issues") or [])
    repaired = result.get("new_generated_question") if isinstance(result.get("new_generated_question"), dict) else None
    if not debug:
        return {
            "id": result.get("id"),
            "is_good": bool(result.get("is_good")),
            "issues": [compact_issue(issue) for issue in issues],
            "new_generated_question": repaired,
        }

    debug_result: dict[str, Any] = {
        "id": result.get("id"),
        "is_good": bool(result.get("is_good")),
        "issues": issues,
        "new_generated_question": repaired,
    }
    anchor = result.get("solution_anchor_result")
    if isinstance(anchor, dict):
        debug_result["solution_anchor_result"] = anchor
    if result.get("repair_status") and result.get("repair_status") != "skipped":
        debug_result["repair_status"] = result["repair_status"]
    if isinstance(result.get("selected_issue"), dict):
        debug_result["selected_issue"] = result["selected_issue"]
    if result.get("patches"):
        debug_result["repair_patches"] = result["patches"]
    if result.get("repair_loop_count"):
        debug_result["loop_count"] = result["repair_loop_count"]
    if result.get("repair_stop_reason"):
        debug_result["stop_reason"] = result["repair_stop_reason"]
    return debug_result


def public_generated_question_output(result: dict[str, Any], *, debug: bool) -> dict[str, Any]:
    if "results" not in result:
        return public_generated_question_result(result, debug=debug)
    rows = [item for item in result.get("results") or [] if isinstance(item, dict)]
    return {
        "is_good": all(bool(item.get("is_good")) for item in rows),
        "summary": result.get("summary") or {},
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
    anchor_result: dict[str, Any],
    *,
    strict_mode: bool,
    generated_question: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    merged = normalize_generated_question_result(
        {
            "id": base_result.get("id"),
            "is_good": base_result.get("is_good", True),
            "issues": canonical_issues([*(base_result.get("issues") or []), *(anchor_result.get("issues") or [])]),
        },
        strict_mode=strict_mode,
        generated_question=generated_question,
        index=index,
    )
    merged["solution_anchor_result"] = anchor_result
    return merged


def without_judge_solution_semantics(result: dict[str, Any]) -> dict[str, Any]:
    """The generic judge must not duplicate resolver-owned solution semantics."""

    filtered = dict(result)
    filtered_issues: list[dict[str, Any]] = []
    for issue in result.get("issues") or []:
        if not isinstance(issue, dict):
            continue
        category = str(issue.get("category") or "")
        intent = str(issue.get("repair_intent") or "")
        if category in {"answer_internal_consistency", "solution_anchor_consistency"} or intent == "fix_correct_option":
            continue
        if category == "solution_quality" and intent != "clean_solution_reasoning":
            continue
        if category == "hint_quality" and intent == "align_hint_to_solution":
            continue
        filtered_issues.append(issue)
    filtered["issues"] = filtered_issues
    return filtered


def suppress_downstream_issues_for_manual_anchor(
    result: dict[str, Any],
    anchor: dict[str, Any],
) -> dict[str, Any]:
    if anchor.get("resolver_status") != "needs_manual_review":
        return result
    filtered = dict(result)
    filtered["issues"] = [
        issue
        for issue in result.get("issues") or []
        if isinstance(issue, dict) and issue.get("category") not in {"solution_quality", "hint_quality"}
    ]
    return filtered


def text_node_map(generated_question: dict[str, Any]) -> dict[str, str]:
    return {
        str(node.get("path") or ""): str(node.get("text") or "")
        for node in extract_text_nodes_for_spelling(generated_question)
        if str(node.get("path") or "")
    }


def changed_text_paths(original: dict[str, Any], repaired: dict[str, Any]) -> set[str]:
    before, after = text_node_map(original), text_node_map(repaired)
    return {path for path, text in after.items() if before.get(path) != text}


def validate_repaired_text(
    original: dict[str, Any],
    repair_result: dict[str, Any],
    *,
    config: AppConfig,
    client: LLMClient,
    debug: bool,
) -> dict[str, Any]:
    candidate = repair_result.get("new_generated_question")
    if repair_result.get("repair_status") != "repaired" or not isinstance(candidate, dict):
        return repair_result
    changed_paths = changed_text_paths(original, candidate)
    if not changed_paths:
        return repair_result
    spelling = check_spelling_and_wording(candidate, config=config, client=client, debug=debug)
    generated_text_issues = [
        issue for issue in spelling.get("issues") or []
        if str(issue.get("location") or "") in changed_paths
    ]
    if not generated_text_issues:
        return repair_result
    return {
        **repair_result,
        "repair_status": "failed",
        "failed_reason": ["Text do repair sinh ra không đạt spelling/render-safety."],
        "suggestions": ["Review scoped patch vừa sinh; giữ nguyên math/LaTeX và sửa wording."],
        "new_generated_question": None,
    }


def select_repair_issue(check_result: dict[str, Any]) -> dict[str, Any] | None:
    issues = [issue for issue in check_result.get("issues") or [] if isinstance(issue, dict)]
    ordered = sorted(issues, key=issue_sort_key)
    repairable = {
        "align_fields_to_solution",
        "align_hint_to_solution",
        "clean_solution_reasoning",
        "fix_schema",
    }
    return next(
        (issue for issue in ordered if issue.get("repair_intent") in repairable),
        ordered[0] if ordered else None,
    )


def compact_selected_issue(issue: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(issue, dict):
        return None
    return {
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
    anchor = check_result.get("solution_anchor_result")
    anchor = anchor if isinstance(anchor, dict) else {}
    if anchor.get("resolver_status") == "needs_manual_review":
        selected = select_repair_issue(check_result)
        return {
            "repair_status": "needs_manual_review",
            "failed_reason": ["Solution chưa đủ rõ để tự động căn chỉnh đáp án."],
            "suggestions": ["Review thủ công solution và answerSpec/options."],
            "new_generated_question": None,
            "patches": [],
            "selected_issue": compact_selected_issue(selected),
        }

    fixes = anchor.get("fields_to_fix") if isinstance(anchor.get("fields_to_fix"), list) else []
    has_align_intent = any(
        isinstance(issue, dict) and issue.get("repair_intent") == "align_fields_to_solution"
        for issue in check_result.get("issues") or []
    )
    if anchor.get("resolver_status") == "resolved" and fixes and has_align_intent:
        selected = next(
            (
                issue for issue in check_result.get("issues") or []
                if isinstance(issue, dict) and issue.get("repair_intent") == "align_fields_to_solution"
            ),
            None,
        )
        patches = [
            {"op": "replace", "path": fix["path"], "value": fix["value"]}
            for fix in fixes
            if isinstance(fix, dict) and fix.get("path") and "value" in fix
        ]
        result = normalize_scoped_repair_result(
            {"repair_status": "repaired", "failed_reason": [], "suggestions": [], "patches": patches},
            generated_question=generated_question,
            index=index,
        )
        result["selected_issue"] = compact_selected_issue(selected)
        return result

    issue = select_repair_issue(check_result)
    if not issue:
        return {"repair_status": "failed", "new_generated_question": None, "patches": []}
    if issue.get("repair_intent") not in {
        "align_hint_to_solution",
        "clean_solution_reasoning",
        "fix_schema",
    }:
        return {
            "repair_status": "needs_manual_review",
            "failed_reason": ["Issue không thể sửa an toàn bằng scoped patch."],
            "suggestions": ["Review thủ công; full-object repair đã bị vô hiệu hóa."],
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
    last_success: dict[str, Any] | None = None
    last_result: dict[str, Any] | None = None
    stop_reason = "max_loop_reached"
    loop_count = 0

    while loop_count < clamp_loop_count(max_loop) and not current_result.get("is_good"):
        last_result = validate_repaired_text(
            current_question,
            repair_once(current_question, current_result, config=config, client=client, index=index, debug=debug),
            config=config,
            client=client,
            debug=debug,
        )
        loop_count += 1
        candidate = last_result.get("new_generated_question")
        if last_result.get("repair_status") != "repaired" or not isinstance(candidate, dict):
            stop_reason = str(last_result.get("repair_status") or "repair_failed")
            break
        last_success = last_result
        if loop_count >= clamp_loop_count(max_loop):
            break
        current_question = candidate
        current_result = evaluate_generated_question_object(
            current_question,
            strict_mode=strict_mode,
            config=config,
            client=client,
            debug=debug,
            index=index,
            auto_repair=False,
            max_loop=1,
        )
        if current_result.get("is_good"):
            stop_reason = "recheck_good"
            break

    chosen = last_success if last_result and last_result.get("repair_status") != "repaired" and last_success else last_result
    return merge_repair_result(check_result, chosen or {}, loop_count, stop_reason)


def evaluate_generated_question_object(
    generated_question: dict[str, Any],
    *,
    strict_mode: bool = True,
    config: AppConfig | None = None,
    client: LLMClient | None = None,
    debug: bool = False,
    index: int = 0,
    auto_repair: bool = False,
    max_loop: int = 1,
) -> dict[str, Any]:
    try:
        schema_result = validate_generated_question_object(generated_question, index)
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
            anchor = resolve_solution_anchor_consistency(
                generated_question,
                config=config,
                client=client,
                debug=debug,
            )
            judge_result = suppress_downstream_issues_for_manual_anchor(
                without_judge_solution_semantics(
                    judge_generated_question_object(
                        generated_question,
                        schema_result,
                        config,
                        client,
                        strict_mode=strict_mode,
                        index=index,
                        debug=debug,
                    )
                ),
                anchor,
            )
            checked = merge_generated_question_results(
                schema_issues=schema_issues,
                llm_result=judge_result,
                strict_mode=strict_mode,
                generated_question=generated_question,
                index=index,
            )
            checked = merge_anchor_result(
                checked,
                anchor,
                strict_mode=strict_mode,
                generated_question=generated_question,
                index=index,
            )
        return maybe_repair_generated_question(
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
    max_loop: int = 1,
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
    results: list[dict[str, Any]] = []
    for index, generated_question in enumerate(generated_questions):
        if progress_callback:
            progress_callback(index + 1, len(generated_questions), generated_question)
        results.append(
            evaluate_generated_question_object(
                generated_question,
                strict_mode=strict_mode,
                config=config,
                client=client,
                debug=debug,
                index=index,
                auto_repair=auto_repair,
                max_loop=max_loop,
            )
        )

    if should_return_aggregate(payload, generated_questions):
        return public_generated_question_output(aggregate_generated_question_results(results), debug=debug)
    if results:
        return public_generated_question_output(results[0], debug=debug)
    return public_generated_question_output(aggregate_generated_question_results(results), debug=debug)

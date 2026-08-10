"""Run the one-operation-per-transition stability benchmark with stage capture."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.question_plan.flows import generated_question_service as service
from src.question_plan.benchmarks.grade_solution_fixtures import (
    load_grade_solution_fixtures,
)
from src.question_plan.logic import generated_question_judge as judge


def make_case(
    template: dict[str, Any],
    *,
    case_id: str,
    instruction: str,
    solution_texts: list[str],
    expected: Any,
) -> dict[str, Any]:
    result = copy.deepcopy(template)
    result["_id"] = f"stability-{case_id}"
    result["aiId"] = f"stability-{case_id}"
    result["instruction"][0]["text"] = instruction
    result["solutions"][0]["solutionContent"] = [
        {
            "id": f"solution-{case_id}-{index}",
            "type": "text",
            "text": text,
        }
        for index, text in enumerate(solution_texts)
    ]
    result["questionItems"][0]["answerSpecs"][0]["expected"] = {
        "correctValue": expected
    }
    return result


def benchmark_cases() -> dict[str, dict[str, Any]]:
    fixtures = load_grade_solution_fixtures(ROOT)
    by_id = {item["_id"]: item for item in fixtures}
    template = by_id["grade-7-pipeline-missing-major-step"]
    return {
        "A": copy.deepcopy(template),
        "B": make_case(
            template,
            case_id="B",
            instruction="Giải phương trình x + 3 = 16.",
            solution_texts=["Ta có x + 3 = 16.", "Suy ra x = 13."],
            expected=13,
        ),
        "C": make_case(
            template,
            case_id="C",
            instruction="Giải phương trình 3x = 12.",
            solution_texts=["Ta có 3x = 12.", "Suy ra x = 4."],
            expected=4,
        ),
        "D": make_case(
            template,
            case_id="D",
            instruction="Giải phương trình x^2 = 9.",
            solution_texts=[
                "Ta có x^2 = 9.",
                "Suy ra x = -3 hoặc x = 3.",
            ],
            expected=[-3, 3],
        ),
        "E": make_case(
            template,
            case_id="E",
            instruction="Giải phương trình 2x^3 + 3 = 19.",
            solution_texts=[
                "Ta có 2x^3 + 3 = 19.",
                "Suy ra x^3 = 8.",
                "Suy ra x = 2.",
            ],
            expected=2,
        ),
        "F": make_case(
            template,
            case_id="F",
            instruction="Giải phương trình 2x^3 = 16.",
            solution_texts=["Ta có 2x^3 = 16.", "Suy ra x = 2."],
            expected=2,
        ),
        "G": copy.deepcopy(
            by_id["grade-7-pipeline-separated-transitions-valid"]
        ),
        "H": make_case(
            template,
            case_id="H",
            instruction="Tính 7 + 5.",
            solution_texts=["Ta có 7 + 5 = 12."],
            expected=12,
        ),
        "I": make_case(
            template,
            case_id="I",
            instruction="Khai triển (x + 1)^2.",
            solution_texts=[
                "Ta có (x + 1)^2.",
                "Kết quả là x^2 + 2x + 1.",
            ],
            expected="x^2 + 2x + 1",
        ),
        "J": make_case(
            template,
            case_id="J",
            instruction="Rút gọn 2(x + 1)^2 - 2.",
            solution_texts=[
                "Ta có 2(x + 1)^2 - 2.",
                "Kết quả là 2x^2 + 4x.",
            ],
            expected="2x^2 + 4x",
        ),
        "K": make_case(
            template,
            case_id="K",
            instruction="Giải phương trình 2x^3 + 3 = 19.",
            solution_texts=["Biến đổi trực tiếp phương trình, ta được x = 2."],
            expected=2,
        ),
        "L": copy.deepcopy(
            by_id["grade-12-pipeline-integral-missing-major-step"]
        ),
    }


def final_error_type(result: dict[str, Any]) -> str | None:
    transition = result.get("first_invalid_transition")
    if isinstance(transition, dict):
        return transition.get("error_type")
    context = result.get("context_issue")
    if isinstance(context, dict):
        return context.get("error_type")
    issue = result.get("issue")
    if isinstance(issue, dict):
        return issue.get("error_type")
    return None


def final_anchor(result: dict[str, Any]) -> dict[str, Any] | None:
    transition = result.get("first_invalid_transition")
    if isinstance(transition, dict):
        return {
            key: transition.get(key)
            for key in (
                "solution_index",
                "from_order",
                "to_order",
                "source_path",
                "evidence_text",
            )
        }
    context = result.get("context_issue")
    if isinstance(context, dict):
        return {
            "scope": "context",
            "source_path": context.get("source_path"),
            "evidence_text": context.get("evidence_text"),
        }
    return None


def run_benchmark(
    repeats: int,
    selected_cases: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = benchmark_cases()
    if selected_cases:
        cases = {
            case_id: value
            for case_id, value in cases.items()
            if case_id in selected_cases
        }
    records: list[dict[str, Any]] = []
    case_for_record: dict[str, str] = {}
    for run_index in range(1, repeats + 1):
        for case_id, base in cases.items():
            item = copy.deepcopy(base)
            record_id = f"stability-{case_id}-run-{run_index:02d}"
            item["_id"] = record_id
            item["aiId"] = record_id
            records.append(item)
            case_for_record[record_id] = case_id

    lock = Lock()
    trace: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "splitter_attempts": [],
            "code_splitter_used": False,
            "correctness_attempts": [],
            "presentation_attempts": [],
            "aggregate": None,
            "resolver": None,
        }
    )
    original_splitter = judge._call_solution_splitter
    original_code_splitter = judge.split_solution_with_code
    original_correctness = judge._call_transition_judge
    original_presentation = judge._call_process_presentation_judge
    original_aggregate = judge.aggregate_specialized_judge_results
    original_resolver = service.resolve_solution_anchor_consistency

    def capture_splitter(
        generated_question: dict[str, Any], *args: Any, **kwargs: Any
    ) -> tuple[dict[str, Any] | None, str]:
        result, error = original_splitter(generated_question, *args, **kwargs)
        with lock:
            trace[generated_question["_id"]]["splitter_attempts"].append(
                {"valid": result is not None, "error": error}
            )
        return result, error

    def capture_code_splitter(
        generated_question: dict[str, Any], *args: Any, **kwargs: Any
    ) -> tuple[dict[str, Any] | None, str]:
        result, error = original_code_splitter(generated_question, *args, **kwargs)
        with lock:
            trace[generated_question["_id"]]["code_splitter_used"] = True
            trace[generated_question["_id"]]["code_splitter_valid"] = (
                result is not None
            )
            trace[generated_question["_id"]]["code_splitter_error"] = error
        return result, error

    def capture_correctness(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_correctness(*args, **kwargs)
        generated_question = kwargs["generated_question"]
        with lock:
            trace[generated_question["_id"]]["correctness_attempts"].append(
                copy.deepcopy(result)
            )
        return result

    def capture_presentation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_presentation(*args, **kwargs)
        generated_question = kwargs["generated_question"]
        with lock:
            trace[generated_question["_id"]]["presentation_attempts"].append(
                copy.deepcopy(result)
            )
        return result

    def capture_aggregate(
        correctness_result: dict[str, Any],
        presentation_result: dict[str, Any],
        ordered_solution: dict[str, Any],
        generated_question: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        result = original_aggregate(
            correctness_result,
            presentation_result,
            ordered_solution,
            generated_question,
            **kwargs,
        )
        selected_source = None
        selected_type = None
        issues = result.get("issues") or []
        selected_reason = issues[0].get("reason") if issues else None
        if selected_reason and selected_reason == correctness_result.get("reason"):
            selected_source = "correctness"
            selected_type = final_error_type(correctness_result)
        elif selected_reason and selected_reason == presentation_result.get("reason"):
            selected_source = "process_presentation"
            selected_type = final_error_type(presentation_result)
        with lock:
            trace[generated_question["_id"]]["aggregate"] = {
                "is_good": result.get("is_good"),
                "selected_source": selected_source,
                "selected_issue_type": selected_type,
                "selected_issue_path": (
                    issues[0].get("location") if issues else None
                ),
            }
        return result

    def capture_resolver(
        generated_question: dict[str, Any], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        result = original_resolver(generated_question, *args, **kwargs)
        with lock:
            trace[generated_question["_id"]]["resolver"] = copy.deepcopy(result)
        return result

    judge._call_solution_splitter = capture_splitter
    judge.split_solution_with_code = capture_code_splitter
    judge._call_transition_judge = capture_correctness
    judge._call_process_presentation_judge = capture_presentation
    judge.aggregate_specialized_judge_results = capture_aggregate
    service.resolve_solution_anchor_consistency = capture_resolver
    try:
        evaluated = service.evaluate_generated_questions(
            records,
            workers=2,
            debug=False,
            auto_repair=False,
        )
    finally:
        judge._call_solution_splitter = original_splitter
        judge.split_solution_with_code = original_code_splitter
        judge._call_transition_judge = original_correctness
        judge._call_process_presentation_judge = original_presentation
        judge.aggregate_specialized_judge_results = original_aggregate
        service.resolve_solution_anchor_consistency = original_resolver

    rows_by_id = {row["id"]: row for row in evaluated["results"]}
    runs: list[dict[str, Any]] = []
    for record in records:
        record_id = record["_id"]
        case_id = case_for_record[record_id]
        run_index = int(record_id.rsplit("-", 1)[1])
        item_trace = trace[record_id]
        correctness = (
            item_trace["correctness_attempts"][-1]
            if item_trace["correctness_attempts"]
            else {}
        )
        presentation = (
            item_trace["presentation_attempts"][-1]
            if item_trace["presentation_attempts"]
            else {}
        )
        aggregate = item_trace["aggregate"] or {}
        resolver = item_trace["resolver"]
        final = rows_by_id[record_id]
        runtime_issues = [
            issue
            for issue in final.get("issues") or []
            if issue.get("category") == "runtime"
        ]
        contract_errors = [
            attempt.get("contract_error")
            for attempt in (
                item_trace["correctness_attempts"]
                + item_trace["presentation_attempts"]
            )
            if attempt.get("contract_valid") is False
        ]
        if isinstance(resolver, dict) and resolver.get("resolver_contract_error"):
            contract_errors.append(resolver["resolver_contract_error"])
        runs.append(
            {
                "run_index": run_index,
                "case_id": case_id,
                "record_id": record_id,
                "splitter_status": (
                    "valid"
                    if item_trace["splitter_attempts"]
                    and (
                        item_trace["splitter_attempts"][-1]["valid"]
                        or item_trace.get("code_splitter_valid")
                    )
                    else "invalid"
                ),
                "splitter_fallback": bool(
                    item_trace["code_splitter_used"]
                    or len(item_trace["splitter_attempts"]) > 1
                ),
                "correctness_verdict": correctness.get("verdict"),
                "correctness_error_type": final_error_type(correctness),
                "correctness_anchor": final_anchor(correctness),
                "presentation_verdict": presentation.get("verdict"),
                "aggregate_verdict": (
                    "good" if aggregate.get("is_good") else "bad"
                ),
                "selected_issue_type": aggregate.get("selected_issue_type"),
                "selected_issue_path": aggregate.get("selected_issue_path"),
                "resolver_called": resolver is not None,
                "resolver_status": (
                    resolver.get("resolver_status")
                    if isinstance(resolver, dict)
                    else None
                ),
                "resolver_fallback": bool(
                    resolver.get("resolver_fallback_called")
                    if isinstance(resolver, dict)
                    else False
                ),
                "final_is_good": final.get("is_good"),
                "runtime_step": (
                    runtime_issues[0].get("reason") if runtime_issues else None
                ),
                "contract_error": (
                    " | ".join(str(value) for value in contract_errors if value)
                    or None
                ),
            }
        )
    return runs, evaluated.get("summary") or {}


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    expected_bad = {"A", "E", "F", "J", "K", "L"}
    summary: dict[str, Any] = {}
    for case_id in sorted({run["case_id"] for run in runs}):
        case_runs = [run for run in runs if run["case_id"] == case_id]
        bad_expected = case_id in expected_bad
        matching_final = sum(
            run["final_is_good"] is (not bad_expected) for run in case_runs
        )
        matching_correctness = sum(
            (
                run["correctness_verdict"] == "bad"
                and run["correctness_error_type"] == "missing_major_step"
            )
            if bad_expected
            else run["correctness_verdict"] == "good"
            for run in case_runs
        )
        resolver_gate_matches = sum(
            (not run["resolver_called"]) if bad_expected else run["resolver_called"]
            for run in case_runs
        )
        summary[case_id] = {
            "runs": len(case_runs),
            "expected": "bad/missing_major_step" if bad_expected else "good",
            "matching_final": matching_final,
            "matching_correctness": matching_correctness,
            "matching_resolver_gate": resolver_gate_matches,
            "splitter_fallbacks": sum(
                bool(run["splitter_fallback"]) for run in case_runs
            ),
            "resolver_fallbacks": sum(
                bool(run["resolver_fallback"]) for run in case_runs
            ),
            "runtime": sum(run["runtime_step"] is not None for run in case_runs),
            "contract_errors": sum(
                run["contract_error"] is not None for run in case_runs
            ),
            "stable": (
                matching_final == len(case_runs)
                and matching_correctness == len(case_runs)
                and resolver_gate_matches == len(case_runs)
                and all(run["runtime_step"] is None for run in case_runs)
                and all(run["contract_error"] is None for run in case_runs)
            ),
        }
    return summary


def report_markdown(summary: dict[str, Any], runs: list[dict[str, Any]]) -> str:
    expected_bad = {"A", "E", "F", "J", "K", "L"}
    lines = [
        "# One-operation stability report",
        "",
        "| Case | Expected | Final | Correctness | Resolver gate | Splitter fallback | Resolver fallback | Runtime | Contract | Stable |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case_id, item in summary.items():
        lines.append(
            f"| {case_id} | {item['expected']} | {item['matching_final']}/{item['runs']} "
            f"| {item['matching_correctness']}/{item['runs']} "
            f"| {item['matching_resolver_gate']}/{item['runs']} "
            f"| {item['splitter_fallbacks']} | {item['resolver_fallbacks']} "
            f"| {item['runtime']} | {item['contract_errors']} "
            f"| {'yes' if item['stable'] else 'no'} |"
        )
    unstable = [
        run
        for run in runs
        if (
            run["runtime_step"]
            or run["contract_error"]
            or (
                run["case_id"] in expected_bad
                and (
                    run["correctness_verdict"] != "bad"
                    or run["correctness_error_type"] != "missing_major_step"
                    or run["final_is_good"]
                    or run["resolver_called"]
                )
            )
            or (
                run["case_id"] not in expected_bad
                and (
                    run["correctness_verdict"] != "good"
                    or not run["final_is_good"]
                    or not run["resolver_called"]
                )
            )
        )
    ]
    lines.extend(["", f"Unstable runs: {len(unstable)}", ""])
    for run in unstable:
        lines.append(
            f"- {run['record_id']}: correctness={run['correctness_verdict']}/"
            f"{run['correctness_error_type']}, final={run['final_is_good']}, "
            f"resolver_called={run['resolver_called']}, runtime={run['runtime_step']}, "
            f"contract={run['contract_error']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--cases",
        default="A,B,C,D,E,F,G,H,I,J",
        help="Comma-separated benchmark case IDs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "one_operation_stability_runs.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "one_operation_stability_report.md",
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    selected_cases = {
        value.strip() for value in args.cases.split(",") if value.strip()
    }
    runs, service_summary = run_benchmark(args.repeats, selected_cases)
    stability = summarize(runs)
    payload = {
        "policy": "Mỗi transition chỉ chứa tối đa một phép biến đổi toán học hoặc một bước suy luận chính.",
        "repeats": args.repeats,
        "service_summary": service_summary,
        "stability": stability,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.report.write_text(
        report_markdown(stability, runs),
        encoding="utf-8",
    )
    print(json.dumps(stability, ensure_ascii=False))


if __name__ == "__main__":
    main()

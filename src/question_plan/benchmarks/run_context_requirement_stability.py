"""Run Context & Transition Builder stability cases with internal stage capture."""

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


def _numeric_question(
    case_id: str,
    instruction: list[dict[str, Any]],
    solution: str,
    answer: int,
) -> dict[str, Any]:
    return {
        "_id": case_id,
        "aiId": case_id,
        "difficulty": "medium",
        "bloom": "apply",
        "interactionTypes": ["short_answer"],
        "instruction": instruction,
        "questionItems": [{
            "id": f"item-{case_id}",
            "stem": [{"id": f"stem-{case_id}", "type": "text", "text": "Nhập đáp án."}],
            "interactions": [{
                "id": "answer",
                "type": "short_answer",
                "config": {"inputMode": "numeric"},
                "display": {"layout": "auto"},
            }],
            "answerSpecs": [{
                "interactionId": "answer",
                "type": "short_answer",
                "expected": [{
                    "inputMode": "numeric",
                    "value": {"correctValue": [answer], "acceptableValues": []},
                    "equivalence": {"type": "numeric_equivalence"},
                }],
            }],
        }],
        "solutions": [{
            "solverName": "default",
            "solutionContent": [{
                "id": f"solution-{case_id}",
                "type": "text",
                "text": solution,
            }],
        }],
    }


def cases() -> dict[str, dict[str, Any]]:
    fixtures = load_grade_solution_fixtures(ROOT)
    by_id = {item["_id"]: item for item in fixtures}
    case_017 = copy.deepcopy(
        by_id["grade-9-pipeline-external-graph-missing"]
    )
    case_017["_id"] = "context-case-017"
    case_017["aiId"] = "context-case-017"
    missing_graph = _numeric_question(
        "context-missing-graph",
        [{
            "id": "intro",
            "type": "text",
            "text": "Quan sát đồ thị hàm số dưới đây và xác định nghiệm của f(x)=0.",
        }],
        "Quan sát đồ thị, nghiệm của f(x)=0 là x = 2.",
        2,
    )
    asset_only = _numeric_question(
        "context-asset-only",
        [
            {
                "id": "intro",
                "type": "text",
                "text": "Dựa vào bảng biến thiên sau, xác định số khoảng đồng biến.",
            },
            {
                "id": "variation-image",
                "type": "image",
                "assetId": "variation-001",
            },
        ],
        "Theo bảng biến thiên, hàm số có 2 khoảng đồng biến.",
        2,
    )
    geometry_figure_missing = _numeric_question(
        "context-geometry-figure-missing",
        [{
            "id": "intro",
            "type": "text",
            "text": (
                "Trong hình bên, biết hai đường thẳng song song. "
                "Tính số đo góc x."
            ),
        }],
        "Theo hình, góc x bằng 60 độ.",
        60,
    )
    text_table = _numeric_question(
        "context-text-table",
        [{
            "id": "intro",
            "type": "text",
            "text": (
                "Dựa vào bảng số liệu sau để tính trung bình: "
                "Giá trị: 2 | 4 | 6; Tần số: 1 | 1 | 1."
            ),
        }],
        "Trung bình cộng là (2 + 4 + 6)/3 = 4.",
        4,
    )
    geometry = copy.deepcopy(by_id["grade-8-thales-valid"])
    geometry["_id"] = "context-geometry-text-complete"
    geometry["aiId"] = "context-geometry-text-complete"
    return {
        "case_017": case_017,
        "missing_graph": missing_graph,
        "asset_only": asset_only,
        "geometry_figure_missing": geometry_figure_missing,
        "text_table": text_table,
        "geometry_text_complete": geometry,
    }


def _error_type(result: dict[str, Any]) -> str | None:
    issue = result.get("context_issue") or result.get("first_invalid_transition")
    return issue.get("error_type") if isinstance(issue, dict) else None


def run(repeats: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark = cases()
    records: list[dict[str, Any]] = []
    metadata: dict[str, tuple[str, int]] = {}
    for run_index in range(1, repeats + 1):
        for case_id, source in benchmark.items():
            item = copy.deepcopy(source)
            record_id = f"context-{case_id}-run-{run_index:02d}"
            item["_id"] = record_id
            item["aiId"] = record_id
            records.append(item)
            metadata[record_id] = (case_id, run_index)

    lock = Lock()
    traces: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "splitter_attempts": [],
            "code_splitter": None,
            "correctness_attempts": [],
            "resolver": None,
        }
    )
    original_splitter = judge._call_solution_splitter
    original_code_splitter = judge.split_solution_with_code
    original_correctness = judge._call_transition_judge
    original_resolver = service.resolve_solution_anchor_consistency

    def capture_splitter(
        generated_question: dict[str, Any], *args: Any, **kwargs: Any
    ) -> tuple[dict[str, Any] | None, str]:
        result, error = original_splitter(generated_question, *args, **kwargs)
        with lock:
            traces[generated_question["_id"]]["splitter_attempts"].append({
                "valid": result is not None,
                "context_requirements": copy.deepcopy(
                    result.get("context_requirements") if result else None
                ),
                "error": error or None,
            })
        return result, error

    def capture_code_splitter(
        generated_question: dict[str, Any], *args: Any, **kwargs: Any
    ) -> tuple[dict[str, Any] | None, str]:
        result, error = original_code_splitter(generated_question, *args, **kwargs)
        with lock:
            traces[generated_question["_id"]]["code_splitter"] = {
                "valid": result is not None,
                "context_requirements": copy.deepcopy(
                    result.get("context_requirements") if result else None
                ),
                "error": error or None,
            }
        return result, error

    def capture_correctness(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_correctness(*args, **kwargs)
        with lock:
            traces[kwargs["generated_question"]["_id"]][
                "correctness_attempts"
            ].append(copy.deepcopy(result))
        return result

    def capture_resolver(
        generated_question: dict[str, Any], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        result = original_resolver(generated_question, *args, **kwargs)
        with lock:
            traces[generated_question["_id"]]["resolver"] = copy.deepcopy(result)
        return result

    judge._call_solution_splitter = capture_splitter
    judge.split_solution_with_code = capture_code_splitter
    judge._call_transition_judge = capture_correctness
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
        service.resolve_solution_anchor_consistency = original_resolver

    public_by_id = {item["id"]: item for item in evaluated["results"]}
    rows: list[dict[str, Any]] = []
    for record in records:
        record_id = record["_id"]
        case_id, run_index = metadata[record_id]
        trace = traces[record_id]
        splitter_attempts = trace["splitter_attempts"]
        chosen_splitter = next(
            (
                attempt
                for attempt in reversed(splitter_attempts)
                if attempt["valid"]
            ),
            trace["code_splitter"],
        ) or {}
        correctness_attempts = trace["correctness_attempts"]
        correctness = correctness_attempts[-1] if correctness_attempts else {}
        public = public_by_id[record_id]
        context_issue = correctness.get("context_issue")
        runtime = next(
            (
                issue.get("reason")
                for issue in public.get("issues") or []
                if issue.get("category") == "runtime"
            ),
            None,
        )
        contract_errors = [
            attempt.get("error")
            for attempt in splitter_attempts
            if not attempt["valid"] and attempt.get("error")
        ]
        contract_errors.extend(
            attempt.get("contract_error")
            for attempt in correctness_attempts
            if attempt.get("contract_valid") is False
            and attempt.get("contract_error")
        )
        rows.append({
            "run_index": run_index,
            "case_id": case_id,
            "record_id": record_id,
            "builder_context_requirements": chosen_splitter.get(
                "context_requirements"
            ),
            "builder_attempts": len(splitter_attempts),
            "builder_code_fallback": trace["code_splitter"] is not None,
            "correctness_verdict": correctness.get("verdict"),
            "correctness_error_type": _error_type(correctness),
            "context_issue": copy.deepcopy(context_issue),
            "transition_evaluation_started": not (
                correctness.get("verdict") == "uncertain"
                and isinstance(context_issue, dict)
            ),
            "resolver_called": trace["resolver"] is not None,
            "final_is_good": public.get("is_good"),
            "runtime": runtime,
            "contract_error": " | ".join(contract_errors) or None,
        })
    return rows, evaluated.get("summary") or {}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    blocking_cases = {
        "case_017",
        "missing_graph",
        "asset_only",
        "geometry_figure_missing",
    }
    expected_types = {
        "case_017": "graph",
        "missing_graph": "graph",
        "asset_only": "variation_table",
        "geometry_figure_missing": "geometry_figure",
    }
    summary: dict[str, Any] = {}
    for case_id in sorted({row["case_id"] for row in rows}):
        case_rows = [row for row in rows if row["case_id"] == case_id]
        blocking = case_id in blocking_cases
        matches = 0
        for row in case_rows:
            requirements = row["builder_context_requirements"] or []
            availabilities = {
                requirement.get("availability")
                for requirement in requirements
            }
            builder_match = (
                bool(requirements)
                and bool(availabilities & {"missing", "insufficient"})
                if blocking
                else not bool(availabilities & {"missing", "insufficient"})
            )
            if blocking:
                builder_match = builder_match and {
                    requirement.get("requirement_type")
                    for requirement in requirements
                } == {expected_types[case_id]}
            if case_id == "asset_only":
                builder_match = (
                    bool(requirements)
                    and availabilities == {"insufficient"}
                )
            correctness_match = (
                row["correctness_verdict"] == "uncertain"
                and row["correctness_error_type"] == "insufficient_context"
                and not row["transition_evaluation_started"]
                and not row["resolver_called"]
                if blocking
                else row["correctness_verdict"] != "uncertain"
            )
            if builder_match and correctness_match:
                matches += 1
        summary[case_id] = {
            "runs": len(case_rows),
            "matching": matches,
            "builder_fallbacks": sum(
                row["builder_attempts"] > 1 or row["builder_code_fallback"]
                for row in case_rows
            ),
            "runtime": sum(row["runtime"] is not None for row in case_rows),
            "contract_errors": sum(
                row["contract_error"] is not None for row in case_rows
            ),
            "stable": (
                matches == len(case_rows)
                and all(row["runtime"] is None for row in case_rows)
                and all(row["contract_error"] is None for row in case_rows)
            ),
        }
    return summary


def markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Context requirement stability report",
        "",
        "| Case | Match | Builder fallback | Runtime | Contract | Stable |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for case_id, item in summary.items():
        lines.append(
            f"| {case_id} | {item['matching']}/{item['runs']} "
            f"| {item['builder_fallbacks']} | {item['runtime']} "
            f"| {item['contract_errors']} | {'yes' if item['stable'] else 'no'} |"
        )
    lines.extend(["", "## Runs", ""])
    for row in rows:
        requirements = json.dumps(
            row["builder_context_requirements"],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        lines.append(
            f"- {row['record_id']}: builder={requirements}; "
            f"correctness={row['correctness_verdict']}/"
            f"{row['correctness_error_type']}; "
            f"transition_started={row['transition_evaluation_started']}; "
            f"resolver={row['resolver_called']}; runtime={row['runtime']}; "
            f"contract={row['contract_error']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "context_requirement_stability_runs.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "context_requirement_stability_report.md",
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    rows, service_summary = run(args.repeats)
    stability = summarize(rows)
    payload = {
        "repeats": args.repeats,
        "service_summary": service_summary,
        "stability": stability,
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.report.write_text(markdown(stability, rows), encoding="utf-8")
    print(json.dumps(stability, ensure_ascii=False))


if __name__ == "__main__":
    main()

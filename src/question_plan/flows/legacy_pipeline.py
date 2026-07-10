"""Pipeline cũ để đánh giá chất lượng question_plan theo từng source record."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ...config import AppConfig
from ...llm_client import LLMClient
from ..schemas.eval_schema import (
    build_plan_eval_summary,
    skipped_due_to_source_issue_result,
    structural_error_result,
)
from ..logic.judge import judge_question_plan
from ..logic.repair_suggester import suggest_question_plan_repair
from ..logic.rule_validator import validate_question_plan_structure


SOURCE_BLOCKING_STATUSES = {"warning", "needs_review", "critical", "skipped", "unchecked", "unknown"}
DEFAULT_SOURCE_CHECK_SUMMARY_PATHS = (
    "results/outputs/source_record/run_summary.json",
    "results/outputs/source_record/source_record_problem_report.json",
    "results/outputs/report/source_record_problem_report.json",
    "results/quick_check/source_record_problem_report.json",
    "results/quick_check/source_alignment_summary.json",
    "results/quick_check/source_record_pipeline_summary.csv",
)
SOURCE_STATUS_FIELDS = (
    "source_alignment_status",
    "severity",
    "status",
    "overall_status",
    "final_status",
    "source_record_status",
)
REPAIR_ELIGIBLE_STATUSES = {"warning", "needs_review", "bad", "structural_error"}


def default_repair_result(manual_review_required: bool = False) -> dict[str, Any]:
    return {
        "repair_suggestions": [],
        "rewritten_question_plan_preview": None,
        "preview_strategy": "not_safe",
        "repair_confidence": "high",
        "manual_review_required": manual_review_required,
        "repair_error": "",
        "preview_validation_error": "",
        "repair_json_parse_ok": None,
        "repair_model": "",
        "repair_latency_seconds": 0,
    }


def attach_repair_result(
    result: dict[str, Any],
    record: dict[str, Any],
    config: AppConfig,
    client: LLMClient,
) -> dict[str, Any]:
    if result.get("overall_status") not in REPAIR_ELIGIBLE_STATUSES:
        return {
            **result,
            **default_repair_result(manual_review_required=False),
        }
    try:
        repair_result = suggest_question_plan_repair(record, result, config, client)
    except Exception as exc:
        repair_result = {
            **default_repair_result(manual_review_required=True),
            "repair_confidence": "low",
            "repair_error": str(exc),
            "repair_json_parse_ok": False,
        }
    return {
        **result,
        **repair_result,
    }


def is_false_like(value: Any) -> bool:
    return value is False or str(value).strip().lower() == "false"


def is_true_like(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def record_id_from_mapping(row: dict[str, Any]) -> str:
    return str(row.get("record_id") or row.get("_id") or row.get("id") or "").strip()


def source_status_from_mapping(row: dict[str, Any]) -> str:
    for field in SOURCE_STATUS_FIELDS:
        value = row.get(field)
        if value not in (None, ""):
            return str(value).strip().lower()
    if is_false_like(row.get("source_record_valid")) or is_true_like(row.get("need_human_review")):
        return "needs_review"
    return ""


def collect_source_issue_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []

    rows: list[dict[str, Any]] = []
    for key in (
        "critical_cases",
        "warning_cases",
        "needs_review_cases",
        "skipped_unchecked_unknown_cases",
        "issue_cases",
        "bad_cases",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))

    results = payload.get("results")
    if isinstance(results, dict):
        rows.extend(collect_source_issue_rows(results))
    elif isinstance(results, list):
        rows.extend(row for row in results if isinstance(row, dict))
    return rows


def resolve_source_check_summary_path(source_check_summary: str | None, root_dir: Path | None = None) -> Path | None:
    candidates = [source_check_summary] if source_check_summary else list(DEFAULT_SOURCE_CHECK_SUMMARY_PATHS)
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = (root_dir or Path.cwd()) / path
        if path.is_file():
            return path
    return None


def load_source_issue_lookup(
    source_check_summary: str | None,
    root_dir: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], str]:
    path = resolve_source_check_summary_path(source_check_summary, root_dir)
    if path is None:
        return {}, ""
    if not path.is_absolute():
        path = (root_dir or Path.cwd()) / path
    if not path.is_file():
        return {}, ""

    rows: list[dict[str, Any]] = []
    try:
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as file:
                rows = [row for row in csv.DictReader(file)]
        else:
            with path.open("r", encoding="utf-8-sig") as file:
                rows = collect_source_issue_rows(json.load(file))
    except (OSError, ValueError, csv.Error):
        return {}, str(path)

    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = record_id_from_mapping(row)
        status = source_status_from_mapping(row)
        if record_id and status in SOURCE_BLOCKING_STATUSES:
            lookup[record_id] = row
    return lookup, str(path)


def run_question_plan_eval_case(
    record: dict[str, Any],
    record_index: int,
    config: AppConfig,
    client: LLMClient,
    source_issue_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source_issue_lookup = source_issue_lookup or {}
    record_id = (
        record.get("_id") or record.get("record_id") or record.get("id") or f"record_{record_index + 1}"
        if isinstance(record, dict)
        else f"record_{record_index + 1}"
    )
    if str(record_id) in source_issue_lookup:
        result = skipped_due_to_source_issue_result(record, source_issue_lookup[str(record_id)])
        return {
            **result,
            **default_repair_result(manual_review_required=False),
            "judge_model": "",
            "json_parse_ok": None,
            "judge_error": "",
            "fallback_called": False,
            "fallback_model": None,
            "final_decision_policy": "source_check_skip_llm",
            "latency_seconds": 0,
            "record": record,
        }

    structural = validate_question_plan_structure(record, record_index)
    if not structural.get("llm_judge_required"):
        result = structural_error_result(record, structural)
        result = {
            **result,
            "judge_model": "",
            "json_parse_ok": None,
            "judge_error": "",
            "fallback_called": False,
            "fallback_model": None,
            "final_decision_policy": "structural_gate_skip_llm",
            "latency_seconds": 0,
            "record": record,
        }
        return attach_repair_result(result, record, config, client)

    result = judge_question_plan(record, config, client, structural)
    result = {
        **result,
        "record": record,
    }
    return attach_repair_result(result, record, config, client)


def run_question_plan_eval_pipeline(
    records: list[dict[str, Any]],
    config: AppConfig,
    source_check_summary: str | None = None,
) -> dict[str, Any]:
    client = LLMClient(config)
    source_issue_lookup, source_check_summary_path = load_source_issue_lookup(source_check_summary, config.root_dir)
    if not source_check_summary_path:
        print("Không tìm thấy source-check summary, chạy đánh giá question_plan cho tất cả records.")
    results = []
    for index, record in enumerate(records):
        record_id = (
            record.get("_id") or record.get("record_id") or record.get("id") or f"record_{index + 1}"
            if isinstance(record, dict)
            else f"record_{index + 1}"
        )
        print(f"Running question_plan quality case: {record_id}")
        results.append(run_question_plan_eval_case(record, index, config, client, source_issue_lookup))
    summary = build_plan_eval_summary(results)
    summary["source_check_summary"] = source_check_summary_path
    summary["source_check_skipped_count"] = sum(
        1 for row in results if row.get("overall_status") == "skipped_due_to_source_issue"
    )
    issue_details = [row for row in results if row.get("overall_status") != "ok"]
    return {
        "results": results,
        "summary": summary,
        "issue_details": issue_details,
    }


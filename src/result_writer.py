"""Ghi kết quả pipeline ra JSON/CSV/Markdown.

File này gom toàn bộ logic xuất file cho real-question, source-record,
label-assignment và repair report để các pipeline chỉ cần trả về dữ liệu.
"""

import csv
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .real_schema import VALID_REAL_INTERACTION_TYPES

INTERACTION_TYPE_OUTPUT_VERSION = "2026-07-question-level-outputs"
ISSUE_STATUSES = {"bad", "warning", "needs_review", "structural_error"}
BAD_STATUSES = {"bad", "structural_error"}
WARNING_STATUSES = {"warning", "needs_review"}
SKIPPED_SOURCE_STATUSES = {"skipped_due_to_source_issue"}


REAL_QUESTION_FIELDS = [
    "case_id",
    "record_id",
    "record_name",
    "generated_question_id",
    "question_item_id",
    "interaction_id",
    "interaction_index",
    "start_page",
    "end_page",
    "declared_interaction_type",
    "planned_interaction_type",
    "detected_interaction_type",
    "suggested_interaction_type",
    "schema_valid",
    "rule_passed",
    "object_structure_valid",
    "interaction_schema_valid",
    "question_text_valid",
    "interaction_type_valid",
    "interaction_type_mismatch",
    "soft_mismatch",
    "warning_type",
    "suggested_normalized_type",
    "policy_override_applied",
    "plan_alignment_valid",
    "source_question_grounded",
    "pdf_grounded",
    "classifier_model",
    "classifier_confidence",
    "classifier_json_parse_ok",
    "classifier_fallback_called",
    "classifier_error",
    "judge_model",
    "judge_confidence",
    "judge_json_parse_ok",
    "judge_fallback_called",
    "judge_error",
    "answer_check_skipped",
    "need_human_review",
    "error_type",
    "reason",
    "evidence",
]


SOURCE_RECORD_FIELDS = [
    "case_id",
    "record_id",
    "record_name",
    "start_page",
    "end_page",
    "record_structure_valid",
    "raw_question_text_valid",
    "raw_answer_text_valid",
    "raw_answer_matches_raw_question",
    "pdf_extraction_available",
    "pdf_question_grounded",
    "pdf_answer_grounded",
    "pdf_page_range_relevant",
    "plan_evaluation_skipped",
    "label_quality",
    "label_issue_count",
    "source_record_valid",
    "need_human_review",
    "severity",
    "error_type",
    "reason",
    "evidence",
    "judge_model",
    "judge_json_parse_ok",
    "judge_fallback_called",
    "latency_seconds",
]


SOURCE_ALIGNMENT_FIELDS = [
    "case_id",
    "record_id",
    "record_name",
    "start_page",
    "end_page",
    "record_structure_valid",
    "raw_question_text_valid",
    "raw_answer_text_valid",
    "raw_answer_matches_raw_question",
    "pdf_extraction_available",
    "pdf_question_grounded",
    "pdf_answer_grounded",
    "pdf_page_range_relevant",
    "plan_evaluation_skipped",
    "source_alignment_status",
    "source_alignment_quality",
    "source_alignment_issue_type",
    "source_alignment_issue_explanation",
    "source_record_valid",
    "need_human_review",
    "short_reason",
    "evidence_brief",
    "judge_model",
    "judge_json_parse_ok",
    "judge_fallback_called",
    "latency_seconds",
]


SOURCE_ALIGNMENT_ISSUE_FIELDS = [
    "record_id",
    "record_name",
    "severity",
    "source_alignment_issue_type",
    "source_alignment_issue_explanation",
    "short_reason",
    "evidence_brief",
]


LABEL_ASSIGNMENT_FIELDS = [
    "case_id",
    "record_id",
    "record_name",
    "question_order",
    "item_order",
    "interaction_order",
    "raw_question_short",
    "question_statement",
    "item_requirement",
    "interaction_requirement",
    "assigned_interaction_type",
    "generated_interaction_type",
    "plan_generated_type_consistent",
    "plan_generated_type_mismatch",
    "rule_validation_status",
    "structural_valid",
    "structural_error_type",
    "missing_fields",
    "invalid_fields",
    "llm_judge_required",
    "semantic_judgement_status",
    "semantic_judgement_valid",
    "semantic_quality",
    "semantic_error_type",
    "needs_human_review",
    "semantic_confidence",
    "decision_basis",
    "observed_mismatch",
    "repair_stage_required",
    "repair_hint",
    "candidate_issue_summary",
    "reason",
    "evidence",
    "judge_model",
    "judge_json_parse_ok",
    "judge_fallback_called",
    "judge_disagreement",
    "final_decision_policy",
    "repair_needed",
    "repair_status",
    "repair_priority",
    "repair_action_type",
    "recommended_interaction_type",
    "recommended_input_mode",
    "recommended_change",
    "repair_reason",
    "repair_evidence",
    "repair_confidence",
    "repair_decision_basis",
    "manual_review_required",
    "requires_config_rebuild",
    "requires_answer_specs_rebuild",
    "patch_preview",
    "final_status",
    "final_reason",
    "checker_source",
    "latency_seconds",
]


LABEL_ASSIGNMENT_RECORD_FIELDS = [
    "record_id",
    "record_name",
    "scope_diagnostic_status",
    "scope_diagnostic_note",
    "detected_raw_subparts",
    "covered_subparts",
    "missing_subparts",
    "total_assigned_interactions",
    "issue_count",
    "final_ok_count",
    "final_warning_count",
    "final_needs_review_count",
    "final_bad_count",
    "final_structural_error_count",
]


PLAN_INTERACTION_TYPE_ISSUE_FIELDS = [
    "record_id",
    "record_name",
    "case_id",
    "item_order",
    "interaction_order",
    "assigned_interaction_type",
    "generated_interaction_type",
    "final_status",
    "rule_validation_status",
    "structural_error_type",
    "semantic_judgement_status",
    "semantic_error_type",
    "semantic_confidence",
    "decision_basis",
    "observed_mismatch",
    "reason",
    "evidence",
    "repair_needed",
    "repair_status",
    "repair_action_type",
    "recommended_interaction_type",
    "recommended_input_mode",
    "recommended_change",
    "repair_reason",
    "repair_confidence",
    "manual_review_required",
    "requires_config_rebuild",
    "requires_answer_specs_rebuild",
    "patch_preview",
]


PLAN_INTERACTION_TYPE_REPAIR_FIELDS = [
    "record_id",
    "record_name",
    "case_id",
    "item_order",
    "interaction_order",
    "assigned_interaction_type",
    "semantic_judgement_status",
    "semantic_error_type",
    "repair_needed",
    "repair_status",
    "repair_priority",
    "repair_action_type",
    "recommended_interaction_type",
    "recommended_input_mode",
    "recommended_change",
    "repair_reason",
    "repair_evidence",
    "repair_confidence",
    "repair_decision_basis",
    "manual_review_required",
    "requires_config_rebuild",
    "requires_answer_specs_rebuild",
    "patch_preview",
]


PLAN_INTERACTION_TYPE_QUESTION_SUMMARY_FIELDS = [
    "record_id",
    "record_name",
    "question_status",
    "total_semantic_cases",
    "ok_case_count",
    "warning_case_count",
    "needs_review_case_count",
    "bad_case_count",
    "structural_error_case_count",
    "repair_needed_count",
    "repair_suggested_count",
    "manual_review_required_count",
    "requires_config_rebuild_count",
    "requires_answer_specs_rebuild_count",
    "issue_locations",
    "main_issue_summary",
    "recommended_actions",
    "worst_semantic_error_type",
    "assigned_interaction_types",
    "recommended_interaction_types",
]


QUESTION_ISSUE_CSV_FIELDS = [
    "record_id",
    "record_name",
    "question_status",
    "total_semantic_cases",
    "bad_case_count",
    "structural_error_case_count",
    "warning_case_count",
    "needs_review_case_count",
    "repair_needed_count",
    "manual_review_required_count",
    "requires_config_rebuild_count",
    "requires_answer_specs_rebuild_count",
    "main_issue_summary",
    "issue_locations",
    "recommended_actions",
    "assigned_interaction_types",
    "recommended_interaction_types",
]


INTERACTION_ISSUE_CSV_FIELDS = [
    "case_id",
    "record_id",
    "record_name",
    "location",
    "question_order",
    "item_order",
    "interaction_order",
    "assigned_interaction_type",
    "generated_interaction_type",
    "plan_generated_type_consistent",
    "semantic_judgement_status",
    "semantic_quality",
    "semantic_error_type",
    "semantic_confidence",
    "problem_summary",
    "reason",
    "evidence",
    "observed_mismatch",
    "repair_needed",
    "repair_status",
    "repair_priority",
    "repair_action_type",
    "recommended_interaction_type",
    "recommended_input_mode",
    "recommended_change",
    "repair_reason",
    "repair_evidence",
    "repair_confidence",
    "manual_review_required",
    "requires_config_rebuild",
    "requires_answer_specs_rebuild",
]


LEGACY_INTERACTION_TYPE_OUTPUT_FILES = [
    "label_assignment_pipeline_results.json",
    "label_assignment_pipeline_summary.csv",
    "label_assignment_problem_report.json",
    "label_assignment_record_summary.csv",
    "plan_interaction_type_eval.csv",
    "plan_interaction_type_eval_report.md",
    "plan_interaction_type_eval_summary.json",
    "plan_interaction_type_interaction_issues.csv",
    "plan_interaction_type_issues.csv",
    "plan_interaction_type_question_summary.csv",
    "plan_interaction_type_repair_report.md",
    "plan_interaction_type_repair_suggestions.csv",
    "plan_interaction_type_repair_summary.json",
]


SOURCE_RECORD_QUICK_CHECK_FILES = [
    "source_record_pipeline_results.json",
    "source_record_pipeline_summary.csv",
    "source_record_problem_report.json",
    "source_alignment_summary.json",
    "source_alignment_issues.csv",
    "source_alignment_report.md",
]


STALE_COMPACT_INTERACTION_TYPE_OUTPUT_FILES = [
    "outputs/report/question_summary.csv",
    "outputs/report/repair_suggestions.csv",
    "outputs/report/question_issues.csv",
    "outputs/report/question_warning.csv",
    "outputs/report/question_issue_details.md",
    "outputs/report/general_report.md",
    "outputs/evidence/interaction_evidence.csv",
    "outputs/evidence/interaction_bad.csv",
    "outputs/evidence/interaction_issues.csv",
    "outputs/evidence/interaction_warning.csv",
    "outputs/evidence/interaction_warning_needs_review.csv",
]


PLAN_QUALITY_QUESTION_FIELDS = [
    "record_id",
    "record_name",
    "overall_status",
    "coverage_status",
    "source_to_plan_mapping",
    "source_subparts",
    "covered_subparts",
    "missing_subparts",
    "selected_scope_summary",
    "plan_quality_summary",
    "judge_consistency_warnings",
    "issue_count",
    "issue_types",
    "issue_summaries",
    "issue_evidence",
    "issue_impacts",
    "impact_on_generation",
    "suggested_fixes",
    "repair_specific_changes",
    "repair_primary_decisions",
    "manual_review_required",
    "has_rewritten_preview",
    "preview_strategy",
    "recommended_actions",
    "confidence",
    "judge_model",
    "json_parse_ok",
    "fallback_called",
    "final_decision_policy",
    "latency_seconds",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fieldnames})


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def clean_cell(value: Any, limit: int = 500) -> str:
    text = str(value or "").replace("\n", " ").replace("|", "\\|").strip()
    return text[: limit - 3] + "..." if len(text) > limit else text


def joined_text(values: list[Any], *, sep: str = " | ", limit: int = 1200) -> str:
    parts = [str(value).strip() for value in values if str(value or "").strip()]
    return clean_cell(sep.join(parts), limit)


def issue_field_text(issues: list[dict[str, Any]], field: str, *, sep: str = " | ", limit: int = 1200) -> str:
    return joined_text([issue.get(field) for issue in issues], sep=sep, limit=limit)


def repair_field_text(
    suggestions: list[dict[str, Any]],
    field: str,
    *,
    fallback_field: str | None = None,
    sep: str = " | ",
    limit: int = 1200,
) -> str:
    values = []
    for suggestion in suggestions:
        value = suggestion.get(field)
        if not value and fallback_field:
            value = suggestion.get(fallback_field)
        if value:
            values.append(value)
    return joined_text(values, sep=sep, limit=limit)


def compact_question_label(row: dict[str, Any]) -> str:
    return str(row.get("record_name") or row.get("record_id") or "").strip()


def constrained_question_cell(row: dict[str, Any]) -> str:
    label = clean_cell(compact_question_label(row), 500)
    return f'<div style="max-width:220px; white-space:normal; word-break:break-word;">{label}</div>'


def write_json(path: Path, results: list[dict[str, Any]] | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": results,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def plan_quality_issue_records(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in results:
        if row.get("overall_status") == "ok":
            continue
        records.append(
            {
                "record_id": row.get("record_id"),
                "record_name": row.get("record_name"),
                "overall_status": row.get("overall_status"),
                "coverage_status": row.get("coverage_status"),
                "source_to_plan_mapping": row.get("source_to_plan_mapping") or [],
                "source_subparts": row.get("source_subparts") or [],
                "covered_subparts": row.get("covered_subparts") or [],
                "missing_subparts": row.get("missing_subparts") or [],
                "selected_scope_summary": row.get("selected_scope_summary"),
                "plan_quality_summary": row.get("plan_quality_summary"),
                "judge_consistency_warnings": row.get("judge_consistency_warnings") or [],
                "issues": row.get("issues") or [],
                "repair_suggestions": row.get("repair_suggestions") or [],
                "rewritten_question_plan_preview": row.get("rewritten_question_plan_preview"),
                "preview_strategy": row.get("preview_strategy"),
                "repair_confidence": row.get("repair_confidence"),
                "manual_review_required": row.get("manual_review_required"),
                "repair_error": row.get("repair_error"),
                "preview_validation_error": row.get("preview_validation_error"),
                "repair_json_parse_ok": row.get("repair_json_parse_ok"),
                "repair_model": row.get("repair_model"),
                "repair_latency_seconds": row.get("repair_latency_seconds"),
                "recommended_actions": row.get("recommended_actions") or [],
                "confidence": row.get("confidence"),
                "judge_model": row.get("judge_model"),
                "json_parse_ok": row.get("json_parse_ok"),
                "fallback_called": row.get("fallback_called"),
                "final_decision_policy": row.get("final_decision_policy"),
                "latency_seconds": row.get("latency_seconds"),
                "structural_result": row.get("structural_result"),
            }
        )
    return records


def plan_quality_question_rows(results: list[dict[str, Any]], statuses: set[str]) -> list[dict[str, Any]]:
    rows = []
    for row in results:
        if row.get("overall_status") not in statuses:
            continue
        issues = row.get("issues") or []
        repair_suggestions = row.get("repair_suggestions") or []
        rows.append(
            {
                "record_id": row.get("record_id"),
                "record_name": row.get("record_name"),
                "overall_status": row.get("overall_status"),
                "coverage_status": row.get("coverage_status"),
                "source_to_plan_mapping": row.get("source_to_plan_mapping") or [],
                "source_subparts": row.get("source_subparts") or [],
                "covered_subparts": row.get("covered_subparts") or [],
                "missing_subparts": row.get("missing_subparts") or [],
                "selected_scope_summary": row.get("selected_scope_summary"),
                "plan_quality_summary": row.get("plan_quality_summary"),
                "judge_consistency_warnings": row.get("judge_consistency_warnings") or [],
                "issue_count": len(issues),
                "issues": issues,
                "issue_types": sorted({issue.get("issue_type") for issue in issues if issue.get("issue_type")}),
                "issue_summaries": [issue.get("summary") for issue in issues if issue.get("summary")],
                "issue_evidence": [issue.get("evidence") for issue in issues if issue.get("evidence")],
                "issue_impacts": [
                    issue.get("impact_on_generation") for issue in issues if issue.get("impact_on_generation")
                ],
                "impact_on_generation": [
                    issue.get("impact_on_generation") for issue in issues if issue.get("impact_on_generation")
                ],
                "suggested_fixes": [issue.get("suggested_fix") for issue in issues if issue.get("suggested_fix")],
                "repair_specific_changes": [
                    suggestion.get("specific_change")
                    for suggestion in repair_suggestions
                    if suggestion.get("specific_change")
                ],
                "repair_primary_decisions": [
                    suggestion.get("primary_decision")
                    for suggestion in repair_suggestions
                    if suggestion.get("primary_decision")
                ],
                "manual_review_required": row.get("manual_review_required"),
                "has_rewritten_preview": row.get("rewritten_question_plan_preview") is not None,
                "preview_strategy": row.get("preview_strategy"),
                "repair_suggestions": repair_suggestions,
                "rewritten_question_plan_preview": row.get("rewritten_question_plan_preview"),
                "recommended_actions": row.get("recommended_actions") or [],
                "confidence": row.get("confidence"),
                "judge_model": row.get("judge_model"),
                "json_parse_ok": row.get("json_parse_ok"),
                "fallback_called": row.get("fallback_called"),
                "final_decision_policy": row.get("final_decision_policy"),
                "latency_seconds": row.get("latency_seconds"),
            }
        )
    return rows


def build_question_plan_quality_markdown(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    status_counts = summary.get("status_counts") or {}
    coverage_counts = summary.get("coverage_status_counts") or {}
    bad_rows = plan_quality_question_rows(results, {"bad", "structural_error"})
    warning_rows = plan_quality_question_rows(results, {"warning", "needs_review"})
    skipped_rows = plan_quality_question_rows(results, SKIPPED_SOURCE_STATUSES)
    lines = [
        "# Báo Cáo Chất Lượng Question Plan",
        "",
        "## Tổng Quan",
        "",
        (
            "Tóm tắt chất lượng question_plan: "
            f"{status_counts.get('ok', 0)} OK, "
            f"{status_counts.get('bad', 0)} lỗi, "
            f"{status_counts.get('warning', 0)} cảnh báo, "
            f"{status_counts.get('needs_review', 0)} cần xem lại, "
            f"{status_counts.get('structural_error', 0)} lỗi cấu trúc, "
            f"{status_counts.get('skipped_due_to_source_issue', 0)} bỏ qua do source/raw lỗi."
        ),
        "",
        f"- Tổng số bản ghi nguồn: {summary.get('total_records')}",
        f"- OK: {status_counts.get('ok', 0)}",
        f"- Cảnh báo: {status_counts.get('warning', 0)}",
        f"- Cần xem lại: {status_counts.get('needs_review', 0)}",
        f"- Lỗi nghiêm trọng: {status_counts.get('bad', 0)}",
        f"- Lỗi cấu trúc: {status_counts.get('structural_error', 0)}",
        f"- Bỏ qua do source/raw lỗi: {status_counts.get('skipped_due_to_source_issue', 0)}",
        f"- Kiểm tra tổng số hợp lệ: {summary.get('count_consistency_valid')}",
        f"- Bao phủ đầy đủ: {coverage_counts.get('full', 0)}",
        f"- Bao phủ một phần: {coverage_counts.get('partial', 0)}",
        f"- Chưa rõ mức bao phủ: {coverage_counts.get('unclear', 0)}",
        f"- Không đánh giá được bao phủ: {coverage_counts.get('not_evaluable', 0)}",
        f"- Số bản ghi có vấn đề: {summary.get('issue_records_count')}",
        f"- Tổng số vấn đề: {summary.get('issue_count')}",
        "",
        "Kết luận chính được tính ở cấp source record/question.",
        "Các recommended actions chỉ là gợi ý review; pipeline không tự sửa dữ liệu gốc.",
        "",
        "## Câu Hỏi Lỗi / Lỗi Cấu Trúc",
        "",
        *plan_quality_markdown_table(bad_rows),
        "",
        "## Câu Hỏi Cảnh Báo / Cần Xem Lại",
        "",
        *plan_quality_markdown_table(warning_rows),
        "",
        "## Bỏ Qua Do Source/Raw Có Vấn Đề",
        "",
        *plan_quality_skipped_markdown_table(skipped_rows),
        "",
        "## Thống Kê Theo Nhóm Vấn Đề",
        "",
    ]
    issue_counts = summary.get("issue_type_counts") or {}
    if issue_counts:
        for issue_type, count in sorted(issue_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {issue_type}: {count}")
    else:
        lines.append("- Không có vấn đề.")
    lines.extend(
        [
            "",
            "## Các File Output",
            "",
            "- `question_plan_issue_details.json`: JSON chi tiết, chỉ chứa các source records/questions có vấn đề hoặc bị bỏ qua.",
            "- `question_bad.csv`: CSV cấp question cho lỗi nghiêm trọng hoặc lỗi cấu trúc.",
            "- `question_warning_needs_review.csv`: CSV cấp question cho cảnh báo hoặc cần xem lại.",
            "- `question_skipped_due_to_source_issue.csv`: CSV các bản ghi bị bỏ qua do source/raw có vấn đề.",
            "- `question_plan_replacement_records_all.json`: Full source records đã thay `question_plan`, gồm mọi replacement có preview hợp lệ.",
            "",
        ]
    )
    return "\n".join(lines)


def plan_quality_issue_detail_sections(results: list[dict[str, Any]], statuses: set[str]) -> list[str]:
    selected = [row for row in results if row.get("overall_status") in statuses]
    lines: list[str] = []
    if not selected:
        lines.append("Không có issue chi tiết.")
        return lines
    for row in selected:
        issues = row.get("issues") or []
        if not issues:
            continue
        title = str(row.get("record_name") or row.get("record_id") or "").replace("\n", " ").strip()
        lines.extend([f"### {title}", ""])
        for issue in issues:
            summary = str(issue.get("summary") or "").strip()
            evidence = str(issue.get("evidence") or "").strip()
            impact = str(issue.get("impact_on_generation") or "").strip()
            suggested_fix = str(issue.get("suggested_fix") or "").strip()
            issue_type = str(issue.get("issue_type") or "").strip()
            severity = str(issue.get("severity") or "").strip()
            lines.extend(
                [
                    f"- Vấn đề: {summary or issue_type or 'Chưa có mô tả cụ thể.'}",
                    f"- Loại/mức độ: `{issue_type}` / `{severity}`",
                    f"- Bằng chứng: {evidence or 'Chưa có bằng chứng cụ thể.'}",
                    f"- Ảnh hưởng: {impact or 'Chưa mô tả rõ ảnh hưởng đến generation/render/chấm.'}",
                    f"- Hướng xử lý: {suggested_fix or issue.get('recommended_action') or 'Cần reviewer quyết định.'}",
                    "",
                ]
            )
    return lines or ["Không có issue chi tiết."]


def plan_quality_markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Câu hỏi | Trạng thái | Bao phủ | Vấn đề chính | Bằng chứng | Ảnh hưởng | Gợi ý sửa cụ thể | Preview sau sửa | Cần review |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    if not rows:
        lines.append("| Không có câu hỏi. |  |  |  |  |  |  |  |  |")
        return lines
    for row in rows:
        issues = row.get("issues") or []
        repair_suggestions = row.get("repair_suggestions") or []
        issue_summary = issue_field_text(issues, "summary", sep="<br>", limit=1200)
        evidence = issue_field_text(issues, "evidence", sep="<br>", limit=1200)
        impact = issue_field_text(issues, "impact_on_generation", sep="<br>", limit=1200)
        repair_text = repair_field_text(
            repair_suggestions,
            "specific_change",
            fallback_field="primary_decision",
            sep="<br>",
            limit=1200,
        )
        primary_decision_text = repair_field_text(
            repair_suggestions,
            "primary_decision",
            sep="<br>",
            limit=1200,
        )
        if primary_decision_text:
            repair_text = (repair_text + "<br>" if repair_text else "") + primary_decision_text
        if not repair_text:
            repair_text = "Chưa có repair_suggestions; cần chạy lại pipeline để sinh gợi ý sửa cụ thể."
        preview_text = (
            "Có object question_plan sau sửa trong JSON."
            if row.get("rewritten_question_plan_preview") is not None and row.get("preview_strategy") == "full_object"
            else "Không tạo preview an toàn, cần review thủ công."
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    constrained_question_cell(row),
                    clean_cell(row.get("overall_status"), 80),
                    clean_cell(row.get("coverage_status"), 80),
                    issue_summary or clean_cell(row.get("plan_quality_summary"), 1000),
                    evidence,
                    impact,
                    repair_text or clean_cell(row.get("plan_quality_summary"), 800),
                    preview_text,
                    "" if row.get("manual_review_required") is None else str(row.get("manual_review_required")),
                ]
            )
            + " |"
        )
    return lines


def plan_quality_skipped_markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Câu hỏi | Trạng thái | Lý do bỏ qua |", "|---|---|---|"]
    if not rows:
        lines.append("| Không có câu hỏi. |  |  |")
        return lines
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    constrained_question_cell(row),
                    clean_cell(row.get("overall_status"), 80),
                    clean_cell(row.get("plan_quality_summary") or row.get("selected_scope_summary"), 1200),
                ]
            )
            + " |"
        )
    return lines


QUESTION_PLAN_SCHEMA_FIELDS = {
    "root": {"type", "plan"},
    "question": {"questionOrder", "questionStatement", "questionItems"},
    "item": {"itemOrder", "requirement", "interactions"},
    "interaction": {"interactionOrder", "interactionType", "interactionRequirement"},
}
AMBIGUOUS_REPLACEMENT_PHRASES = ("hoặc", "có thể", "cân nhắc", "nếu muốn", "tùy")
DISALLOWED_REPLACEMENT_FIELDS = {"suggested_change", "before", "after", "patch_preview"}


def question_plan_replacement_validation_errors(question_plan: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(question_plan, dict):
        return ["new_question_plan không phải object."]
    for key in question_plan:
        if key not in QUESTION_PLAN_SCHEMA_FIELDS["root"]:
            errors.append(f"Field lạ ở question_plan root: {key}.")
    if question_plan.get("type") != "advanced_question_plan":
        errors.append("question_plan.type phải là advanced_question_plan.")
    plan = question_plan.get("plan")
    if not isinstance(plan, list) or not plan:
        errors.append("question_plan.plan phải là list không rỗng.")
        return errors

    def scan_disallowed(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in DISALLOWED_REPLACEMENT_FIELDS:
                    errors.append(f"Field dạng patch không hợp lệ tại {path or '$'}: {key}.")
                scan_disallowed(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                scan_disallowed(child, f"{path}[{index}]")

    def scan_ambiguous_text(value: Any, path: str) -> None:
        text = str(value or "").lower()
        for phrase in AMBIGUOUS_REPLACEMENT_PHRASES:
            if phrase in text:
                errors.append(f"Nội dung tại {path} chứa cụm mơ hồ: {phrase}.")

    scan_disallowed(question_plan)

    for question_index, question in enumerate(plan, start=1):
        if not isinstance(question, dict):
            errors.append(f"plan[{question_index}] không phải object.")
            continue
        for key in question:
            if key not in QUESTION_PLAN_SCHEMA_FIELDS["question"]:
                errors.append(f"Field lạ ở plan[{question_index}]: {key}.")
        if not str(question.get("questionStatement") or "").strip():
            errors.append(f"plan[{question_index}].questionStatement bị thiếu hoặc rỗng.")
        items = question.get("questionItems")
        if not isinstance(items, list) or not items:
            errors.append(f"plan[{question_index}].questionItems phải là list không rỗng.")
            continue
        for item_index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                errors.append(f"questionItems[{item_index}] không phải object.")
                continue
            for key in item:
                if key not in QUESTION_PLAN_SCHEMA_FIELDS["item"]:
                    errors.append(f"Field lạ ở questionItems[{item_index}]: {key}.")
            requirement = item.get("requirement")
            if not str(requirement or "").strip():
                errors.append(f"questionItems[{item_index}].requirement bị thiếu hoặc rỗng.")
            scan_ambiguous_text(requirement, f"questionItems[{item_index}].requirement")
            interactions = item.get("interactions")
            if not isinstance(interactions, list) or not interactions:
                errors.append(f"questionItems[{item_index}].interactions phải là list không rỗng.")
                continue
            for interaction_index, interaction in enumerate(interactions, start=1):
                if not isinstance(interaction, dict):
                    errors.append(f"interactions[{interaction_index}] không phải object.")
                    continue
                for key in interaction:
                    if key not in QUESTION_PLAN_SCHEMA_FIELDS["interaction"]:
                        errors.append(f"Field lạ ở interactions[{interaction_index}]: {key}.")
                interaction_type = interaction.get("interactionType")
                if interaction_type not in VALID_REAL_INTERACTION_TYPES:
                    errors.append(f"interactionType không hợp lệ: {interaction_type}.")
                interaction_requirement = interaction.get("interactionRequirement")
                if not str(interaction_requirement or "").strip():
                    errors.append(f"interactions[{interaction_index}].interactionRequirement bị thiếu hoặc rỗng.")
                scan_ambiguous_text(
                    interaction_requirement,
                    f"interactions[{interaction_index}].interactionRequirement",
                )
    return errors


def summarize_plan_quality_issues(issues: list[dict[str, Any]]) -> str:
    if not issues:
        return ""
    return joined_text(
        [
            issue.get("summary") or issue.get("issue_type") or issue.get("recommended_action")
            for issue in issues
        ],
        sep=" | ",
        limit=2000,
    )


def summarize_plan_repairs(repair_suggestions: list[dict[str, Any]]) -> str:
    if not repair_suggestions:
        return ""
    return joined_text(
        [
            suggestion.get("primary_decision")
            or suggestion.get("specific_change")
            or suggestion.get("problem_summary")
            or suggestion.get("action_code")
            for suggestion in repair_suggestions
        ],
        sep=" | ",
        limit=2000,
    )


def build_question_plan_replacement_item(row: dict[str, Any]) -> dict[str, Any]:
    preview = row.get("rewritten_question_plan_preview")
    preview_strategy = str(row.get("preview_strategy") or "")
    preview_validation_error = str(row.get("preview_validation_error") or "").strip()
    repair_confidence = str(row.get("repair_confidence") or "")
    validation_errors = question_plan_replacement_validation_errors(preview)
    preview_is_valid = (
        preview_strategy == "full_object"
        and not preview_validation_error
        and not validation_errors
    )

    if preview_is_valid:
        if row.get("manual_review_required") or repair_confidence == "low":
            replacement_status = "review_before_copy"
            replacement_reason = "Preview hợp lệ nhưng cần reviewer đọc lại trước khi thay vào data chính."
        else:
            replacement_status = "copy_ready"
            replacement_reason = "Preview hợp lệ và không yêu cầu review thủ công."
        new_question_plan = preview
    else:
        replacement_status = "not_safe"
        reason_parts = []
        if preview_strategy != "full_object":
            reason_parts.append("preview_strategy không phải full_object")
        if preview is None:
            reason_parts.append("không có rewritten_question_plan_preview")
        if preview_validation_error:
            reason_parts.append(f"preview_validation_error: {preview_validation_error}")
        if validation_errors:
            reason_parts.append("validate new_question_plan lỗi: " + "; ".join(validation_errors))
        replacement_reason = "; ".join(reason_parts) or "Không có rewritten_question_plan_preview hợp lệ."
        new_question_plan = None

    issues = row.get("issues") or []
    repair_suggestions = row.get("repair_suggestions") or []
    return {
        "record_id": row.get("record_id"),
        "record_name": row.get("record_name"),
        "overall_status": row.get("overall_status"),
        "coverage_status": row.get("coverage_status"),
        "replacement_status": replacement_status,
        "replacement_reason": replacement_reason,
        "original_issue_summary": summarize_plan_quality_issues(issues),
        "repair_summary": summarize_plan_repairs(repair_suggestions),
        "new_question_plan": new_question_plan,
        "source_eval_result_ref": {
            "has_issues": bool(issues),
            "issue_count": len(issues),
            "preview_strategy": preview_strategy,
            "manual_review_required": row.get("manual_review_required"),
            "repair_confidence": repair_confidence,
            "preview_validation_error": preview_validation_error,
        },
    }


def source_record_keys(record: dict[str, Any]) -> list[str]:
    values = [
        record.get("_id"),
        record.get("id"),
        record.get("record_id"),
        record.get("name"),
        record.get("record_name"),
    ]
    return [str(value) for value in values if value is not None and str(value).strip()]


def build_source_record_lookup(source_records: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for record in source_records or []:
        if not isinstance(record, dict):
            continue
        for key in source_record_keys(record):
            lookup.setdefault(key, record)
    return lookup


def find_original_source_record(
    row_or_item: dict[str, Any],
    source_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for key in (
        row_or_item.get("record_id"),
        row_or_item.get("_id"),
        row_or_item.get("id"),
        row_or_item.get("record_name"),
        row_or_item.get("name"),
    ):
        if key is None:
            continue
        record = source_lookup.get(str(key))
        if record is not None:
            return record
    return None


def build_question_plan_replacement_records_all(
    items: list[dict[str, Any]],
    source_records: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    source_lookup = build_source_record_lookup(source_records)
    records: list[dict[str, Any]] = []

    for item in items:
        preview_status = item.get("replacement_status")
        preview = item.get("new_question_plan")
        original_record = find_original_source_record(item, source_lookup)
        has_valid_preview = preview_status in {"copy_ready", "review_before_copy"} and preview is not None
        if has_valid_preview and original_record is not None:
            replacement_record = deepcopy(original_record)
            replacement_record["question_plan"] = deepcopy(preview)
            records.append(replacement_record)
    return records


def write_question_plan_replacement_files(
    results: list[dict[str, Any]],
    output_dir: Path,
    source_records: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    stale_names = (
        "question_plan_replacement_bundle.json",
        "question_plan_replacement_ready.json",
        "question_plan_replacement_review_required.json",
        "question_plan_replacement_not_safe.json",
        "question_plan_replacement_records_ready.json",
        "question_plan_replacement_records_review_required.json",
        "question_plan_replacement_manifest.json",
        "question_plan_replacement_report.md",
    )
    for stale_name in stale_names:
        stale_path = output_dir / stale_name
        if stale_path.is_file():
            stale_path.unlink()

    items = [
        build_question_plan_replacement_item(row)
        for row in results
        if row.get("overall_status") != "ok" or row.get("rewritten_question_plan_preview") is not None
    ]
    records_all_path = output_dir / "question_plan_replacement_records_all.json"

    all_records = build_question_plan_replacement_records_all(
        items,
        source_records,
    )

    write_plain_json(records_all_path, all_records)

    return {
        "question_plan_replacement_records_all": records_all_path,
    }


def write_question_plan_eval_output_files(
    results_dir: Path,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    source_records: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    output_dir = results_dir / "outputs" / "plan_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    general_report_path = output_dir / "general_report.md"
    issue_details_path = output_dir / "question_plan_issue_details.json"
    question_bad_path = output_dir / "question_bad.csv"
    question_warning_path = output_dir / "question_warning_needs_review.csv"
    question_skipped_path = output_dir / "question_skipped_due_to_source_issue.csv"
    run_summary_path = output_dir / "run_summary.json"

    for stale_name in ("interaction_bad.csv", "interaction_warning_needs_review.csv"):
        stale_path = output_dir / stale_name
        if stale_path.is_file():
            stale_path.unlink()

    general_report_path.write_text(build_question_plan_quality_markdown(summary, results), encoding="utf-8")
    write_json(issue_details_path, plan_quality_issue_records(results))
    write_plain_json(run_summary_path, summary)
    write_csv(question_bad_path, plan_quality_question_rows(results, BAD_STATUSES), PLAN_QUALITY_QUESTION_FIELDS)
    write_csv(
        question_warning_path,
        plan_quality_question_rows(results, WARNING_STATUSES),
        PLAN_QUALITY_QUESTION_FIELDS,
    )
    write_csv(
        question_skipped_path,
        plan_quality_question_rows(results, SKIPPED_SOURCE_STATUSES),
        PLAN_QUALITY_QUESTION_FIELDS,
    )
    replacement_paths = write_question_plan_replacement_files(results, output_dir, source_records)
    output_paths = {
        "general_report": general_report_path,
        "question_plan_issue_details": issue_details_path,
        "question_bad": question_bad_path,
        "question_warning_needs_review": question_warning_path,
        "question_skipped_due_to_source_issue": question_skipped_path,
        "run_summary": run_summary_path,
    }
    output_paths.update(replacement_paths)
    return output_paths


def write_real_question_result_files(results_dir: Path, results: list[dict[str, Any]]) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "real_question_pipeline_results.json"
    csv_path = results_dir / "real_question_pipeline_summary.csv"
    write_json(json_path, results)
    write_csv(csv_path, results, REAL_QUESTION_FIELDS)
    return json_path, csv_path


def write_source_record_result_files(
    results_dir: Path,
    results: list[dict[str, Any]],
    problem_report: dict[str, Any],
) -> tuple[Path, Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    for filename in SOURCE_RECORD_QUICK_CHECK_FILES:
        legacy_path = results_dir / filename
        if legacy_path.is_file():
            legacy_path.unlink()

    quick_check_dir = results_dir / "quick_check"
    quick_check_dir.mkdir(parents=True, exist_ok=True)
    json_path = quick_check_dir / "source_record_pipeline_results.json"
    csv_path = quick_check_dir / "source_record_pipeline_summary.csv"
    report_path = quick_check_dir / "source_record_problem_report.json"
    alignment_summary_path = quick_check_dir / "source_alignment_summary.json"
    alignment_issues_path = quick_check_dir / "source_alignment_issues.csv"
    alignment_report_path = quick_check_dir / "source_alignment_report.md"
    write_json(json_path, results)
    write_csv(csv_path, results, SOURCE_ALIGNMENT_FIELDS)
    report_path.write_text(json.dumps(problem_report, ensure_ascii=False, indent=2), encoding="utf-8")
    alignment_summary_path.write_text(json.dumps(problem_report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(alignment_issues_path, problem_report.get("issue_cases") or [], SOURCE_ALIGNMENT_ISSUE_FIELDS)
    alignment_report_path.write_text(build_source_alignment_markdown(problem_report), encoding="utf-8")
    return json_path, csv_path, report_path


def build_source_alignment_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Source Alignment Report",
        "",
        "## Summary",
        "",
        f"- Total records: {summary.get('total_records')}",
        f"- Valid: {summary.get('valid_count')}",
        f"- Warning: {summary.get('warning_count')}",
        f"- Needs review: {summary.get('needs_review_count')}",
        f"- Critical: {summary.get('critical_count')}",
        f"- Skipped: {summary.get('skipped_count')}",
        f"- Unchecked: {summary.get('unchecked_count')}",
        f"- Unknown: {summary.get('unknown_count')}",
        f"- Count consistency: {summary.get('count_consistency_valid')} - {summary.get('count_consistency_formula')}",
        "",
    ]
    sections = [
        ("Critical cases", summary.get("critical_cases") or []),
        ("Warning cases", summary.get("warning_cases") or []),
        ("Needs review cases", summary.get("needs_review_cases") or []),
        ("Skipped / unchecked / unknown cases", summary.get("skipped_unchecked_unknown_cases") or []),
    ]
    for title, cases in sections:
        lines.extend([f"## {title}", ""])
        if not cases:
            lines.append("No cases.")
            lines.append("")
            continue
        for index, case in enumerate(cases, start=1):
            lines.extend(
                [
                    f"### {index}. {case.get('record_name')}",
                    "",
                    f"- Record ID: `{case.get('record_id')}`",
                    f"- Severity: `{case.get('severity')}`",
                    f"- Issue type: `{case.get('source_alignment_issue_type')}`",
                    f"- Issue explanation: {case.get('source_alignment_issue_explanation')}",
                    f"- Short reason: {case.get('short_reason')}",
                    f"- Evidence: {case.get('evidence_brief')}",
                    "",
                ]
            )
    return "\n".join(lines)


def write_label_assignment_result_files(
    results_dir: Path,
    results: list[dict[str, Any]],
    problem_report: dict[str, Any],
    record_summary: list[dict[str, Any]],
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "label_assignment_pipeline_results.json"
    csv_path = results_dir / "label_assignment_pipeline_summary.csv"
    report_path = results_dir / "label_assignment_problem_report.json"
    record_csv_path = results_dir / "label_assignment_record_summary.csv"
    eval_csv_path = results_dir / "plan_interaction_type_eval.csv"
    issues_csv_path = results_dir / "plan_interaction_type_interaction_issues.csv"
    legacy_issues_csv_path = results_dir / "plan_interaction_type_issues.csv"
    question_summary_csv_path = results_dir / "plan_interaction_type_question_summary.csv"
    eval_summary_path = results_dir / "plan_interaction_type_eval_summary.json"
    eval_report_path = results_dir / "plan_interaction_type_eval_report.md"
    issue_cases = problem_report.get("issue_cases") or []
    question_summary = build_question_level_summary(results, record_summary)
    eval_summary = build_plan_interaction_type_summary(results, question_summary)
    write_json(json_path, results)
    write_csv(csv_path, results, LABEL_ASSIGNMENT_FIELDS)
    write_csv(record_csv_path, record_summary, LABEL_ASSIGNMENT_RECORD_FIELDS)
    report_path.write_text(json.dumps(problem_report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(eval_csv_path, results, LABEL_ASSIGNMENT_FIELDS)
    write_csv(issues_csv_path, issue_cases, PLAN_INTERACTION_TYPE_ISSUE_FIELDS)
    write_csv(legacy_issues_csv_path, issue_cases, PLAN_INTERACTION_TYPE_ISSUE_FIELDS)
    write_csv(question_summary_csv_path, question_summary, PLAN_INTERACTION_TYPE_QUESTION_SUMMARY_FIELDS)
    eval_summary_path.write_text(json.dumps(eval_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    eval_report_path.write_text(
        build_plan_interaction_type_eval_markdown(eval_summary, question_summary),
        encoding="utf-8",
    )
    return json_path, csv_path, report_path, record_csv_path, issues_csv_path, question_summary_csv_path, eval_summary_path


def write_label_repair_result_files(
    results_dir: Path,
    suggestions: list[dict[str, Any]],
    repair_summary: dict[str, Any],
) -> tuple[Path, Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    suggestions_csv_path = results_dir / "plan_interaction_type_repair_suggestions.csv"
    summary_json_path = results_dir / "plan_interaction_type_repair_summary.json"
    report_md_path = results_dir / "plan_interaction_type_repair_report.md"
    write_csv(suggestions_csv_path, suggestions, PLAN_INTERACTION_TYPE_REPAIR_FIELDS)
    summary_json_path.write_text(json.dumps(repair_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md_path.write_text(build_plan_interaction_type_repair_markdown(repair_summary, suggestions), encoding="utf-8")
    return suggestions_csv_path, summary_json_path, report_md_path


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relative_output(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir)).replace("\\", "/")
    except ValueError:
        return str(path)


def clean_legacy_interaction_type_outputs(results_dir: Path) -> None:
    for filename in LEGACY_INTERACTION_TYPE_OUTPUT_FILES:
        path = results_dir / filename
        if path.is_file():
            path.unlink()


def write_plain_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_record_summary_lookup(record_summary: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("record_id") or ""): row for row in record_summary}


def build_run_summary(
    case_results: list[dict[str, Any]],
    question_summary: list[dict[str, Any]],
    repair_suggestions: list[dict[str, Any]],
    repair_summary: dict[str, Any],
    output_files: dict[str, str],
) -> dict[str, Any]:
    eval_summary = build_plan_interaction_type_summary(case_results, question_summary)
    question_status_counts = Counter(row.get("question_status") or "unknown" for row in question_summary)
    semantic_status_counts = Counter(row.get("semantic_judgement_status") or "not_run" for row in case_results)
    final_status_counts = Counter(case_assessment_status(row) for row in case_results)
    repair_status_counts = repair_summary.get("count_by_repair_status") or dict(
        Counter(row.get("repair_status") or "none" for row in repair_suggestions)
    )
    return {
        "generated_at": utc_timestamp(),
        "pipeline_version": INTERACTION_TYPE_OUTPUT_VERSION,
        "total_records": len(question_summary),
        "total_semantic_cases": len(case_results),
        "question_status_counts": dict(question_status_counts),
        "semantic_judgement_status_counts": dict(semantic_status_counts),
        "interaction_status_counts": dict(final_status_counts),
        "repair_status_counts": dict(repair_status_counts),
        "question_level": eval_summary.get("question_level", {}),
        "interaction_level": eval_summary.get("interaction_level", {}),
        "repair": {
            **(eval_summary.get("repair") or {}),
            "repair_summary": repair_summary,
        },
        "count_by_assigned_interaction_type": eval_summary.get("count_by_assigned_interaction_type", {}),
        "count_by_recommended_interaction_type": eval_summary.get("count_by_recommended_interaction_type", {}),
        "count_by_semantic_error_type": eval_summary.get("count_by_semantic_error_type", {}),
        "count_by_repair_action_type": eval_summary.get("count_by_repair_action_type", {}),
        "output_files": output_files,
    }


def repair_suggestion_rows(suggestions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in suggestions
        if row.get("repair_needed")
        or row.get("repair_stage_required")
        or row.get("repair_status") in {"suggested", "needs_human_review", "no_auto_fix"}
    ]


def question_issue_rows(question_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in question_summary if row.get("question_status") in ISSUE_STATUSES]


def interaction_issue_rows(case_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in case_results if case_assessment_status(row) in ISSUE_STATUSES]


def question_rows_by_status_group(question_summary: list[dict[str, Any]], statuses: set[str]) -> list[dict[str, Any]]:
    return [row for row in question_summary if row.get("question_status") in statuses]


def interaction_rows_by_status_group(case_results: list[dict[str, Any]], statuses: set[str]) -> list[dict[str, Any]]:
    return [row for row in case_results if case_assessment_status(row) in statuses]


def truncate_text(value: Any, max_chars: int = 800) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text[: max_chars - 3] + "..." if len(text) > max_chars else text


def full_export_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def is_explicit_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value is False
    return str(value).strip().lower() == "false"


def problem_summary(case: dict[str, Any], max_chars: int = 260) -> str:
    return truncate_text(
        case.get("candidate_issue_summary")
        or case.get("observed_mismatch")
        or case.get("semantic_error_type")
        or case.get("reason")
        or "InteractionType may need review",
        max_chars,
    )


def has_interaction_issue(case: dict[str, Any]) -> bool:
    return bool(
        case.get("semantic_judgement_status") in {"warning", "needs_review", "bad"}
        or case.get("rule_validation_status") == "structural_error"
        or is_explicit_false(case.get("structural_valid"))
        or bool_value(case.get("repair_needed"))
        or bool_value(case.get("manual_review_required"))
    )


def question_recommended_actions(issue_cases: list[dict[str, Any]]) -> list[str]:
    actions = []
    statuses = {case_assessment_status(case) for case in issue_cases}
    if statuses & BAD_STATUSES:
        actions.append("Prioritize manual review for this question.")
    if any(case.get("repair_action_type") == "change_interaction_type" for case in issue_cases):
        actions.append("Review assigned interactionType and consider the recommended interactionType.")
    if any(bool_value(case.get("requires_config_rebuild")) for case in issue_cases):
        actions.append("If interactionType is changed, rebuild config.")
    if any(bool_value(case.get("requires_answer_specs_rebuild")) for case in issue_cases):
        actions.append("If interactionType is changed, rebuild answerSpecs.")
    if not actions or statuses <= WARNING_STATUSES:
        actions.append("Review evidence before changing data.")
    return unique_nonempty(actions)


def compact_interaction_issue(case: dict[str, Any], *, for_json: bool = False) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "record_id": case.get("record_id"),
        "record_name": case.get("record_name"),
        "location": case_location(case),
        "question_order": case.get("question_order"),
        "item_order": case.get("item_order"),
        "interaction_order": case.get("interaction_order"),
        "assigned_interaction_type": case.get("assigned_interaction_type"),
        "generated_interaction_type": case.get("generated_interaction_type"),
        "plan_generated_type_consistent": case.get("plan_generated_type_consistent"),
        "semantic_judgement_status": case.get("semantic_judgement_status"),
        "semantic_quality": case.get("semantic_quality"),
        "semantic_error_type": case.get("semantic_error_type"),
        "semantic_confidence": case.get("semantic_confidence"),
        "problem_summary": problem_summary(case, 320 if for_json else 220),
        "reason": full_export_text(case.get("reason") or case.get("final_reason")),
        "evidence": full_export_text(case.get("evidence")),
        "observed_mismatch": full_export_text(case.get("observed_mismatch")),
        "repair_needed": case.get("repair_needed"),
        "repair_status": case.get("repair_status"),
        "repair_priority": case.get("repair_priority"),
        "repair_action_type": case.get("repair_action_type"),
        "recommended_interaction_type": case.get("recommended_interaction_type"),
        "recommended_input_mode": case.get("recommended_input_mode"),
        "recommended_change": full_export_text(case.get("recommended_change")),
        "repair_reason": full_export_text(case.get("repair_reason")),
        "repair_evidence": full_export_text(case.get("repair_evidence")),
        "repair_confidence": case.get("repair_confidence"),
        "manual_review_required": case.get("manual_review_required"),
        "requires_config_rebuild": case.get("requires_config_rebuild"),
        "requires_answer_specs_rebuild": case.get("requires_answer_specs_rebuild"),
    }


def compact_interaction_issue_csv_rows(case_results: list[dict[str, Any]], statuses: set[str]) -> list[dict[str, Any]]:
    rows = []
    for case in case_results:
        if case_assessment_status(case) not in statuses:
            continue
        compacted = compact_interaction_issue(case)
        rows.append({field: compacted.get(field, "") for field in INTERACTION_ISSUE_CSV_FIELDS})
    return rows


def build_question_issue_details_json(
    question_summary: list[dict[str, Any]],
    case_results: list[dict[str, Any]],
    *,
    generated_at: str,
    output_files: dict[str, str],
) -> dict[str, Any]:
    grouped_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_results:
        grouped_cases[str(row.get("record_id") or "")].append(row)

    issue_questions = question_issue_rows(question_summary)
    question_counts = Counter(row.get("question_status") or "unknown" for row in issue_questions)
    interaction_issue_cases = [case for case in case_results if has_interaction_issue(case)]
    interaction_counts = Counter(case_assessment_status(case) for case in interaction_issue_cases)
    issue_payload = []
    for row in issue_questions:
        record_id = str(row.get("record_id") or "")
        issue_cases = [case for case in grouped_cases.get(record_id, []) if has_interaction_issue(case)]
        issue_payload.append(
            {
                "record_id": record_id,
                "record_name": row.get("record_name"),
                "question_status": row.get("question_status"),
                "total_semantic_cases": row.get("total_semantic_cases"),
                "status_counts": {
                    "ok": row.get("ok_case_count", 0),
                    "warning": row.get("warning_case_count", 0),
                    "needs_review": row.get("needs_review_case_count", 0),
                    "bad": row.get("bad_case_count", 0),
                    "structural_error": row.get("structural_error_case_count", 0),
                },
                "main_issue_summary": row.get("main_issue_summary"),
                "recommended_actions": question_recommended_actions(issue_cases),
                "issue_locations": [compact_interaction_issue(case, for_json=True) for case in issue_cases],
            }
        )

    return {
        "run_info": {
            "generated_at": generated_at,
            "pipeline_version": INTERACTION_TYPE_OUTPUT_VERSION,
            "total_source_questions": len(question_summary),
            "total_semantic_cases": len(case_results),
            "included_issue_questions": len(issue_payload),
        },
        "summary": {
            "question_issue_counts": {
                "bad": question_counts.get("bad", 0),
                "structural_error": question_counts.get("structural_error", 0),
                "warning": question_counts.get("warning", 0),
                "needs_review": question_counts.get("needs_review", 0),
            },
            "interaction_issue_counts": {
                "bad": interaction_counts.get("bad", 0),
                "structural_error": interaction_counts.get("structural_error", 0),
                "warning": interaction_counts.get("warning", 0),
                "needs_review": interaction_counts.get("needs_review", 0),
            },
            "repair_counts": {
                "repair_needed": sum(1 for case in interaction_issue_cases if bool_value(case.get("repair_needed"))),
                "repair_suggested": sum(1 for case in interaction_issue_cases if case.get("repair_status") == "suggested"),
                "manual_review_required": sum(1 for case in interaction_issue_cases if bool_value(case.get("manual_review_required"))),
                "requires_config_rebuild": sum(1 for case in interaction_issue_cases if bool_value(case.get("requires_config_rebuild"))),
                "requires_answer_specs_rebuild": sum(
                    1 for case in interaction_issue_cases if bool_value(case.get("requires_answer_specs_rebuild"))
                ),
            },
        },
        "issue_questions": issue_payload,
        "output_files": output_files,
    }


def write_plan_interaction_type_output_files(
    results_dir: Path,
    results: list[dict[str, Any]],
    problem_report: dict[str, Any],
    record_summary: list[dict[str, Any]],
    repair_suggestions: list[dict[str, Any]],
    repair_summary: dict[str, Any],
    *,
    include_debug: bool = False,
    legacy_outputs: bool = False,
) -> dict[str, Path]:
    """Write the compact interactionType output layout.

    Default output is issue-first:
    report/question_issue_details.json plus compact bad/warning CSV extracts.
    Debug and legacy files are opt-in so the root results folder stays readable.
    """

    results_dir.mkdir(parents=True, exist_ok=True)
    if not legacy_outputs:
        clean_legacy_interaction_type_outputs(results_dir)
    for filename in STALE_COMPACT_INTERACTION_TYPE_OUTPUT_FILES:
        stale_path = results_dir / filename
        if stale_path.is_file():
            stale_path.unlink()

    output_root = results_dir / "outputs"
    report_dir = output_root / "report"
    evidence_dir = output_root / "evidence"
    debug_dir = output_root / "debug"
    output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    issue_cases = problem_report.get("issue_cases") or []
    question_summary = build_question_level_summary(results, record_summary)
    filtered_repair_suggestions = repair_suggestion_rows(repair_suggestions)

    question_summary_path = report_dir / "question_summary.csv"
    general_report_path = report_dir / "general_report.md"
    repair_suggestions_path = report_dir / "repair_suggestions.csv"
    question_issue_details_path = report_dir / "question_issue_details.json"
    question_bad_path = report_dir / "question_bad.csv"
    question_warning_needs_review_path = report_dir / "question_warning_needs_review.csv"
    interaction_evidence_path = evidence_dir / "interaction_evidence.csv"
    interaction_bad_path = report_dir / "interaction_bad.csv"
    interaction_warning_needs_review_path = report_dir / "interaction_warning_needs_review.csv"
    run_summary_path = evidence_dir / "run_summary.json"

    output_files = {
        "question_issue_details_json": relative_output(question_issue_details_path, output_root),
        "question_bad_csv": relative_output(question_bad_path, output_root),
        "question_warning_needs_review_csv": relative_output(question_warning_needs_review_path, output_root),
        "interaction_bad_csv": relative_output(interaction_bad_path, output_root),
        "interaction_warning_needs_review_csv": relative_output(interaction_warning_needs_review_path, output_root),
        "general_report": relative_output(general_report_path, output_root),
        "run_summary": relative_output(run_summary_path, output_root),
    }
    if include_debug:
        output_files.update(
            {
                "raw_pipeline_results": "debug/raw_pipeline_results.json",
                "raw_problem_report": "debug/raw_problem_report.json",
                "eval_debug_report": "debug/eval_debug_report.md",
                "debug_interaction_issues": "debug/interaction_issues.csv",
                "debug_repair_summary": "debug/repair_summary.json",
                "question_summary": "debug/question_summary.csv",
                "repair_suggestions": "debug/repair_suggestions.csv",
                "interaction_evidence": "debug/interaction_evidence.csv",
            }
        )

    generated_at = utc_timestamp()
    run_summary = build_run_summary(results, question_summary, filtered_repair_suggestions, repair_summary, output_files)
    run_summary["generated_at"] = generated_at

    general_report_path.write_text(build_interaction_type_general_markdown(run_summary, question_summary), encoding="utf-8")
    write_plain_json(
        question_issue_details_path,
        build_question_issue_details_json(
            question_summary,
            results,
            generated_at=generated_at,
            output_files=output_files,
        ),
    )
    write_csv(question_bad_path, question_rows_by_status_group(question_summary, BAD_STATUSES), QUESTION_ISSUE_CSV_FIELDS)
    write_csv(
        question_warning_needs_review_path,
        question_rows_by_status_group(question_summary, WARNING_STATUSES),
        QUESTION_ISSUE_CSV_FIELDS,
    )
    write_csv(interaction_bad_path, compact_interaction_issue_csv_rows(results, BAD_STATUSES), INTERACTION_ISSUE_CSV_FIELDS)
    write_csv(
        interaction_warning_needs_review_path,
        compact_interaction_issue_csv_rows(results, WARNING_STATUSES),
        INTERACTION_ISSUE_CSV_FIELDS,
    )
    write_plain_json(run_summary_path, run_summary)

    written_paths = {
        "general_report": general_report_path,
        "question_issue_details": question_issue_details_path,
        "question_bad": question_bad_path,
        "question_warning_needs_review": question_warning_needs_review_path,
        "interaction_bad": interaction_bad_path,
        "interaction_warning_needs_review": interaction_warning_needs_review_path,
        "run_summary": run_summary_path,
    }

    if include_debug:
        raw_pipeline_results_path = debug_dir / "raw_pipeline_results.json"
        raw_problem_report_path = debug_dir / "raw_problem_report.json"
        eval_debug_report_path = debug_dir / "eval_debug_report.md"
        interaction_issues_path = debug_dir / "interaction_issues.csv"
        repair_summary_path = debug_dir / "repair_summary.json"
        debug_question_summary_path = debug_dir / "question_summary.csv"
        debug_repair_suggestions_path = debug_dir / "repair_suggestions.csv"
        debug_interaction_evidence_path = debug_dir / "interaction_evidence.csv"
        write_json(raw_pipeline_results_path, results)
        write_plain_json(raw_problem_report_path, problem_report)
        eval_summary = build_plan_interaction_type_summary(results, question_summary)
        eval_debug_report_path.write_text(
            build_plan_interaction_type_legacy_eval_markdown(problem_report)
            + "\n\n"
            + build_plan_interaction_type_eval_markdown(eval_summary, question_summary),
            encoding="utf-8",
        )
        write_csv(interaction_issues_path, issue_cases, PLAN_INTERACTION_TYPE_ISSUE_FIELDS)
        write_plain_json(repair_summary_path, repair_summary)
        write_csv(debug_question_summary_path, question_summary, PLAN_INTERACTION_TYPE_QUESTION_SUMMARY_FIELDS)
        write_csv(debug_repair_suggestions_path, filtered_repair_suggestions, PLAN_INTERACTION_TYPE_REPAIR_FIELDS)
        write_csv(debug_interaction_evidence_path, results, LABEL_ASSIGNMENT_FIELDS)
        written_paths.update(
            {
                "raw_pipeline_results": raw_pipeline_results_path,
                "raw_problem_report": raw_problem_report_path,
                "eval_debug_report": eval_debug_report_path,
                "debug_interaction_issues": interaction_issues_path,
                "debug_repair_summary": repair_summary_path,
                "question_summary": debug_question_summary_path,
                "repair_suggestions": debug_repair_suggestions_path,
                "interaction_evidence": debug_interaction_evidence_path,
            }
        )

    if legacy_outputs:
        legacy_assignment_paths = write_label_assignment_result_files(results_dir, results, problem_report, record_summary)
        legacy_repair_paths = write_label_repair_result_files(results_dir, filtered_repair_suggestions, repair_summary)
        for index, path in enumerate([*legacy_assignment_paths, *legacy_repair_paths], start=1):
            written_paths[f"legacy_{index}"] = path

    return written_paths


def case_assessment_status(case: dict[str, Any]) -> str:
    final_status = str(case.get("final_status") or "").strip()
    semantic_status = str(case.get("semantic_judgement_status") or "").strip()
    rule_status = str(case.get("rule_validation_status") or "").strip()
    if final_status == "structural_error" or rule_status in {"structural_error", "insufficient_input"}:
        return "structural_error"
    if final_status == "bad" or semantic_status == "bad":
        return "bad"
    if final_status == "needs_review" or semantic_status == "needs_review" or case.get("manual_review_required"):
        return "needs_review"
    if final_status == "warning" or semantic_status == "warning":
        return "warning"
    if final_status == "ok" or semantic_status == "ok":
        return "ok"
    return "unknown"


def derive_question_status(cases: list[dict[str, Any]]) -> str:
    if not cases:
        return "unknown"
    statuses = [case_assessment_status(case) for case in cases]
    for status in ("structural_error", "bad", "needs_review", "warning"):
        if status in statuses:
            return status
    if statuses and all(status == "ok" for status in statuses):
        return "ok"
    return "unknown"


def compact_text(value: Any, limit: int = 220) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text[: limit - 3] + "..." if len(text) > limit else text


def unique_nonempty(values: list[Any]) -> list[str]:
    seen = set()
    items = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            items.append(text)
    return items


def case_location(case: dict[str, Any]) -> str:
    plan_order = case.get("question_order") or 1
    item_order = case.get("item_order") or "?"
    interaction_order = case.get("interaction_order") or "?"
    return f"plan_{plan_order}.item_{item_order}.interaction_{interaction_order}"


def format_issue_location(case: dict[str, Any]) -> str:
    status = case_assessment_status(case)
    location = case_location(case)
    structural_error = case.get("structural_error_type")
    assigned = case.get("assigned_interaction_type") or case.get("generated_interaction_type") or "missing"
    recommended = case.get("recommended_interaction_type")
    if status == "structural_error":
        return f"{location}: structural_error {structural_error or 'unknown'}"
    if recommended:
        return f"{location}: {assigned} -> {recommended} [{status}]"
    return f"{location}: {assigned} [{status}]"


def build_main_issue_summary(question_status: str, cases: list[dict[str, Any]]) -> str:
    issue_cases = [
        case
        for case in cases
        if case_assessment_status(case) != "ok"
        or case.get("repair_needed")
        or case.get("manual_review_required")
    ]
    if question_status == "ok" and not issue_cases:
        return "Không phát hiện vấn đề ở các interactionType trong question này."
    if question_status == "unknown":
        return "Không có semantic case hợp lệ để kết luận cho question này."

    counts = Counter(case_assessment_status(case) for case in issue_cases)
    transitions = Counter(
        (
            case.get("assigned_interaction_type") or "missing",
            case.get("recommended_interaction_type") or "",
        )
        for case in issue_cases
        if case.get("recommended_interaction_type")
    )
    parts = []
    for status in ("structural_error", "bad", "needs_review", "warning"):
        if counts.get(status):
            parts.append(f"{counts[status]} {status}")
    summary = f"Question này có {', '.join(parts)} interaction cần chú ý."
    if transitions:
        transition_text = "; ".join(
            f"{count} interaction từ {assigned} sang {recommended}"
            for (assigned, recommended), count in transitions.most_common(3)
        )
        summary += f" Repair gợi ý: {transition_text}."
    else:
        reasons = [
            compact_text(
                case.get("candidate_issue_summary")
                or case.get("observed_mismatch")
                or case.get("reason")
                or case.get("repair_reason"),
                140,
            )
            for case in issue_cases[:2]
        ]
        reasons = [reason for reason in reasons if reason]
        if reasons:
            summary += " " + " ".join(reasons)
    return compact_text(summary, 420)


def build_recommended_actions(cases: list[dict[str, Any]]) -> str:
    issue_cases = [
        case
        for case in cases
        if case_assessment_status(case) != "ok"
        or case.get("repair_needed")
        or case.get("manual_review_required")
    ]
    if not issue_cases:
        return "Không cần sửa."

    if any(case_assessment_status(case) == "needs_review" or case.get("manual_review_required") for case in issue_cases):
        review_prefix = "Review và cân nhắc"
    else:
        review_prefix = "Cân nhắc"

    transitions = Counter(
        (
            case.get("assigned_interaction_type") or "missing",
            case.get("recommended_interaction_type") or "",
        )
        for case in issue_cases
        if case.get("recommended_interaction_type")
    )
    actions = []
    if transitions:
        for (assigned, recommended), count in transitions.most_common(3):
            if count == 1:
                matching_case = next(
                    case
                    for case in issue_cases
                    if (case.get("assigned_interaction_type") or "missing", case.get("recommended_interaction_type") or "")
                    == (assigned, recommended)
                )
                actions.append(f"{review_prefix} đổi {case_location(matching_case)} từ {assigned} sang {recommended}")
            else:
                actions.append(f"{review_prefix} đổi {count} interaction từ {assigned} sang {recommended}")

    if any(case.get("requires_config_rebuild") or case.get("requires_answer_specs_rebuild") for case in issue_cases):
        actions.append("rebuild config/answerSpecs tương ứng")

    if any(case.get("repair_action_type") in {"mark_for_human_review", "no_auto_fix"} for case in issue_cases):
        actions.append("chuyển human review vì evidence chưa đủ chắc hoặc không nên auto fix")

    return compact_text("; ".join(actions) + ".", 520) if actions else "Chuyển human review."


def worst_semantic_error_type(cases: list[dict[str, Any]]) -> str:
    severity = {"structural_error": 5, "bad": 4, "needs_review": 3, "warning": 2, "ok": 1, "unknown": 0}
    sorted_cases = sorted(cases, key=lambda case: severity.get(case_assessment_status(case), 0), reverse=True)
    for case in sorted_cases:
        error_type = case.get("semantic_error_type") or case.get("structural_error_type")
        if error_type:
            return str(error_type)
    return ""


def build_question_level_summary(
    case_results: list[dict[str, Any]],
    record_summary: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    record_names: dict[str, str] = {}
    for case in case_results:
        record_id = str(case.get("record_id") or "")
        grouped[record_id].append(case)
        record_names[record_id] = str(case.get("record_name") or "")

    ordered_record_ids = []
    for row in record_summary or []:
        record_id = str(row.get("record_id") or "")
        if record_id and record_id not in ordered_record_ids:
            ordered_record_ids.append(record_id)
            record_names.setdefault(record_id, str(row.get("record_name") or ""))
    for case in case_results:
        record_id = str(case.get("record_id") or "")
        if record_id not in ordered_record_ids:
            ordered_record_ids.append(record_id)

    rows = []
    for record_id in ordered_record_ids:
        cases = grouped.get(record_id, [])
        status_counts = Counter(case_assessment_status(case) for case in cases)
        question_status = derive_question_status(cases)
        issue_cases = [
            case
            for case in cases
            if case_assessment_status(case) != "ok"
            or case.get("repair_needed")
            or case.get("manual_review_required")
        ]
        rows.append(
            {
                "record_id": record_id,
                "record_name": record_names.get(record_id, ""),
                "question_status": question_status,
                "total_semantic_cases": len(cases),
                "ok_case_count": status_counts.get("ok", 0),
                "warning_case_count": status_counts.get("warning", 0),
                "needs_review_case_count": status_counts.get("needs_review", 0),
                "bad_case_count": status_counts.get("bad", 0),
                "structural_error_case_count": status_counts.get("structural_error", 0),
                "repair_needed_count": sum(1 for case in cases if case.get("repair_needed")),
                "repair_suggested_count": sum(1 for case in cases if case.get("repair_status") == "suggested"),
                "manual_review_required_count": sum(1 for case in cases if case.get("manual_review_required")),
                "requires_config_rebuild_count": sum(1 for case in cases if case.get("requires_config_rebuild")),
                "requires_answer_specs_rebuild_count": sum(1 for case in cases if case.get("requires_answer_specs_rebuild")),
                "issue_locations": "\n".join(format_issue_location(case) for case in issue_cases),
                "main_issue_summary": build_main_issue_summary(question_status, cases),
                "recommended_actions": build_recommended_actions(cases),
                "worst_semantic_error_type": worst_semantic_error_type(cases),
                "assigned_interaction_types": unique_nonempty(
                    [case.get("assigned_interaction_type") for case in cases]
                ),
                "recommended_interaction_types": unique_nonempty(
                    [case.get("recommended_interaction_type") for case in cases]
                ),
            }
        )
    return rows


def build_plan_interaction_type_summary(
    case_results: list[dict[str, Any]],
    question_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    question_counts = Counter(row.get("question_status") or "unknown" for row in question_summary)
    interaction_counts = Counter(case_assessment_status(case) for case in case_results)
    return {
        "total_source_questions": len(question_summary),
        "total_semantic_cases": len(case_results),
        "question_level": {
            "ok_count": question_counts.get("ok", 0),
            "warning_count": question_counts.get("warning", 0),
            "needs_review_count": question_counts.get("needs_review", 0),
            "bad_count": question_counts.get("bad", 0),
            "structural_error_count": question_counts.get("structural_error", 0),
            "unknown_count": question_counts.get("unknown", 0),
        },
        "interaction_level": {
            "ok_count": interaction_counts.get("ok", 0),
            "warning_count": interaction_counts.get("warning", 0),
            "needs_review_count": interaction_counts.get("needs_review", 0),
            "bad_count": interaction_counts.get("bad", 0),
            "structural_error_count": interaction_counts.get("structural_error", 0),
            "unknown_count": interaction_counts.get("unknown", 0),
        },
        "repair": {
            "repair_checked_cases": sum(1 for case in case_results if case.get("repair_status")),
            "repair_needed_count": sum(1 for case in case_results if case.get("repair_needed")),
            "repair_suggested_count": sum(1 for case in case_results if case.get("repair_status") == "suggested"),
            "manual_review_required_count": sum(1 for case in case_results if case.get("manual_review_required")),
            "requires_config_rebuild_count": sum(1 for case in case_results if case.get("requires_config_rebuild")),
            "requires_answer_specs_rebuild_count": sum(
                1 for case in case_results if case.get("requires_answer_specs_rebuild")
            ),
        },
        "count_by_assigned_interaction_type": dict(
            Counter(case.get("assigned_interaction_type") or "missing" for case in case_results)
        ),
        "count_by_recommended_interaction_type": dict(
            Counter(case.get("recommended_interaction_type") or "none" for case in case_results)
        ),
        "count_by_semantic_error_type": dict(
            Counter(case.get("semantic_error_type") or "none" for case in case_results)
        ),
        "count_by_repair_action_type": dict(
            Counter(case.get("repair_action_type") or "none" for case in case_results)
        ),
    }


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Question | Status | Issue locations | Main issue | Recommended actions |", "|---|---|---|---|---|"]
    if not rows:
        lines.append("| No questions. |  |  |  |  |")
        return lines
    for row in rows:
        issue_locations = str(row.get("issue_locations") or "").replace("\n", "<br>")
        lines.append(
            "| "
            + " | ".join(
                [
                    compact_text(row.get("record_name") or row.get("record_id"), 90).replace("|", "\\|"),
                    str(row.get("question_status") or "").replace("|", "\\|"),
                    compact_text(issue_locations, 260).replace("|", "\\|"),
                    compact_text(row.get("main_issue_summary"), 260).replace("|", "\\|"),
                    compact_text(row.get("recommended_actions"), 260).replace("|", "\\|"),
                ]
            )
            + " |"
        )
    return lines


def common_issue_group_lines(question_summary: list[dict[str, Any]]) -> list[str]:
    transition_counts = Counter()
    error_counts = Counter()
    for row in question_summary:
        if row.get("question_status") == "ok":
            continue
        error_type = row.get("worst_semantic_error_type")
        if error_type:
            error_counts[str(error_type)] += 1
        for line in str(row.get("issue_locations") or "").splitlines():
            match = re.search(r":\s*([a-z_]+)\s*->\s*([a-z_]+)\s*\[", line)
            if match:
                transition_counts[(match.group(1), match.group(2))] += 1

    lines = []
    if transition_counts:
        lines.append("### Type-change patterns")
        lines.append("")
        for (assigned, recommended), count in transition_counts.most_common(6):
            lines.append(f"- {assigned} -> {recommended}: {count} interaction(s)")
        lines.append("")
    if error_counts:
        lines.append("### Semantic error types")
        lines.append("")
        for error_type, count in error_counts.most_common(6):
            lines.append(f"- {error_type}: {count} question(s)")
        lines.append("")
    if not lines:
        lines.extend(["No repeated issue groups detected.", ""])
    return lines


def build_interaction_type_general_markdown(
    summary: dict[str, Any],
    question_summary: list[dict[str, Any]],
) -> str:
    question_status_counts = summary.get("question_status_counts") or {}
    interaction_status_counts = summary.get("interaction_status_counts") or {}
    repair = summary.get("repair") or {}
    issue_questions = [row for row in question_summary if row.get("question_status") != "ok"]
    bad_questions = [row for row in question_summary if row.get("question_status") in {"bad", "structural_error"}]
    needs_review_questions = [row for row in question_summary if row.get("question_status") == "needs_review"]
    warning_questions = [row for row in question_summary if row.get("question_status") == "warning"]
    lines = [
        "# Interaction Type general Report",
        "",
        "## Overall Summary",
        "",
        f"- Total source questions: {summary.get('total_records')}",
        f"- Total semantic / interaction-level cases: {summary.get('total_semantic_cases')}",
        f"- Questions OK: {question_status_counts.get('ok', 0)}",
        f"- Questions with warning: {question_status_counts.get('warning', 0)}",
        f"- Questions needs review: {question_status_counts.get('needs_review', 0)}",
        f"- Questions bad: {question_status_counts.get('bad', 0)}",
        f"- Questions structural error: {question_status_counts.get('structural_error', 0)}",
        f"- Interaction cases OK: {interaction_status_counts.get('ok', 0)}",
        f"- Interaction cases warning: {interaction_status_counts.get('warning', 0)}",
        f"- Interaction cases needs review: {interaction_status_counts.get('needs_review', 0)}",
        f"- Interaction cases bad: {interaction_status_counts.get('bad', 0)}",
        f"- Repair checked cases: {repair.get('repair_checked_cases', 0)}",
        f"- Repair needed: {repair.get('repair_needed_count', 0)}",
        "",
        "Main conclusion is question-level: each source question is classified from all semantic cases under that record.",
        "Interaction-level cases are technical evidence for tracing the exact plan/item/interaction location.",
        "Repair suggestions are recommendations only; this pipeline does not mutate source data automatically.",
        "",
        "## Question-Level Issues",
        "",
        "### Bad / Structural Error Questions",
        "",
        *markdown_table(bad_questions),
        "",
        "### Needs Review Questions",
        "",
        *markdown_table(needs_review_questions),
        "",
        "### Warning Questions",
        "",
        *markdown_table(warning_questions),
        "",
        "## Common Issue Groups",
        "",
        *common_issue_group_lines(issue_questions),
        "## Recommended Actions",
        "",
    ]
    if issue_questions:
        for row in issue_questions:
            lines.append(
                f"- {compact_text(row.get('record_name') or row.get('record_id'), 110)}: "
                f"{compact_text(row.get('recommended_actions'), 260)}"
            )
    else:
        lines.append("- Không cần sửa.")
    lines.extend(
        [
            "",
            "## Evidence Files",
            "",
            "- `question_issue_details.json`: main detailed issue report, grouped by source question.",
            "- `question_bad.csv` and `interaction_bad.csv`: compact extracts for confirmed bad/structural-error cases.",
            "- `question_warning_needs_review.csv` and `interaction_warning_needs_review.csv`: compact extracts for warning/needs-review cases.",
            "- Full interaction evidence is available with `--include-debug` at `../debug/interaction_evidence.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_plan_interaction_type_eval_markdown(
    summary: dict[str, Any],
    question_summary: list[dict[str, Any]] | None = None,
) -> str:
    if question_summary is None:
        return build_plan_interaction_type_legacy_eval_markdown(summary)
    question_level = summary.get("question_level") or {}
    interaction_level = summary.get("interaction_level") or {}
    repair = summary.get("repair") or {}
    bad_questions = [row for row in question_summary if row.get("question_status") in {"bad", "structural_error"}]
    warning_questions = [row for row in question_summary if row.get("question_status") == "warning"]
    needs_review_questions = [row for row in question_summary if row.get("question_status") == "needs_review"]
    lines = [
        "# Interaction Type Evaluation Report",
        "",
        "## Overall summary",
        "",
        f"- Total source questions: {summary.get('total_source_questions')}",
        f"- Total semantic cases: {summary.get('total_semantic_cases')}",
        f"- Questions OK: {question_level.get('ok_count')}",
        f"- Questions with warning: {question_level.get('warning_count')}",
        f"- Questions needs review: {question_level.get('needs_review_count')}",
        f"- Questions bad: {question_level.get('bad_count')}",
        f"- Questions structural error: {question_level.get('structural_error_count')}",
        f"- Questions unknown: {question_level.get('unknown_count')}",
        f"- Interaction warnings: {interaction_level.get('warning_count')}",
        f"- Interaction needs review: {interaction_level.get('needs_review_count')}",
        f"- Interaction bad: {interaction_level.get('bad_count')}",
        f"- Repair checked cases: {repair.get('repair_checked_cases')}",
        f"- Repair needed: {repair.get('repair_needed_count')}",
        "",
        "Pipeline vẫn kiểm tra chi tiết theo từng interactionType, nhưng kết luận chính được tổng hợp theo từng source question.",
        "",
        "## Question-level issues",
        "",
        "### Bad questions",
        "",
        *markdown_table(bad_questions),
        "",
        "### Warning questions",
        "",
        *markdown_table(warning_questions),
        "",
        "### Needs review questions",
        "",
        *markdown_table(needs_review_questions),
        "",
        "## Interaction-level details",
        "",
        "Interaction-level cases are used as evidence. See `plan_interaction_type_interaction_issues.csv`.",
        "",
    ]
    return "\n".join(lines)


def build_plan_interaction_type_legacy_eval_markdown(summary: dict[str, Any]) -> str:
    bad_cases = summary.get("bad_cases") or []
    warning_cases = summary.get("warning_cases") or []
    needs_review_cases = summary.get("needs_review_cases") or []
    structural_error_cases = summary.get("structural_error_cases") or []
    repair_suggestions = summary.get("repair_suggestions") or []
    lines = [
        "# Interaction Type Evaluation Report",
        "",
        "## Summary",
        "",
        f"- Total records: {summary.get('total_records')}",
        f"- Total cases: {summary.get('total_cases')}",
        f"- Structural errors: {summary.get('structural_error_count')}",
        f"- Semantic OK: {summary.get('semantic_ok_count')}",
        f"- Warning: {summary.get('final_warning_count')}",
        f"- Needs review: {summary.get('final_needs_review_count')}",
        f"- Bad: {summary.get('final_bad_count')}",
        f"- Repair needed: {summary.get('repair_needed_count')}",
        f"- Manual review required: {summary.get('manual_review_required_count')}",
        "",
        "## Structural errors",
        "",
    ]
    if not structural_error_cases:
        lines.append("No structural errors.")
    for index, case in enumerate(structural_error_cases, start=1):
        lines.extend(markdown_issue_case(index, case, include_repair=False))
    lines.extend(
        [
            "",
            "## Bad cases",
            "",
        ]
    )
    if not bad_cases:
        lines.append("No bad cases.")
    for index, case in enumerate(bad_cases, start=1):
        lines.extend(markdown_issue_case(index, case, include_repair=False))
    lines.extend(
        [
            "",
            "## Warning cases",
            "",
        ]
    )
    if not warning_cases:
        lines.append("No warning cases.")
    for index, case in enumerate(warning_cases, start=1):
        lines.extend(markdown_issue_case(index, case, include_repair=False))
    lines.extend(
        [
            "",
            "## Needs review cases",
            "",
        ]
    )
    if not needs_review_cases:
        lines.append("No needs-review cases.")
    for index, case in enumerate(needs_review_cases, start=1):
        lines.extend(markdown_issue_case(index, case, include_repair=False))
    lines.extend(
        [
            "",
            "## Repair suggestions",
            "",
        ]
    )
    if not repair_suggestions:
        lines.append("No repair suggestions.")
    for index, case in enumerate(repair_suggestions, start=1):
        lines.extend(markdown_issue_case(index, case, include_repair=True))
    lines.append("")
    return "\n".join(lines)


def markdown_issue_case(index: int, case: dict[str, Any], *, include_repair: bool) -> list[str]:
    lines = [
        f"### {index}. {case.get('record_name')}",
        "",
        f"- Case ID: `{case.get('case_id')}`",
        f"- Assigned type: `{case.get('assigned_interaction_type')}`",
        f"- Generated type: `{case.get('generated_interaction_type')}`",
        f"- Final status: `{case.get('final_status')}`",
        f"- Semantic status: `{case.get('semantic_judgement_status')}`",
        f"- Semantic error: `{case.get('semantic_error_type')}`",
        f"- Reason: {case.get('short_reason')}",
        f"- Evidence: {case.get('evidence_brief')}",
    ]
    if include_repair:
        lines.extend(
            [
                f"- Repair status: `{case.get('repair_status')}`",
                f"- Repair action: `{case.get('repair_action_type')}`",
                f"- Recommended type: `{case.get('recommended_interaction_type')}`",
                f"- Recommended input mode: `{case.get('recommended_input_mode')}`",
                f"- Recommended change: {case.get('recommended_change')}",
                f"- Repair reason: {case.get('repair_reason')}",
                f"- Repair confidence: `{case.get('repair_confidence')}`",
            ]
        )
    lines.append("")
    return lines


def build_plan_interaction_type_error_markdown(summary: dict[str, Any]) -> str:
    return build_plan_interaction_type_eval_markdown(summary)


def build_plan_interaction_type_repair_markdown(
    summary: dict[str, Any],
    suggestions: list[dict[str, Any]],
) -> str:
    question_rows = build_question_level_summary(suggestions)
    repair_issue_questions = [row for row in question_rows if row.get("question_status") != "ok"]
    lines = [
        "# Interaction Type Repair Suggestions",
        "",
        "## Summary",
        "",
        f"- Total records: {summary.get('total_records')}",
        f"- Total semantic cases: {summary.get('total_semantic_cases')}",
        f"- Repair checked cases: {summary.get('repair_checked_cases')}",
        f"- Repair suggestions: {summary.get('repair_suggestion_count')}",
        f"- Repair needed: {summary.get('repair_needed_count')}",
        f"- Needs human review: {summary.get('needs_human_review_count')}",
        f"- No auto fix: {summary.get('no_auto_fix_count')}",
        f"- Change interactionType: {summary.get('change_interaction_type_count')}",
        f"- Rebuild config/answerSpecs: {summary.get('rebuild_config_answer_specs_count')}",
        "",
        "## Question-level repair summary",
        "",
        "Repair suggestions are grouped by source question first. Interaction-level suggestions below are evidence.",
        "",
        *markdown_table(repair_issue_questions),
        "",
    ]
    if not suggestions:
        lines.extend(["## Suggestions", "", "No repair suggestions.", ""])
        return "\n".join(lines)

    for title, rows in (
        ("High priority repairs", [row for row in suggestions if row.get("repair_priority") == "high"]),
        ("Medium priority repairs", [row for row in suggestions if row.get("repair_priority") == "medium"]),
        ("Low priority repairs", [row for row in suggestions if row.get("repair_priority") == "low"]),
    ):
        lines.extend([f"## {title}", ""])
        if not rows:
            lines.extend(["No cases.", ""])
            continue
        for index, row in enumerate(rows, start=1):
            lines.extend(
                [
                    f"### {index}. {row.get('record_name')}",
                    "",
                    f"- Case ID: `{row.get('case_id')}`",
                    f"- Semantic status: `{row.get('semantic_judgement_status')}`",
                    f"- Assigned type: `{row.get('assigned_interaction_type')}`",
                    f"- Repair status: `{row.get('repair_status')}`",
                    f"- Action: `{row.get('repair_action_type')}`",
                    f"- Recommended interaction type: `{row.get('recommended_interaction_type')}`",
                    f"- Recommended input mode: `{row.get('recommended_input_mode')}`",
                    f"- Recommended change: {row.get('recommended_change')}",
                    f"- Reason: {row.get('repair_reason')}",
                    f"- Evidence: {row.get('repair_evidence')}",
                    f"- Repair decision basis: `{row.get('repair_decision_basis')}`",
                    f"- Manual review required: {row.get('manual_review_required')}",
                    f"- Requires config rebuild: {row.get('requires_config_rebuild')}",
                    f"- Requires answerSpecs rebuild: {row.get('requires_answer_specs_rebuild')}",
                    f"- Repair confidence: `{row.get('repair_confidence')}`",
                    "",
                ]
            )
    return "\n".join(lines)

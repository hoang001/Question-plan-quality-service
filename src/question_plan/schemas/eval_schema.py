"""Chuẩn hóa schema cho kết quả đánh giá chất lượng question_plan."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

from ...real_schema import content_to_text


PLAN_EVAL_STATUSES = {
    "ok",
    "warning",
    "needs_review",
    "bad",
    "structural_error",
    "skipped_due_to_source_issue",
}
COVERAGE_STATUSES = {"full", "partial", "unclear", "not_evaluable"}
ISSUE_LEVELS = {"plan", "question", "item", "interaction"}
ISSUE_SEVERITIES = {"warning", "needs_review", "bad", "structural_error"}
ISSUE_TYPES = {
    "coverage_issue",
    "source_fidelity_issue",
    "selected_scope_issue",
    "decomposition_issue",
    "unsupported_format_adaptation_issue",
    "requirement_clarity_issue",
    "interaction_type_issue",
    "choice_planning_issue",
    "answerability_issue",
    "pedagogical_flow_issue",
    "other_plan_quality_issue",
    "structural_error",
}
CONFIDENCES = {"high", "medium", "low"}
COVERAGE_STATES = {"present", "absent", "unclear"}
QUALITY_STATES = {"valid", "present_but_invalid", "unclear"}
STATUS_SEVERITY = {
    "ok": 0,
    "skipped_due_to_source_issue": 1,
    "warning": 2,
    "needs_review": 3,
    "bad": 4,
    "structural_error": 5,
}

REQUIRED_RESULT_FIELDS = {
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
    "issues",
    "recommended_actions",
    "confidence",
}

REQUIRED_ISSUE_FIELDS = {
    "issue_id",
    "issue_level",
    "location",
    "severity",
    "issue_type",
    "summary",
    "evidence",
    "impact_on_generation",
    "recommended_action",
    "suggested_fix",
    "requires_human_review",
    "confidence",
}

TAXONOMY_TERMS = {
    "coverage_issue",
    "source_fidelity_issue",
    "selected_scope_issue",
    "decomposition_issue",
    "unsupported_format_adaptation_issue",
    "requirement_clarity_issue",
    "interaction_type_issue",
    "choice_planning_issue",
    "answerability_issue",
    "pedagogical_flow_issue",
    "other_plan_quality_issue",
    "structural_error",
    "coverage",
    "fidelity",
    "clarity",
    "adaptation",
}
GENERIC_SUMMARY_WARNING = (
    "Lưu ý: summary của issue còn chung chung hoặc chỉ nêu taxonomy; reviewer cần đọc evidence/plan/raw_question "
    "để xác định vấn đề cụ thể."
)


def as_text(value: Any) -> str:
    return content_to_text(value) if not isinstance(value, str) else value.strip()


def as_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [as_text(item) for item in value if as_text(item)]


def as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else False


def is_generic_issue_summary(summary: str, issue_type: str) -> bool:
    normalized = summary.strip().lower()
    if not normalized:
        return False
    compact = normalized.replace(" ", "_")
    if compact == issue_type:
        return True
    if len(normalized.split()) <= 4 and any(term in compact for term in TAXONOMY_TERMS):
        return True
    generic_phrases = [
        "có vấn đề về",
        "vấn đề về",
        "thiếu coverage",
        "coverage issue",
        "fidelity issue",
        "clarity issue",
        "có vấn đề adaptation",
    ]
    return any(phrase in normalized for phrase in generic_phrases)


def normalize_issue(raw_issue: Any, index: int) -> dict[str, Any]:
    issue = raw_issue if isinstance(raw_issue, dict) else {}
    severity = issue.get("severity")
    issue_type = issue.get("issue_type")
    issue_level = issue.get("issue_level")
    confidence = issue.get("confidence")
    summary = as_text(issue.get("summary"))
    impact_on_generation = as_text(issue.get("impact_on_generation"))
    if is_generic_issue_summary(summary, issue_type if issue_type in ISSUE_TYPES else ""):
        impact_on_generation = (
            f"{impact_on_generation} {GENERIC_SUMMARY_WARNING}".strip()
            if impact_on_generation
            else GENERIC_SUMMARY_WARNING
        )
    return {
        "issue_id": as_text(issue.get("issue_id")) or f"issue_{index}",
        "issue_level": issue_level if issue_level in ISSUE_LEVELS else "plan",
        "location": issue.get("location") if isinstance(issue.get("location"), dict) else {},
        "severity": severity if severity in ISSUE_SEVERITIES else "needs_review",
        "issue_type": issue_type if issue_type in ISSUE_TYPES else "selected_scope_issue",
        "summary": summary,
        "evidence": as_text(issue.get("evidence")),
        "impact_on_generation": impact_on_generation,
        "recommended_action": as_text(issue.get("recommended_action")),
        "suggested_fix": as_text(issue.get("suggested_fix")),
        "requires_human_review": as_bool(issue.get("requires_human_review")),
        "confidence": confidence if confidence in CONFIDENCES else "medium",
    }


def normalize_location(value: Any) -> dict[str, Any]:
    location = value if isinstance(value, dict) else {}
    return {
        "questionOrder": location.get("questionOrder"),
        "itemOrder": location.get("itemOrder"),
        "interactionOrder": location.get("interactionOrder"),
    }


def normalize_mapping_location(value: Any) -> dict[str, Any]:
    location = normalize_location(value)
    location["evidence"] = as_text(value.get("evidence")) if isinstance(value, dict) else ""
    return location


def normalize_source_to_plan_mapping(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    mappings: list[dict[str, Any]] = []
    for index, raw_mapping in enumerate(value, start=1):
        mapping = raw_mapping if isinstance(raw_mapping, dict) else {}
        coverage_state = mapping.get("coverage_state")
        quality_state = mapping.get("quality_state")
        locations = mapping.get("matched_plan_locations")
        if not isinstance(locations, list):
            locations = []
        normalized_locations = [normalize_mapping_location(item) for item in locations]
        if normalized_locations and coverage_state == "absent":
            coverage_state = "present"
        mappings.append(
            {
                "source_subpart": as_text(mapping.get("source_subpart")) or f"source_subpart_{index}",
                "matched_plan_locations": normalized_locations,
                "coverage_state": coverage_state if coverage_state in COVERAGE_STATES else "unclear",
                "quality_state": quality_state if quality_state in QUALITY_STATES else "unclear",
                "quality_issue_type": as_text(mapping.get("quality_issue_type")),
                "notes": as_text(mapping.get("notes")),
            }
        )
    return mappings


def derive_coverage_from_mapping(
    mapping: list[dict[str, Any]],
    fallback_source_subparts: list[str],
    fallback_covered_subparts: list[str],
    fallback_missing_subparts: list[str],
    fallback_coverage_status: str,
) -> tuple[list[str], list[str], list[str], str]:
    if not mapping:
        return fallback_source_subparts, fallback_covered_subparts, fallback_missing_subparts, fallback_coverage_status

    source_subparts = [item["source_subpart"] for item in mapping if item.get("source_subpart")]
    covered_subparts = [
        item["source_subpart"]
        for item in mapping
        if item.get("source_subpart") and item.get("coverage_state") == "present"
    ]
    missing_subparts = [
        item["source_subpart"]
        for item in mapping
        if item.get("source_subpart") and item.get("coverage_state") == "absent"
    ]
    unclear_count = sum(1 for item in mapping if item.get("coverage_state") == "unclear")
    absent_count = len(missing_subparts)
    if absent_count:
        coverage_status = "partial"
    elif unclear_count:
        coverage_status = "unclear"
    else:
        coverage_status = "full"
    return source_subparts, covered_subparts, missing_subparts, coverage_status


def text_mentions_subpart(text: str, source_subpart: str) -> bool:
    text_norm = text.lower()
    subpart_norm = source_subpart.lower()
    if subpart_norm and subpart_norm in text_norm:
        return True
    raw_tokens = subpart_norm.replace(")", " ").replace("(", " ").split()
    marker_tokens = [token for token in raw_tokens if len(token) == 1 and token.isalnum()]
    if marker_tokens:
        for token in marker_tokens:
            marker_patterns = [
                rf"hệ phương trình\s+{re.escape(token)}\b",
                rf"ý\s+{re.escape(token)}\b",
                rf"câu\s+{re.escape(token)}\b",
                rf"phần\s+{re.escape(token)}\b",
                rf"\b{re.escape(token)}[\)\.,:]",
            ]
            if any(re.search(pattern, text_norm) for pattern in marker_patterns):
                return True
        return False
    tokens = [token for token in raw_tokens if len(token) > 1]
    return bool(tokens and all(token in text_norm for token in tokens))


def apply_mapping_consistency(
    issues: list[dict[str, Any]],
    mapping: list[dict[str, Any]],
    missing_subparts: list[str],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    warnings: list[str] = []
    present_subparts = {
        item.get("source_subpart", "")
        for item in mapping
        if item.get("coverage_state") == "present" or item.get("matched_plan_locations")
    }
    cleaned_missing = [subpart for subpart in missing_subparts if subpart not in present_subparts]
    removed_missing = sorted(set(missing_subparts) - set(cleaned_missing))
    for subpart in removed_missing:
        warnings.append(
            f"Đã xóa `{subpart}` khỏi missing_subparts vì mapping có matched_plan_locations/coverage_state=present."
        )

    cleaned_issues: list[dict[str, Any]] = []
    for issue in issues:
        if issue.get("issue_type") != "coverage_issue":
            cleaned_issues.append(issue)
            continue
        issue_text = " ".join(
            str(issue.get(field) or "")
            for field in ("summary", "evidence", "suggested_fix", "recommended_action")
        )
        matched_present = [
            subpart
            for subpart in present_subparts
            if subpart and text_mentions_subpart(issue_text, subpart)
        ]
        if matched_present:
            warnings.append(
                "Đã bỏ coverage_issue vì subpart đã có trong plan theo source_to_plan_mapping: "
                + ", ".join(matched_present)
            )
            continue
        cleaned_issues.append(issue)
    return cleaned_issues, cleaned_missing, warnings


def worst_status(statuses: list[str]) -> str:
    if not statuses:
        return "ok"
    return max(statuses, key=lambda status: STATUS_SEVERITY.get(status, 2))


def normalize_question_plan_eval_result(
    parsed: dict[str, Any],
    *,
    record: dict[str, Any],
    structural_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    structural_result = structural_result or {}
    issues = [normalize_issue(issue, index) for index, issue in enumerate(parsed.get("issues") or [], start=1)]
    source_to_plan_mapping = normalize_source_to_plan_mapping(parsed.get("source_to_plan_mapping"))
    coverage_status = parsed.get("coverage_status")
    if coverage_status not in COVERAGE_STATUSES:
        coverage_status = "not_evaluable" if not structural_result.get("structural_valid", True) else "unclear"
    source_subparts, covered_subparts, missing_subparts, coverage_status = derive_coverage_from_mapping(
        source_to_plan_mapping,
        as_text_list(parsed.get("source_subparts")),
        as_text_list(parsed.get("covered_subparts")),
        as_text_list(parsed.get("missing_subparts")),
        coverage_status,
    )
    issues, missing_subparts, judge_consistency_warnings = apply_mapping_consistency(
        issues,
        source_to_plan_mapping,
        missing_subparts,
    )
    if source_to_plan_mapping:
        if missing_subparts:
            coverage_status = "partial"
        elif any(item.get("coverage_state") == "unclear" for item in source_to_plan_mapping):
            coverage_status = "unclear"
        else:
            coverage_status = "full"
        covered_subparts = [
            item["source_subpart"]
            for item in source_to_plan_mapping
            if item.get("source_subpart") and item.get("source_subpart") not in missing_subparts
            and item.get("coverage_state") == "present"
        ]

    issue_status = worst_status([issue["severity"] for issue in issues])
    overall_status = parsed.get("overall_status")
    if overall_status not in PLAN_EVAL_STATUSES or judge_consistency_warnings:
        overall_status = "ok" if not issues else issue_status
    confidence = parsed.get("confidence")
    if confidence not in CONFIDENCES:
        confidence = "medium"

    record_id = as_text(parsed.get("record_id")) or as_text(record.get("_id") or record.get("id"))
    record_name = as_text(parsed.get("record_name")) or as_text(record.get("name"))
    recommended_actions = parsed.get("recommended_actions")
    if isinstance(recommended_actions, str):
        recommended_actions = [recommended_actions]
    recommended_actions = as_text_list(recommended_actions)

    return {
        "record_id": record_id,
        "record_name": record_name,
        "overall_status": overall_status,
        "coverage_status": coverage_status,
        "source_to_plan_mapping": source_to_plan_mapping,
        "source_subparts": source_subparts,
        "covered_subparts": covered_subparts,
        "missing_subparts": missing_subparts,
        "judge_consistency_warnings": judge_consistency_warnings,
        "selected_scope_summary": as_text(parsed.get("selected_scope_summary")),
        "plan_quality_summary": as_text(parsed.get("plan_quality_summary")),
        "issues": issues,
        "recommended_actions": recommended_actions,
        "confidence": confidence,
        "structural_result": structural_result,
    }


def structural_error_result(record: dict[str, Any], structural_result: dict[str, Any]) -> dict[str, Any]:
    issue = normalize_issue(
        {
            "issue_id": "structural_1",
            "issue_level": "plan",
            "location": {},
            "severity": "structural_error",
            "issue_type": "structural_error",
            "summary": structural_result.get("reason") or "question_plan có lỗi cấu trúc.",
            "evidence": structural_result.get("evidence") or "",
            "impact_on_generation": "Không đủ schema để đánh giá semantic hoặc generation ổn định.",
            "recommended_action": "Sửa cấu trúc question_plan theo schema.",
            "suggested_fix": structural_result.get("reason") or "",
            "requires_human_review": True,
            "confidence": "high",
        },
        1,
    )
    return normalize_question_plan_eval_result(
        {
            "record_id": record.get("_id") or record.get("record_id") or record.get("id"),
            "record_name": record.get("name") or record.get("record_name", ""),
            "overall_status": "structural_error",
            "coverage_status": "not_evaluable",
            "source_to_plan_mapping": [],
            "source_subparts": [],
            "covered_subparts": [],
            "missing_subparts": [],
            "selected_scope_summary": "",
            "plan_quality_summary": "Không đánh giá semantic vì question_plan lỗi cấu trúc.",
            "issues": [issue],
            "recommended_actions": ["Sửa cấu trúc question_plan trước khi chạy LLM judge."],
            "confidence": "high",
        },
        record=record,
        structural_result=structural_result,
    )


def skipped_due_to_source_issue_result(
    record: dict[str, Any],
    source_issue: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_issue = source_issue or {}
    return normalize_question_plan_eval_result(
        {
            "record_id": record.get("_id") or record.get("id"),
            "record_name": record.get("name", ""),
            "overall_status": "skipped_due_to_source_issue",
            "coverage_status": "not_evaluable",
            "source_to_plan_mapping": [],
            "source_subparts": [],
            "covered_subparts": [],
            "missing_subparts": [],
            "selected_scope_summary": "",
            "plan_quality_summary": "Bỏ qua đánh giá question_plan vì source/raw record chưa hợp lệ.",
            "issues": [],
            "recommended_actions": ["Kiểm tra và xử lý source/raw record trước khi đánh giá question_plan."],
            "confidence": "high",
        },
        record=record,
        structural_result={
            "structural_valid": True,
            "rule_validation_status": "skipped_due_to_source_issue",
            "source_issue": source_issue,
        },
    )


def build_plan_eval_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(row.get("overall_status") or "unknown" for row in results)
    coverage_counts = Counter(row.get("coverage_status") or "unknown" for row in results)
    issue_type_counts = Counter(
        issue.get("issue_type") or "unknown"
        for row in results
        for issue in row.get("issues") or []
    )
    severity_counts = Counter(
        issue.get("severity") or "unknown"
        for row in results
        for issue in row.get("issues") or []
    )
    total_records = len(results)
    counted_total = (
        status_counts.get("ok", 0)
        + status_counts.get("warning", 0)
        + status_counts.get("needs_review", 0)
        + status_counts.get("bad", 0)
        + status_counts.get("structural_error", 0)
        + status_counts.get("skipped_due_to_source_issue", 0)
    )
    return {
        "total_records": total_records,
        "count_by_status": dict(status_counts),
        "status_counts": dict(status_counts),
        "coverage_status_counts": dict(coverage_counts),
        "issue_type_counts": dict(issue_type_counts),
        "issue_severity_counts": dict(severity_counts),
        "ok_count": status_counts.get("ok", 0),
        "warning_count": status_counts.get("warning", 0),
        "needs_review_count": status_counts.get("needs_review", 0),
        "bad_count": status_counts.get("bad", 0),
        "structural_error_count": status_counts.get("structural_error", 0),
        "skipped_due_to_source_issue_count": status_counts.get("skipped_due_to_source_issue", 0),
        "counted_status_total": counted_total,
        "count_consistency_valid": counted_total == total_records,
        "issue_records_count": sum(1 for row in results if row.get("overall_status") != "ok"),
        "issue_count": sum(len(row.get("issues") or []) for row in results),
    }


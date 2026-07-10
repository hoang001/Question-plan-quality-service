"""LLM judge đánh giá chất lượng tổng thể của `question_plan`."""

from __future__ import annotations

import json
from typing import Any

from ..infra.config import AppConfig
from ..infra.llm_client import LLMClient
from ..knowledge.interaction_type_knowledge import compact_interaction_knowledge
from ..knowledge.plan_knowledge import QUESTION_PLAN_EVAL_KNOWLEDGE
from ..knowledge.prompts import (
    JUDGE_ANSWERABILITY_POLICY,
    JUDGE_CHOICE_POLICY,
    JUDGE_COVERAGE_ISSUE_POLICY,
    JUDGE_COVERAGE_POLICY,
    JUDGE_FORMAT_POLICY,
    JUDGE_MAPPING_POLICY,
    JUDGE_SEVERITY_POLICY,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_TASK_INSTRUCTIONS,
    JUDGE_WRITING_EXAMPLES,
    JUDGE_WRITING_POLICY,
)
from ..schemas.eval_schema import (
    CONFIDENCES,
    COVERAGE_STATUSES,
    ISSUE_LEVELS,
    ISSUE_SEVERITIES,
    ISSUE_TYPES,
    PLAN_EVAL_STATUSES,
    REQUIRED_ISSUE_FIELDS,
    REQUIRED_RESULT_FIELDS,
    normalize_question_plan_eval_result,
)
from ..shared.real_schema import content_to_text
from ..shared.utils import parse_json_output


def truncate(value: Any, limit: int = 6000) -> str:
    text = content_to_text(value) if not isinstance(value, str) else value
    text = str(text or "").strip()
    return text[: limit - 3] + "..." if len(text) > limit else text


def compact_record_payload(record: dict[str, Any], structural_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record.get("_id") or record.get("id") or "",
        "record_name": record.get("name", ""),
        "raw_question": truncate(record.get("question"), 6000),
        "raw_answer": truncate(record.get("answer"), 5000),
        "question_plan": record.get("question_plan"),
        "structural_validation": structural_result,
    }


def build_question_plan_judge_messages(
    record: dict[str, Any],
    structural_result: dict[str, Any],
) -> list[dict[str, str]]:
    payload = compact_record_payload(record, structural_result)
    status_values = " | ".join(sorted(PLAN_EVAL_STATUSES))
    coverage_values = " | ".join(sorted(COVERAGE_STATUSES))
    issue_level_values = " | ".join(sorted(ISSUE_LEVELS))
    issue_severity_values = " | ".join(sorted(ISSUE_SEVERITIES))
    issue_type_values = " | ".join(sorted(ISSUE_TYPES))
    confidence_values = " | ".join(sorted(CONFIDENCES))
    return [
        {
            "role": "system",
            "content": JUDGE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"{JUDGE_TASK_INSTRUCTIONS}\n\n"
                f"{JUDGE_COVERAGE_POLICY}\n\n"
                f"{JUDGE_CHOICE_POLICY}\n\n"
                f"{JUDGE_SEVERITY_POLICY}\n\n"
                f"{JUDGE_WRITING_POLICY}\n\n"
                f"{JUDGE_ANSWERABILITY_POLICY}\n\n"
                f"{JUDGE_COVERAGE_ISSUE_POLICY}\n\n"
                f"{JUDGE_MAPPING_POLICY}\n\n"
                f"{JUDGE_FORMAT_POLICY}\n\n"
                f"{JUDGE_WRITING_EXAMPLES}\n\n"
                "Allowed enums:\n"
                f"- overall_status: {status_values}\n"
                f"- coverage_status: {coverage_values}\n"
                f"- issue_level: {issue_level_values}\n"
                f"- severity của issue: {issue_severity_values}\n"
                f"- issue_type: {issue_type_values}\n"
                f"- confidence: {confidence_values}\n\n"
                "Các field bắt buộc ở cấp record:\n"
                f"{json.dumps(sorted(REQUIRED_RESULT_FIELDS), ensure_ascii=False)}\n\n"
                "Các field bắt buộc ở cấp issue:\n"
                f"{json.dumps(sorted(REQUIRED_ISSUE_FIELDS), ensure_ascii=False)}\n\n"
                "Schema JSON đầu ra:\n"
                "{\n"
                '  "record_id": "",\n'
                '  "record_name": "",\n'
                f'  "overall_status": "{status_values}",\n'
                f'  "coverage_status": "{coverage_values}",\n'
                '  "source_to_plan_mapping": [\n'
                "    {\n"
                '      "source_subpart": "",\n'
                '      "matched_plan_locations": [\n'
                "        {\n"
                '          "questionOrder": null,\n'
                '          "itemOrder": null,\n'
                '          "interactionOrder": null,\n'
                '          "evidence": ""\n'
                "        }\n"
                "      ],\n"
                '      "coverage_state": "present|absent|unclear",\n'
                '      "quality_state": "valid|present_but_invalid|unclear",\n'
                '      "quality_issue_type": "",\n'
                '      "notes": ""\n'
                "    }\n"
                "  ],\n"
                '  "source_subparts": [],\n'
                '  "covered_subparts": [],\n'
                '  "missing_subparts": [],\n'
                '  "selected_scope_summary": "",\n'
                '  "plan_quality_summary": "",\n'
                '  "issues": [\n'
                "    {\n"
                '      "issue_id": "",\n'
                f'      "issue_level": "{issue_level_values}",\n'
                '      "location": {"questionOrder": null, "itemOrder": null, "interactionOrder": null},\n'
                f'      "severity": "{issue_severity_values}",\n'
                f'      "issue_type": "{issue_type_values}",\n'
                '      "summary": "",\n'
                '      "evidence": "",\n'
                '      "impact_on_generation": "",\n'
                '      "recommended_action": "",\n'
                '      "suggested_fix": "",\n'
                '      "requires_human_review": false,\n'
                f'      "confidence": "{confidence_values}"\n'
                "    }\n"
                "  ],\n"
                '  "recommended_actions": [],\n'
                f'  "confidence": "{confidence_values}"\n'
                "}\n\n"
                "Knowledge cho question_plan:\n"
                f"{json.dumps(QUESTION_PLAN_EVAL_KNOWLEDGE, ensure_ascii=False, indent=2)}\n\n"
                "Knowledge cho interactionType:\n"
                f"{json.dumps(compact_interaction_knowledge(), ensure_ascii=False, indent=2)}\n\n"
                "PAYLOAD ĐẦU VÀO:\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def invalid_judge_result(
    *,
    model: str,
    error: str,
    record: dict[str, Any],
    structural_result: dict[str, Any],
    raw_response: Any = None,
    latency_seconds: float = 0,
) -> dict[str, Any]:
    parsed = normalize_question_plan_eval_result(
        {
            "record_id": record.get("_id") or record.get("id"),
            "record_name": record.get("name", ""),
            "overall_status": "needs_review",
            "coverage_status": "unclear",
            "source_to_plan_mapping": [],
            "source_subparts": [],
            "covered_subparts": [],
            "missing_subparts": [],
            "selected_scope_summary": "",
            "plan_quality_summary": "LLM judge không trả về JSON/schema hợp lệ.",
            "issues": [
                {
                    "issue_id": "judge_output_invalid",
                    "issue_level": "plan",
                    "location": {},
                    "severity": "needs_review",
                    "issue_type": "selected_scope_issue",
                    "summary": "Không parse được output của LLM judge.",
                    "evidence": error,
                    "impact_on_generation": "Không thể kết luận tự động về chất lượng plan.",
                    "recommended_action": "Chạy lại judge hoặc review thủ công.",
                    "suggested_fix": "",
                    "requires_human_review": True,
                    "confidence": "low",
                }
            ],
            "recommended_actions": ["Review thủ công vì judge output không hợp lệ."],
            "confidence": "low",
        },
        record=record,
        structural_result=structural_result,
    )
    return {
        **parsed,
        "judge_model": model,
        "json_parse_ok": False,
        "judge_error": error,
        "raw_response": raw_response,
        "latency_seconds": latency_seconds,
    }


def normalize_judge_response(
    *,
    model: str,
    response: dict[str, Any] | None,
    record: dict[str, Any],
    structural_result: dict[str, Any],
    error: str = "",
) -> dict[str, Any]:
    if error:
        return invalid_judge_result(
            model=model,
            error=error,
            record=record,
            structural_result=structural_result,
        )

    assert response is not None
    content = str(response.get("content") or "")
    parsed, ok, parse_error = parse_json_output(content)
    errors = []
    if content.strip().startswith("```"):
        errors.append("Output chứa markdown/code fence.")
    if not isinstance(parsed, dict):
        errors.append(parse_error or "Không parse được JSON.")
    else:
        missing = sorted(REQUIRED_RESULT_FIELDS - set(parsed))
        if missing:
            errors.append("Thiếu field: " + ", ".join(missing))
        if parsed.get("overall_status") not in PLAN_EVAL_STATUSES:
            errors.append(f"overall_status không hợp lệ: {parsed.get('overall_status')}")
        if parsed.get("coverage_status") not in COVERAGE_STATUSES:
            errors.append(f"coverage_status không hợp lệ: {parsed.get('coverage_status')}")
        if parsed.get("confidence") not in CONFIDENCES:
            errors.append(f"confidence không hợp lệ: {parsed.get('confidence')}")
        if not isinstance(parsed.get("issues"), list):
            errors.append("issues phải là list.")
        else:
            for index, issue in enumerate(parsed.get("issues") or [], start=1):
                if not isinstance(issue, dict):
                    errors.append(f"issues[{index}] phải là object.")
                    continue
                missing_issue = sorted(REQUIRED_ISSUE_FIELDS - set(issue))
                if missing_issue:
                    errors.append(f"issues[{index}] thiếu field: {', '.join(missing_issue)}")

    if errors:
        return invalid_judge_result(
            model=model,
            error="; ".join(errors),
            record=record,
            structural_result=structural_result,
            raw_response=response.get("raw_response"),
            latency_seconds=response.get("latency_seconds", 0),
        )

    normalized = normalize_question_plan_eval_result(parsed, record=record, structural_result=structural_result)
    return {
        **normalized,
        "judge_model": model,
        "json_parse_ok": bool(ok),
        "judge_error": "",
        "raw_response": response.get("raw_response"),
        "latency_seconds": response.get("latency_seconds", 0),
    }


def call_question_plan_judge(
    client: LLMClient,
    model: str,
    record: dict[str, Any],
    structural_result: dict[str, Any],
) -> dict[str, Any]:
    try:
        response = client.chat_completion(
            model=model,
            messages=build_question_plan_judge_messages(record, structural_result),
            temperature=0,
        )
        return normalize_judge_response(
            model=model,
            response=response,
            record=record,
            structural_result=structural_result,
        )
    except Exception as exc:
        return normalize_judge_response(
            model=model,
            response=None,
            record=record,
            structural_result=structural_result,
            error=str(exc),
        )


def should_call_fallback(result: dict[str, Any], config: AppConfig) -> bool:
    if not config.use_fallback_judge:
        return False
    return bool(
        result.get("judge_error")
        or not result.get("json_parse_ok")
        or result.get("confidence") == "low"
        or result.get("overall_status") in {"needs_review", "bad"}
    )


def choose_result(primary: dict[str, Any], fallback: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    if fallback is None:
        return primary, "primary_only"
    if primary.get("json_parse_ok") and not fallback.get("json_parse_ok"):
        return primary, "primary_used_fallback_invalid"
    if fallback.get("json_parse_ok") and not primary.get("json_parse_ok"):
        return fallback, "fallback_used_primary_invalid"
    severity = {"ok": 0, "warning": 1, "needs_review": 2, "bad": 3, "structural_error": 4}
    if severity.get(fallback.get("overall_status"), 2) > severity.get(primary.get("overall_status"), 2):
        return fallback, "conservative_more_severe_fallback"
    return primary, "primary_retained"


def judge_question_plan(
    record: dict[str, Any],
    config: AppConfig,
    client: LLMClient,
    structural_result: dict[str, Any],
) -> dict[str, Any]:
    primary = call_question_plan_judge(client, config.primary_judge_model, record, structural_result)
    fallback = None
    if should_call_fallback(primary, config):
        fallback = call_question_plan_judge(client, config.fallback_judge_model, record, structural_result)
    chosen, policy = choose_result(primary, fallback)
    return {
        **chosen,
        "judge_model": primary.get("judge_model"),
        "fallback_called": fallback is not None,
        "fallback_model": fallback.get("judge_model") if fallback else None,
        "final_decision_policy": policy,
        "primary_result": primary,
        "fallback_result": fallback,
        "latency_seconds": (primary.get("latency_seconds", 0) or 0) + ((fallback or {}).get("latency_seconds", 0) or 0),
    }



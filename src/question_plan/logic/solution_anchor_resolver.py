"""LLM resolver for solution-anchored generated-question semantics."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..infra.config import (
    AppConfig,
    generated_question_fast_model,
    generated_question_gemma_runs,
    generated_question_reasoning_model,
)
from ..infra.debug import debug_llm_messages
from ..infra.llm_client import LLMClient
from ..shared.real_schema import content_to_text
from ..shared.utils import parse_json_output
from ..schemas.generated_question_contracts import (
    SolutionResolverOutput,
    contract_schema_text,
    validation_error_text,
)
from ..utils.json_pointer import JsonPointerError, get_by_json_pointer
from .generated_question_schema import (
    generated_question_id,
    get_answer_specs,
    get_interaction_id,
    get_interaction_type,
    get_options,
    location_to_json_pointer,
    make_issue,
    option_id,
    option_text,
)


KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"
RESOLVER_RULES_PATH = KNOWLEDGE_DIR / "solution_anchor_resolver_rules.md"
VALID_RESOLVER_STATUSES = {"resolved", "needs_manual_review"}
VALID_CATEGORIES = {"solution_anchor_consistency", "solution_quality", "hint_quality"}
VALID_INTENTS = {
    "align_fields_to_solution",
    "align_hint_to_solution",
    "clean_solution_reasoning",
    "needs_manual_review",
}
OPTION_INTERACTION_TYPES = {"single_choice", "multiple_choice", "choice_blank_fill", "matching"}


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def deep_content_text(value: Any) -> str:
    """Flatten content blocks structurally without interpreting their meaning."""

    if value is None:
        return ""
    direct = content_to_text(value)
    parts = [direct] if direct else []
    if isinstance(value, list):
        parts.extend(deep_content_text(item) for item in value)
    elif isinstance(value, dict):
        for key in ("solutionContent", "content", "stem", "instruction", "hints"):
            if key in value:
                parts.append(deep_content_text(value.get(key)))
    result: list[str] = []
    for part in parts:
        text = str(part or "").strip()
        if text and text not in result:
            result.append(text)
    return "\n".join(result)


def solution_text(generated_question: dict[str, Any]) -> str:
    return deep_content_text(generated_question.get("solutions"))


def compact_generated_question_for_solution_anchor(generated_question: dict[str, Any]) -> dict[str, Any]:
    contexts: list[dict[str, Any]] = []
    question_items = generated_question.get("questionItems")
    question_items = question_items if isinstance(question_items, list) else []
    for item_index, item in enumerate(question_items):
        if not isinstance(item, dict):
            continue
        interactions = item.get("interactions")
        interactions = interactions if isinstance(interactions, list) else []
        specs = get_answer_specs(item, generated_question)
        for interaction_index, interaction in enumerate(interactions):
            if not isinstance(interaction, dict):
                continue
            interaction_type = get_interaction_type(interaction)
            if interaction_type == "essay":
                continue
            interaction_id = get_interaction_id(interaction)
            spec_index = next(
                (
                    index
                    for index, spec in enumerate(specs)
                    if isinstance(spec, dict) and str(spec.get("interactionId") or "") == interaction_id
                ),
                None,
            )
            answer_spec = specs[spec_index] if spec_index is not None else None
            if answer_spec is None:
                continue
            context = {
                "question_item_index": item_index,
                "interaction_index": interaction_index,
                "interaction_id": interaction_id,
                "interaction_type": interaction_type,
                "stem": item.get("stem") or [],
                "answerSpec": answer_spec,
                "answerSpec_path": f"/questionItems/{item_index}/answerSpecs/{spec_index}",
            }
            if interaction_type in OPTION_INTERACTION_TYPES:
                context["options"] = [
                    {"id": option_id(option), "text": option_text(option)}
                    for option in get_options(interaction)
                    if isinstance(option, dict)
                ]
            if item.get("hints"):
                context["hints"] = item["hints"]
                context["hints_path"] = f"/questionItems/{item_index}/hints"
            contexts.append(context)
    return {
        "generated_question_id": generated_question_id(generated_question),
        "solution": solution_text(generated_question),
        "solution_path": "/solutions",
        "interaction_contexts": contexts,
    }


def build_solution_anchor_resolver_messages(
    generated_question: dict[str, Any],
    rules_text: str,
) -> list[dict[str, str]]:
    payload = compact_generated_question_for_solution_anchor(generated_question)
    output_schema_text = contract_schema_text(SolutionResolverOutput)
    return [
        {
            "role": "system",
            "content": (
                "Bạn là Solution Resolver cho generated question. Chỉ solution quyết định canonical answer. "
                "Không tự giải lại instruction/stem, không kiểm tra phép tính đúng sai và không dùng answerSpec "
                "để phủ định hoặc sửa solution. Bạn đọc tiếng Việt/LaTeX, map kết luận solution sang option, "
                "đối chiếu answerSpec và chỉ kiểm tra hint alignment khi solution đã resolved. Chỉ trả JSON hợp lệ."
            ),
        },
        {
            "role": "user",
            "content": (
                "Áp dụng đúng rules/schema. Nhiều số, phương trình hoặc option trung gian không phải nhiều đáp án cuối. "
                "Chỉ resolve các interaction_contexts được cung cấp; bỏ qua mọi phần solution thuộc câu essay hoặc context đã bị loại. "
                "single_choice chỉ needs_manual_review khi kết luận cuối thật sự có nhiều đáp án; multiple_choice được "
                "phép có nhiều đáp án. Nếu không có kết luận cụ thể, không đoán và không đề xuất sửa answerSpec/options/hints. "
                "Generic Quality Judge sẽ xử lý dài dòng/thử-sai/tự vấn khi final answer vẫn rõ, nên resolver không emit "
                "trùng lỗi trình bày đó. Mọi reason/suggestion phải là tiếng Việt có dấu.\n\n"
                f"RESOLVER RULES:\n{rules_text}\n\n"
                f"OUTPUT SCHEMA:\n{output_schema_text}\n\n"
                f"DYNAMIC PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def _issue(
    *, severity: str, category: str, location: str, reason: str, suggestion: str, repair_intent: str
) -> dict[str, Any]:
    return make_issue(
        severity=severity,
        category=category,
        location=location,
        reason=reason,
        suggestion=suggestion,
        repair_intent=repair_intent,
    )


def manual_review_result(
    reason: str,
    suggestion: str = "Review thủ công solution trước khi sửa các field khác.",
    *,
    contract_error: str = "",
) -> dict[str, Any]:
    result = {
        "resolver_status": "needs_manual_review",
        "final_answer": {
            "text": None,
            "matched_option_id": None,
            "correctOptionIds": [],
            "expected": None,
            "evidence_from_solution": "",
        },
        "answerSpec_matches_solution": False,
        "fields_to_fix": [],
        "issues": [
            _issue(
                severity="needs_review",
                category="solution_quality",
                location="/solutions",
                reason=reason,
                suggestion=suggestion,
                repair_intent="needs_manual_review",
            )
        ],
    }
    if contract_error:
        result["resolver_contract_error"] = contract_error
    return result


def resolver_contract_error(error: str) -> dict[str, Any]:
    return manual_review_result(
        f"Bộ phân giải lời giải trả kết quả không đúng cấu trúc: {error}",
        "Kiểm tra cấu trúc dữ liệu của bộ phân giải lời giải và chạy lại trước khi sửa các trường nghiệp vụ.",
        contract_error=error,
    )


def _valid_fix_path(generated_question: dict[str, Any], path: str) -> bool:
    tokens = [token for token in path.split("/") if token]
    if len(tokens) < 5 or tokens[0] != "questionItems" or tokens[2] != "answerSpecs":
        return False
    if not tokens[1].isdigit() or not tokens[3].isdigit() or tokens[4] != "expected":
        return False
    if len(tokens) > 5 and tokens[-1] not in {"correctOptionId", "correctOptionIds", "correctValue", "value"}:
        return False
    try:
        get_by_json_pointer(generated_question, path)
    except JsonPointerError:
        return False
    return True


def _known_option_ids(generated_question: dict[str, Any]) -> set[str]:
    payload = compact_generated_question_for_solution_anchor(generated_question)
    return {
        str(option.get("id") or "")
        for context in payload["interaction_contexts"]
        for option in context.get("options") or []
        if str(option.get("id") or "")
    }


def _normalize_issue(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    category = str(value.get("category") or "solution_anchor_consistency")
    if category not in VALID_CATEGORIES:
        return None
    intent = str(value.get("repair_intent") or "needs_manual_review")
    if intent not in VALID_INTENTS:
        intent = "needs_manual_review"
    if category == "solution_anchor_consistency":
        intent = "align_fields_to_solution"
    elif category == "hint_quality":
        intent = "align_hint_to_solution"
    severity = str(value.get("severity") or "needs_review")
    if severity not in {"warning", "needs_review", "bad"}:
        severity = "needs_review"
    return _issue(
        severity=severity,
        category=category,
        location=location_to_json_pointer(str(value.get("location") or "/solutions")),
        reason=str(value.get("reason") or "Resolver phát hiện vấn đề semantic cần review.").strip(),
        suggestion=str(value.get("suggestion") or "Review thủ công trường liên quan.").strip(),
        repair_intent=intent,
    )


def _dedupe_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for issue in issues:
        reason = " ".join(str(issue.get("reason") or "").lower().split())
        key = (
            str(issue.get("category") or ""),
            str(issue.get("location") or ""),
            str(issue.get("repair_intent") or ""),
            reason,
        )
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result


def normalize_solution_anchor_result(parsed: Any, generated_question: dict[str, Any]) -> dict[str, Any]:
    if isinstance(parsed, dict):
        parsed = dict(parsed)
        raw_answer = parsed.get("final_answer")
        if isinstance(raw_answer, dict):
            answer = dict(raw_answer)
        else:
            answer = {
                "text": raw_answer if isinstance(raw_answer, str) else None,
                "expected": raw_answer if raw_answer is not None and not isinstance(raw_answer, str) else None,
            }
        if answer.get("correctOptionIds") is None:
            answer["correctOptionIds"] = []
        if parsed.get("resolver_status") == "resolved" and not str(answer.get("evidence_from_solution") or "").strip():
            answer["evidence_from_solution"] = solution_text(generated_question)
        else:
            answer.setdefault("evidence_from_solution", "")
        parsed["final_answer"] = answer
    try:
        parsed = SolutionResolverOutput.model_validate(parsed).model_dump()
    except ValidationError as exc:
        return resolver_contract_error(validation_error_text(exc))
    status = parsed["resolver_status"]
    answer = parsed["final_answer"]

    final_answer = {
        "text": answer.get("text"),
        "matched_option_id": answer.get("matched_option_id"),
        "correctOptionIds": answer.get("correctOptionIds") if isinstance(answer.get("correctOptionIds"), list) else [],
        "expected": answer.get("expected"),
        "evidence_from_solution": str(answer.get("evidence_from_solution") or "").strip(),
    }
    if status == "resolved" and not any(
        final_answer[key] not in (None, "", [])
        for key in ("text", "matched_option_id", "correctOptionIds", "expected")
    ):
        return resolver_contract_error("resolver_status=resolved nhưng final_answer không có kết luận cụ thể")
    if status == "resolved" and not final_answer["evidence_from_solution"]:
        return resolver_contract_error("resolver_status=resolved nhưng final_answer.evidence_from_solution rỗng")

    matched_option_id = str(final_answer.get("matched_option_id") or "")
    if matched_option_id and matched_option_id not in _known_option_ids(generated_question):
        return resolver_contract_error("final_answer.matched_option_id không tồn tại trong options")

    issues = [issue for issue in (_normalize_issue(value) for value in parsed.get("issues") or []) if issue]
    if status == "needs_manual_review":
        manual_issue = next(
            (
                issue for issue in issues
                if issue.get("category") == "solution_quality" and issue.get("repair_intent") == "needs_manual_review"
            ),
            None,
        )
        if manual_issue is None:
            return manual_review_result("Solution không có kết luận cuối phù hợp với interaction type.")
        return {
            "resolver_status": "needs_manual_review",
            "final_answer": final_answer,
            "answerSpec_matches_solution": False,
            "fields_to_fix": [],
            "issues": [manual_issue],
        }

    fixes: list[dict[str, Any]] = []
    invalid_fix_count = 0
    for value in parsed["fields_to_fix"]:
        if not isinstance(value, dict) or "path" not in value or "value" not in value:
            invalid_fix_count += 1
            continue
        path = location_to_json_pointer(str(value.get("path") or ""))
        if _valid_fix_path(generated_question, path):
            fixes.append(
                {
                    "path": path,
                    "value": value.get("value"),
                    "reason": str(value.get("reason") or "").strip(),
                    "suggestion": str(value.get("suggestion") or "").strip(),
                }
            )
        else:
            invalid_fix_count += 1
    if invalid_fix_count:
        return resolver_contract_error("fields_to_fix chứa path/value không hợp lệ")
    answer_spec_matches = parsed["answerSpec_matches_solution"]
    if answer_spec_matches and fixes:
        return resolver_contract_error("answerSpec_matches_solution=true nhưng fields_to_fix không rỗng")
    if not answer_spec_matches and not fixes:
        return resolver_contract_error("answerSpec_matches_solution=false nhưng fields_to_fix rỗng")
    issues = [
        issue
        for issue in issues
        if (
            issue.get("category") == "solution_anchor_consistency"
            and issue.get("repair_intent") == "align_fields_to_solution"
        )
        or (
            issue.get("category") == "hint_quality"
            and issue.get("repair_intent") == "align_hint_to_solution"
        )
    ]
    return {
        "resolver_status": "resolved",
        "final_answer": final_answer,
        "answerSpec_matches_solution": answer_spec_matches,
        "fields_to_fix": fixes,
        "issues": _dedupe_issues(issues),
    }


def _call_solution_resolver(
    *,
    generated_question: dict[str, Any],
    messages: list[dict[str, str]],
    model: str,
    client: LLMClient,
    debug: bool,
) -> dict[str, Any]:
    try:
        debug_llm_messages(step="solution_anchor_resolver", model=model, messages=messages, debug=debug)
        response = client.chat_completion(model=model, messages=messages, temperature=0)
        parsed, ok, parse_error = parse_json_output(str(response.get("content") or ""))
        if not ok:
            return resolver_contract_error(parse_error or "Không parse được output của Solution Resolver.")
        return normalize_solution_anchor_result(parsed, generated_question)
    except Exception as exc:
        return resolver_contract_error(str(exc))


def _resolver_result_valid(result: dict[str, Any]) -> bool:
    return not result.get("resolver_contract_error")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _is_simple_answer(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    return isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value)


def _resolver_signature(result: dict[str, Any]) -> tuple[Any, ...]:
    answer = result.get("final_answer") or {}
    text_answer = " ".join(str(answer.get("text") or "").lower().split())
    expected_answer = answer.get("expected")
    option_answer = str(answer.get("matched_option_id") or "")
    option_answers = tuple(sorted(str(value) for value in answer.get("correctOptionIds") or []))
    fixes = tuple(
        sorted((str(fix.get("path") or ""), _freeze(fix.get("value"))) for fix in result.get("fields_to_fix") or [])
    )
    issues = tuple(
        sorted(
            (
                str(issue.get("location") or ""),
                str(issue.get("category") or ""),
                str(issue.get("severity") or ""),
                str(issue.get("repair_intent") or ""),
            )
            for issue in result.get("issues") or []
        )
    )
    return (
        str(result.get("resolver_status") or ""),
        option_answer,
        option_answers,
        (
            _freeze(expected_answer)
            if _is_simple_answer(expected_answer) or not (text_answer or option_answer or option_answers)
            else text_answer
        ),
        bool(result.get("answerSpec_matches_solution")),
        fixes,
        issues,
    )


def resolve_solution_anchor_consistency(
    generated_question: dict[str, Any],
    *,
    config: AppConfig | None = None,
    client: LLMClient | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    if not solution_text(generated_question):
        return manual_review_result("Generated question không có solution với kết luận cụ thể.")
    if config is None or client is None:
        return manual_review_result("Không có LLM client/config để chạy Solution Resolver.")

    messages = build_solution_anchor_resolver_messages(
        generated_question,
        load_text(RESOLVER_RULES_PATH),
    )
    fast_model = generated_question_fast_model(config)
    reasoning_model = generated_question_reasoning_model(config)
    runs = generated_question_gemma_runs(config)

    def call_fast() -> dict[str, Any]:
        return _call_solution_resolver(
            generated_question=generated_question,
            messages=messages,
            model=fast_model,
            client=client,
            debug=debug,
        )

    if runs == 2:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(call_fast) for _ in range(2)]
            gemma_results = [future.result() for future in futures]
    else:
        gemma_results = [call_fast()]

    result = gemma_results[0]
    agreement: bool | None = None
    fallback_reason: str | None = None
    if not _resolver_result_valid(result):
        fallback_reason = "gemma_run_1_invalid"
    elif runs == 2 and not _resolver_result_valid(gemma_results[1]):
        fallback_reason = "gemma_run_2_invalid"
    elif runs == 2:
        agreement = _resolver_signature(result) == _resolver_signature(gemma_results[1])
        fallback_reason = None if agreement else "gemma_disagreement"
    else:
        reason = " ".join(str(issue.get("reason") or "") for issue in result.get("issues") or []).lower()
        if result.get("resolver_status") == "needs_manual_review" and any(
            marker in reason for marker in ("mơ hồ", "quá phức tạp", "không chắc chắn")
        ):
            fallback_reason = "gemma_requires_fallback"

    fallback_called = bool(
        fallback_reason
        and getattr(config, "use_fallback_judge", True)
        and reasoning_model
        and reasoning_model != fast_model
    )
    if fallback_called:
        result = _call_solution_resolver(
            generated_question=generated_question,
            messages=messages,
            model=reasoning_model,
            client=client,
            debug=debug,
        )
    elif fallback_reason == "gemma_disagreement":
        result = resolver_contract_error("Hai lần đánh giá Gemma không thống nhất và không có Qwen fallback.")

    result["resolver_model"] = reasoning_model if fallback_called else fast_model
    result["resolver_attempt_count"] = runs + int(fallback_called)
    result["resolver_fallback_called"] = fallback_called
    result["resolver_gemma_run_count"] = runs
    result["resolver_gemma_agreement"] = agreement
    result["resolver_fallback_reason"] = fallback_reason
    return result

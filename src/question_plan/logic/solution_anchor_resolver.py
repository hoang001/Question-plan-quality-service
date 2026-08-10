"""LLM resolver for solution-anchored generated-question semantics."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..infra.config import (
    AppConfig,
    generated_question_resolver_model,
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
from ..utils.json_pointer import (
    JsonPointerError,
    get_by_json_pointer,
    parse_json_pointer,
    set_by_json_pointer,
)
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
    validate_generated_question_object,
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
RESOLVER_OUTPUT_INVARIANTS = """ĐIỀU KIỆN BẤT BIẾN CỦA ĐẦU RA:
- `final_answer` bắt buộc phải là một JSON object; không được là string, array hoặc null.
- `final_answer` phải có bốn field semantic: `text`, `matched_option_id`, `correctOptionIds`, `expected`; không được bỏ bất kỳ field semantic nào.
- `final_answer.text` bắt buộc phải tồn tại và phải là JSON string.
- `matched_option_id` là JSON string hoặc null; `correctOptionIds` luôn là JSON array.
- `evidence_from_solution` là metadata tùy chọn. Nếu có, code sẽ thử grounding và tự khôi phục đoạn nguyên văn; field này không quyết định resolver contract.
- Khi `resolver_status="resolved"`, `final_answer.text` phải là chuỗi không rỗng.
- Khi `resolver_status="needs_manual_review"`, `final_answer.text` phải là chuỗi rỗng `""`.
- `answer_spec_alignment` phải là `matched`, `equivalent` hoặc `mismatched`.
- `matched`/`equivalent` chỉ dùng khi `answerSpec_matches_solution=true`; `mismatched` chỉ dùng khi `answerSpec_matches_solution=false`.
- Dùng `equivalent` khi answerSpec khác cách biểu diễn nhưng tương đương semantic với kết luận solution.
- Không được bỏ field `final_answer` hoặc `final_answer.text`.
- Không dùng chuỗi `"None"`, `"null"` hoặc `"N/A"`.
- Chỉ trả một JSON object đúng schema; không viết markdown, code fence hoặc nội dung bên ngoài JSON."""


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


def _normalize_resolver_evidence(value: str) -> str:
    """Normalize formatting-only differences without semantic equivalence."""

    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = re.sub(r"\\(?:left|right)\b", "", normalized)
    normalized = normalized.replace(r"\(", "").replace(r"\)", "")
    normalized = normalized.replace(r"\[", "").replace(r"\]", "")
    normalized = normalized.replace("$", "")
    normalized = re.sub(r"\\(?:,|;|!| )", "", normalized)
    normalized = re.sub(r"\\(?:times|cdot)\b", "*", normalized)
    normalized = normalized.translate(str.maketrans({"×": "*", "·": "*", "÷": "/", "−": "-"}))
    return re.sub(r"\s+", "", normalized)


def _solution_evidence_excerpts(generated_question: dict[str, Any]) -> list[str]:
    """Return exact source excerpts, preferring paragraphs over whole blocks."""

    excerpts: list[str] = []
    solutions = generated_question.get("solutions")
    solutions = solutions if isinstance(solutions, list) else []
    for solution in solutions:
        if not isinstance(solution, dict):
            continue
        blocks = solution.get("solutionContent")
        blocks = blocks if isinstance(blocks, list) else []
        for block in blocks:
            text = deep_content_text(block).strip()
            if not text:
                continue
            for paragraph in re.split(r"(?:\r?\n\s*){2,}", text):
                paragraph = paragraph.strip()
                if paragraph and paragraph not in excerpts:
                    excerpts.append(paragraph)
            if text not in excerpts:
                excerpts.append(text)
    return excerpts


def _ground_resolver_evidence(
    evidence: str,
    generated_question: dict[str, Any],
) -> str | None:
    """Recover an exact source excerpt after harmless formatting changes."""

    source = solution_text(generated_question)
    if evidence in source:
        return evidence
    normalized_evidence = _normalize_resolver_evidence(evidence)
    if len(normalized_evidence) < 3:
        return None
    for excerpt in _solution_evidence_excerpts(generated_question):
        if normalized_evidence in _normalize_resolver_evidence(excerpt):
            return excerpt
    return None


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
                "Chỉ đối chiếu answerSpec với một kết luận explicit mà solution khẳng định là đáp án cuối cho đúng đại lượng/interaction được hỏi; "
                "không dùng số xuất hiện cuối như một heuristic. Kết luận explicit không bắt buộc có từ 'Vậy'. "
                "Chỉ resolve các interaction_contexts được cung cấp; bỏ qua mọi phần solution thuộc câu essay hoặc context đã bị loại. "
                "single_choice chỉ needs_manual_review khi kết luận cuối thật sự có nhiều đáp án; multiple_choice được "
                "phép có nhiều đáp án. Nếu không có kết luận cụ thể, không đoán và không đề xuất sửa answerSpec/options/hints. "
                "Generic Quality Judge sẽ xử lý dài dòng/thử-sai/tự vấn khi final answer vẫn rõ, nên resolver không emit "
                "trùng lỗi trình bày đó. Mọi reason/suggestion phải là tiếng Việt có dấu.\n\n"
                f"QUY TẮC RESOLVER:\n{rules_text}\n\n"
                f"LƯỢC ĐỒ ĐẦU RA:\n{output_schema_text}\n\n"
                f"DỮ LIỆU ĐỘNG:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
                f"{RESOLVER_OUTPUT_INVARIANTS}\n"
                "Chỉ trả một JSON object đúng schema: không thêm field, không bỏ field bắt buộc, "
                "dùng null thay vì chuỗi \"None\", enum đúng giá trị trong schema, không markdown. "
                "Không cần sao chép evidence_from_solution; code tự dựng metadata grounding khi có thể."
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
            "text": "",
            "matched_option_id": None,
            "correctOptionIds": [],
            "expected": None,
            "evidence_from_solution": "",
        },
        "answerSpec_matches_solution": False,
        "answer_spec_alignment": "mismatched",
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
    result = manual_review_result(
        f"Bộ phân giải lời giải trả kết quả không đúng cấu trúc: {error}",
        "Kiểm tra cấu trúc dữ liệu của bộ phân giải lời giải và chạy lại trước khi sửa các trường nghiệp vụ.",
        contract_error=error,
    )
    return result


def resolver_runtime_error(error: str) -> dict[str, Any]:
    result = manual_review_result(
        "Runtime error khi gọi Solution Resolver.",
        "Kiểm tra kết nối/model và chạy lại trước khi sửa các trường nghiệp vụ.",
    )
    result["issues"] = [
        _issue(
            severity="needs_review",
            category="runtime",
            location="/solutions",
            reason="Runtime error khi gọi Solution Resolver.",
            suggestion="Kiểm tra kết nối/model và chạy lại service.",
            repair_intent="needs_manual_review",
        )
    ]
    result["resolver_runtime_error"] = error
    return result


def _compatible_replacement_type(current: Any, replacement: Any) -> bool:
    if current is None:
        return replacement is None
    if isinstance(current, bool):
        return isinstance(replacement, bool)
    if isinstance(current, (int, float)):
        return isinstance(replacement, (int, float)) and not isinstance(
            replacement, bool
        )
    return isinstance(replacement, type(current))


def _valid_fix_path(
    generated_question: dict[str, Any],
    path: str,
    replacement: Any,
) -> bool:
    try:
        tokens = parse_json_pointer(path)
    except JsonPointerError:
        return False
    if len(tokens) < 5 or tokens[0] != "questionItems" or tokens[2] != "answerSpecs":
        return False
    if not tokens[1].isdigit() or not tokens[3].isdigit() or tokens[4] != "expected":
        return False
    allowed_roots = {
        f"{context['answerSpec_path']}/expected"
        for context in compact_generated_question_for_solution_anchor(generated_question)["interaction_contexts"]
    }
    if not any(path == root or path.startswith(root + "/") for root in allowed_roots):
        return False
    try:
        current = get_by_json_pointer(generated_question, path)
        if not _compatible_replacement_type(current, replacement):
            return False
        candidate = set_by_json_pointer(generated_question, path, replacement)
    except JsonPointerError:
        return False
    return bool(validate_generated_question_object(candidate).get("valid"))


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
    try:
        parsed = SolutionResolverOutput.model_validate(parsed).model_dump()
    except ValidationError as exc:
        return resolver_contract_error(validation_error_text(exc))
    status = parsed["resolver_status"]
    answer = parsed["final_answer"]
    answer_spec_matches = parsed["answerSpec_matches_solution"]
    answer_spec_alignment = parsed.get("answer_spec_alignment") or (
        "matched" if answer_spec_matches else "mismatched"
    )
    if answer_spec_alignment in {"matched", "equivalent"} and not answer_spec_matches:
        return resolver_contract_error(
            "answer_spec_alignment matched/equivalent yêu cầu answerSpec_matches_solution=true"
        )
    if answer_spec_alignment == "mismatched" and answer_spec_matches:
        return resolver_contract_error(
            "answer_spec_alignment=mismatched yêu cầu answerSpec_matches_solution=false"
        )

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
    if status == "resolved" and not final_answer["text"].strip():
        return resolver_contract_error(
            "resolver_status=resolved nhưng final_answer.text rỗng"
        )
    if status == "resolved":
        supplied_evidence = final_answer["evidence_from_solution"]
        final_answer["evidence_from_solution"] = (
            _ground_resolver_evidence(supplied_evidence, generated_question)
            if supplied_evidence
            else None
        ) or ""

    matched_option_id = str(final_answer.get("matched_option_id") or "")
    if matched_option_id and matched_option_id not in _known_option_ids(generated_question):
        return resolver_contract_error("final_answer.matched_option_id không tồn tại trong options")
    if any(
        str(option_id) not in _known_option_ids(generated_question)
        for option_id in final_answer["correctOptionIds"]
    ):
        return resolver_contract_error("final_answer.correctOptionIds chứa option không tồn tại")

    for value in parsed["issues"]:
        raw_location = str(value["location"])
        if location_to_json_pointer(raw_location) != raw_location:
            return resolver_contract_error("issue.location không phải JSON Pointer chuẩn")
        try:
            get_by_json_pointer(generated_question, raw_location)
        except JsonPointerError:
            return resolver_contract_error("issue.location không tồn tại trong generated question")
    issues = [issue for issue in (_normalize_issue(value) for value in parsed["issues"]) if issue]
    if status == "needs_manual_review":
        if final_answer["text"]:
            return resolver_contract_error(
                "resolver_status=needs_manual_review phải có final_answer.text rỗng"
            )
        manual_issue = next(
            (
                issue for issue in issues
                if issue.get("category") == "solution_quality" and issue.get("repair_intent") == "needs_manual_review"
            ),
            None,
        )
        if manual_issue is None:
            return resolver_contract_error(
                "resolver_status=needs_manual_review nhưng thiếu solution_quality issue tương ứng"
            )
        if parsed["answerSpec_matches_solution"] or parsed["fields_to_fix"]:
            return resolver_contract_error(
                "needs_manual_review phải có answerSpec_matches_solution=false và fields_to_fix rỗng"
            )
        if len(issues) != 1:
            return resolver_contract_error(
                "needs_manual_review phải có đúng một solution_quality issue"
            )
        return {
            "resolver_status": "needs_manual_review",
            "final_answer": final_answer,
            "answerSpec_matches_solution": False,
            "answer_spec_alignment": "mismatched",
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
        if _valid_fix_path(generated_question, path, value.get("value")):
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
    if answer_spec_matches and fixes:
        return resolver_contract_error("answerSpec_matches_solution=true nhưng fields_to_fix không rỗng")
    if not answer_spec_matches and not fixes:
        return resolver_contract_error("answerSpec_matches_solution=false nhưng fields_to_fix rỗng")
    align_issues = [
        issue
        for issue in issues
        if issue.get("category") == "solution_anchor_consistency"
        and issue.get("repair_intent") == "align_fields_to_solution"
    ]
    if answer_spec_matches and align_issues:
        return resolver_contract_error(
            "answerSpec_matches_solution=true nhưng vẫn có issue căn chỉnh answerSpec"
        )
    if not answer_spec_matches and not align_issues:
        return resolver_contract_error(
            "answerSpec_matches_solution=false nhưng thiếu issue căn chỉnh answerSpec"
        )
    if any(issue.get("category") == "solution_quality" for issue in issues):
        return resolver_contract_error(
            "resolver_status=resolved không được chứa solution_quality issue"
        )
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
        "answer_spec_alignment": answer_spec_alignment,
        "fields_to_fix": fixes,
        "issues": _dedupe_issues(issues),
    }


def _call_solution_resolver(
    *,
    generated_question: dict[str, Any],
    messages: list[dict[str, Any]],
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
        return resolver_runtime_error(str(exc))


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
    resolver_model = generated_question_resolver_model(config)
    result = _call_solution_resolver(
        generated_question=generated_question,
        messages=messages,
        model=resolver_model,
        client=client,
        debug=debug,
    )
    result["resolver_model"] = resolver_model
    result["resolver_attempt_count"] = 1
    result["resolver_fallback_called"] = False
    result["resolver_gemma_run_count"] = 1
    result["resolver_fallback_reason"] = None
    return result

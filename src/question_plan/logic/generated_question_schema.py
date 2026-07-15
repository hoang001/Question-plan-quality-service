"""Schema helper cho service đánh giá generated question object.

Module này chỉ kiểm tra nội bộ generated question: schema, interaction,
answerSpecs, options, hints, solutions và render safety. Nó không yêu cầu
question_plan, không dùng raw question/raw answer và không so sánh với source.
"""

from __future__ import annotations

import re
from typing import Any

from ..knowledge.interaction_type_knowledge import INTERACTION_TYPE_KNOWLEDGE
from ..shared.real_schema import content_to_text


SEVERITIES = {"warning", "needs_review", "bad"}
CATEGORIES = {
    "solution_anchor_consistency",
    "answer_internal_consistency",
    "interaction_schema",
    "choice_quality",
    "hint_quality",
    "solution_quality",
    "spelling_wording",
    "render_schema",
    "pedagogical_quality",
    "runtime",
}
STRICT_BLOCKING_SEVERITIES = {"warning", "needs_review", "bad"}
NON_STRICT_BLOCKING_SEVERITIES = {"bad"}
VALID_GENERATED_INTERACTION_TYPES = set(INTERACTION_TYPE_KNOWLEDGE) | {"essay"}
GENERATED_OBJECT_HINT_KEYS = {"instruction", "questionItems", "interactions", "solutions", "interactionTypes"}


def location_to_json_pointer(location: str) -> str:
    text = str(location or "").strip()
    if not text:
        return ""
    if text in {"$", "/$"}:
        return ""
    if text.startswith("/$/"):
        text = text[2:]
    elif text.startswith("$/"):
        text = text[1:]
    elif text.startswith("$."):
        text = text[2:]
    if text.startswith("/"):
        return text
    if text == "generatedQuestion":
        return ""
    if text.startswith("generatedQuestion."):
        text = text[len("generatedQuestion.") :]
    if text in {"input", "generatedQuestions"}:
        return text
    parts: list[str] = []
    for chunk in text.split("."):
        if not chunk:
            continue
        match = re.match(r"^([^\[]+)((?:\[\d+\])*)$", chunk)
        if not match:
            return str(location or "").strip()
        key, indexes = match.groups()
        parts.append(key)
        parts.extend(re.findall(r"\[(\d+)\]", indexes))
    return "/" + "/".join(parts) if parts else ""


def pointer_tokens(pointer: str) -> list[str]:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return []
    return pointer.strip("/").split("/") if pointer else []


def question_item_index_from_pointer(pointer: str) -> str | None:
    tokens = pointer_tokens(pointer)
    for index, token in enumerate(tokens[:-1]):
        if token == "questionItems" and tokens[index + 1].isdigit():
            return tokens[index + 1]
    return None


def solution_index_from_pointer(pointer: str) -> str | None:
    tokens = pointer_tokens(pointer)
    for index, token in enumerate(tokens[:-1]):
        if token == "solutions" and tokens[index + 1].isdigit():
            return tokens[index + 1]
    return None


def default_repair_intent(category: str, reason: str = "", location: str = "") -> str:
    del reason, location
    if category == "solution_anchor_consistency":
        return "align_fields_to_solution"
    if category == "answer_internal_consistency":
        return "needs_manual_review"
    if category == "spelling_wording":
        return "fix_spelling_wording"
    if category == "choice_quality":
        return "improve_distractors"
    if category == "hint_quality":
        return "reduce_hint_leakage"
    if category == "solution_quality":
        return "clean_solution_reasoning"
    if category in {"interaction_schema", "render_schema", "runtime"}:
        return "fix_schema"
    if category == "pedagogical_quality":
        return "improve_stem_clarity"
    return "needs_manual_review"


def default_context_paths(category: str, location: str) -> list[str]:
    pointer = location_to_json_pointer(location)
    paths: list[str] = []
    if pointer.startswith("/"):
        paths.append(pointer)

    item_index = question_item_index_from_pointer(pointer)
    if item_index is not None:
        item_prefix = f"/questionItems/{item_index}"
        if category in {"solution_anchor_consistency", "answer_internal_consistency", "choice_quality"}:
            paths.extend(
                [
                    "/instruction",
                    f"{item_prefix}/stem",
                    f"{item_prefix}/interactions",
                    f"{item_prefix}/answerSpecs",
                    "/solutions",
                ]
            )
        elif category == "hint_quality":
            paths.extend(
                [
                    f"{item_prefix}/stem",
                    f"{item_prefix}/hints",
                    f"{item_prefix}/interactions",
                    f"{item_prefix}/answerSpecs",
                    "/solutions",
                ]
            )
        elif category == "pedagogical_quality":
            paths.extend(
                [
                    "/instruction",
                    f"{item_prefix}/stem",
                    f"{item_prefix}/interactions",
                    f"{item_prefix}/answerSpecs",
                ]
            )
        elif category in {"interaction_schema", "render_schema"}:
            paths.extend([item_prefix, f"{item_prefix}/interactions", f"{item_prefix}/answerSpecs"])

    solution_index = solution_index_from_pointer(pointer)
    if category == "solution_quality":
        paths.append("/instruction")
        if item_index is not None:
            item_prefix = f"/questionItems/{item_index}"
            paths.extend([f"{item_prefix}/stem", f"{item_prefix}/interactions", f"{item_prefix}/answerSpecs"])
        else:
            paths.append("/questionItems")
        if solution_index is not None:
            paths.append(f"/solutions/{solution_index}")
        elif pointer.startswith("/solutions"):
            paths.append("/solutions")

    result: list[str] = []
    for path in paths:
        if path and path not in result:
            result.append(path)
    return result


def error_snippet_from_text(*values: Any, limit: int = 240) -> str:
    text = " ".join(str(value or "").strip() for value in values if str(value or "").strip())
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def make_issue(
    *,
    severity: str,
    category: str,
    location: str,
    reason: str,
    suggestion: str,
    error_snippet: str = "",
    required_context_paths: list[str] | None = None,
    repair_intent: str = "",
) -> dict[str, Any]:
    normalized_category = category if category in CATEGORIES else "runtime"
    normalized_location = location_to_json_pointer(location)
    normalized_paths = required_context_paths or default_context_paths(normalized_category, normalized_location)
    return {
        "severity": severity if severity in SEVERITIES else "needs_review",
        "category": normalized_category,
        "location": normalized_location,
        "reason": reason,
        "suggestion": suggestion,
        "error_snippet": error_snippet or error_snippet_from_text(reason, suggestion),
        "required_context_paths": normalized_paths,
        "repair_intent": repair_intent or default_repair_intent(normalized_category, reason, normalized_location),
    }


def runtime_issue(
    reason: str = "Runtime error hoặc LLM output không hợp lệ.",
    *,
    location: str = "generatedQuestion",
) -> dict[str, str]:
    return make_issue(
        severity="needs_review",
        category="runtime",
        location=location,
        reason=reason,
        suggestion="Cần kiểm tra thủ công hoặc chạy lại service.",
    )


def generated_question_id(generated_question: Any, index: int = 0) -> str:
    if isinstance(generated_question, dict):
        value = generated_question.get("id") or generated_question.get("_id")
        if str(value or "").strip():
            return str(value).strip()
    return f"item[{index}]"


def fail_closed_output(
    reason: str = "Không đánh giá được generated question do lỗi runtime hoặc LLM output không hợp lệ.",
    *,
    generated_question: Any | None = None,
    index: int = 0,
) -> dict[str, Any]:
    suggestion = "Cần kiểm tra thủ công hoặc chạy lại service."
    return {
        "id": generated_question_id(generated_question, index),
        "is_good": False,
        "failed_reason": [reason],
        "suggestions": [suggestion],
        "issues": [runtime_issue("Runtime error hoặc LLM output không hợp lệ.")],
    }


def text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        value = [value]
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def normalize_issue(value: Any, index: int) -> dict[str, Any]:
    issue = value if isinstance(value, dict) else {}
    severity = str(issue.get("severity") or "").strip()
    category = str(issue.get("category") or "").strip()
    if category == "answer_validity":
        category = "solution_anchor_consistency"
    raw_paths = issue.get("required_context_paths")
    required_context_paths = None
    if isinstance(raw_paths, list):
        required_context_paths = []
        for path in raw_paths:
            pointer = location_to_json_pointer(str(path or "").strip())
            if pointer.startswith("/") and pointer not in required_context_paths:
                required_context_paths.append(pointer)
    return make_issue(
        severity=severity if severity in SEVERITIES else "needs_review",
        category=category if category in CATEGORIES else "runtime",
        location=str(issue.get("location") or f"issues[{index}]").strip(),
        reason=str(issue.get("reason") or "Issue thiếu reason rõ ràng.").strip(),
        suggestion=str(issue.get("suggestion") or "Cần review thủ công.").strip(),
        error_snippet=str(issue.get("error_snippet") or "").strip(),
        required_context_paths=required_context_paths,
        repair_intent=str(issue.get("repair_intent") or "").strip(),
    )
    return make_issue(
        severity=severity if severity in SEVERITIES else "needs_review",
        category=category if category in CATEGORIES else "runtime",
        location=str(issue.get("location") or f"issues[{index}]").strip(),
        reason=str(issue.get("reason") or "Issue thiếu reason rõ ràng.").strip(),
        suggestion=str(issue.get("suggestion") or "Cần review thủ công.").strip(),
    )


def is_ignored_metadata_issue(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    category = str(value.get("category") or "").strip()
    if category == "difficulty_fit":
        return True
    text = " ".join(
        str(value.get(key) or "")
        for key in ("location", "reason", "suggestion")
    ).lower()
    return "difficulty" in text or "bloom" in text or "độ khó" in text


def is_ignored_metadata_text(value: Any) -> bool:
    text = str(value or "").lower()
    return "difficulty" in text or "bloom" in text or "độ khó" in text or "do kho" in text


def normalize_generated_question_result(
    result: Any,
    *,
    strict_mode: bool = True,
    generated_question: Any | None = None,
    index: int = 0,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        return fail_closed_output(generated_question=generated_question, index=index)
    if not isinstance(result.get("is_good"), bool):
        return fail_closed_output(
            "LLM output không có field is_good dạng boolean.",
            generated_question=generated_question,
            index=index,
        )

    raw_issues = result.get("issues") or []
    if not isinstance(raw_issues, list):
        return fail_closed_output(
            "LLM output có field issues không phải list.",
            generated_question=generated_question,
            index=index,
        )

    filtered_issues = [issue for issue in raw_issues if not is_ignored_metadata_issue(issue)]
    issues = [normalize_issue(issue, issue_index) for issue_index, issue in enumerate(filtered_issues)]
    blocking = STRICT_BLOCKING_SEVERITIES if strict_mode else NON_STRICT_BLOCKING_SEVERITIES
    is_good = not any(issue["severity"] in blocking for issue in issues)

    failed_reason = [text for text in text_list(result.get("failed_reason")) if not is_ignored_metadata_text(text)]
    suggestions = [text for text in text_list(result.get("suggestions")) if not is_ignored_metadata_text(text)]
    if not is_good:
        for issue in issues:
            if issue["severity"] in blocking and issue["reason"] not in failed_reason:
                failed_reason.append(issue["reason"])
            if issue["suggestion"] and issue["suggestion"] not in suggestions:
                suggestions.append(issue["suggestion"])

    return {
        "id": str(result.get("id") or generated_question_id(generated_question, index)),
        "is_good": is_good,
        "failed_reason": failed_reason,
        "suggestions": suggestions,
        "issues": issues,
    }


def merge_generated_question_results(
    *,
    schema_issues: list[dict[str, str]],
    llm_result: dict[str, Any] | None,
    strict_mode: bool,
    generated_question: Any | None = None,
    index: int = 0,
) -> dict[str, Any]:
    normalized_llm = normalize_generated_question_result(
        llm_result or {"is_good": True, "failed_reason": [], "suggestions": [], "issues": []},
        strict_mode=strict_mode,
        generated_question=generated_question,
        index=index,
    )
    issues = [*schema_issues, *normalized_llm["issues"]]
    return normalize_generated_question_result(
        {
            "id": generated_question_id(generated_question, index),
            "is_good": normalized_llm["is_good"],
            "failed_reason": normalized_llm["failed_reason"],
            "suggestions": normalized_llm["suggestions"],
            "issues": issues,
        },
        strict_mode=strict_mode,
        generated_question=generated_question,
        index=index,
    )


def is_generated_question_object(value: Any) -> bool:
    return isinstance(value, dict) and bool(GENERATED_OBJECT_HINT_KEYS & set(value))


def normalize_generated_question_input(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item if isinstance(item, dict) else {"__invalid_item__": item} for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("generatedQuestions"), list):
        return [
            item if isinstance(item, dict) else {"__invalid_item__": item}
            for item in payload.get("generatedQuestions", [])
        ]
    if is_generated_question_object(payload):
        return [payload]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("Input phải là generated question object, list object, hoặc wrapper có generatedQuestions.")


def should_return_aggregate(payload: Any, generated_questions: list[dict[str, Any]]) -> bool:
    return isinstance(payload, list) or len(generated_questions) != 1


def get_question_items(generated_question: Any) -> list[Any]:
    if not isinstance(generated_question, dict):
        return []
    items = generated_question.get("questionItems")
    return items if isinstance(items, list) else []


def get_interactions(item: Any) -> list[Any]:
    if not isinstance(item, dict):
        return []
    interactions = item.get("interactions")
    return interactions if isinstance(interactions, list) else []


def get_answer_specs(item: Any, generated_question: Any | None = None) -> list[Any]:
    specs: list[Any] = []
    if isinstance(item, dict) and isinstance(item.get("answerSpecs"), list):
        specs.extend(item["answerSpecs"])
    if isinstance(generated_question, dict) and isinstance(generated_question.get("answerSpecs"), list):
        specs.extend(generated_question["answerSpecs"])
    return specs


def find_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for item_key, item_value in value.items():
            if item_key == key:
                found.append(item_value)
            found.extend(find_values(item_value, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(find_values(item, key))
    return found


def option_id(option: Any) -> str:
    if isinstance(option, dict):
        return str(option.get("id") or option.get("value") or option.get("key") or "").strip()
    return str(option or "").strip()


def option_text(option: Any) -> str:
    if isinstance(option, dict):
        return content_to_text(option) or str(option.get("id") or "").strip()
    return str(option or "").strip()


def get_options(interaction: dict[str, Any]) -> list[Any]:
    config = interaction.get("config") if isinstance(interaction.get("config"), dict) else {}
    for container in (config, interaction):
        for key in ("options", "choices", "items"):
            value = container.get(key)
            if isinstance(value, list):
                return value
    return []


def get_interaction_id(interaction: Any) -> str:
    if not isinstance(interaction, dict):
        return ""
    return str(interaction.get("id") or interaction.get("_id") or interaction.get("interactionId") or "").strip()


def get_interaction_type(interaction: Any) -> str:
    if not isinstance(interaction, dict):
        return ""
    return str(interaction.get("type") or interaction.get("interactionType") or "").strip()


def expected_option_ids(answer_spec: Any, field_name: str) -> list[str]:
    values = find_values(answer_spec, field_name)
    ids: list[str] = []
    for value in values:
        if isinstance(value, list):
            ids.extend(str(item).strip() for item in value if str(item).strip())
        elif str(value).strip():
            ids.append(str(value).strip())
    return ids


def expected_boolean_values(answer_spec: Any) -> list[Any]:
    values: list[Any] = []
    for key in ("correct", "correctValue", "value", "answer"):
        values.extend(find_values(answer_spec, key))
    return values


def expected_coordinate_slot_ids(answer_spec: Any) -> list[str]:
    return [str(value).strip() for value in find_values(answer_spec, "slotId") if str(value).strip()]


def has_correct_value(answer_spec: Any) -> bool:
    return bool(find_values(answer_spec, "correctValue") or find_values(answer_spec, "acceptableValues"))


def has_essay_rubric_or_grading(generated_question: Any, item: Any, specs: list[Any]) -> bool:
    if specs:
        return True
    for value in (item, generated_question):
        if isinstance(value, dict):
            for key in ("rubric", "grading", "gradingConfig", "modelAnswer", "sampleAnswer"):
                if value.get(key):
                    return True
    return False


def has_solution(generated_question: Any) -> bool:
    return isinstance(generated_question, dict) and bool(generated_question.get("solutions"))


def interaction_location(g_index: int, item_index: int, interaction_index: int) -> str:
    return f"generatedQuestion.questionItems[{item_index}].interactions[{interaction_index}]"


def answer_spec_location(g_index: int, item_index: int, spec_index: int) -> str:
    return f"generatedQuestion.questionItems[{item_index}].answerSpecs[{spec_index}]"


def content_block_location(base_location: str, index: int) -> str:
    return f"{base_location}[{index}]"


def validate_content_blocks(
    value: Any,
    *,
    location: str,
    issues: list[dict[str, str]],
    require_text_for_text_block: bool = True,
) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        issues.append(
            make_issue(
                severity="bad",
                category="render_schema",
                location=location,
                reason="Content block container phải là list.",
                suggestion="Chuẩn hóa field content/stem/instruction/solutionContent thành list content block.",
            )
        )
        return
    for index, block in enumerate(value):
        block_location = content_block_location(location, index)
        if not isinstance(block, dict):
            issues.append(
                make_issue(
                    severity="bad",
                    category="render_schema",
                    location=block_location,
                    reason="Content block phải là object.",
                    suggestion="Chuẩn hóa content block thành object có id/type và nội dung phù hợp.",
                )
            )
            continue
        block_type = str(block.get("type") or "").strip()
        if not str(block.get("id") or "").strip():
            issues.append(
                make_issue(
                    severity="bad",
                    category="render_schema",
                    location=block_location,
                    reason="Content block thiếu id.",
                    suggestion="Bổ sung id ổn định cho content block.",
                )
            )
        if not block_type:
            issues.append(
                make_issue(
                    severity="bad",
                    category="render_schema",
                    location=block_location,
                    reason="Content block thiếu type.",
                    suggestion="Bổ sung type phù hợp cho content block.",
                )
            )
        if require_text_for_text_block and block_type == "text" and not str(block.get("text") or "").strip():
            issues.append(
                make_issue(
                    severity="bad",
                    category="render_schema",
                    location=f"{block_location}.text",
                    reason="Content block type text thiếu text.",
                    suggestion="Bổ sung text hiển thị cho content block.",
                )
            )
        if block_type in {"graph", "chart"} and not isinstance(block.get("display"), dict):
            issues.append(
                make_issue(
                    severity="needs_review",
                    category="render_schema",
                    location=f"{block_location}.display",
                    reason=f"Content block type {block_type} thiếu display object.",
                    suggestion="Review và bổ sung display/config cần thiết cho visual block.",
                )
            )


def count_generated_items(generated_questions: list[Any]) -> tuple[int, int]:
    item_count = 0
    interaction_count = 0
    for generated_question in generated_questions:
        for item in get_question_items(generated_question):
            item_count += 1
            interaction_count += len([interaction for interaction in get_interactions(item) if isinstance(interaction, dict)])
    return item_count, interaction_count


def validate_input_record(payload: Any) -> list[dict[str, str]]:
    try:
        generated_questions = normalize_generated_question_input(payload)
    except ValueError as exc:
        return [
            make_issue(
                severity="bad",
                category="render_schema",
                location="$",
                reason=str(exc),
                suggestion="Gửi generated question object, list generated question object, hoặc wrapper có generatedQuestions.",
            )
        ]
    if not generated_questions:
        return [
            make_issue(
                severity="bad",
                category="render_schema",
                location="generatedQuestions",
                reason="Không có generated question nào để đánh giá.",
                suggestion="Gửi ít nhất một generated question object.",
            )
        ]
    return []


def validate_generated_question_object(generated_question: Any, index: int = 0) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    if not isinstance(generated_question, dict) or "__invalid_item__" in generated_question:
        issues.append(
            make_issue(
                severity="bad",
                category="render_schema",
                location=f"item[{index}]",
                reason="Mỗi generated question phải là object.",
                suggestion="Chuẩn hóa input thành list object.",
            )
        )
        return {
            "valid": False,
            "issues": issues,
            "summary": {
                "generated_question_count": 1,
                "generated_item_count": 0,
                "generated_interaction_count": 0,
                "schema_issue_count": len(issues),
                "schema_bad_count": 1,
            },
        }

    instruction = generated_question.get("instruction")
    if instruction is None:
        issues.append(
            make_issue(
                severity="needs_review",
                category="pedagogical_quality",
                location="generatedQuestion.instruction",
                reason="Generated question thiếu instruction.",
                suggestion="Review xem questionItems/stem đã đủ ngữ cảnh để học sinh làm bài chưa.",
            )
        )
    else:
        validate_content_blocks(instruction, location="generatedQuestion.instruction", issues=issues)

    solutions = generated_question.get("solutions")
    if solutions is not None:
        if not isinstance(solutions, list):
            issues.append(
                make_issue(
                    severity="bad",
                    category="solution_quality",
                    location="generatedQuestion.solutions",
                    reason="solutions phải là list.",
                    suggestion="Chuẩn hóa solutions thành danh sách solution object.",
                )
            )
        else:
            for solution_index, solution in enumerate(solutions):
                solution_location = f"generatedQuestion.solutions[{solution_index}]"
                if not isinstance(solution, dict):
                    issues.append(
                        make_issue(
                            severity="bad",
                            category="solution_quality",
                            location=solution_location,
                            reason="Mỗi solution phải là object.",
                            suggestion="Chuẩn hóa solution thành object có solverName và solutionContent.",
                        )
                    )
                    continue
                if "solutionContent" in solution:
                    validate_content_blocks(
                        solution.get("solutionContent"),
                        location=f"{solution_location}.solutionContent",
                        issues=issues,
                        require_text_for_text_block=False,
                    )

    question_items = get_question_items(generated_question)
    if not question_items:
        issues.append(
            make_issue(
                severity="bad",
                category="render_schema",
                location="generatedQuestion.questionItems",
                reason="Generated question thiếu questionItems hoặc questionItems rỗng.",
                suggestion="Bổ sung questionItems cho generated question.",
            )
        )
        if instruction is None:
            issues.append(
                make_issue(
                    severity="bad",
                    category="render_schema",
                    location="generatedQuestion",
                    reason="Generated question thiếu cả instruction và questionItems.",
                    suggestion="Bổ sung instruction/questionItems trước khi đánh giá.",
                )
            )

    generated_question_interaction_ids: set[str] = set()
    generated_interaction_count = 0
    for item_index, item in enumerate(question_items):
        if not isinstance(item, dict):
            issues.append(
                make_issue(
                    severity="bad",
                    category="render_schema",
                    location=f"generatedQuestion.questionItems[{item_index}]",
                    reason="questionItem phải là object.",
                    suggestion="Chuẩn hóa questionItems thành list object.",
                )
            )
            continue

        if not str(item.get("id") or "").strip():
            issues.append(
                make_issue(
                    severity="bad",
                    category="render_schema",
                    location=f"generatedQuestion.questionItems[{item_index}].id",
                    reason="questionItem thiếu id.",
                    suggestion="Bổ sung id ổn định cho questionItem.",
                )
            )

        stem = item.get("stem")
        if stem is None:
            issues.append(
                make_issue(
                    severity="bad",
                    category="render_schema",
                    location=f"generatedQuestion.questionItems[{item_index}].stem",
                    reason="questionItem thiếu stem.",
                    suggestion="Bổ sung stem để học sinh thấy yêu cầu cần trả lời.",
                )
            )
        else:
            validate_content_blocks(
                stem,
                location=f"generatedQuestion.questionItems[{item_index}].stem",
                issues=issues,
            )

        hints = item.get("hints")
        if hints is not None:
            if not isinstance(hints, list):
                issues.append(
                    make_issue(
                        severity="bad",
                        category="hint_quality",
                        location=f"generatedQuestion.questionItems[{item_index}].hints",
                        reason="hints phải là list.",
                        suggestion="Chuẩn hóa hints thành danh sách hint object.",
                    )
                )
            else:
                for hint_index, hint in enumerate(hints):
                    hint_location = f"generatedQuestion.questionItems[{item_index}].hints[{hint_index}]"
                    if not isinstance(hint, dict):
                        issues.append(
                            make_issue(
                                severity="bad",
                                category="hint_quality",
                                location=hint_location,
                                reason="Mỗi hint phải là object.",
                                suggestion="Chuẩn hóa hint thành object có name/content.",
                            )
                        )
                        continue
                    if "content" in hint:
                        validate_content_blocks(hint.get("content"), location=f"{hint_location}.content", issues=issues)

        interactions = get_interactions(item)
        if not interactions:
            issues.append(
                make_issue(
                    severity="bad",
                    category="interaction_schema",
                    location=f"generatedQuestion.questionItems[{item_index}].interactions",
                    reason="questionItem thiếu interactions hoặc interactions rỗng.",
                    suggestion="Bổ sung ít nhất một interaction cho questionItem.",
                )
            )
            continue

        interaction_ids: list[str] = []
        interaction_types: dict[str, str] = {}
        interaction_by_id: dict[str, dict[str, Any]] = {}
        for interaction_index, interaction in enumerate(interactions):
            generated_interaction_count += 1
            location = interaction_location(0, item_index, interaction_index)
            if not isinstance(interaction, dict):
                issues.append(
                    make_issue(
                        severity="bad",
                        category="interaction_schema",
                        location=location,
                        reason="interaction phải là object.",
                        suggestion="Chuẩn hóa interactions thành list object.",
                    )
                )
                continue

            interaction_id = get_interaction_id(interaction)
            interaction_type = get_interaction_type(interaction)
            if not interaction_id:
                issues.append(
                    make_issue(
                        severity="bad",
                        category="interaction_schema",
                        location=location,
                        reason="Interaction thiếu id.",
                        suggestion="Bổ sung id ổn định cho interaction.",
                    )
                )
            elif interaction_id in generated_question_interaction_ids:
                issues.append(
                    make_issue(
                        severity="bad",
                        category="render_schema",
                        location=location,
                        reason=f"Interaction id `{interaction_id}` bị trùng trong cùng generated question.",
                        suggestion="Đổi id interaction để unique trong cùng generated question.",
                    )
                )
            else:
                interaction_ids.append(interaction_id)
                generated_question_interaction_ids.add(interaction_id)
                interaction_by_id[interaction_id] = interaction

            if not interaction_type:
                issues.append(
                    make_issue(
                        severity="bad",
                        category="interaction_schema",
                        location=location,
                        reason="Interaction thiếu type.",
                        suggestion="Bổ sung type hợp lệ cho interaction.",
                    )
                )
            elif interaction_type not in VALID_GENERATED_INTERACTION_TYPES:
                issues.append(
                    make_issue(
                        severity="bad",
                        category="interaction_schema",
                        location=location,
                        reason=f"Interaction type `{interaction_type}` không nằm trong danh sách hợp lệ.",
                        suggestion="Đổi interaction type sang type được hệ thống hỗ trợ hoặc cập nhật allowed types nếu mentor xác nhận type mới.",
                    )
                )

            if "config" not in interaction or not isinstance(interaction.get("config"), dict):
                issues.append(
                    make_issue(
                        severity="bad",
                        category="interaction_schema",
                        location=f"{location}.config",
                        reason="Interaction thiếu config object.",
                        suggestion="Bổ sung config object phù hợp với interaction type.",
                    )
                )
            if "display" not in interaction or not isinstance(interaction.get("display"), dict):
                issues.append(
                    make_issue(
                        severity="bad",
                        category="interaction_schema",
                        location=f"{location}.display",
                        reason="Interaction thiếu display object.",
                        suggestion="Bổ sung display object để UI render ổn định.",
                    )
                )

            if interaction_id:
                interaction_types[interaction_id] = interaction_type

            if interaction_type in {"single_choice", "multiple_choice"}:
                options = get_options(interaction)
                option_ids = [option_id(option) for option in options if option_id(option)]
                duplicate_option_ids = sorted({item_id for item_id in option_ids if option_ids.count(item_id) > 1})
                min_options = 2
                if not options:
                    issues.append(
                        make_issue(
                            severity="bad",
                            category="interaction_schema",
                            location=f"{location}.config.options",
                            reason=f"{interaction_type} thiếu options.",
                            suggestion=f"Bổ sung danh sách options cho {interaction_type}.",
                        )
                    )
                elif len(options) < min_options:
                    issues.append(
                        make_issue(
                            severity="bad",
                            category="interaction_schema",
                            location=f"{location}.config.options",
                            reason=f"{interaction_type} có ít hơn {min_options} options.",
                            suggestion=f"Bổ sung đủ options cho {interaction_type}.",
                        )
                    )
                elif duplicate_option_ids:
                    issues.append(
                        make_issue(
                            severity="bad",
                            category="interaction_schema",
                            location=f"{location}.config.options",
                            reason=f"{interaction_type} có option id bị trùng: {', '.join(duplicate_option_ids)}.",
                            suggestion="Đổi option id để unique trong cùng interaction.",
                        )
                    )
                elif len(option_ids) < len(options):
                    issues.append(
                        make_issue(
                            severity="bad",
                            category="interaction_schema",
                            location=f"{location}.config.options",
                            reason=f"Một số option của {interaction_type} thiếu id.",
                            suggestion="Bổ sung id ổn định cho mọi option.",
                        )
                    )

                normalized_texts = [option_text(option).casefold() for option in options if option_text(option)]
                if normalized_texts and len(normalized_texts) != len(set(normalized_texts)):
                    issues.append(
                        make_issue(
                            severity="needs_review",
                            category="choice_quality",
                            location=f"{location}.config.options",
                            reason=f"{interaction_type} có option trùng nội dung hiển thị.",
                            suggestion="Review và đổi các option trùng nội dung.",
                        )
                    )

        answer_specs = get_answer_specs(item, generated_question)
        answer_spec_interaction_ids = {
            str(spec.get("interactionId")).strip()
            for spec in answer_specs
            if isinstance(spec, dict) and str(spec.get("interactionId") or "").strip()
        }
        for interaction_id in interaction_ids:
            interaction_type = interaction_types.get(interaction_id, "")
            if interaction_id in answer_spec_interaction_ids:
                continue
            if interaction_type == "essay" and (
                has_essay_rubric_or_grading(generated_question, item, answer_specs)
                or has_solution(generated_question)
            ):
                continue
            severity = "needs_review" if interaction_type in {"essay", "short_answer"} else "bad"
            issues.append(
                make_issue(
                    severity=severity,
                    category="answer_internal_consistency",
                    location=f"generatedQuestion.questionItems[{item_index}].answerSpecs",
                    reason=f"Interaction `{interaction_id}` không có answerSpec tương ứng.",
                    suggestion="Bổ sung answerSpec nếu cần chấm tự động, hoặc bổ sung rubric/grading nếu là câu tự luận.",
                )
            )

        for spec_index, spec in enumerate(answer_specs):
            location = answer_spec_location(0, item_index, spec_index)
            if not isinstance(spec, dict):
                issues.append(
                    make_issue(
                        severity="bad",
                        category="answer_internal_consistency",
                        location=location,
                        reason="answerSpec phải là object.",
                        suggestion="Chuẩn hóa answerSpecs thành list object.",
                    )
                )
                continue

            interaction_id = str(spec.get("interactionId") or "").strip()
            if interaction_id and interaction_id not in interaction_ids:
                issues.append(
                    make_issue(
                        severity="bad",
                        category="answer_internal_consistency",
                        location=f"{location}.interactionId",
                        reason=f"answerSpec trỏ tới interactionId `{interaction_id}` không tồn tại trong questionItem.",
                        suggestion="Sửa interactionId hoặc bổ sung interaction tương ứng.",
                    )
                )
                continue
            if not interaction_id:
                issues.append(
                    make_issue(
                        severity="bad",
                        category="answer_internal_consistency",
                        location=f"{location}.interactionId",
                        reason="answerSpec thiếu interactionId.",
                        suggestion="Bổ sung interactionId để map answerSpec với interaction cần chấm.",
                    )
                )
                continue

            interaction_type = interaction_types.get(interaction_id, "")
            spec_type = str(spec.get("type") or "").strip()
            if spec_type and interaction_type and spec_type != interaction_type:
                issues.append(
                    make_issue(
                        severity="bad",
                        category="answer_internal_consistency",
                        location=f"{location}.type",
                        reason=f"answerSpec type `{spec_type}` không khớp interaction type `{interaction_type}`.",
                        suggestion="Đổi answerSpec type để tương thích với interaction type.",
                    )
                )

            interaction = interaction_by_id.get(interaction_id, {})
            if interaction_type == "single_choice":
                option_ids = [option_id(option) for option in get_options(interaction) if option_id(option)]
                correct_option_ids = expected_option_ids(spec, "correctOptionId")
                if not correct_option_ids:
                    issues.append(
                        make_issue(
                            severity="bad",
                            category="answer_internal_consistency",
                            location=location,
                            reason="single_choice answerSpec thiếu correctOptionId.",
                            suggestion="Bổ sung expected.correctOptionId cho single_choice.",
                        )
                    )
                elif len(set(correct_option_ids)) > 1:
                    issues.append(
                        make_issue(
                            severity="bad",
                            category="answer_internal_consistency",
                            location=location,
                            reason="single_choice có nhiều correctOptionId.",
                            suggestion="single_choice chỉ được có đúng một đáp án đúng.",
                        )
                    )
                for correct_option_id in correct_option_ids:
                    if option_ids and correct_option_id not in option_ids:
                        issues.append(
                            make_issue(
                                severity="bad",
                                category="answer_internal_consistency",
                                location=location,
                                reason=f"correctOptionId `{correct_option_id}` không tồn tại trong options.",
                                suggestion="Sửa correctOptionId hoặc option id để khớp nhau.",
                            )
                        )

            if interaction_type == "multiple_choice":
                option_ids = [option_id(option) for option in get_options(interaction) if option_id(option)]
                correct_option_ids = expected_option_ids(spec, "correctOptionIds")
                if not correct_option_ids:
                    issues.append(
                        make_issue(
                            severity="bad",
                            category="answer_internal_consistency",
                            location=location,
                            reason="multiple_choice answerSpec thiếu correctOptionIds.",
                            suggestion="Bổ sung expected.correctOptionIds cho multiple_choice.",
                        )
                    )
                elif len(set(correct_option_ids)) < 2:
                    issues.append(
                        make_issue(
                            severity="bad",
                            category="answer_internal_consistency",
                            location=location,
                            reason="multiple_choice có ít hơn 2 đáp án đúng.",
                            suggestion="Nếu chỉ có 1 đáp án đúng, đổi sang single_choice hoặc bổ sung các đáp án đúng thật sự.",
                        )
                    )
                for correct_option_id in correct_option_ids:
                    if option_ids and correct_option_id not in option_ids:
                        issues.append(
                            make_issue(
                                severity="bad",
                                category="answer_internal_consistency",
                                location=location,
                                reason=f"correctOptionIds chứa `{correct_option_id}` không tồn tại trong options.",
                                suggestion="Sửa correctOptionIds hoặc option id để khớp nhau.",
                            )
                        )

            if interaction_type == "true_false":
                boolean_values = expected_boolean_values(spec)
                if not any(isinstance(value, bool) for value in boolean_values):
                    issues.append(
                        make_issue(
                            severity="bad",
                            category="answer_internal_consistency",
                            location=location,
                            reason="true_false answerSpec không xác định rõ expected true/false.",
                            suggestion="Bổ sung expected boolean rõ ràng cho true_false.",
                        )
                    )

            if interaction_type == "short_answer" and not has_correct_value(spec):
                issues.append(
                    make_issue(
                        severity="needs_review",
                        category="answer_internal_consistency",
                        location=location,
                        reason="short_answer answerSpec thiếu correctValue hoặc acceptableValues để chấm tự động.",
                        suggestion="Bổ sung expected.value.correctValue hoặc acceptableValues nếu cần chấm tự động.",
                    )
                )

            if interaction_type == "coordinate_input":
                config = interaction.get("config") if isinstance(interaction.get("config"), dict) else {}
                dimensions = config.get("dimensions")
                if dimensions is not None and (not isinstance(dimensions, int) or dimensions < 1):
                    issues.append(
                        make_issue(
                            severity="bad",
                            category="interaction_schema",
                            location=f"{location}.config.dimensions",
                            reason="coordinate_input config.dimensions không hợp lệ.",
                            suggestion="Đặt dimensions là số nguyên dương phù hợp với số slot tọa độ.",
                        )
                    )
                slots = config.get("slots") if isinstance(config.get("slots"), list) else []
                slot_ids = {
                    str(slot.get("id")).strip()
                    for slot in slots
                    if isinstance(slot, dict) and str(slot.get("id") or "").strip()
                }
                expected_slot_ids = expected_coordinate_slot_ids(spec)
                for slot_id in expected_slot_ids:
                    if slot_ids and slot_id not in slot_ids:
                        issues.append(
                            make_issue(
                                severity="bad",
                                category="answer_internal_consistency",
                                location=location,
                                reason=f"expected.coordinates.slotId `{slot_id}` không tồn tại trong config.slots.",
                                suggestion="Sửa slotId trong expected hoặc bổ sung slot tương ứng trong config.",
                            )
                        )

        for interaction_index, interaction in enumerate(interactions):
            interaction_type = get_interaction_type(interaction)
            if interaction_type != "essay":
                continue
            location = interaction_location(0, item_index, interaction_index)
            stem_text = content_to_text(item.get("stem"))
            specs_for_item = get_answer_specs(item, generated_question)
            if len(stem_text.strip()) < 20:
                issues.append(
                    make_issue(
                        severity="needs_review",
                        category="pedagogical_quality",
                        location=f"generatedQuestion.questionItems[{item_index}].stem",
                        reason="essay có stem quá ngắn hoặc chưa rõ hướng trả lời.",
                        suggestion="Bổ sung yêu cầu trả lời/rubric để học sinh biết cần viết gì.",
                    )
                )
            if not has_essay_rubric_or_grading(generated_question, item, specs_for_item) and not has_solution(generated_question):
                issues.append(
                    make_issue(
                        severity="needs_review",
                        category="answer_internal_consistency",
                        location=location,
                        reason="essay thiếu rubric/grading/answerSpec/solution để đối chiếu chất lượng câu trả lời.",
                        suggestion="Bổ sung rubric, model answer, grading hoặc solution nếu hệ thống cần chấm/review tự động.",
                    )
                )

    return {
        "valid": not any(issue["severity"] == "bad" for issue in issues),
        "issues": issues,
        "summary": {
            "generated_question_count": 1,
            "generated_item_count": len(question_items),
            "generated_interaction_count": generated_interaction_count,
            "schema_issue_count": len(issues),
            "schema_bad_count": len([issue for issue in issues if issue["severity"] == "bad"]),
        },
    }


def validate_generated_question_schema(payload: Any) -> dict[str, Any]:
    try:
        generated_questions = normalize_generated_question_input(payload)
    except ValueError as exc:
        issues = validate_input_record(payload)
        return {
            "valid": False,
            "issues": issues,
            "summary": {
                "generated_question_count": 0,
                "generated_item_count": 0,
                "generated_interaction_count": 0,
                "schema_issue_count": len(issues),
                "schema_bad_count": len([issue for issue in issues if issue["severity"] == "bad"]),
            },
            "error": str(exc),
        }

    all_issues: list[dict[str, str]] = []
    item_total = 0
    interaction_total = 0
    for index, generated_question in enumerate(generated_questions):
        result = validate_generated_question_object(generated_question, index)
        all_issues.extend(result.get("issues") or [])
        summary = result.get("summary") or {}
        item_total += int(summary.get("generated_item_count") or 0)
        interaction_total += int(summary.get("generated_interaction_count") or 0)

    return {
        "valid": not any(issue["severity"] == "bad" for issue in all_issues),
        "issues": all_issues,
        "summary": {
            "generated_question_count": len(generated_questions),
            "generated_item_count": item_total,
            "generated_interaction_count": interaction_total,
            "schema_issue_count": len(all_issues),
            "schema_bad_count": len([issue for issue in all_issues if issue["severity"] == "bad"]),
        },
    }


def result_status(result: dict[str, Any]) -> str:
    severities = {issue.get("severity") for issue in result.get("issues") or [] if isinstance(issue, dict)}
    if "bad" in severities:
        return "bad"
    if "needs_review" in severities:
        return "needs_review"
    if "warning" in severities:
        return "warning"
    return "good"


def aggregate_generated_question_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "total": len(results),
        "good": 0,
        "bad": 0,
        "needs_review": 0,
        "warning": 0,
        "repaired": 0,
        "repair_failed": 0,
    }
    failed_reason: list[str] = []
    suggestions: list[str] = []
    for result in results:
        status = result_status(result)
        summary[status] += 1
        if result.get("repair_status") == "repaired" and result.get("new_generated_question") is not None:
            summary["repaired"] += 1
        elif result.get("repair_status") == "failed":
            summary["repair_failed"] += 1
        for reason in result.get("failed_reason") or []:
            if reason and reason not in failed_reason:
                failed_reason.append(reason)
        for suggestion in result.get("suggestions") or []:
            if suggestion and suggestion not in suggestions:
                suggestions.append(suggestion)
    return {
        "is_good": all(result.get("is_good") for result in results),
        "failed_reason": failed_reason,
        "suggestions": suggestions,
        "summary": summary,
        "results": results,
    }


def count_generated_payload(payload: Any) -> dict[str, int]:
    try:
        generated_questions = normalize_generated_question_input(payload)
    except ValueError:
        generated_questions = []
    item_count, interaction_count = count_generated_items(generated_questions)
    schema_result = validate_generated_question_schema(generated_questions) if generated_questions else {"issues": []}
    return {
        "generated_question_count": len(generated_questions),
        "question_item_count": item_count,
        "interaction_count": interaction_count,
        "schema_issue_count": len(schema_result.get("issues") or []),
    }

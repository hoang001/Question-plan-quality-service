import json
from pathlib import Path
from types import SimpleNamespace

from cli import (
    extract_repaired_generated_questions,
    format_generated_question_markdown,
    write_generated_question_cli_outputs,
)
from src.generated_question.real_interaction_classifier import build_real_classifier_messages
from src.question_plan.flows.generated_question_service import evaluate_generated_questions
from src.question_plan.logic.generated_question_judge import build_generated_question_judge_messages
from src.question_plan.logic.generated_question_schema import (
    normalize_generated_question_input,
    normalize_generated_question_result,
)
from src.question_plan.logic.generated_question_repair import (
    build_generated_question_scoped_repair_messages,
    build_scoped_repair_context,
    normalize_scoped_repair_result,
)
from src.question_plan.logic.generated_question_spelling import protect_math_segments, restore_math_segments
from src.question_plan.utils.json_pointer import apply_json_patch, get_by_json_pointer


ROOT_DIR = Path(__file__).resolve().parents[1]


def fake_config():
    return SimpleNamespace(primary_judge_model="fake-primary", fallback_judge_model="", use_fallback_judge=False)


class FakeClient:
    def __init__(self, content: str):
        self.content = content
        self.messages = []

    def chat_completion(self, **kwargs):
        self.messages.append(kwargs.get("messages") or [])
        return {"content": self.content, "latency_seconds": 0}


class SequenceFakeClient:
    def __init__(self, contents: list[str]):
        self.contents = list(contents)
        self.messages = []

    def chat_completion(self, **kwargs):
        self.messages.append(kwargs.get("messages") or [])
        content = self.contents.pop(0) if self.contents else ok_llm_payload()
        return {"content": content, "latency_seconds": 0}


def ok_llm_payload() -> str:
    return json.dumps({"is_good": True, "failed_reason": [], "suggestions": [], "issues": []}, ensure_ascii=False)


def repair_llm_payload(new_generated_question: dict | None, status: str = "repaired") -> str:
    return json.dumps(
        {
            "repair_status": status,
            "failed_reason": [] if status == "repaired" else ["Không sửa tự động an toàn."],
            "suggestions": ["Đã sửa generated question."] if status == "repaired" else ["Review thủ công."],
            "new_generated_question": new_generated_question,
        },
        ensure_ascii=False,
    )


def scoped_repair_payload(patches: list[dict], status: str = "repaired") -> str:
    return json.dumps(
        {
            "repair_status": status,
            "failed_reason": [] if status == "repaired" else ["Can full repair."],
            "suggestions": ["Da sua bang patch."] if status == "repaired" else ["Dung full repair."],
            "patches": patches,
        },
        ensure_ascii=False,
    )


def solution_anchor_payload(correct_option_id: str = "B") -> str:
    return json.dumps(
        {
            "resolver_status": "resolved",
            "confidence": "high",
            "solution_derived_answer": {
                "interaction_type": "single_choice",
                "answer": "$x = 2$",
                "correctOptionId": correct_option_id,
                "correctOptionIds": [],
                "expected": {"correctOptionId": correct_option_id},
                "evidence_from_solution": "Solution kết luận x = 2.",
            },
            "field_comparison": {
                "answerSpecs_match_solution": False,
                "options_match_solution": True,
                "hints_match_solution": True,
            },
            "fields_to_fix": [
                {
                    "path": "/questionItems/0/answerSpecs/0/expected/correctOptionId",
                    "reason": "answerSpec hiện tại không khớp solution.",
                    "suggestion": f"Đổi correctOptionId thành {correct_option_id} theo solution.",
                }
            ],
            "issues": [
                {
                    "severity": "bad",
                    "category": "solution_anchor_consistency",
                    "location": "/questionItems/0/answerSpecs/0",
                    "reason": "answerSpec không khớp đáp án trong solution.",
                    "suggestion": f"Đổi correctOptionId thành {correct_option_id} theo solution.",
                    "repair_intent": "align_fields_to_solution",
                }
            ],
        },
        ensure_ascii=False,
    )


def make_single_choice_generated(question_id: str = "generated_1") -> dict:
    return {
        "_id": question_id,
        "aiId": "ai-flow",
        "difficulty": "easy",
        "bloom": "apply",
        "interactionTypes": ["single_choice"],
        "instruction": [{"id": "intro", "type": "text", "text": "Giải phương trình 2x + 3 = 7."}],
        "questionItems": [
            {
                "id": "item_1",
                "stem": [{"id": "stem_1", "type": "text", "text": "Chọn giá trị đúng của x."}],
                "interactions": [
                    {
                        "id": "choice_1",
                        "type": "single_choice",
                        "config": {
                            "options": [
                                {"id": "A", "content": [{"id": "opt_a", "type": "text", "text": "$x = 1$"}]},
                                {"id": "B", "content": [{"id": "opt_b", "type": "text", "text": "$x = 2$"}]},
                                {"id": "C", "content": [{"id": "opt_c", "type": "text", "text": "$x = 3$"}]},
                            ],
                            "shuffleOptions": False,
                        },
                        "display": {"layout": "vertical"},
                    }
                ],
                "answerSpecs": [
                    {
                        "interactionId": "choice_1",
                        "type": "single_choice",
                        "expected": {"correctOptionId": "B"},
                    }
                ],
                "hints": [
                    {
                        "name": "Gợi ý 1",
                        "content": [{"id": "hint_1", "type": "text", "text": "Chuyển 3 sang vế phải."}],
                    }
                ],
            }
        ],
        "solutions": [
            {
                "solverName": "default",
                "solutionContent": [{"id": "sol_1", "type": "text", "text": "Ta có 2x = 4 nên x = 2."}],
            }
        ],
    }


def make_short_answer_generated(question_id: str = "generated_short") -> dict:
    return {
        "_id": question_id,
        "aiId": "ai-flow",
        "difficulty": "easy",
        "bloom": "apply",
        "interactionTypes": ["short_answer"],
        "instruction": [{"id": "intro", "type": "text", "text": "Giải phương trình 2x + 3 = 7."}],
        "questionItems": [
            {
                "id": "item_1",
                "stem": [{"id": "stem_1", "type": "text", "text": "Nhập giá trị của x."}],
                "interactions": [
                    {
                        "id": "x_value",
                        "type": "short_answer",
                        "config": {"inputMode": "numeric"},
                        "display": {"layout": "auto"},
                    }
                ],
                "answerSpecs": [
                    {
                        "interactionId": "x_value",
                        "type": "short_answer",
                        "expected": [
                            {
                                "inputMode": "numeric",
                                "value": {"correctValue": 2, "acceptableValues": []},
                                "equivalence": {"type": "numeric_equivalence"},
                            }
                        ],
                    }
                ],
            }
        ],
        "solutions": [
            {
                "solverName": "default",
                "solutionContent": [{"id": "sol_1", "type": "text", "text": "Ta có x = 2."}],
            }
        ],
    }


def make_wrapper(generated_questions: list[dict] | None = None) -> dict:
    return {
        "_id": "source_record_1",
        "name": "Wrapper demo",
        "question": "RAW_QUESTION_SHOULD_NOT_BE_USED",
        "answer": "RAW_ANSWER_SHOULD_NOT_BE_USED",
        "question_plan": {"type": "advanced_question_plan", "plan": []},
        "generatedQuestions": generated_questions if generated_questions is not None else [make_single_choice_generated()],
    }


def test_single_generated_object_input_works_and_uses_id_alias():
    client = FakeClient(ok_llm_payload())
    result = evaluate_generated_questions(
        make_single_choice_generated("direct_id"),
        config=fake_config(),
        client=client,
    )

    assert result["id"] == "direct_id"
    assert result["is_good"] is True
    assert "summary" not in result
    assert len(client.messages) == 1


def test_list_input_returns_aggregate_and_calls_llm_per_object():
    client = FakeClient(ok_llm_payload())
    payload = [make_single_choice_generated("g1"), make_short_answer_generated("g2")]

    result = evaluate_generated_questions(payload, config=fake_config(), client=client)

    assert result["is_good"] is True
    assert result["summary"]["total"] == 2
    assert result["summary"]["good"] == 2
    assert len(result["results"]) == 2
    assert len(client.messages) == 2


def test_wrapper_input_uses_generated_questions_without_requiring_plan():
    client = FakeClient(ok_llm_payload())
    result = evaluate_generated_questions(make_wrapper(), config=fake_config(), client=client)

    assert result["is_good"] is True
    assert result["id"] == "generated_1"
    assert len(client.messages) == 1


def test_normalize_generated_question_input_accepts_list_single_and_wrapper():
    direct = make_single_choice_generated("direct")
    wrapper = make_wrapper([direct])

    assert normalize_generated_question_input(direct)[0]["_id"] == "direct"
    assert normalize_generated_question_input([direct])[0]["_id"] == "direct"
    assert normalize_generated_question_input(wrapper)[0]["_id"] == "direct"


def test_json_pointer_get_and_apply_patch_without_mutating_original():
    original = {"a": [{"b": 1}], "keep": True}
    patched = apply_json_patch(original, [{"op": "replace", "path": "/a/0/b", "value": 2}])

    assert get_by_json_pointer(original, "/a/0/b") == 1
    assert get_by_json_pointer(patched, "/a/0/b") == 2
    assert original["a"][0]["b"] == 1


def test_checker_issue_output_has_scoped_repair_fields():
    generated = make_single_choice_generated()
    generated["questionItems"][0]["answerSpecs"][0]["expected"]["correctOptionId"] = "Z"

    result = evaluate_generated_questions(generated, output_mode="debug")
    issue = result["issues"][0]

    assert issue["location"].startswith("/")
    assert issue["required_context_paths"]
    assert issue["repair_intent"] in {"fix_answer_spec", "fix_correct_option", "align_fields_to_solution"}
    assert "error_snippet" in issue


def test_build_scoped_repair_context_uses_required_paths():
    generated = make_single_choice_generated()
    issue = {
        "severity": "bad",
        "category": "answer_internal_consistency",
        "location": "/questionItems/0/answerSpecs/0",
        "reason": "Sai correctOptionId.",
        "suggestion": "Sua correctOptionId.",
        "error_snippet": "correctOptionId Z",
        "required_context_paths": [
            "/questionItems/0/stem",
            "/questionItems/0/interactions",
            "/questionItems/0/answerSpecs",
        ],
        "repair_intent": "fix_correct_option",
    }

    scoped = build_scoped_repair_context(generated, issue, {"id": "g", "issues": [issue]})

    assert scoped["fallback_needed"] is False
    context = scoped["scoped_payload"]["extracted_context"]
    assert "/questionItems/0/stem" in context
    assert "/questionItems/0/interactions" in context
    assert "/questionItems/0/answerSpecs" in context


def test_generated_checker_does_not_require_raw_question_answer_or_question_plan():
    client = FakeClient(ok_llm_payload())
    generated = make_short_answer_generated()

    result = evaluate_generated_questions(generated, config=fake_config(), client=client, output_mode="debug")

    assert result["is_good"] is True
    assert not result["failed_reason"]


def test_generated_checker_default_output_is_compact_and_hides_diagnostics():
    client = FakeClient(ok_llm_payload())
    generated = make_short_answer_generated("compact_default")

    result = evaluate_generated_questions(generated, config=fake_config(), client=client)

    assert result["is_good"] is True
    assert "failed_reason" not in result
    assert "suggestions" not in result
    assert "solution_anchor_result" not in result
    assert "answer_verifier_result" not in result


def test_debug_output_hides_deprecated_alias_unless_requested():
    generated = make_single_choice_generated("debug_deprecated_alias")

    hidden = evaluate_generated_questions(
        generated,
        config=fake_config(),
        client=FakeClient(ok_llm_payload()),
        output_mode="debug",
    )
    visible = evaluate_generated_questions(
        generated,
        config=fake_config(),
        client=FakeClient(ok_llm_payload()),
        output_mode="debug",
        include_deprecated=True,
    )

    assert "answer_verifier_result" not in hidden
    assert visible["answer_verifier_result"]["deprecated"] is True


def test_input_spelling_check_is_off_by_default():
    generated = make_single_choice_generated("spelling_default_off")
    generated["questionItems"][0]["stem"][0]["text"] = "Chọn chọn giá trị đúng của x."

    result = evaluate_generated_questions(
        generated,
        config=fake_config(),
        client=FakeClient(ok_llm_payload()),
        answer_verify_mode="off",
        output_mode="debug",
    )

    assert result["spelling_result"]["spelling_status"] == "skipped"
    assert not any(issue["category"] == "spelling_wording" for issue in result["issues"])


def test_single_choice_solution_final_value_matches_option_content():
    generated = make_single_choice_generated("value_match")
    interaction = generated["questionItems"][0]["interactions"][0]
    interaction["config"]["options"] = [
        {"id": "opt_a", "content": [{"id": "opt_a_text", "type": "text", "text": "$2$"}]},
        {"id": "opt_b", "content": [{"id": "opt_b_text", "type": "text", "text": "$5$"}]},
        {"id": "opt_c", "content": [{"id": "opt_c_text", "type": "text", "text": "$3$"}]},
        {"id": "opt_d", "content": [{"id": "opt_d_text", "type": "text", "text": "$0$"}]},
    ]
    generated["questionItems"][0]["answerSpecs"][0]["expected"]["correctOptionId"] = "opt_b"
    generated["solutions"][0]["solutionContent"][0]["text"] = (
        "Giá trị lớn nhất là $2$, giá trị nhỏ nhất là $-2$. "
        "Tổng hai giá trị là $2 + (-2) = 0$. Vậy đáp án đúng là $0$."
    )

    result = evaluate_generated_questions(
        generated,
        config=fake_config(),
        client=FakeClient(ok_llm_payload()),
        solution_anchor_mode="always",
        output_mode="debug",
    )

    anchor = result["solution_anchor_result"]
    assert anchor["resolver_status"] == "resolved"
    assert anchor["solution_derived_answer"]["correctOptionId"] == "opt_d"
    assert anchor["solution_derived_answer"]["answer"] == "$0$"
    assert anchor["solution_derived_answer"]["matched_option_text"] == "$0$"
    assert anchor["fields_to_fix"][0]["value"] == "opt_d"
    assert not any(issue.get("repair_intent") == "needs_manual_review" for issue in result["issues"])


def test_public_output_deduplicates_repeated_issues():
    issue = {
        "severity": "bad",
        "category": "answer_internal_consistency",
        "location": "/questionItems/0/answerSpecs/0",
        "reason": "answerSpec không khớp với option đúng.",
        "suggestion": "Sửa correctOptionId.",
        "repair_intent": "fix_correct_option",
    }
    client = FakeClient(
        json.dumps(
            {
                "is_good": False,
                "failed_reason": [issue["reason"], issue["reason"]],
                "suggestions": [issue["suggestion"], issue["suggestion"]],
                "issues": [issue, dict(issue)],
            },
            ensure_ascii=False,
        )
    )

    result = evaluate_generated_questions(
        make_short_answer_generated("dedupe_issues"),
        config=fake_config(),
        client=client,
        solution_anchor_mode="off",
    )

    assert len(result["issues"]) == 1
    assert result["issues"][0]["category"] == "answer_internal_consistency"


def test_solution_anchor_skips_when_no_conflict_in_on_conflict_mode():
    client = FakeClient(ok_llm_payload())
    generated = make_single_choice_generated()

    result = evaluate_generated_questions(
        generated,
        config=fake_config(),
        client=client,
        solution_anchor_mode="on_conflict",
        output_mode="debug",
    )

    assert result["is_good"] is True
    assert result["solution_anchor_result"]["resolver_status"] == "skipped"
    assert len(client.messages) == 1


def test_solution_anchor_adds_consistency_issue_from_solution():
    generated = make_single_choice_generated("solution_anchor_bad")
    generated["questionItems"][0]["answerSpecs"][0]["expected"]["correctOptionId"] = "A"
    client = FakeClient(ok_llm_payload())

    result = evaluate_generated_questions(
        generated,
        config=fake_config(),
        client=client,
        solution_anchor_mode="on_conflict",
        output_mode="debug",
    )

    assert result["is_good"] is False
    assert result["solution_anchor_result"]["resolver_status"] == "resolved"
    assert result["solution_anchor_result"]["solution_derived_answer"]["correctOptionId"] == "B"
    assert any(issue["category"] == "solution_anchor_consistency" for issue in result["issues"])


def test_deprecated_answer_verify_mode_maps_to_solution_anchor():
    generated = make_single_choice_generated("deprecated_answer_verify_mode")
    generated["questionItems"][0]["answerSpecs"][0]["expected"]["correctOptionId"] = "A"
    client = FakeClient(ok_llm_payload())

    result = evaluate_generated_questions(
        generated,
        config=fake_config(),
        client=client,
        answer_verify_mode="all",
        include_deprecated=True,
        output_mode="debug",
    )

    assert result["solution_anchor_result"]["resolver_status"] == "resolved"
    assert result["answer_verifier_result"]["deprecated"] is True
    assert result["answer_verifier_result"]["verifier_status"] == "verified"


def test_spelling_rule_based_detects_repeated_word_without_llm_spelling_call():
    client = FakeClient(ok_llm_payload())
    generated = make_single_choice_generated("spell_bad")
    generated["questionItems"][0]["stem"][0]["text"] = "Chọn chọn giá trị đúng của x."

    result = evaluate_generated_questions(
        generated,
        config=fake_config(),
        client=client,
        answer_verify_mode="off",
        spelling_check_mode="on_issue",
        output_mode="debug",
    )

    assert result["is_good"] is False
    assert result["spelling_result"]["spelling_status"] == "checked"
    assert any(issue["category"] == "spelling_wording" for issue in result["issues"])


def test_math_segments_can_be_protected_and_restored_for_spelling():
    text = "Tính $a^2 + b^2$ rồi trả lời."
    protected, segments = protect_math_segments(text)

    assert "$a^2 + b^2$" not in protected
    assert restore_math_segments(protected, segments) == text


def test_scoped_text_repair_rejects_math_segment_changes():
    generated = make_single_choice_generated("math_guard")
    generated["instruction"][0]["text"] = "Tính $a^2$."

    result = normalize_scoped_repair_result(
        {
            "repair_status": "repaired",
            "failed_reason": [],
            "suggestions": [],
            "patches": [{"op": "replace", "path": "/instruction/0/text", "value": "Tính $a^3$."}],
        },
        generated_question=generated,
    )

    assert result["repair_status"] == "needs_full_repair"
    assert "math/LaTeX" in result["failed_reason"][0]


def test_generated_checker_ignores_difficulty_and_bloom_issues():
    result = normalize_generated_question_result(
        {
            "id": "metadata_only",
            "is_good": False,
            "failed_reason": ["Độ khó medium không phù hợp.", "Bloom remember chưa phù hợp."],
            "suggestions": ["Sửa difficulty xuống easy.", "Sửa bloom sang apply."],
            "issues": [
                {
                    "severity": "warning",
                    "category": "difficulty_fit",
                    "location": "difficulty,bloom",
                    "reason": "Difficulty/Bloom có vẻ chưa phù hợp.",
                    "suggestion": "Sửa difficulty hoặc bloom.",
                }
            ],
        }
    )

    assert result["is_good"] is True
    assert result["issues"] == []
    assert result["failed_reason"] == []
    assert result["suggestions"] == []


def test_generated_question_prompt_excludes_wrapper_fields():
    wrapper = make_wrapper()
    generated = normalize_generated_question_input(wrapper)[0]

    messages = build_generated_question_judge_messages(
        generated,
        "criteria",
        "output schema",
    )
    prompt = "\n".join(message["content"] for message in messages)

    assert "RAW_QUESTION_SHOULD_NOT_BE_USED" not in prompt
    assert "RAW_ANSWER_SHOULD_NOT_BE_USED" not in prompt
    assert '"question_plan"' not in prompt


def test_generated_question_llm_prompts_require_vietnamese_output():
    generated = make_single_choice_generated()
    check_result = {
        "id": "generated_1",
        "is_good": False,
        "failed_reason": ["Wrong answerSpec."],
        "suggestions": ["Fix the correct option."],
        "issues": [
            {
                "severity": "bad",
                "category": "answer_internal_consistency",
                "location": "/questionItems/0/answerSpecs/0",
                "reason": "Wrong answerSpec.",
                "suggestion": "Fix answerSpec.",
                "repair_intent": "fix_correct_option",
                "required_context_paths": ["/questionItems/0/answerSpecs"],
            }
        ],
    }
    scoped_payload = {
        "generated_question_id": "generated_1",
        "issue": check_result["issues"][0],
        "check_result": check_result,
        "extracted_context": {"/questionItems/0/answerSpecs": generated["questionItems"][0]["answerSpecs"]},
    }

    judge_prompt = "\n".join(
        message["content"]
        for message in build_generated_question_judge_messages(
            generated,
            "criteria",
            "output schema",
        )
    )
    scoped_prompt = "\n".join(
        message["content"]
        for message in build_generated_question_scoped_repair_messages(
            scoped_payload,
            "repair rules",
            "output schema",
        )
    )

    for prompt in (judge_prompt, scoped_prompt):
        assert "tiếng Việt có dấu" in prompt
        assert "Không dùng câu tiếng Anh" in prompt
        assert "failed_reason" in prompt
        assert "suggestions" in prompt


def test_single_choice_correct_option_id_must_exist_in_options():
    generated = make_single_choice_generated()
    generated["questionItems"][0]["answerSpecs"][0]["expected"]["correctOptionId"] = "Z"

    result = evaluate_generated_questions(generated)

    assert result["is_good"] is False
    assert any(issue["category"] == "answer_internal_consistency" for issue in result["issues"])
    assert any("correctOptionId" in issue["reason"] for issue in result["issues"])


def test_multiple_choice_with_one_correct_option_returns_bad_without_llm():
    generated = make_single_choice_generated()
    generated["interactionTypes"] = ["multiple_choice"]
    interaction = generated["questionItems"][0]["interactions"][0]
    interaction["type"] = "multiple_choice"
    spec = generated["questionItems"][0]["answerSpecs"][0]
    spec["type"] = "multiple_choice"
    spec["expected"] = {"correctOptionIds": ["B"]}

    result = evaluate_generated_questions(generated)

    assert result["is_good"] is False
    assert any(issue["category"] == "answer_internal_consistency" for issue in result["issues"])


def test_essay_is_allowed_and_not_reported_as_unsupported_type():
    client = FakeClient(ok_llm_payload())
    generated = make_short_answer_generated("essay_ok")
    generated["interactionTypes"] = ["essay"]
    generated["questionItems"][0]["stem"][0]["text"] = "Trình bày lời giải chi tiết cho phương trình đã cho."
    generated["questionItems"][0]["interactions"][0] = {
        "id": "essay_1",
        "type": "essay",
        "config": {},
        "display": {"layout": "auto"},
    }
    generated["questionItems"][0]["answerSpecs"] = [
        {"interactionId": "essay_1", "type": "essay", "rubric": "Có lập luận và kết luận đúng."}
    ]

    result = evaluate_generated_questions(generated, config=fake_config(), client=client)

    assert result["is_good"] is True
    assert not any("unsupported" in issue["reason"].lower() for issue in result["issues"])


def test_essay_missing_rubric_or_solution_needs_review_not_crash():
    client = FakeClient(ok_llm_payload())
    generated = {
        "_id": "essay_review",
        "difficulty": "medium",
        "bloom": "analyze",
        "interactionTypes": ["essay"],
        "instruction": [{"id": "intro", "type": "text", "text": "Trả lời câu hỏi sau."}],
        "questionItems": [
            {
                "id": "item_1",
                "stem": [{"id": "stem_1", "type": "text", "text": "Trình bày lời giải chi tiết."}],
                "interactions": [{"id": "essay_1", "type": "essay", "config": {}, "display": {"layout": "auto"}}],
                "answerSpecs": [],
            }
        ],
        "solutions": [],
    }

    result = evaluate_generated_questions(generated, config=fake_config(), client=client)

    assert result["is_good"] is False
    assert any(issue["severity"] == "needs_review" for issue in result["issues"])


def test_essay_with_solution_can_have_empty_answer_specs():
    client = FakeClient(ok_llm_payload())
    generated = {
        "_id": "essay_with_solution",
        "difficulty": "medium",
        "bloom": "remember",
        "interactionTypes": ["essay"],
        "instruction": [
            {
                "id": "q1_block_001",
                "type": "text",
                "text": "Biểu đồ tranh biểu diễn số lượng máy cày của 5 xã. Em hãy lập bảng thống kê tương ứng.",
            }
        ],
        "questionItems": [
            {
                "id": "item_1",
                "stem": [
                    {
                        "id": "stem_1",
                        "type": "text",
                        "text": "Lập bảng thống kê tương ứng với số lượng máy cày của 5 xã dựa trên biểu đồ tranh.",
                    }
                ],
                "interactions": [
                    {
                        "id": "essay_table_crop_count",
                        "type": "essay",
                        "config": {},
                        "display": {"layout": "vertical", "width": "full"},
                    }
                ],
                "answerSpecs": [],
            }
        ],
        "solutions": [
            {
                "solverName": "default",
                "solutionContent": [
                    {
                        "id": "sol_001",
                        "type": "text",
                        "text": "Lời giải mẫu trình bày cách đọc biểu đồ tranh và lập bảng thống kê.",
                    }
                ],
            }
        ],
    }

    result = evaluate_generated_questions(generated, config=fake_config(), client=client)

    assert result["is_good"] is True
    assert not any("answerSpec" in issue["reason"] for issue in result["issues"])
    assert not any("unsupported" in issue["reason"].lower() for issue in result["issues"])


def test_answer_spec_unknown_interaction_id_returns_bad_without_llm():
    generated = make_short_answer_generated()
    generated["questionItems"][0]["answerSpecs"][0]["interactionId"] = "missing_interaction"

    result = evaluate_generated_questions(generated)

    assert result["is_good"] is False
    assert any(issue["category"] == "answer_internal_consistency" for issue in result["issues"])


def test_unsupported_interaction_type_returns_bad_without_llm():
    generated = make_short_answer_generated()
    generated["questionItems"][0]["interactions"][0]["type"] = "unknown_type"
    generated["questionItems"][0]["answerSpecs"][0]["type"] = "unknown_type"

    result = evaluate_generated_questions(generated)

    assert result["is_good"] is False
    assert any(issue["category"] == "interaction_schema" for issue in result["issues"])


def test_llm_parse_error_is_fail_closed():
    result = evaluate_generated_questions(
        make_short_answer_generated(),
        config=fake_config(),
        client=FakeClient("not json"),
    )

    assert result["is_good"] is False
    assert any(issue["category"] == "runtime" for issue in result["issues"])


def test_math_9_bt_test_json_list_can_run_with_fake_client():
    path = ROOT_DIR / "data" / "processed" / "math_9_bt_test.json"
    payload = json.loads(path.read_text(encoding="utf-8-sig"))[:3]

    result = evaluate_generated_questions(payload, config=fake_config(), client=FakeClient(ok_llm_payload()))

    assert result["summary"]["total"] == 3
    assert len(result["results"]) == 3


def test_generated_question_markdown_report_includes_summary_table_and_issues():
    result = {
        "is_good": False,
        "failed_reason": ["Có lỗi."],
        "suggestions": ["Sửa lỗi."],
        "summary": {"total": 1, "good": 0, "bad": 1, "needs_review": 0, "warning": 0},
        "results": [
            {
                "id": "g1",
                "is_good": False,
                "failed_reason": ["Sai answerSpec."],
                "suggestions": ["Sửa correctOptionId."],
                "issues": [
                    {
                        "severity": "bad",
                        "category": "answer_internal_consistency",
                        "location": "questionItems[0].answerSpecs[0]",
                        "reason": "correctOptionId không tồn tại.",
                        "suggestion": "Đổi correctOptionId.",
                    }
                ],
            }
        ],
    }

    markdown = format_generated_question_markdown(result, source_name="input.json")

    assert "# Generated Question Quality Report" in markdown
    assert "Source file: `input.json`" in markdown
    assert "| 1 | g1 | bad | skipped | skipped | skipped | skipped | 0 | 1 |" in markdown
    assert "correctOptionId không tồn tại." in markdown

def test_scoped_repair_applies_patch_and_keeps_original_object_unchanged():
    generated = make_single_choice_generated("scoped_bad")
    generated["questionItems"][0]["answerSpecs"][0]["expected"]["correctOptionId"] = "Z"
    original_bad_value = generated["questionItems"][0]["answerSpecs"][0]["expected"]["correctOptionId"]
    client = FakeClient(
        scoped_repair_payload(
            [{"op": "replace", "path": "/questionItems/0/answerSpecs/0/expected/correctOptionId", "value": "B"}]
        )
    )

    result = evaluate_generated_questions(
        generated,
        config=fake_config(),
        client=client,
        auto_repair=True,
        repair_mode="scoped",
        output_mode="debug",
    )

    assert result["repair_status"] == "repaired"
    assert result["repair_mode_used"] == "scoped"
    assert result["patch_count"] == 1
    assert result["new_generated_question"]["questionItems"][0]["answerSpecs"][0]["expected"]["correctOptionId"] == "B"
    assert generated["questionItems"][0]["answerSpecs"][0]["expected"]["correctOptionId"] == original_bad_value


def test_solution_anchor_repair_aligns_answer_spec_without_changing_solution():
    generated = make_single_choice_generated("solution_anchor_repair")
    generated["questionItems"][0]["answerSpecs"][0]["expected"]["correctOptionId"] = "A"
    original_solution = generated["solutions"][0]["solutionContent"][0]["text"]
    client = SequenceFakeClient(
        [
            ok_llm_payload(),
            scoped_repair_payload(
                [{"op": "replace", "path": "/questionItems/0/answerSpecs/0/expected/correctOptionId", "value": "B"}]
            ),
        ]
    )

    result = evaluate_generated_questions(
        generated,
        config=fake_config(),
        client=client,
        auto_repair=True,
        repair_mode="scoped",
        solution_anchor_mode="on_conflict",
        output_mode="debug",
    )

    assert result["repair_status"] == "repaired"
    assert result["new_generated_question"]["questionItems"][0]["answerSpecs"][0]["expected"]["correctOptionId"] == "B"
    assert result["new_generated_question"]["solutions"][0]["solutionContent"][0]["text"] == original_solution


def test_solution_anchor_does_not_report_answer_validity_when_solution_and_answer_spec_match():
    generated = make_single_choice_generated("solution_and_spec_match")
    generated["instruction"][0]["text"] = "Giải phương trình 2x + 3 = 7."
    generated["questionItems"][0]["answerSpecs"][0]["expected"]["correctOptionId"] = "A"
    generated["solutions"][0]["solutionContent"][0]["text"] = "Ta có 2x = 2 nên x = 1. Vậy chọn đáp án A."
    client = FakeClient(ok_llm_payload())

    result = evaluate_generated_questions(
        generated,
        config=fake_config(),
        client=client,
        solution_anchor_mode="on_conflict",
        output_mode="debug",
    )

    assert result["is_good"] is True
    assert not any(issue["category"] == "answer_validity" for issue in result["issues"])
    assert not any(issue["category"] == "solution_anchor_consistency" for issue in result["issues"])


def test_single_choice_option_analysis_with_one_final_answer_is_valid():
    generated = make_single_choice_generated("single_choice_option_analysis")
    generated["questionItems"][0]["answerSpecs"][0]["expected"]["correctOptionId"] = "C"
    generated["solutions"][0]["solutionContent"][0]["text"] = (
        "Đáp án A: sai vì chưa thỏa mãn phương trình.\n"
        "Đáp án B: sai vì thay vào không đúng.\n"
        "Đáp án C: đúng.\n"
        "Vậy đáp án đúng là C."
    )

    result = evaluate_generated_questions(
        generated,
        config=fake_config(),
        client=FakeClient(ok_llm_payload()),
        solution_anchor_mode="always",
        output_mode="debug",
    )

    assert result["solution_anchor_result"]["resolver_status"] == "resolved"
    assert result["solution_anchor_result"]["solution_derived_answer"]["correctOptionId"] == "C"
    assert result["is_good"] is True
    assert not any(issue["repair_intent"] == "needs_manual_review" for issue in result["issues"])


def test_single_choice_analysis_and_final_answer_contradiction_needs_review():
    generated = make_single_choice_generated("single_choice_contradiction")
    generated["questionItems"][0]["answerSpecs"][0]["expected"]["correctOptionId"] = "B"
    generated["solutions"][0]["solutionContent"][0]["text"] = (
        "Đáp án A: sai.\n"
        "Đáp án B: sai.\n"
        "Đáp án C: đúng.\n"
        "Vậy đáp án đúng là B."
    )

    result = evaluate_generated_questions(
        generated,
        config=fake_config(),
        client=FakeClient(ok_llm_payload()),
        solution_anchor_mode="always",
        output_mode="debug",
    )

    flags = result["solution_anchor_result"]["solution_quality_flags"]
    assert result["is_good"] is False
    assert result["solution_anchor_result"]["resolver_status"] == "needs_manual_review"
    assert result["solution_anchor_result"]["fields_to_fix"] == []
    assert any(flag["type"] == "internal_contradiction" for flag in flags)


def test_single_choice_multiple_final_answers_needs_review():
    generated = make_single_choice_generated("single_choice_ambiguous")
    generated["solutions"][0]["solutionContent"][0]["text"] = "A hoặc C đều đúng, có thể chọn một trong hai phương án."

    result = evaluate_generated_questions(
        generated,
        config=fake_config(),
        client=FakeClient(ok_llm_payload()),
        solution_anchor_mode="always",
        output_mode="debug",
    )

    flags = result["solution_anchor_result"]["solution_quality_flags"]
    assert result["is_good"] is False
    assert result["solution_anchor_result"]["resolver_status"] == "needs_manual_review"
    assert any(flag["type"] == "ambiguous_final_answer" for flag in flags)


def test_solution_internal_reasoning_is_clean_solution_issue_not_answer_change():
    generated = make_single_choice_generated("solution_internal_reasoning")
    generated["solutions"][0]["solutionContent"][0]["text"] = (
        "Không, cách này phức tạp. Hãy thử cách khác.\n"
        "Ta có 2x + 3 = 7 nên x = 2. Vậy đáp án đúng là B."
    )

    result = evaluate_generated_questions(
        generated,
        config=fake_config(),
        client=FakeClient(ok_llm_payload()),
        solution_anchor_mode="always",
        output_mode="debug",
    )

    flags = result["solution_anchor_result"]["solution_quality_flags"]
    assert result["solution_anchor_result"]["resolver_status"] == "resolved"
    assert result["solution_anchor_result"]["solution_derived_answer"]["correctOptionId"] == "B"
    assert result["solution_anchor_result"]["fields_to_fix"] == []
    assert any(flag["type"] == "internal_reasoning" for flag in flags)
    assert any(issue["repair_intent"] == "clean_solution_reasoning" for issue in result["issues"])
    assert not any(issue["category"] == "solution_anchor_consistency" for issue in result["issues"])


def test_multiple_choice_single_solution_answer_needs_review_not_invented():
    generated = make_single_choice_generated("mc_one_solution")
    generated["interactionTypes"] = ["multiple_choice"]
    generated["questionItems"][0]["interactions"][0]["type"] = "multiple_choice"
    generated["questionItems"][0]["answerSpecs"][0]["type"] = "multiple_choice"
    generated["questionItems"][0]["answerSpecs"][0]["expected"] = {"correctOptionIds": ["B"]}
    generated["solutions"][0]["solutionContent"][0]["text"] = "Tính được x = 2. Vậy chọn đáp án B."

    result = evaluate_generated_questions(
        generated,
        config=fake_config(),
        client=FakeClient(ok_llm_payload()),
        solution_anchor_mode="on_conflict",
        output_mode="debug",
    )

    assert result["is_good"] is False
    assert result["solution_anchor_result"]["resolver_status"] == "needs_manual_review"
    assert result["solution_anchor_result"]["fields_to_fix"] == []
    assert any(issue["repair_intent"] == "needs_manual_review" for issue in result["issues"])


def test_auto_repair_falls_back_to_full_when_scoped_requests_full_repair():
    generated = make_single_choice_generated("fallback_bad")
    generated["questionItems"][0]["answerSpecs"][0]["expected"]["correctOptionId"] = "Z"
    repaired = make_single_choice_generated("fallback_bad")
    client = SequenceFakeClient(
        [
            scoped_repair_payload([], "needs_full_repair"),
            repair_llm_payload(repaired, "repaired"),
        ]
    )

    result = evaluate_generated_questions(
        generated,
        config=fake_config(),
        client=client,
        auto_repair=True,
        repair_mode="auto",
        answer_verify_mode="off",
        output_mode="debug",
    )

    assert result["repair_status"] == "repaired"
    assert result["repair_mode_used"] == "full_fallback"
    assert result["new_generated_question"]["_id"] == "fallback_bad"


def test_generated_question_repair_outputs_write_repaired_objects_and_mapping(tmp_path):
    bad_1 = make_single_choice_generated("bad_1")
    bad_1["schemaVersion"] = 4
    bad_1["questionItems"][0]["answerSpecs"][0]["expected"]["correctOptionId"] = "Z"
    repaired_1 = make_single_choice_generated("bad_1")

    bad_2 = make_single_choice_generated("bad_2")
    bad_2["questionItems"][0]["answerSpecs"][0]["expected"]["correctOptionId"] = "Z"

    client = SequenceFakeClient(
        [
            repair_llm_payload(repaired_1, "repaired"),
            repair_llm_payload(None, "failed"),
        ]
    )
    result = evaluate_generated_questions(
        [bad_1, bad_2],
        config=fake_config(),
        client=client,
        auto_repair=True,
        repair_mode="full",
        answer_verify_mode="off",
    )

    assert result["summary"]["total"] == 2
    assert result["summary"]["repaired"] == 1
    assert result["summary"]["repair_failed"] == 1
    assert len(result["results"]) == 2

    full_output = tmp_path / "full.json"
    report_output = tmp_path / "report.md"
    repaired_output = tmp_path / "repaired.json"
    write_generated_question_cli_outputs(
        result=result,
        source_name="input.json",
        output_path=full_output,
        report_output_path=report_output,
        repaired_output_path=repaired_output,
    )

    full_result = json.loads(full_output.read_text(encoding="utf-8"))
    repaired_objects = json.loads(repaired_output.read_text(encoding="utf-8"))

    assert len(full_result["results"]) == 2
    assert len(repaired_objects) == 1
    assert repaired_objects[0]["_id"] == "bad_1"
    assert repaired_objects[0]["schemaVersion"] == 4
    assert "issues" not in repaired_objects[0]
    assert "repair_status" not in repaired_objects[0]
    assert report_output.read_text(encoding="utf-8").startswith("# Generated Question Quality Report")


def test_repaired_objects_output_is_empty_when_no_repair_success(tmp_path):
    result = {
        "is_good": False,
        "failed_reason": [],
        "suggestions": [],
        "summary": {
            "total": 1,
            "good": 0,
            "bad": 1,
            "needs_review": 0,
            "warning": 0,
            "repaired": 0,
            "repair_failed": 1,
        },
        "results": [
            {
                "id": "g1",
                "is_good": False,
                "failed_reason": [],
                "suggestions": [],
                "issues": [],
                "repair_status": "failed",
                "new_generated_question": None,
            }
        ],
    }
    repaired_output = tmp_path / "repaired.json"

    write_generated_question_cli_outputs(
        result=result,
        source_name="input.json",
        output_path=tmp_path / "full.json",
        report_output_path=None,
        repaired_output_path=repaired_output,
    )

    assert json.loads(repaired_output.read_text(encoding="utf-8")) == []
    assert extract_repaired_generated_questions(result) == []


def test_interaction_classifier_prompt_uses_full_question_plan_knowledge():
    messages = build_real_classifier_messages(
        {
            "raw_question": "Ghép các biểu thức tương đương.",
            "question_plan": {},
            "instruction": [],
            "stem": [],
            "declared_interaction_type": "matching",
            "planned_interaction_type": "matching",
            "interaction": {"id": "m1", "type": "matching", "config": {}},
        }
    )
    user_prompt = messages[1]["content"]

    assert "INTERACTION KNOWLEDGE" in user_prompt
    assert "matching" in user_prompt
    assert "drag_drop" in user_prompt
    assert "graph_draw" in user_prompt

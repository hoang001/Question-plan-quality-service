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
    validate_generated_question_schema,
)


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


def test_generated_checker_does_not_require_raw_question_answer_or_question_plan():
    client = FakeClient(ok_llm_payload())
    generated = make_short_answer_generated()

    result = evaluate_generated_questions(generated, config=fake_config(), client=client)

    assert result["is_good"] is True
    assert not result["failed_reason"]


def test_generated_question_prompt_excludes_wrapper_fields():
    wrapper = make_wrapper()
    generated = normalize_generated_question_input(wrapper)[0]
    schema_result = validate_generated_question_schema(generated)

    messages = build_generated_question_judge_messages(
        generated,
        schema_result,
        "criteria",
        "type rules",
        "output schema",
    )
    prompt = "\n".join(message["content"] for message in messages)

    assert "RAW_QUESTION_SHOULD_NOT_BE_USED" not in prompt
    assert "RAW_ANSWER_SHOULD_NOT_BE_USED" not in prompt
    assert '"question_plan"' not in prompt


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
    assert "| 1 | g1 | bad | skipped | 1 |" in markdown
    assert "correctOptionId không tồn tại." in markdown

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
    )

    assert result["summary"]["total"] == 2
    assert result["summary"]["repaired"] == 1
    assert result["summary"]["repair_failed"] == 1
    assert len(result["results"]) == 2

    full_output = tmp_path / "full.json"
    report_output = tmp_path / "report.md"
    repaired_output = tmp_path / "repaired.json"
    mapping_output = tmp_path / "mapping.json"
    write_generated_question_cli_outputs(
        result=result,
        source_name="input.json",
        output_path=full_output,
        report_output_path=report_output,
        repaired_output_path=repaired_output,
        repair_mapping_output_path=mapping_output,
    )

    full_result = json.loads(full_output.read_text(encoding="utf-8"))
    repaired_objects = json.loads(repaired_output.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_output.read_text(encoding="utf-8"))

    assert len(full_result["results"]) == 2
    assert len(repaired_objects) == 1
    assert repaired_objects[0]["_id"] == "bad_1"
    assert repaired_objects[0]["schemaVersion"] == 4
    assert "issues" not in repaired_objects[0]
    assert "repair_status" not in repaired_objects[0]
    assert report_output.read_text(encoding="utf-8").startswith("# Generated Question Quality Report")
    assert mapping[0]["repair_status"] == "repaired"
    assert mapping[0]["new_generated_question_included"] is True
    assert mapping[1]["repair_status"] == "failed"


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
        repair_mapping_output_path=None,
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

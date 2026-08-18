import json
import re
from copy import deepcopy
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace

from cli import (
    default_generated_question_report_path,
    extract_repaired_generated_questions,
    format_generated_question_markdown,
)
from src.question_plan.flows.generated_question_service import (
    evaluate_generated_question_object,
    evaluate_generated_questions,
    public_generated_question_output,
    public_generated_question_result,
)
from src.question_plan.infra.llm_client import ApiError, LLMClient
from src.question_plan.logic.generated_question_judge import (
    build_direct_correctness_messages,
    build_direct_process_presentation_messages,
    compact_generated_question_payload,
    judge_generated_question_object,
)
from src.question_plan.logic.generated_question_repair import normalize_scoped_repair_result
from src.question_plan.logic.generated_question_schema import (
    aggregate_generated_question_results,
    normalize_generated_question_input,
    normalize_generated_question_result,
)
from src.question_plan.logic.generated_question_spelling import protect_math_segments, restore_math_segments
from src.question_plan.logic.solution_anchor_resolver import (
    _valid_fix_path,
    build_solution_anchor_resolver_messages,
    compact_generated_question_for_solution_anchor,
    normalize_solution_anchor_result,
    resolve_solution_anchor_consistency,
)
from src.question_plan.schemas.generated_question_contracts import (
    ResolverFieldFix,
    ScopedRepairOutput,
    SolutionResolverOutput,
    SpellingIssue,
    SpellingOutput,
)
from src.question_plan.utils.json_pointer import apply_json_patch, get_by_json_pointer


def fake_config(*, gemma_runs: int = 1):
    return SimpleNamespace(
        primary_judge_model="fast-model",
        fallback_judge_model="reasoning-model",
        use_fallback_judge=True,
        gemma_self_consistency_runs=gemma_runs,
        gemma_evaluation_concurrency=4,
    )


def test_routing_metrics_are_internal_unless_debug_output_is_requested():
    aggregate = {
        "summary": {
            "total": 1,
            "good": 1,
            "primary_12b_calls": 1,
            "fallback_26b_calls": 0,
        },
        "results": [{"id": "x", "is_good": True, "issues": []}],
    }

    public = public_generated_question_output(aggregate, debug=False)
    debug = public_generated_question_output(aggregate, debug=True)

    assert public["summary"] == {"total": 1, "good": 1}
    assert debug["summary"]["primary_12b_calls"] == 1


def test_reasoning_http_504_retries_same_request_exactly_once(monkeypatch):
    config = SimpleNamespace(
        api_key="test",
        base_url="https://example.invalid/",
        chat_completions_endpoint="/v1/chat/completions",
        request_timeout_seconds=30,
        gemma_request_timeout_seconds=30,
        primary_judge_model="fast-model",
        llm_top_p=0.1,
    )
    client = LLMClient(config)
    calls = []

    def post(endpoints, payload, **kwargs):
        calls.append(deepcopy(payload))
        if len(calls) == 1:
            raise ApiError("gateway timeout", status_code=504)
        return {"choices": [{"message": {"content": "{}"}}]}, endpoints[0]

    monkeypatch.setattr(client, "post_json_with_fallback", post)
    response = client.chat_completion(
        model="reasoning-model",
        messages=[{"role": "user", "content": "test"}],
        response_format={"type": "json_schema"},
        max_tokens=2048,
        retry_transient_once=True,
    )

    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert calls[0]["max_tokens"] == 2048
    assert response["http_retry_count"] == 1


def test_default_markdown_report_path_is_scoped_by_input_filename():
    first = default_generated_question_report_path(
        Path("tests/grade_2_solution_test_cases.json")
    )
    second = default_generated_question_report_path(
        Path("tests/grade_3_solution_test_cases.json")
    )
    same_name = default_generated_question_report_path(
        Path("other/grade_2_solution_test_cases.json")
    )

    assert first == Path("results/grade_2_solution_test_cases_report.md")
    assert second == Path("results/grade_3_solution_test_cases_report.md")
    assert same_name == first


class SequenceClient:
    def __init__(self, payloads: list):
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if not self.payloads:
            raise AssertionError("Unexpected LLM call")
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return {"content": json.dumps(payload, ensure_ascii=False), "latency_seconds": 0}


class RoleClient:
    def __init__(self, *, correctness=(), global_quality=(), resolvers=(), reviews=()):
        self.payloads = {
            "correctness": list(correctness),
            "global_quality": list(global_quality),
            "resolver": list(resolvers),
            "review": list(reviews),
        }
        self.calls: list[dict] = []
        self.lock = Lock()

    def chat_completion(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        role = (
            "review" if "trách nhiệm kiểm chứng các nhận xét" in system
            else "correctness" if "trách nhiệm kiểm tra tính đúng đắn" in system
            else "global_quality" if "trách nhiệm xem xét quá trình" in system
            else "resolver" if "Solution Resolver" in system
            else ""
        )
        with self.lock:
            self.calls.append(kwargs)
            if role == "review" and not self.payloads[role]:
                prompt = kwargs["messages"][-1]["content"]
                ids = list(dict.fromkeys(re.findall(r'comment_id[^A-Za-z0-9_-]+([A-Za-z0-9_-]+)', prompt)))
                payload = {
                    "reviewed_comments": [
                        {
                            "comment_id": comment_id,
                            "disposition": "blocking",
                            "review_reason": "Nhận xét được xác nhận trong test.",
                            "review_suggestion": "Sửa nội dung theo nhận xét đã xác nhận.",
                        }
                        for comment_id in ids
                        if comment_id.startswith(("correctness-", "process-"))
                    ],
                }
            elif not role or not self.payloads[role]:
                raise AssertionError(f"Unexpected LLM call: {role or system[:80]}")
            else:
                payload = self.payloads[role].pop(0)
        if isinstance(payload, Exception):
            raise payload
        return {"content": json.dumps(payload, ensure_ascii=False), "latency_seconds": 0}


def judge_ok() -> dict:
    return {
        "opening_checks": [{
            "solution_index": 0,
            "evidence": "Ta có 2x = 4",
            "analysis": "Khẳng định mở đầu phù hợp với đề bài.",
        }],
        "comments": [],
    }


def judge_issue(reason: str = "Lời giải làm mất một nghiệm cần được giữ lại.") -> dict:
    return {
        "opening_checks": [{
            "solution_index": 0,
            "evidence": "Ta có 2x = 4",
            "analysis": "Khẳng định mở đầu phù hợp với đề bài.",
        }],
        "comments": [{"solution_index": 0, "evidence": "Ta có 2x = 4", "reason": reason}],
    }


def global_quality_good() -> dict:
    return {
        "comments": [],
    }


def global_quality_bad() -> dict:
    return {
        "comments": [{
            "error_type": "draft_or_self_questioning",
            "solution_index": 0,
            "evidence": "Ta có 2x = 4",
            "reason": "Lời giải còn đoạn nháp và câu tự vấn.",
            "suggestion": "Loại bỏ đoạn nháp và chỉ giữ lập luận hoàn chỉnh.",
        }],
    }


def resolver_ok() -> dict:
    return {
        "resolver_status": "resolved",
        "final_answer": {
            "text": "x = 2",
            "matched_option_id": "B",
            "correctOptionIds": [],
            "expected": 2,
            "evidence_from_solution": "x = 2.",
        },
        "answerSpec_matches_solution": True,
        "answer_spec_alignment": "matched",
        "fields_to_fix": [],
        "issues": [],
    }


def resolver_mismatch() -> dict:
    return {
        "resolver_status": "resolved",
        "final_answer": {
            "text": "x = 2",
            "matched_option_id": "B",
            "correctOptionIds": [],
            "expected": 2,
            "evidence_from_solution": "x = 2.",
        },
        "answerSpec_matches_solution": False,
        "answer_spec_alignment": "mismatched",
        "fields_to_fix": [
            {
                "path": "/questionItems/0/answerSpecs/0/expected/correctOptionId",
                "value": "B",
                "reason": "answerSpec lệch solution.",
                "suggestion": "Đổi correctOptionId thành B.",
            }
        ],
        "issues": [
            {
                "severity": "bad",
                "category": "solution_anchor_consistency",
                "location": "/questionItems/0/answerSpecs/0/expected/correctOptionId",
                "reason": "answerSpec lệch solution.",
                "suggestion": "Đổi correctOptionId thành B.",
                "repair_intent": "align_fields_to_solution",
            }
        ],
    }


def resolver_equivalent() -> dict:
    result = resolver_ok()
    result["answer_spec_alignment"] = "equivalent"
    return result


def resolver_manual(reason: str) -> dict:
    return {
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
        "issues": [{
            "severity": "needs_review",
            "category": "solution_quality",
            "location": "/solutions",
            "reason": reason,
            "suggestion": "Review thủ công solution.",
            "repair_intent": "needs_manual_review",
        }],
    }


def generated_question(question_id: str = "generated-1") -> dict:
    return {
        "_id": question_id,
        "difficulty": "easy",
        "bloom": "apply",
        "interactionTypes": ["single_choice"],
        "instruction": [{"id": "intro", "type": "text", "text": "Giải phương trình 2x + 3 = 7."}],
        "questionItems": [
            {
                "id": "item-1",
                "stem": [{"id": "stem", "type": "text", "text": "Chọn giá trị đúng của x."}],
                "interactions": [
                    {
                        "id": "choice-1",
                        "type": "single_choice",
                        "config": {
                            "options": [
                                {"id": "A", "content": [{"id": "a", "type": "text", "text": "x = 1"}]},
                                {"id": "B", "content": [{"id": "b", "type": "text", "text": "x = 2"}]},
                            ]
                        },
                        "display": {"layout": "vertical"},
                    }
                ],
                "answerSpecs": [
                    {
                        "interactionId": "choice-1",
                        "type": "single_choice",
                        "expected": {"correctOptionId": "B"},
                    }
                ],
                "hints": [],
            }
        ],
        "solutions": [
            {
                "solverName": "default",
                "solutionContent": [{"id": "solution", "type": "text", "text": "Ta có 2x = 4 ⇒ x = 2."}],
            }
        ],
    }


def coordinate_generated_question() -> dict:
    payload = generated_question("coordinate-question")
    payload["interactionTypes"] = ["coordinate_input"]
    interaction = payload["questionItems"][0]["interactions"][0]
    interaction.update(
        id="coordinate-1",
        type="coordinate_input",
        config={
            "dimensions": 3,
            "countMode": "fixed",
            "defaultInputMode": "numeric",
            "slots": [{"id": "coordinate-slot", "label": "M"}],
        },
        display={"layout": "row"},
    )
    payload["questionItems"][0]["answerSpecs"] = [{
        "interactionId": "coordinate-1",
        "type": "coordinate_input",
        "expected": {
            "coordinates": [{
                "slotId": "coordinate-slot",
                "inputMode": "numeric",
                "components": [1, 3, 4],
            }]
        },
    }]
    return payload


def test_resolver_fix_path_accepts_structured_coordinate_components():
    payload = coordinate_generated_question()
    path = (
        "/questionItems/0/answerSpecs/0/expected/"
        "coordinates/0/components"
    )

    assert _valid_fix_path(payload, path, [1, 3, 5]) is True
    assert get_by_json_pointer(payload, path) == [1, 3, 4]


def test_resolver_fix_path_rejects_missing_outside_or_wrong_typed_values():
    payload = coordinate_generated_question()
    path = (
        "/questionItems/0/answerSpecs/0/expected/"
        "coordinates/0/components"
    )

    assert _valid_fix_path(payload, path + "/9", 5) is False
    assert (
        _valid_fix_path(
            payload,
            "/questionItems/0/interactions/0/config/dimensions",
            2,
        )
        is False
    )
    assert _valid_fix_path(payload, path, "1, 3, 5") is False


def test_resolver_fix_path_revalidates_patched_generated_question():
    payload = generated_question()
    path = "/questionItems/0/answerSpecs/0/expected/correctOptionId"

    assert _valid_fix_path(payload, path, "A") is True
    assert _valid_fix_path(payload, path, "missing-option") is False


def test_resolver_legacy_fix_paths_remain_valid():
    single_choice = generated_question()
    assert _valid_fix_path(
        single_choice,
        "/questionItems/0/answerSpecs/0/expected/correctOptionId",
        "A",
    )

    multiple_choice = deepcopy(single_choice)
    multiple_choice["interactionTypes"] = ["multiple_choice"]
    multiple_choice["questionItems"][0]["interactions"][0]["type"] = "multiple_choice"
    multiple_choice["questionItems"][0]["answerSpecs"][0].update(
        type="multiple_choice",
        expected={"correctOptionIds": ["A"]},
    )
    assert _valid_fix_path(
        multiple_choice,
        "/questionItems/0/answerSpecs/0/expected/correctOptionIds",
        ["A", "B"],
    )

    short_answer = deepcopy(single_choice)
    short_answer["interactionTypes"] = ["short_answer"]
    short_answer["questionItems"][0]["interactions"][0].update(
        type="short_answer",
        config={"inputMode": "numeric"},
        display={"layout": "auto"},
    )
    short_answer["questionItems"][0]["answerSpecs"][0].update(
        type="short_answer",
        expected={"correctValue": 2},
    )
    assert _valid_fix_path(
        short_answer,
        "/questionItems/0/answerSpecs/0/expected/correctValue",
        3,
    )


def test_input_accepts_single_list_and_wrapper():
    question = generated_question()

    assert normalize_generated_question_input(question) == [question]
    assert normalize_generated_question_input([question]) == [question]
    assert normalize_generated_question_input({"generatedQuestions": [question]}) == [question]


def test_single_object_runs_judge_then_resolver_and_returns_compact_output():
    client = RoleClient(
        correctness=[judge_ok()],
        global_quality=[global_quality_good()],
        resolvers=[resolver_ok()],
    )

    result = evaluate_generated_questions(generated_question("direct"), config=fake_config(), client=client)

    assert result == {
        "id": "direct",
        "is_good": True,
        "issues": [],
        "correctness_comments": [],
        "process_comments": [],
        "new_generated_question": None,
    }
    assert len(client.calls) == 3
    assert all(call["model"] == "fast-model" for call in client.calls)


def test_resolver_matched_does_not_trigger_correctness_fallback():
    client = RoleClient(
        correctness=[judge_ok()],
        global_quality=[global_quality_good()],
        resolvers=[resolver_ok()],
    )

    result = evaluate_generated_question_object(
        generated_question("resolver-matched"), config=fake_config(), client=client
    )

    assert result["is_good"] is True
    assert result["solution_anchor_result"]["answer_spec_alignment"] == "matched"
    assert result["correctness_fallback_called"] is False
    assert sum(call["model"] == "reasoning-model" for call in client.calls) == 0


def test_resolver_mismatch_keeps_solution_based_fix_without_correctness_recheck():
    client = RoleClient(
        correctness=[judge_ok(), judge_ok()],
        global_quality=[global_quality_good()],
        resolvers=[resolver_mismatch()],
    )

    result = evaluate_generated_question_object(
        generated_question("resolver-mismatch-fallback-good"),
        config=fake_config(),
        client=client,
    )

    assert result["is_good"] is False
    assert result["issues"][0]["category"] == "solution_anchor_consistency"
    assert result["issues"][0]["suggestion"] == "Đổi correctOptionId thành B."
    assert result["solution_anchor_result"]["answer_spec_alignment"] == "mismatched"
    assert result["solution_anchor_result"]["resolver_status"] == "resolved"
    assert result["solution_anchor_result"]["final_answer"]["text"] == "x = 2"
    assert result["solution_anchor_result"]["fields_to_fix"][0]["value"] == "B"
    assert result["correctness_fallback_called"] is False
    assert result["judge_fallback_reason"] is None
    reasoning_calls = [
        call for call in client.calls
        if call["model"] == "reasoning-model"
    ]
    assert reasoning_calls == []
    primary_correctness_call = next(
        call for call in client.calls
        if call["model"] == "fast-model"
        and "trách nhiệm kiểm tra tính đúng đắn" in call["messages"][0]["content"]
    )
    assert primary_correctness_call["response_format"]["type"] == "json_schema"
    primary_schema = primary_correctness_call["response_format"]["json_schema"]["schema"]
    assert "oneOf" not in primary_schema
    assert list(primary_schema["properties"]) == [
        "opening_checks",
        "comments",
    ]


def test_resolver_mismatch_keeps_fix_and_does_not_consume_second_correctness_candidate():
    client = RoleClient(
        correctness=[judge_ok(), judge_issue("Fallback phát hiện solution sai.")],
        global_quality=[global_quality_good()],
        resolvers=[resolver_mismatch()],
    )

    result = evaluate_generated_question_object(
        generated_question("resolver-mismatch-fallback-bad"),
        config=fake_config(),
        client=client,
    )

    assert result["is_good"] is False
    assert result["issues"][0]["category"] == "solution_anchor_consistency"
    assert result["solution_anchor_result"]["resolver_status"] == "resolved"
    assert result["solution_anchor_result"]["fields_to_fix"]
    assert result["correctness_fallback_called"] is False


def test_resolver_mismatch_does_not_consume_noncanonical_second_candidate():
    client = RoleClient(
        correctness=[judge_ok(), {"verdict": "good"}],
        global_quality=[global_quality_good()],
        resolvers=[resolver_mismatch()],
    )

    result = evaluate_generated_question_object(
        generated_question("resolver-mismatch-fallback-invalid"),
        config=fake_config(),
        client=client,
    )

    assert result["is_good"] is False
    assert result["issues"][0]["category"] == "solution_anchor_consistency"
    assert result["solution_anchor_result"]["resolver_status"] == "resolved"
    assert result["solution_anchor_result"]["fields_to_fix"]
    assert result["correctness_fallback_called"] is False


def test_correctness_runtime_stops_before_resolver_without_semantic_fallback():
    client = RoleClient(
        correctness=[RuntimeError("primary timeout"), judge_ok()],
        global_quality=[global_quality_good()],
        resolvers=[resolver_mismatch()],
    )

    result = evaluate_generated_question_object(
        generated_question("resolver-mismatch-existing-fallback"),
        config=fake_config(),
        client=client,
    )

    assert result["is_good"] is False
    assert result["issues"][0]["category"] == "runtime"
    assert result["solution_anchor_result"] is None
    assert result["correctness_fallback_called"] is False
    correctness_calls = [
        call for call in client.calls
        if "trách nhiệm kiểm tra tính đúng đắn" in call["messages"][0]["content"]
    ]
    assert len(correctness_calls) == 1


def test_process_runtime_enables_retry_and_does_not_report_unrun_review():
    client = RoleClient(
        correctness=[judge_ok()],
        global_quality=[RuntimeError("process timeout")],
        resolvers=[resolver_ok()],
    )

    result = evaluate_generated_question_object(
        generated_question("process-runtime"),
        config=fake_config(),
        client=client,
    )

    process_calls = [
        call for call in client.calls
        if "trách nhiệm xem xét quá trình"
        in call["messages"][0]["content"]
    ]
    assert len(process_calls) == 1
    assert process_calls[0]["retry_transient_once"] is True
    assert result["issues"][0]["category"] == "runtime"
    assert "process timeout" in result["failed_reason"][0]
    assert "comments_review" not in result["failed_reason"][0]
    assert result["solution_anchor_result"] is None
    assert all(
        "trách nhiệm kiểm chứng các nhận xét" not in call["messages"][0]["content"]
        for call in client.calls
    )


def test_resolver_equivalent_does_not_trigger_correctness_fallback():
    client = RoleClient(
        correctness=[judge_ok()],
        global_quality=[global_quality_good()],
        resolvers=[resolver_equivalent()],
    )

    result = evaluate_generated_question_object(
        generated_question("resolver-equivalent"), config=fake_config(), client=client
    )

    assert result["is_good"] is True
    assert result["solution_anchor_result"]["answer_spec_alignment"] == "equivalent"
    assert result["correctness_fallback_called"] is False
    assert sum(call["model"] == "reasoning-model" for call in client.calls) == 0
    debug_public = public_generated_question_result(result, debug=True)
    assert "answer_spec_alignment" not in debug_public["solution_anchor_result"]
    assert debug_public["correctness_fallback_called"] is False


def test_direct_judges_call_correctness_process_and_resolver():
    client = RoleClient(
        correctness=[judge_ok()],
        global_quality=[global_quality_good()],
        resolvers=[resolver_ok()],
    )

    result = evaluate_generated_questions(
        generated_question("missing-context"),
        config=fake_config(),
        client=client,
    )

    assert len(client.calls) == 3
    assert result["is_good"] is True


def test_debug_output_exposes_runtime_detail_without_changing_compact_output():
    internal = {
        "id": "runtime-case",
        "is_good": False,
        "issues": [{
            "severity": "needs_review",
            "category": "runtime",
            "location": "/solutions",
            "reason": "Runtime error hoặc LLM output không hợp lệ.",
            "suggestion": "Chạy lại.",
            "repair_intent": "needs_manual_review",
        }],
        "failed_reason": ["Correctness Judge trả output không canonical."],
    }

    compact = public_generated_question_result(internal, debug=False)
    debug = public_generated_question_result(internal, debug=True)

    assert compact["issues"][0]["reason"] == "Runtime error hoặc LLM output không hợp lệ."
    assert debug["issues"][0]["reason"] == "Correctness Judge trả output không canonical."
    assert debug["failed_reason"] == ["Correctness Judge trả output không canonical."]


def test_wrapper_source_fields_are_not_sent_to_solution_judge():
    question = generated_question()
    wrapper = {
        "question": "raw question",
        "answer": "raw answer",
        "question_plan": {},
        "generatedQuestions": [question],
    }
    client = RoleClient(
        correctness=[judge_ok()],
        global_quality=[global_quality_good()],
        resolvers=[resolver_ok()],
    )

    result = evaluate_generated_questions(wrapper, config=fake_config(), client=client)
    payload = compact_generated_question_payload(question)

    assert result["is_good"] is True
    serialized = json.dumps(payload)
    assert "answerSpecs" not in serialized
    assert "expected" not in serialized
    assert "options" not in serialized
    assert "hints" not in serialized
    assert "interactionTypes" not in payload
    assert "solverName" not in serialized
    assert "question_plan" not in serialized
    assert "raw answer" not in serialized


def test_essay_requirements_reach_both_judges_but_not_resolver():
    question = {
        "_id": "essay-context-test",
        "instruction": [{"type": "text", "text": "Chứng minh mệnh đề bằng quy nạp."}],
        "questionItems": [{
            "id": "essay-item",
            "stem": [{"type": "text", "text": "Trình bày đầy đủ bốn phần."}],
            "interactions": [{
                "id": "essay-response",
                "type": "essay",
                "config": {"minWords": 80, "maxWords": 500},
                "requirements": ["Nêu rõ giả thiết quy nạp."],
            }],
            "rubric": [
                "Kiểm tra bước cơ sở.",
                "Nêu giả thiết và hoàn thành bước quy nạp.",
            ],
        }],
        "solutions": [{
            "solutionContent": [{"type": "text", "text": "Bước cơ sở đúng. Giả sử mệnh đề đúng tại k, rồi chứng minh tại k+1."}],
        }],
    }
    compact = compact_generated_question_payload(question)
    essay = compact["questionItems"][0]["essayInteractions"][0]
    assert essay["config"] == {"minWords": 80, "maxWords": 500}
    assert essay["requirements"] == ["Nêu rõ giả thiết quy nạp."]
    assert compact["questionItems"][0]["rubric"] == question["questionItems"][0]["rubric"]

    correctness_prompt = build_direct_correctness_messages(question)[-1]["content"]
    process_prompt = build_direct_process_presentation_messages(
        question,
    )[-1]["content"]
    for prompt in (correctness_prompt, process_prompt):
        assert '"essayInteractions"' in prompt
        assert '"minWords":80' in prompt
        assert "Kiểm tra bước cơ sở." in prompt
    assert '"type":"essay"' in correctness_prompt
    assert '"rubric"' in process_prompt

    resolver_payload = compact_generated_question_for_solution_anchor(question)
    assert resolver_payload["interaction_contexts"] == []


def test_bad_structure_stops_before_llm():
    question = generated_question()
    del question["questionItems"][0]["interactions"][0]["config"]
    client = SequenceClient([])

    result = evaluate_generated_questions(question, config=fake_config(), client=client)

    assert result["is_good"] is False
    assert any(issue["category"] == "interaction_schema" for issue in result["issues"])
    assert client.calls == []


def test_unknown_answer_option_and_unsupported_type_fail_without_llm():
    bad_option = generated_question("bad-option")
    bad_option["questionItems"][0]["answerSpecs"][0]["expected"]["correctOptionId"] = "Z"
    unsupported = generated_question("bad-type")
    unsupported["questionItems"][0]["interactions"][0]["type"] = "unknown_type"

    for question in (bad_option, unsupported):
        client = SequenceClient([])
        result = evaluate_generated_questions(question, config=fake_config(), client=client)
        assert result["is_good"] is False
        assert client.calls == []


def test_strict_mode_controls_warning_blocking():
    issue = {
        "severity": "warning",
        "category": "solution_quality",
        "location": "/solutions/0/solutionContent/0/text",
        "reason": "Solution dài dòng.",
        "suggestion": "Làm gọn solution.",
        "repair_intent": "clean_solution_reasoning",
    }

    strict = normalize_generated_question_result({"is_good": True, "issues": [issue]}, strict_mode=True)
    relaxed = normalize_generated_question_result({"is_good": False, "issues": [issue]}, strict_mode=False)

    assert strict["is_good"] is False
    assert relaxed["is_good"] is True


def test_llm_contracts_do_not_normalize_explicit_nulls():
    invalid_resolver = SolutionResolverOutput.model_validate_json
    try:
        invalid_resolver(json.dumps({
            "resolver_status": "resolved",
            "final_answer": {
                "text": "2",
                "matched_option_id": None,
                "correctOptionIds": [],
                "expected": 2,
                "evidence_from_solution": "x = 2",
            },
            "answerSpec_matches_solution": True,
            "fields_to_fix": None,
            "issues": None,
        }))
        assert False, "Resolver contract phải từ chối null tại field list"
    except ValueError:
        pass
    repair = ScopedRepairOutput.model_validate({
        "repair_status": "failed",
        "patches": None,
        "failed_reason": None,
        "suggestions": None,
    })
    assert repair.patches == repair.failed_reason == repair.suggestions == []
    assert SpellingOutput.model_validate({"issues": None}).issues == []
    try:
        ResolverFieldFix.model_validate({
            "path": "/questionItems/0/answerSpecs/0/expected",
            "value": 2,
            "reason": None,
            "suggestion": None,
        })
        assert False, "ResolverFieldFix phải từ chối reason/suggestion=null"
    except ValueError:
        pass
    assert SpellingIssue.model_validate({
        "severity": "warning",
        "location": "/instruction/0/text",
        "reason": "Sai chính tả.",
        "suggestion": "Sửa chính tả.",
        "error_snippet": None,
    }).error_snippet == ""


def test_judge_uses_fast_model_and_keeps_concrete_business_issue():
    client = RoleClient(
        correctness=[judge_issue()],
        global_quality=[global_quality_good()],
    )

    result = judge_generated_question_object(
        generated_question(), {"issues": []}, fake_config(), client
    )

    assert len(client.calls) == 3
    assert all(call["model"] == "fast-model" for call in client.calls)
    assert result["judge_model"] == "fast-model"
    assert result["judge_attempt_count"] == 3
    assert result["judge_fallback_called"] is False


def test_no_candidate_comments_skip_review():
    client = RoleClient(correctness=[judge_ok()], global_quality=[global_quality_good()])

    result = judge_generated_question_object(
        generated_question(), {"issues": []}, fake_config(), client
    )

    assert result["is_good"] is True
    assert len(client.calls) == 2
    assert all(
        "trách nhiệm kiểm chứng các nhận xét" not in call["messages"][0]["content"]
        for call in client.calls
    )


def test_judge_only_corrects_mechanical_contract_failure_with_same_model():
    for invalid, responses, expected_calls, expected_correction in (
        (RuntimeError("timeout"), [RuntimeError("timeout"), judge_issue()], 2, 0),
        ({"is_good": True}, [{"is_good": True}, judge_issue()], 4, 1),
    ):
        client = RoleClient(
            correctness=responses,
            global_quality=[global_quality_good()],
        )

        result = judge_generated_question_object(
            generated_question(), {"issues": []}, fake_config(), client
        )

        assert len(client.calls) == expected_calls
        assert sum(call["model"] == "reasoning-model" for call in client.calls) == 0
        assert result["is_good"] is False
        assert result["judge_model"] == "fast-model"
        assert result["judge_attempt_count"] == expected_calls
        assert result["judge_fallback_called"] is False
        assert result["contract_correction_12b_calls"] == expected_correction
        assert result["fallback_26b_calls"] == 0
        assert result["non_canonical_count"] == int(not isinstance(invalid, RuntimeError))


def test_judge_exhausted_runtime_is_not_relabelled_as_contract_failure():
    client = RoleClient(
        correctness=[RuntimeError("fast timeout"), RuntimeError("fallback timeout")],
        global_quality=[global_quality_good()],
    )

    result = judge_generated_question_object(
        generated_question(), {"issues": []}, fake_config(), client
    )

    assert result["is_good"] is False
    assert result["issues"][0]["category"] == "runtime"
    assert result["judge_fallback_called"] is False
    assert result["judge_fallback_reason"] is None
    assert "fast timeout" in result["failed_reason"][0]


def test_judge_exhausted_invalid_contract_stays_solution_quality():
    client = RoleClient(
        correctness=[{"is_good": True}, {"is_good": True}, {"is_good": True}],
        global_quality=[global_quality_good()],
    )

    result = judge_generated_question_object(
        generated_question(), {"issues": []}, fake_config(), client
    )

    assert result["is_good"] is False
    assert result["issues"][0]["category"] == "solution_quality"
    assert result["judge_fallback_called"] is False
    assert result["judge_fallback_reason"] is None


def test_resolver_contract_error_does_not_fallback():
    legacy = resolver_ok()
    legacy["solution_derived_answer"] = legacy.pop("final_answer")
    client = SequenceClient([legacy, resolver_ok()])

    result = resolve_solution_anchor_consistency(
        generated_question(),
        config=fake_config(),
        client=client,
    )

    assert result["resolver_status"] == "needs_manual_review"
    assert result["resolver_fallback_called"] is False
    assert result["resolver_attempt_count"] == 1
    assert result["resolver_fallback_reason"] is None
    assert [call["model"] for call in client.calls] == ["fast-model"]


def test_resolver_runtime_does_not_fallback_or_relabel_error():
    client = SequenceClient([RuntimeError("resolver timeout")])

    result = resolve_solution_anchor_consistency(
        generated_question(),
        config=fake_config(),
        client=client,
    )

    assert result["resolver_status"] == "needs_manual_review"
    assert result["issues"][0]["category"] == "runtime"
    assert result["resolver_fallback_called"] is False
    assert result["resolver_fallback_reason"] is None
    assert result["resolver_runtime_error"] == "resolver timeout"
    assert len(client.calls) == 1


def test_resolver_rejects_null_options_and_missing_evidence():
    raw = resolver_ok()
    raw["final_answer"]["correctOptionIds"] = None
    raw["final_answer"].pop("evidence_from_solution")

    result = normalize_solution_anchor_result(raw, generated_question())

    assert result["resolver_status"] == "needs_manual_review"
    assert result["issues"][0]["category"] == "solution_quality"
    assert result["issues"][0]["repair_intent"] == "needs_manual_review"
    assert "resolver_contract_error" in result


def test_resolver_final_answer_contract_accepts_object_with_required_text():
    validated = SolutionResolverOutput.model_validate(resolver_ok())

    assert validated.final_answer.text == "x = 2"
    assert validated.answer_spec_alignment == "matched"


def test_resolver_recovers_exact_evidence_after_harmless_math_formatting():
    raw = resolver_ok()
    raw["final_answer"]["evidence_from_solution"] = "$x = 2$"

    result = normalize_solution_anchor_result(raw, generated_question())

    assert result["resolver_status"] == "resolved"
    assert result["final_answer"]["evidence_from_solution"] == "Ta có 2x = 4 ⇒ x = 2."


def test_resolver_ignores_ungrounded_optional_evidence_metadata():
    raw = resolver_ok()
    raw["final_answer"]["evidence_from_solution"] = "x = 999"

    result = normalize_solution_anchor_result(raw, generated_question())

    assert result["resolver_status"] == "resolved"
    assert result["final_answer"]["evidence_from_solution"] == ""


def test_resolver_accepts_missing_optional_evidence_metadata():
    raw = resolver_ok()
    raw["final_answer"].pop("evidence_from_solution")

    result = normalize_solution_anchor_result(raw, generated_question())

    assert result["resolver_status"] == "resolved"
    assert result["final_answer"]["evidence_from_solution"] == ""


def test_resolver_ignores_non_string_optional_evidence_metadata():
    raw = resolver_ok()
    raw["final_answer"]["evidence_from_solution"] = {"model_note": "x = 2"}

    result = normalize_solution_anchor_result(raw, generated_question())

    assert result["resolver_status"] == "resolved"
    assert result["final_answer"]["evidence_from_solution"] == ""


def test_resolver_alignment_accepts_equivalent_and_rejects_boolean_contradiction():
    equivalent = normalize_solution_anchor_result(
        resolver_equivalent(), generated_question()
    )
    assert equivalent["answer_spec_alignment"] == "equivalent"
    assert equivalent["answerSpec_matches_solution"] is True

    contradictory = resolver_mismatch()
    contradictory["answer_spec_alignment"] = "equivalent"
    invalid = normalize_solution_anchor_result(contradictory, generated_question())
    assert invalid["resolver_status"] == "needs_manual_review"
    assert "resolver_contract_error" in invalid


def test_resolver_builds_missing_alignment_issue_from_valid_fix():
    raw = resolver_mismatch()
    raw["issues"] = []

    result = normalize_solution_anchor_result(raw, generated_question())

    assert result["resolver_status"] == "resolved"
    assert "resolver_contract_error" not in result
    assert result["answerSpec_matches_solution"] is False
    assert len(result["issues"]) == 1
    issue = result["issues"][0]
    assert issue["severity"] == "bad"
    assert issue["category"] == "solution_anchor_consistency"
    assert issue["location"] == "/questionItems/0/answerSpecs/0/expected/correctOptionId"
    assert issue["reason"] == "answerSpec lệch solution."
    assert issue["suggestion"] == "Đổi correctOptionId thành B."
    assert issue["repair_intent"] == "align_fields_to_solution"


def test_resolver_reanchors_nonexistent_issue_location_to_valid_fix_path():
    raw = resolver_mismatch()
    raw["issues"][0]["location"] = (
        "/questionItems/0/answerSpecs/0/expected/nonexistent"
    )

    result = normalize_solution_anchor_result(raw, generated_question())

    assert result["resolver_status"] == "resolved"
    assert "resolver_contract_error" not in result
    assert result["issues"][0]["location"] == (
        "/questionItems/0/answerSpecs/0/expected/correctOptionId"
    )


def test_resolver_final_answer_contract_rejects_missing_wrong_or_null_text():
    invalid_answers = [
        {},
        "x = 2",
        None,
        {
            "text": None,
            "matched_option_id": "B",
            "correctOptionIds": [],
            "expected": 2,
            "evidence_from_solution": "x = 2.",
        },
    ]

    for final_answer in invalid_answers:
        payload = resolver_ok()
        payload["final_answer"] = final_answer
        try:
            SolutionResolverOutput.model_validate(payload)
            assert False, f"Contract phải từ chối final_answer={final_answer!r}"
        except ValueError:
            pass


def test_resolver_prompt_states_final_answer_invariants_for_shared_model_messages():
    messages = build_solution_anchor_resolver_messages(
        generated_question(),
    )
    prompt = messages[-1]["content"]

    assert "`final_answer` bắt buộc phải là một JSON object" in prompt
    assert "`final_answer` phải có bốn field semantic" in prompt
    assert "`final_answer.text` bắt buộc phải tồn tại" in prompt
    assert "Không được bỏ field `final_answer` hoặc `final_answer.text`" in prompt
    assert "`answer_spec_alignment` phải là `matched`, `equivalent` hoặc `mismatched`" in prompt
    assert "kết luận explicit" in prompt
    assert "đúng đại lượng/interaction được hỏi" in prompt
    assert "không dùng số xuất hiện cuối như một heuristic" in prompt
    assert 'Không dùng chuỗi `"None"`, `"null"` hoặc `"N/A"`' in prompt
    assert "Chỉ trả một JSON object đúng schema" in prompt


def test_invalid_resolver_handoff_fails_closed_without_field_access_error():
    invalid = resolver_ok()
    invalid["final_answer"] = {}

    result = normalize_solution_anchor_result(invalid, generated_question())

    assert result["resolver_status"] == "needs_manual_review"
    assert result["issues"][0]["category"] == "solution_quality"
    assert result["issues"][0]["repair_intent"] == "needs_manual_review"
    assert "resolver_contract_error" in result
    assert "final_answer.text" in result["resolver_contract_error"]


def test_resolver_uses_fast_model_without_fallback_when_resolved():
    client = SequenceClient([resolver_ok()])

    result = resolve_solution_anchor_consistency(
        generated_question(), config=fake_config(), client=client
    )

    assert [call["model"] for call in client.calls] == ["fast-model"]
    assert result["resolver_model"] == "fast-model"
    assert result["resolver_fallback_called"] is False


def test_resolver_keeps_ambiguous_manual_review_without_fallback():
    ambiguous = SequenceClient([resolver_manual("Kết luận mơ hồ, không chắc chắn."), resolver_ok()])
    manual = resolve_solution_anchor_consistency(
        generated_question(), config=fake_config(), client=ambiguous
    )
    assert manual["resolver_status"] == "needs_manual_review"
    assert manual["resolver_fallback_called"] is False
    assert manual["resolver_fallback_reason"] is None
    assert [call["model"] for call in ambiguous.calls] == ["fast-model"]

    missing = SequenceClient([resolver_manual("Solution không có kết luận cuối.")])
    missing_result = resolve_solution_anchor_consistency(
        generated_question(), config=fake_config(), client=missing
    )
    assert missing_result["resolver_status"] == "needs_manual_review"
    assert [call["model"] for call in missing.calls] == ["fast-model"]


def test_resolver_payload_only_includes_options_and_nonempty_hints_when_needed():
    choice_payload = compact_generated_question_for_solution_anchor(generated_question())
    choice_context = choice_payload["interaction_contexts"][0]
    assert choice_context["options"]
    assert "hints" not in choice_context
    assert "instruction" not in choice_payload

    short_answer = generated_question("short")
    interaction = short_answer["questionItems"][0]["interactions"][0]
    interaction["type"] = "short_answer"
    short_answer["questionItems"][0]["answerSpecs"][0]["type"] = "short_answer"
    short_context = compact_generated_question_for_solution_anchor(short_answer)["interaction_contexts"][0]
    assert "options" not in short_context
    assert "hints" not in short_context


def test_batch_summary_tracks_fallback_rate():
    results = [
        {"is_good": True, "issues": [], "judge_attempt_count": 1, "judge_fallback_called": False, "solution_anchor_result": {"resolver_attempt_count": 1, "resolver_fallback_called": False}},
        {"is_good": True, "issues": [], "judge_attempt_count": 2, "judge_fallback_called": True, "solution_anchor_result": {"resolver_attempt_count": 2, "resolver_fallback_called": True}},
    ]

    summary = aggregate_generated_question_results(results)["summary"]

    assert summary["judge_calls"] == 2
    assert summary["judge_fallbacks"] == 1
    assert summary["judge_fallback_rate"] == 0.5
    assert summary["resolver_calls"] == 2
    assert summary["resolver_fallbacks"] == 1
    assert summary["resolver_fallback_rate"] == 0.5


def test_resolver_mismatch_repair_uses_fix_without_mutating_input():
    original = generated_question("repair")
    original["questionItems"][0]["answerSpecs"][0]["expected"]["correctOptionId"] = "A"
    snapshot = deepcopy(original)
    client = RoleClient(
        correctness=[judge_ok(), judge_ok()],
        global_quality=[global_quality_good(), global_quality_good()],
        resolvers=[resolver_mismatch(), resolver_ok()],
    )

    result = evaluate_generated_questions(
        original,
        config=fake_config(),
        client=client,
        auto_repair=True,
        max_loop=1,
        debug=True,
    )

    assert result["new_generated_question"] is not None
    assert (
        result["new_generated_question"]["questionItems"][0]["answerSpecs"][0]
        ["expected"]["correctOptionId"]
        == "B"
    )
    assert original == snapshot
    assert result["repair_status"] == "repaired"


def test_scoped_repair_rejects_source_path():
    result = normalize_scoped_repair_result(
        {
            "repair_status": "repaired",
            "patches": [{"op": "replace", "path": "/question", "value": "new"}],
        },
        generated_question=generated_question(),
        repair_intent="fix_schema",
    )

    assert result["repair_status"] == "needs_manual_review"
    assert result["new_generated_question"] is None


def test_json_patch_is_immutable():
    original = {"items": [{"value": 1}]}
    patched = apply_json_patch(original, [{"op": "replace", "path": "/items/0/value", "value": 2}])

    assert get_by_json_pointer(patched, "/items/0/value") == 2
    assert get_by_json_pointer(original, "/items/0/value") == 1


def test_math_segments_round_trip():
    text = "Giải $x^2 = 4$ rồi kết luận."
    protected, segments = protect_math_segments(text)

    assert "$x^2 = 4$" not in protected
    assert restore_math_segments(protected, segments) == text


def test_report_and_repaired_object_extraction():
    repaired = generated_question("fixed")
    result = {
        "is_good": False,
        "summary": {"total": 1},
        "results": [
            {
                "id": "fixed",
                "is_good": False,
                "issues": [
                    {
                        "severity": "bad",
                        "category": "solution_anchor_consistency",
                        "reason": "answerSpec lệch solution.",
                        "suggestion": "Căn lại answerSpec.",
                    }
                ],
                "correctness_comments": [{
                    "solution_index": 0,
                    "evidence": "2x^3 = 18",
                    "reason": "Phép biến đổi sai.",
                    "review_disposition": "rejected",
                }],
                "process_comments": [{
                    "solution_index": 0,
                    "error_type": "missing_major_step",
                    "evidence": "2x^3 = 16",
                    "reason": "Thiếu bước chuyển vế.",
                    "suggestion": "Bổ sung bước.",
                    "review_disposition": "blocking",
                }],
                "new_generated_question": repaired,
            }
        ],
    }

    report = format_generated_question_markdown(result, source_name="input.json")

    assert "solution_anchor_consistency(bad)" in report
    assert "Nhận xét Correctness" in report
    assert "[rejected] — “2x^3 = 18” — Phép biến đổi sai." in report
    assert "[blocking] — [missing_major_step] — “2x^3 = 16” — Thiếu bước chuyển vế." in report
    assert extract_repaired_generated_questions(result) == [repaired]


def test_judge_calls_three_specialized_gemma():
    client = RoleClient(
        correctness=[judge_issue("Phép biến đổi chưa hợp lệ.")],
        global_quality=[global_quality_good()],
    )

    result = judge_generated_question_object(
        generated_question(), {"issues": []}, fake_config(gemma_runs=2), client
    )

    assert result["judge_gemma_run_count"] == 3
    assert result["judge_fallback_called"] is False
    assert len(client.calls) == 3
    assert all(call["model"] == "fast-model" for call in client.calls)


def test_judge_contract_failure_never_switches_to_reasoning_model():
    for invalid, responses, expected_calls in (
        (
            {"is_good": True},
            [{"is_good": True}, {"is_good": True}, judge_issue()],
            3,
        ),
        (RuntimeError("timeout"), [RuntimeError("timeout"), judge_issue()], 2),
    ):
        client = RoleClient(
            correctness=responses,
            global_quality=[global_quality_good()],
        )
        result = judge_generated_question_object(
            generated_question(), {"issues": []}, fake_config(gemma_runs=2), client
        )
        assert result["is_good"] is False
        assert result["judge_fallback_called"] is False
        assert len(client.calls) == expected_calls
        assert sum(call["model"] == "reasoning-model" for call in client.calls) == 0


def test_process_presentation_contract_failure_is_corrected_once_with_same_model():
    client = RoleClient(
        correctness=[judge_ok()],
        global_quality=[{"verdict": "good"}, global_quality_bad()],
    )
    result = judge_generated_question_object(
        generated_question(), {"issues": []}, fake_config(), client
    )

    assert result["is_good"] is False
    assert result["issues"][0]["category"] == "solution_quality"
    assert result["judge_fallback_called"] is False
    assert len(client.calls) == 4
    assert result["contract_correction_12b_calls"] == 1
    correction_prompt = next(
        call["messages"][-1]["content"]
        for call in client.calls
        if "SỬA HỢP ĐỒNG ĐẦU RA" in call["messages"][-1]["content"]
    )
    assert "SỬA HỢP ĐỒNG ĐẦU RA" in correction_prompt
    assert "raw_candidate" in correction_prompt
    assert sum(call["model"] == "reasoning-model" for call in client.calls) == 0


def test_resolver_calls_gemma_once():
    client = SequenceClient([resolver_ok()])

    result = resolve_solution_anchor_consistency(
        generated_question(), config=fake_config(gemma_runs=2), client=client
    )

    assert result["resolver_gemma_run_count"] == 1
    assert result["resolver_fallback_called"] is False
    assert [call["model"] for call in client.calls] == ["fast-model"]


def test_resolver_contract_failure_stops_after_primary_call():
    client = SequenceClient([{"resolver_status": "resolved"}, resolver_ok()])
    result = resolve_solution_anchor_consistency(
        generated_question(), config=fake_config(gemma_runs=2), client=client
    )
    assert result["resolver_status"] == "needs_manual_review"
    assert result["resolver_fallback_called"] is False
    assert result["resolver_fallback_reason"] is None
    assert [call["model"] for call in client.calls] == ["fast-model"]


def test_specialized_gemma_batch_caps_calls_at_four_and_preserves_order():
    class ConcurrentClient:
        def __init__(self):
            self.barrier = Barrier(2)
            self.lock = Lock()
            self.active = 0
            self.max_active = 0
            self.calls = []

        def chat_completion(self, **kwargs):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.calls.append(kwargs)
            try:
                self.barrier.wait(timeout=2)
                system_prompt = kwargs["messages"][0]["content"]
                if "Solution Resolver" in system_prompt:
                    payload = resolver_ok()
                elif "trách nhiệm kiểm chứng các nhận xét" in system_prompt:
                    payload = {"reviewed_comments": []}
                elif "trách nhiệm xem xét quá trình" in system_prompt:
                    payload = global_quality_good()
                else:
                    payload = judge_ok()
                return {"content": json.dumps(payload, ensure_ascii=False), "latency_seconds": 0}
            finally:
                with self.lock:
                    self.active -= 1

    first, second = generated_question("first"), generated_question("second")
    client = ConcurrentClient()
    result = evaluate_generated_questions(
        [first, second], config=fake_config(gemma_runs=2), client=client, workers=4
    )

    assert 2 <= client.max_active <= 4
    assert 5 <= len(client.calls) <= 6
    assert [item["id"] for item in result["results"]] == ["first", "second"]

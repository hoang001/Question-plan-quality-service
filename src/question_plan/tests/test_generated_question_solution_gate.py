import json
from copy import deepcopy
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

from src.question_plan.flows import generated_question_service as service
from src.question_plan.benchmarks.grade_solution_fixtures import (
    load_grade_solution_expected_cases,
    load_grade_solution_fixtures,
)
from src.question_plan.logic.generated_question_judge import (
    PROCESS_PRESENTATION_CRITERIA,
    build_direct_correctness_messages,
)
from src.question_plan.logic.generated_question_schema import merge_generated_question_results


def generated_question() -> dict:
    return {
        "_id": "gate-test",
        "instruction": [{"id": "intro", "type": "text", "text": "Giải phương trình 2x + 3 = 7."}],
        "questionItems": [
            {
                "id": "item",
                "stem": [{"id": "stem", "type": "text", "text": "Nhập x."}],
                "interactions": [{"id": "x", "type": "short_answer", "config": {}, "display": {}}],
                "answerSpecs": [{"interactionId": "x", "type": "short_answer", "expected": {"correctValue": 2}}],
            }
        ],
        "solutions": [{"solverName": "default", "solutionContent": [{"id": "s", "type": "text", "text": "Suy ra x = 2."}]}],
    }


def solution_issue(intent: str) -> dict:
    return {
        "severity": "bad" if intent == "needs_manual_review" else "warning",
        "category": "solution_quality",
        "location": "/solutions/0/solutionContent/0/text",
        "reason": "Solution cần xử lý trước.",
        "suggestion": "Xử lý solution trước.",
        "repair_intent": intent,
        "disposition": "blocking" if intent == "needs_manual_review" else "advisory",
    }


def test_judge_prompt_contains_original_solution_directly():
    messages = build_direct_correctness_messages(generated_question())
    prompt = messages[-1]["content"]
    assert "trách nhiệm kiểm tra tính đúng đắn" in messages[0]["content"]
    assert "Suy ra x = 2." in prompt
    assert "transition_id" not in prompt
    assert prompt.count("### TIÊU CHÍ VÀ YÊU CẦU KIỂM TRA") == 1
    assert prompt.count("### ĐỀ BÀI VÀ LỜI GIẢI NGUYÊN VĂN") == 1
    assert '"comments"' in prompt
    assert '"solution_index"' in prompt
    assert "Những vấn đề này thuộc kiểm tra quá trình và trình bày" in prompt






def test_presentation_criteria_does_not_require_a_separate_conclusion():
    criteria = PROCESS_PRESENTATION_CRITERIA

    assert "Không báo lỗi chỉ vì thiếu một câu kết luận riêng" in criteria
    assert "thiếu kết luận;" not in criteria
    assert "Đọc toàn bộ solution theo thứ tự" in criteria
    assert "Không tự xác minh hay phán quyết đúng sai toán học" in criteria
    assert "phép tính con" in criteria
    assert "Không báo lỗi nếu chỉ thiếu phép tính con" in criteria
    assert "Mỗi comment phải có `solution_index`" in criteria
    assert "Không kết luận toàn bộ lời giải tốt hay xấu" in criteria


def test_one_operation_benchmark_oracle_marks_condensed_cases_bad():
    root = Path(__file__).resolve().parents[3]
    fixtures = load_grade_solution_fixtures(root)
    oracle = load_grade_solution_expected_cases(root)
    fixture_ids = {case["_id"] for case in fixtures}
    expected = {case["id"]: case["expected"] for case in oracle}

    for case_id in (
        "grade-7-pipeline-missing-major-step",
        "grade-9-verbal-intermediate-jump-invalid",
        "grade-12-pipeline-integral-missing-major-step",
    ):
        assert case_id in fixture_ids
        assert expected[case_id]["correctness"]["verdict"] == "good"
        assert expected[case_id]["process_presentation"]["verdict"] == "bad"
        assert expected[case_id]["process_presentation"]["error_type"] == "missing_major_step"
        assert expected[case_id]["aggregate"]["selected_source"] == "process_presentation"
        assert expected[case_id]["resolver_should_run"] is False


def test_solution_benchmark_has_complete_main_error_oracle():
    root = Path(__file__).resolve().parents[3]
    fixtures = load_grade_solution_fixtures(root)
    expected_cases = load_grade_solution_expected_cases(root)
    fixtures_by_id = {case["_id"]: case for case in fixtures}
    expected_by_id = {case["id"]: case for case in expected_cases}

    assert len(fixtures) == len(fixtures_by_id)
    assert len(fixtures) >= 370
    assert set(fixtures_by_id) == set(expected_by_id)

    for case_id, oracle in expected_by_id.items():
        expected = oracle["expected"]
        aggregate = expected["aggregate"]
        if aggregate["verdict"] == "good":
            continue

        is_pipeline_contract = "pipeline_contract" in oracle.get("tags", [])
        if not is_pipeline_contract and "main_error" not in oracle:
            continue

        main_error = oracle["main_error"]
        selected_source = aggregate["selected_source"]
        selected_stage = expected[selected_source]
        selected_error_type = aggregate.get(
            "selected_error_type"
        ) or selected_stage.get("error_type")

        assert selected_error_type == selected_stage["error_type"]
        assert main_error["reason"].strip()
        assert main_error["expected_correction"].strip()
        if selected_stage.get("source_path"):
            assert selected_stage["source_path"].startswith("/")


def test_solution_benchmark_covers_requested_complex_math_families():
    root = Path(__file__).resolve().parents[3]
    expected_cases = load_grade_solution_expected_cases(root)
    expected_by_id = {case["id"]: case for case in expected_cases}
    families = {
        "rational_expression": (
            "grade-8-rational-expression-valid",
            "grade-8-rational-expression-missing-domain",
        ),
        "radical": (
            "grade-9-square-root-valid",
            "grade-9-square-root-missing-absolute-value",
            "grade-9-pipeline-extraneous-radical-solution",
        ),
        "inequality": (
            "grade-9-inequality-valid",
            "grade-9-inequality-no-sign-reversal",
            "grade-10-linear-inequality-point-valid",
            "grade-10-strict-inequality-includes-boundary",
        ),
        "expression_transformation": (
            "grade-8-polynomial-multiplication-valid",
            "grade-8-polynomial-wrong-like-terms",
            "grade-8-identity-factorization-valid",
            "grade-8-identity-missing-middle-term",
        ),
    }

    for case_ids in families.values():
        family = [expected_by_id[case_id] for case_id in case_ids]
        verdicts = {
            case["expected"]["aggregate"]["verdict"]
            for case in family
        }
        assert len(family) >= 2
        assert verdicts == {"good", "bad"}

    for case_id in (
        "grade-7-pipeline-missing-major-step",
        "grade-12-pipeline-integral-missing-major-step",
    ):
        case = expected_by_id[case_id]
        assert case["expected"]["correctness"]["verdict"] == "good"
        assert case["expected"]["process_presentation"]["error_type"] == "missing_major_step"
        assert case["expected"]["resolver_should_run"] is False


def test_expected_does_not_treat_missing_separate_conclusion_as_an_error():
    root = Path(__file__).resolve().parents[3]
    expected = {
        case["id"]: case["expected"]
        for case in load_grade_solution_expected_cases(root)
    }

    incorrect_without_separate_conclusion = expected[
        "grade-12-pipeline-first-error-before-later-error"
    ]
    assert (
        incorrect_without_separate_conclusion["correctness"]["error_type"]
        == "calculation_error"
    )
    assert (
        incorrect_without_separate_conclusion["process_presentation"]["verdict"]
        == "good"
    )

    valid_without_separate_conclusion = expected["grade-8-linear-equation-valid"]
    assert valid_without_separate_conclusion["correctness"]["verdict"] == "good"
    assert valid_without_separate_conclusion["aggregate"]["verdict"] == "good"
    assert valid_without_separate_conclusion["resolver_should_run"] is True


def test_batch_workers_run_questions_concurrently_and_preserve_input_order(monkeypatch):
    both_workers_started = Barrier(2)
    progress = []

    def evaluate_one(question, **kwargs):
        both_workers_started.wait(timeout=2)
        return {
            "id": question["_id"],
            "is_good": True,
            "issues": [],
            "failed_reason": [],
            "suggestions": [],
            "new_generated_question": None,
        }

    first = generated_question()
    first["_id"] = "first"
    second = deepcopy(first)
    second["_id"] = "second"
    monkeypatch.setattr(service, "evaluate_generated_question_object", evaluate_one)

    def record_progress(current, total, question):
        progress.append((current, total, question["_id"]))

    result = service.evaluate_generated_questions(
        [first, second],
        workers=2,
        progress_callback=record_progress,
    )

    assert [item["id"] for item in result["results"]] == ["first", "second"]
    assert sorted(item[:2] for item in progress) == [(1, 2), (2, 2)]
    assert service.clamp_generated_question_workers(99) == 2


def test_judge_runs_before_resolver_when_solution_passes(monkeypatch):
    calls = []

    def judge(*args, **kwargs):
        calls.append("judge")
        return {
            "is_good": True,
            "issues": [],
            "failed_reason": [],
            "suggestions": [],
        }

    def resolver(*args, **kwargs):
        calls.append("resolver")
        return {
            "resolver_status": "resolved",
                "final_answer": {"text": "x = 2", "matched_option_id": None, "correctOptionIds": [], "expected": 2},
                "answerSpec_matches_solution": True,
                "answer_spec_alignment": "matched",
                "fields_to_fix": [],
            "issues": [],
        }

    monkeypatch.setattr(service, "judge_generated_question_object", judge)
    monkeypatch.setattr(service, "resolve_solution_anchor_consistency", resolver)

    result = service.evaluate_generated_question_object(generated_question(), config=SimpleNamespace(), client=object())

    assert calls == ["judge", "resolver"]
    assert result["solution_anchor_result"]["resolver_status"] == "resolved"


def test_none_judge_handoff_fails_closed_and_never_calls_resolver(monkeypatch):
    calls = []
    monkeypatch.setattr(service, "judge_generated_question_object", lambda *a, **k: None)
    monkeypatch.setattr(
        service,
        "resolve_solution_anchor_consistency",
        lambda *a, **k: calls.append("resolver"),
    )

    result = service.evaluate_generated_question_object(
        generated_question(), config=SimpleNamespace(), client=object()
    )

    assert calls == []
    assert result["is_good"] is False
    assert result["issues"][0]["category"] == "runtime"


def test_merge_none_judge_result_is_not_treated_as_good():
    result = merge_generated_question_results(
        schema_issues=[],
        llm_result=None,
        strict_mode=True,
        generated_question=generated_question(),
    )

    assert result["is_good"] is False
    assert result["issues"][0]["category"] == "runtime"


def test_invalid_resolver_handoff_fails_closed_without_reading_nested_fields(monkeypatch):
    monkeypatch.setattr(
        service,
        "judge_generated_question_object",
        lambda *a, **k: {
            "is_good": True,
            "issues": [],
            "failed_reason": [],
            "suggestions": [],
        },
    )
    monkeypatch.setattr(
        service,
        "resolve_solution_anchor_consistency",
        lambda *a, **k: {"resolver_status": "resolved", "final_answer": None},
    )

    result = service.evaluate_generated_question_object(
        generated_question(), config=SimpleNamespace(), client=object()
    )

    assert result["is_good"] is False
    assert result["issues"][0]["category"] == "runtime"
    assert result["solution_anchor_result"] is None


def test_manual_solution_issue_blocks_resolver(monkeypatch):
    calls = []

    def judge(*args, **kwargs):
        calls.append("judge")
        return {"is_good": False, "issues": [solution_issue("needs_manual_review")]}

    def resolver(*args, **kwargs):
        calls.append("resolver")
        raise AssertionError("resolver must be blocked")

    monkeypatch.setattr(service, "judge_generated_question_object", judge)
    monkeypatch.setattr(service, "resolve_solution_anchor_consistency", resolver)

    result = service.evaluate_generated_question_object(generated_question(), config=SimpleNamespace(), client=object())

    assert calls == ["judge"]
    assert result["solution_anchor_result"] is None


def test_runtime_issue_blocks_resolver_and_repair(monkeypatch):
    calls = []

    def judge(*args, **kwargs):
        calls.append("judge")
        return {
            "is_good": False,
            "issues": [{
                "severity": "needs_review",
                "category": "runtime",
                "location": "/solutions",
                "reason": "Judge lỗi.",
                "suggestion": "Chạy lại.",
                "repair_intent": "needs_manual_review",
            }],
        }

    monkeypatch.setattr(service, "judge_generated_question_object", judge)
    monkeypatch.setattr(service, "resolve_solution_anchor_consistency", lambda *a, **k: calls.append("resolver"))
    monkeypatch.setattr(service, "repair_generated_question_scoped", lambda *a, **k: calls.append("repair"))

    result = service.evaluate_generated_question_object(
        generated_question(), config=SimpleNamespace(), client=object(), auto_repair=True
    )

    assert calls == ["judge"]
    assert result["repair_status"] == "needs_manual_review"


def test_blocking_alignment_has_priority_over_advisory_cleanup():
    cleanup = solution_issue("clean_solution_reasoning")
    alignment = {
        "severity": "bad",
        "category": "solution_anchor_consistency",
        "location": "/questionItems/0/answerSpecs/0/expected",
        "reason": "answerSpec lệch solution.",
        "suggestion": "Căn answerSpec.",
        "repair_intent": "align_fields_to_solution",
        "disposition": "blocking",
    }

    assert service.select_repair_issue({"issues": [alignment, cleanup]}) == alignment


def test_manual_solution_issue_is_not_auto_repaired():
    result = service.repair_once(
        generated_question(),
        {"issues": [solution_issue("needs_manual_review")]},
        config=SimpleNamespace(),
        client=object(),
        index=0,
        debug=False,
    )

    assert result["repair_status"] == "needs_manual_review"
    assert result["new_generated_question"] is None

from types import SimpleNamespace

from src.question_plan.flows import generated_question_service as service
from src.question_plan.logic.generated_question_judge import (
    build_generated_question_judge_messages,
    compact_generated_question_payload,
)


def generated_question() -> dict:
    return {
        "_id": "solution-gate-test",
        "aiId": "ai-test",
        "difficulty": "easy",
        "bloom": "apply",
        "interactionTypes": ["short_answer"],
        "instruction": [{"id": "intro", "type": "text", "text": "Giải phương trình."}],
        "questionItems": [
            {
                "id": "item",
                "stem": [{"id": "stem", "type": "text", "text": "Nhập x."}],
                "interactions": [
                    {
                        "id": "x",
                        "type": "short_answer",
                        "config": {"inputMode": "numeric"},
                        "display": {"layout": "auto"},
                    }
                ],
                "answerSpecs": [
                    {
                        "interactionId": "x",
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
                "solutionContent": [{"id": "sol", "type": "text", "text": "Ta có x = 2."}],
            }
        ],
    }


def solution_issue(intent: str) -> dict:
    return {
        "severity": "warning",
        "category": "solution_quality",
        "location": "/solutions/0/solutionContent/0/text",
        "reason": "Solution cần được xử lý trước.",
        "suggestion": "Xử lý solution trước các field phụ thuộc.",
        "repair_intent": intent,
    }


def test_judge_prompt_requires_local_transition_audit_without_answer_bias():
    messages = build_generated_question_judge_messages(
        generated_question(),
        "criteria",
        "output schema",
    )
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]

    assert "Được phép và bắt buộc sử dụng kiến thức toán học" in system_prompt
    assert "Không đánh giá answerSpec, option hoặc hint" in system_prompt
    assert "kiểm tra từng mệnh đề từ trên xuống" in user_prompt
    assert "tính lại cục bộ" in user_prompt

    payload = compact_generated_question_payload(generated_question())
    serialized_payload = str(payload)
    assert "answerSpecs" not in serialized_payload
    assert "expected" not in serialized_payload
    assert "options" not in serialized_payload
    assert "hints" not in serialized_payload


def test_judge_runs_before_resolver_when_solution_passes(monkeypatch):
    calls: list[str] = []

    def fake_judge(*args, **kwargs):
        calls.append("judge")
        return {"is_good": True, "issues": []}

    def fake_resolver(*args, **kwargs):
        calls.append("resolver")
        return {
            "resolver_status": "resolved",
            "final_answer": {"text": "2", "correctOptionIds": []},
            "answerSpec_matches_solution": True,
            "fields_to_fix": [],
            "issues": [],
        }

    monkeypatch.setattr(service, "judge_generated_question_object", fake_judge)
    monkeypatch.setattr(service, "resolve_solution_anchor_consistency", fake_resolver)

    result = service.evaluate_generated_question_object(
        generated_question(),
        config=SimpleNamespace(),
        client=object(),
    )

    assert calls == ["judge", "resolver"]
    assert result["solution_anchor_result"]["resolver_status"] == "resolved"


def test_solution_issue_blocks_resolver(monkeypatch):
    calls: list[str] = []

    def fake_judge(*args, **kwargs):
        calls.append("judge")
        return {"is_good": False, "issues": [solution_issue("needs_manual_review")]}

    def unexpected_resolver(*args, **kwargs):
        calls.append("resolver")
        raise AssertionError("Resolver must not run while solution is blocked.")

    monkeypatch.setattr(service, "judge_generated_question_object", fake_judge)
    monkeypatch.setattr(service, "resolve_solution_anchor_consistency", unexpected_resolver)

    result = service.evaluate_generated_question_object(
        generated_question(),
        config=SimpleNamespace(),
        client=object(),
    )

    assert calls == ["judge"]
    assert result["solution_anchor_result"] is None
    assert result["issues"][0]["repair_intent"] == "needs_manual_review"


def test_solution_cleanup_is_selected_before_downstream_alignment():
    cleanup = solution_issue("clean_solution_reasoning")
    alignment = {
        "severity": "bad",
        "category": "solution_anchor_consistency",
        "location": "/questionItems/0/answerSpecs/0/expected",
        "reason": "answerSpec lệch solution.",
        "suggestion": "Căn chỉnh answerSpec.",
        "repair_intent": "align_fields_to_solution",
    }

    assert service.select_repair_issue({"issues": [alignment, cleanup]}) == cleanup


def test_repair_once_does_not_apply_anchor_fix_before_solution_cleanup(monkeypatch):
    cleanup = solution_issue("clean_solution_reasoning")
    alignment = {
        "severity": "bad",
        "category": "solution_anchor_consistency",
        "location": "/questionItems/0/answerSpecs/0/expected",
        "reason": "answerSpec lệch solution.",
        "suggestion": "Căn chỉnh answerSpec.",
        "repair_intent": "align_fields_to_solution",
    }
    selected: list[dict] = []

    def fake_scoped_repair(question, check_result, issue, config, client, **kwargs):
        selected.append(issue)
        return {"repair_status": "failed", "new_generated_question": None, "patches": []}

    monkeypatch.setattr(service, "repair_generated_question_scoped", fake_scoped_repair)
    result = service.repair_once(
        generated_question(),
        {
            "issues": [alignment, cleanup],
            "solution_anchor_result": {
                "resolver_status": "resolved",
                "fields_to_fix": [
                    {
                        "path": "/questionItems/0/answerSpecs/0/expected",
                        "value": [],
                    }
                ],
            },
        },
        config=SimpleNamespace(),
        client=object(),
        index=0,
        debug=False,
    )

    assert result["repair_status"] == "failed"
    assert selected == [cleanup]

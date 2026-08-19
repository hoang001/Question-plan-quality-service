import json
from copy import deepcopy
from types import SimpleNamespace

from src.question_plan.flows import generated_question_service as service


class RepairClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"content": json.dumps(self.payload, ensure_ascii=False)}


def config():
    return SimpleNamespace(
        primary_judge_model="fast-model",
        fallback_judge_model="reasoning-model",
    )


def generated_question() -> dict:
    return {
        "_id": "repair-policy",
        "instruction": [{"id": "intro", "type": "text", "text": "Giải phương trình 2x + 3 = 7."}],
        "questionItems": [
            {
                "id": "item",
                "stem": [{"id": "stem", "type": "text", "text": "Nhập x."}],
                "interactions": [
                    {"id": "x", "type": "short_answer", "config": {}, "display": {}}
                ],
                "answerSpecs": [
                    {
                        "interactionId": "x",
                        "type": "short_answer",
                        "expected": {"correctValue": 2},
                    }
                ],
                "hints": [
                    {
                        "name": "Gợi ý 1",
                        "content": [{"id": "hint", "type": "text", "text": "Chuyển 3 sang vế phải."}],
                    }
                ],
            }
        ],
        "solutions": [
            {
                "solverName": "default",
                "solutionContent": [
                    {"id": "solution", "type": "text", "text": "2x = 3, suy ra x = 2."}
                ],
            }
        ],
    }


def blocking_issue(intent: str, *, category: str = "solution_quality", location: str = "/solutions/0") -> dict:
    return {
        "severity": "bad",
        "disposition": "blocking",
        "category": category,
        "location": location,
        "reason": "Issue đã được Review xác nhận.",
        "suggestion": "Sửa đúng phạm vi.",
        "repair_intent": intent,
    }


def repair_payload(path: str, value):
    return {
        "patches": [
            {
                "op": "replace",
                "path": path,
                "value": value,
                "reason": "Sửa issue đã được Review xác nhận.",
            }
        ]
    }


def test_wrong_solution_with_correct_answer_spec_repairs_only_solution():
    question = generated_question()
    answer_spec = deepcopy(question["questionItems"][0]["answerSpecs"])
    client = RepairClient(
        repair_payload(
            "/solutions/0/solutionContent/0/text",
            "2x = 4, suy ra x = 2.",
        )
    )

    result = service.repair_once(
        question,
        {"issues": [blocking_issue("fix_solution_correctness")]},
        config=config(),
        client=client,
        index=0,
        debug=False,
    )

    assert result["repair_status"] == "repaired"
    assert result["new_generated_question"]["solutions"][0]["solutionContent"][0]["text"] == "2x = 4, suy ra x = 2."
    assert result["new_generated_question"]["questionItems"][0]["answerSpecs"] == answer_spec
    assert len(client.calls) == 1


def test_verified_solution_with_wrong_answer_spec_repairs_only_answer_spec():
    question = generated_question()
    original_solution = deepcopy(question["solutions"])
    issue = blocking_issue(
        "align_fields_to_solution",
        category="solution_anchor_consistency",
        location="/questionItems/0/answerSpecs/0/expected",
    )
    check_result = {
        "issues": [issue],
        "solution_anchor_result": {
            "resolver_status": "resolved",
            "fields_to_fix": [
                {
                    "path": "/questionItems/0/answerSpecs/0/expected",
                    "value": {"correctValue": 3},
                    "reason": "answerSpec lệch solution đã xác minh.",
                }
            ],
        },
    }

    result = service.repair_once(
        question,
        check_result,
        config=config(),
        client=RepairClient({}),
        index=0,
        debug=False,
    )

    assert result["repair_status"] == "repaired"
    assert result["new_generated_question"]["questionItems"][0]["answerSpecs"][0]["expected"] == {"correctValue": 3}
    assert result["new_generated_question"]["solutions"] == original_solution


def test_answer_spec_is_not_repaired_before_resolver_verifies_solution():
    issue = blocking_issue(
        "align_fields_to_solution",
        category="solution_anchor_consistency",
        location="/questionItems/0/answerSpecs/0/expected",
    )
    client = RepairClient({})

    result = service.repair_once(
        generated_question(),
        {"issues": [issue], "solution_anchor_result": None},
        config=config(),
        client=client,
        index=0,
        debug=False,
    )

    assert result["repair_status"] == "needs_manual_review"
    assert result["new_generated_question"] is None
    assert client.calls == []


def test_verified_hint_mismatch_repairs_only_target_hint():
    question = generated_question()
    original_solution = deepcopy(question["solutions"])
    original_answer_specs = deepcopy(question["questionItems"][0]["answerSpecs"])
    issue = blocking_issue(
        "align_hint_to_solution",
        category="hint_quality",
        location="/questionItems/0/hints/0",
    )
    client = RepairClient(
        repair_payload(
            "/questionItems/0/hints/0/content/0/text",
            "Trừ 3 ở cả hai vế.",
        )
    )

    result = service.repair_once(
        question,
        {
            "issues": [issue],
            "solution_anchor_result": {"resolver_status": "resolved", "fields_to_fix": []},
        },
        config=config(),
        client=client,
        index=0,
        debug=False,
    )

    repaired = result["new_generated_question"]
    assert result["repair_status"] == "repaired"
    assert repaired["questionItems"][0]["hints"][0]["content"][0]["text"] == "Trừ 3 ở cả hai vế."
    assert repaired["questionItems"][0]["answerSpecs"] == original_answer_specs
    assert repaired["solutions"] == original_solution


def test_process_missing_step_only_expands_solution():
    question = generated_question()
    untouched_question_items = deepcopy(question["questionItems"])
    client = RepairClient(
        repair_payload(
            "/solutions/0/solutionContent/0/text",
            "Từ 2x + 3 = 7 suy ra 2x = 4, nên x = 2.",
        )
    )

    result = service.repair_once(
        question,
        {"issues": [blocking_issue("fix_solution_process")]},
        config=config(),
        client=client,
        index=0,
        debug=False,
    )

    assert result["repair_status"] == "repaired"
    assert "2x = 4" in result["new_generated_question"]["solutions"][0]["solutionContent"][0]["text"]
    assert result["new_generated_question"]["questionItems"] == untouched_question_items


def test_noncanonical_and_missing_image_are_not_repaired():
    client = RepairClient({})
    noncanonical = {
        "severity": "needs_review",
        "category": "runtime",
        "location": "/solutions",
        "reason": "Judge non-canonical.",
        "suggestion": "Chạy lại.",
        "repair_intent": "needs_manual_review",
    }
    missing_image = blocking_issue("fix_solution_correctness")
    missing_image["reason"] = "Thiếu hình ảnh nên không đủ context."

    for issue in (noncanonical, missing_image):
        result = service.repair_once(
            generated_question(),
            {"issues": [issue]},
            config=config(),
            client=client,
            index=0,
            debug=False,
        )
        assert result["repair_status"] == "needs_manual_review"
        assert result["new_generated_question"] is None
    assert client.calls == []


def test_warning_and_advisory_are_not_repaired():
    client = RepairClient({})
    issue = blocking_issue("fix_solution_process")
    issue.update(severity="warning", disposition="advisory")

    result = service.repair_once(
        generated_question(),
        {"issues": [issue]},
        config=config(),
        client=client,
        index=0,
        debug=False,
    )

    assert result["repair_status"] == "needs_manual_review"
    assert result["new_generated_question"] is None
    assert client.calls == []


def test_patch_outside_allowed_paths_is_rejected():
    client = RepairClient(
        repair_payload(
            "/questionItems/0/answerSpecs/0/expected",
            {"correctValue": 9},
        )
    )

    result = service.repair_once(
        generated_question(),
        {"issues": [blocking_issue("fix_solution_correctness")]},
        config=config(),
        client=client,
        index=0,
        debug=False,
    )

    assert result["repair_status"] == "needs_manual_review"
    assert result["new_generated_question"] is None


def test_patch_that_creates_new_issue_is_rolled_back(monkeypatch):
    original = generated_question()
    candidate = deepcopy(original)
    candidate["solutions"][0]["solutionContent"][0]["text"] = "Nội dung mới nhưng sai."
    target = blocking_issue("fix_solution_correctness")
    new_issue = blocking_issue("fix_solution_process")
    monkeypatch.setattr(
        service,
        "repair_once",
        lambda *args, **kwargs: {
            "repair_status": "repaired",
            "new_generated_question": candidate,
            "patches": [{"op": "replace", "path": "/solutions/0/solutionContent/0/text", "value": "Nội dung mới nhưng sai.", "reason": "test"}],
            "selected_issue": service.compact_selected_issue(target),
        },
    )
    monkeypatch.setattr(
        service,
        "evaluate_generated_question_object",
        lambda *args, **kwargs: {"is_good": False, "issues": [new_issue]},
    )

    result = service.maybe_repair_generated_question(
        original,
        {"is_good": False, "issues": [target]},
        strict_mode=True,
        config=config(),
        client=object(),
        debug=False,
        index=0,
        auto_repair=True,
        max_loop=2,
    )

    assert result["repair_status"] == "needs_manual_review"
    assert result["new_generated_question"] is None
    assert result["repair_stop_reason"] == "new_issue_after_rejudge"


def test_successful_patch_is_committed_only_after_rejudge(monkeypatch):
    original = generated_question()
    candidate = deepcopy(original)
    candidate["solutions"][0]["solutionContent"][0]["text"] = "2x = 4, suy ra x = 2."
    target = blocking_issue("fix_solution_correctness")
    rejudge_calls = []
    monkeypatch.setattr(
        service,
        "repair_once",
        lambda *args, **kwargs: {
            "repair_status": "repaired",
            "new_generated_question": candidate,
            "patches": [{"op": "replace", "path": "/solutions/0/solutionContent/0/text", "value": "2x = 4, suy ra x = 2.", "reason": "test"}],
            "selected_issue": service.compact_selected_issue(target),
        },
    )

    def rejudge(question, **kwargs):
        rejudge_calls.append((question, kwargs))
        return {"is_good": True, "issues": []}

    monkeypatch.setattr(service, "evaluate_generated_question_object", rejudge)

    result = service.maybe_repair_generated_question(
        original,
        {"is_good": False, "issues": [target]},
        strict_mode=True,
        config=config(),
        client=object(),
        debug=False,
        index=0,
        auto_repair=True,
        max_loop=2,
    )

    assert len(rejudge_calls) == 1
    assert rejudge_calls[0][1]["auto_repair"] is False
    assert result["repair_status"] == "repaired"
    assert result["repair_stop_reason"] == "recheck_good"
    assert result["new_generated_question"] == candidate
    assert original != candidate

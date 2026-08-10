import json
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.question_plan.logic import generated_question_judge as judge_module
from src.question_plan.logic.code_transition_analyzer import (
    _normalize_math,
    analyze_code_transition,
    analyze_transition_stages,
)
from src.question_plan.logic.generated_question_judge import (
    aggregate_specialized_judge_results,
    combined_correction_response_format,
    build_generated_question_judge_messages,
    build_process_presentation_judge_messages,
    build_solution_splitter_messages,
    build_transition_stages,
    compact_generated_question_payload,
    enrich_question_stem_with_visual_descriptions,
    judge_generated_question_object,
    normalize_math_evidence,
    normalize_correctness_result,
    split_solution_with_code,
    validate_combined_contract_correction_output,
    validate_process_presentation_judge_output,
    validate_correctness_semantic_output,
    validate_splitter_output,
    validate_transition_judge_output,
)
from src.question_plan.shared.utils import parse_json_output
from src.question_plan.schemas.generated_question_contracts import (
    CodeTransitionAnalysis,
    CombinedJudgeCorrectionOutput,
    CorrectnessJudgeOutput,
    ProcessPresentationJudgeOutput,
    ProcessPresentationSemanticOutput,
    SolutionSplitOutput,
)


def test_combined_correction_uses_flat_schema_and_validates_both_branches():
    schema = combined_correction_response_format(ordered_solution())["json_schema"]["schema"]
    assert "oneOf" not in schema
    assert len(schema["properties"]) == 16
    assert schema["additionalProperties"] is False

    parsed = {
        "correctness_error_type": None,
        "correctness_solution_index": None,
        "correctness_from_order": None,
        "correctness_to_order": None,
        "correctness_reason": None,
        "correctness_suggestion": None,
        "correctness_semantic_role": None,
        "correctness_certificate_disposition": None,
        "correctness_role_evidence": None,
        "correctness_status": "good",
        "process_error_type": None,
        "process_solution_index": None,
        "process_state_order": None,
        "process_reason": "",
        "process_suggestion": "",
        "process_verdict": "good",
    }
    CombinedJudgeCorrectionOutput.model_validate(parsed)
    transition_payload = analyze_transition_stages(
        build_transition_stages(ordered_solution(), question())
    )
    result, error = validate_combined_contract_correction_output(
        parsed,
        ordered_solution(),
        question(),
        transition_payload,
    )

    assert error == ""
    assert result["correctness_result"]["verdict"] == "good"
    assert result["process_presentation_result"]["verdict"] == "good"


def question(text: str = "A ⇒ B ⇒ C") -> dict:
    return {
        "_id": "transition-test",
        "instruction": [{"type": "text", "text": "Đề bài A."}],
        "questionItems": [{"stem": [{"type": "text", "text": "Hãy giải."}]}],
        "solutions": [{"solutionContent": [{"type": "text", "text": text}]}],
    }


def ordered_solution() -> dict:
    return {
        "context_requirements": [],
        "stem": {
            "source_path": "/instruction/0/text",
            "source_text": "Đề bài A.",
        },
        "states": [
            *[
                {
                    "solution_index": 0,
                    "order": order,
                    "source_path": "/solutions/0/solutionContent/0/text",
                    "source_text": text,
                }
                for order, text in enumerate(("A ⇒ ", "B ⇒ ", "C"))
            ],
        ]
    }


def correctness_good() -> dict:
    return {
        "first_invalid_transition": None,
        "context_issue": None,
        "reason": "",
        "suggestion": "",
        "verdict": "good",
    }


def correctness_semantic_good() -> dict:
    return {
        "status": "good",
        "error_type": None,
        "solution_index": None,
        "from_order": None,
        "to_order": None,
        "reason": None,
        "suggestion": None,
        "semantic_role": None,
        "certificate_disposition": None,
        "role_evidence": None,
    }


def correctness_semantic_bad() -> dict:
    return {
        "status": "bad",
        "error_type": "calculation_error",
        "solution_index": 0,
        "from_order": 0,
        "to_order": 1,
        "reason": "Phép tính sai.",
        "suggestion": "Sửa lại phép tính.",
        "semantic_role": None,
        "certificate_disposition": None,
        "role_evidence": None,
    }


def test_semantic_correctness_is_grounded_from_builder_transition():
    analyzed = analyze_transition_stages(
        build_transition_stages(ordered_solution(), question())
    )
    output, error = validate_correctness_semantic_output(
        correctness_semantic_bad(), ordered_solution(), question(), analyzed
    )

    assert error == ""
    assert output["verdict"] == "bad"
    assert output["first_invalid_transition"] == {
        "solution_index": 0,
        "from_order": 0,
        "to_order": 1,
        "error_type": "calculation_error",
        "evidence_text": "B ⇒",
        "calculation_check": None,
    }


def test_semantic_correctness_rejects_nonexistent_anchor():
    candidate = {**correctness_semantic_bad(), "to_order": 99}
    output, error = validate_correctness_semantic_output(
        candidate, ordered_solution(), question()
    )

    assert output is None
    assert "ordered transition canonical" in error


def test_bad_missing_error_type_is_safely_normalized_after_anchor_validation():
    candidate = {
        **correctness_semantic_bad(),
        "error_type": None,
        "reason": "Lời giải tự thêm một điều kiện không có trong đề bài.",
        "suggestion": "Bỏ điều kiện tự thêm và dùng miền của đề bài.",
    }

    output, error = validate_correctness_semantic_output(
        candidate, ordered_solution(), question()
    )

    assert error == ""
    assert output["first_invalid_transition"]["error_type"] == "other_correctness_error"
    assert output["error_type_normalized"] is True


@pytest.mark.parametrize(
    "changes",
    [
        {"reason": ""},
        {"to_order": 99},
        {
            "reason": "Lời giải hoàn toàn đúng.",
            "suggestion": "Giữ nguyên lời giải.",
        },
    ],
)
def test_bad_missing_error_type_is_not_normalized_when_safety_conditions_fail(changes):
    candidate = {
        **correctness_semantic_bad(),
        "error_type": None,
        **changes,
    }

    output, error = validate_correctness_semantic_output(
        candidate, ordered_solution(), question()
    )

    assert output is None
    assert error


def test_good_status_ignores_and_nulls_every_other_model_field():
    candidate = {
        "error_type": "not-an-enum",
        "solution_index": "stale-anchor",
        "from_order": -99,
        "to_order": {"stale": True},
        "reason": "Lời giải đúng.",
        "suggestion": "Không cần sửa.",
        "unexpected": "also ignored",
        "status": "good",
    }

    output, error = validate_correctness_semantic_output(
        candidate, ordered_solution(), question()
    )

    assert error == ""
    assert output == {
        "first_invalid_transition": None,
        "context_issue": None,
        "reason": "",
        "suggestion": "",
        "verdict": "good",
        "semantic_role": None,
        "certificate_disposition": None,
        "role_evidence": None,
    }


@pytest.mark.parametrize("contract_retry_error", [None, "previous contract invalid"])
def test_correctness_call_always_uses_flat_strict_schema(contract_retry_error):
    analyzed = analyze_transition_stages(
        build_transition_stages(ordered_solution(), question())
    )

    class Client:
        def __init__(self):
            self.kwargs = None

        def chat_completion(self, **kwargs):
            self.kwargs = kwargs
            return {"content": json.dumps(correctness_semantic_good())}

    client = Client()
    result = judge_module._call_transition_judge(
        generated_question=question(),
        ordered_solution=ordered_solution(),
        schema_validation_result={"issues": []},
        client=client,
        model="gemma-4-12b-it",
        debug=False,
        transition_payload=analyzed,
        contract_retry_error=contract_retry_error,
    )

    response_format = client.kwargs["response_format"]
    assert result["contract_valid"] is True
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert "oneOf" not in schema
    assert "discriminator" not in schema
    assert schema["additionalProperties"] is False
    assert list(schema["properties"])[-1] == "status"
    assert set(schema["required"]) == set(schema["properties"])
    assert client.kwargs["max_tokens"] == 2048


def correctness_bad(*, from_order: int | None = 0, to_order: int = 1) -> dict:
    return {
        "first_invalid_transition": {
            "solution_index": 0,
            "from_order": from_order,
            "to_order": to_order,
            "error_type": "calculation_error",
            "evidence_text": "A",
            "calculation_check": {
                "expected_result": "B",
                "actual_result": "A",
                "matches": False,
            },
        },
        "context_issue": None,
        "reason": "Phép tính không hợp lệ.",
        "suggestion": "Sửa lại phép tính.",
        "verdict": "bad",
    }


def algebra_question() -> dict:
    return {
        "_id": "algebra-transition-test",
        "instruction": [{"type": "text", "text": "Giải phương trình 2x + 3 = 7."}],
        "questionItems": [{"stem": [{"type": "text", "text": "Tìm x."}]}],
        "solutions": [{
            "solutionContent": [
                {"type": "text", "text": "Ta có 2x = 4."},
                {"type": "text", "text": "Suy ra x = 3."},
            ],
        }],
    }


def algebra_ordered_solution() -> dict:
    return {
        "context_requirements": [],
        "stem": {
            "source_path": "/instruction/0/text",
            "source_text": "Giải phương trình 2x + 3 = 7.",
        },
        "states": [
            {
                "solution_index": 0,
                "order": 0,
                "source_path": "/solutions/0/solutionContent/0/text",
                "source_text": "Ta có 2x = 4.",
            },
            {
                "solution_index": 0,
                "order": 1,
                "source_path": "/solutions/0/solutionContent/1/text",
                "source_text": "Suy ra x = 3.",
            },
        ],
    }


def verified_candidate(
    *,
    after: str = "2x = 4",
    from_order: int | None = None,
    to_order: int = 0,
    error_type: str = "calculation_error",
) -> dict:
    return {
        "first_invalid_transition": {
            "solution_index": 0,
            "from_order": from_order,
            "to_order": to_order,
            "error_type": error_type,
            "evidence_text": after,
            "calculation_check": None,
        },
        "context_issue": None,
        "reason": "Transition đã gộp nhiều phép biến đổi.",
        "suggestion": "Tách transition.",
        "verdict": "bad",
    }


def process_good() -> dict:
    return {"issue": None, "reason": "", "suggestion": "", "verdict": "good"}


def process_bad() -> dict:
    return {
        "issue": {
            "scope": "state",
            "solution_index": 0,
            "state_order": 0,
            "source_path": "/solutions/0/solutionContent/0/text",
            "evidence_text": "A",
            "error_type": "draft_or_self_questioning",
        },
        "reason": "Lời giải còn đoạn tự vấn.",
        "suggestion": "Loại bỏ đoạn tự vấn.",
        "verdict": "bad",
    }


def process_semantic_good() -> dict:
    return {
        "error_type": None,
        "solution_index": None,
        "state_order": None,
        "reason": "",
        "suggestion": "",
        "verdict": "good",
    }


def process_semantic_bad(*, state_order: int | None = 0) -> dict:
    return {
        "error_type": "draft_or_self_questioning",
        "solution_index": 0,
        "state_order": state_order,
        "reason": "Lời giải còn đoạn tự vấn.",
        "suggestion": "Loại bỏ đoạn tự vấn.",
        "verdict": "bad",
    }


def cubic_question(solution_text: str) -> dict:
    return {
        "_id": "cubic-redundancy-test",
        "instruction": [{"type": "text", "text": "Giải phương trình 2x^3 + 3 = 19."}],
        "questionItems": [{"stem": [{"type": "text", "text": "Tìm x."}]}],
        "solutions": [{"solutionContent": [{"type": "text", "text": solution_text}]}],
    }


def cubic_ordered_solution(*states: str) -> dict:
    return {
        "context_requirements": [],
        "stem": {
            "source_path": "/instruction/0/text",
            "source_text": "Giải phương trình 2x^3 + 3 = 19.",
        },
        "states": [
            {
                "solution_index": 0,
                "order": order,
                "source_path": "/solutions/0/solutionContent/0/text",
                "source_text": text,
            }
            for order, text in enumerate(states)
        ],
    }


def test_hard_valid_certificate_blocks_only_direct_math_validity_denial():
    generated = cubic_question(
        "Ta có 2x^3 = 16, 2x^3 + 4 = 20, x^3 = 8, suy ra x = 2."
    )
    ordered = cubic_ordered_solution(
        "Ta có 2x^3 = 16,",
        "2x^3 + 4 = 20,",
        "x^3 = 8,",
        "suy ra x = 2.",
    )
    analyzed = analyze_transition_stages(build_transition_stages(ordered, generated))
    assert analyzed["stages"][1]["code_analysis"]["status"] == "verified_valid"
    assert analyzed["stages"][1]["code_analysis"]["strength"] == "hard"

    correctness_prompt = build_generated_question_judge_messages(
        generated, "Tiêu chí đúng sai.", ordered, analyzed
    )[-1]["content"]
    process_prompt = build_process_presentation_judge_messages(
        generated,
        "Tiêu chí trình bày.",
        ordered,
        transition_payload=analyzed,
    )[-1]["content"]
    assert "status=verified_valid" in correctness_prompt
    assert "CHỨNG NHẬN PHÉP BIẾN ĐỔI HỢP LỆ" in process_prompt
    assert '"status":"verified_valid"' in process_prompt
    assert "ỨNG VIÊN BƯỚC THỪA" in process_prompt
    assert '"state_order":1' in process_prompt
    assert "### RULES" not in process_prompt
    assert "### OUTPUT SCHEMA" not in process_prompt

    denied, error = validate_correctness_semantic_output(
        {
            **correctness_semantic_bad(),
            "error_type": "incorrect_transformation",
            "reason": "Phép biến đổi này không bảo toàn đẳng thức.",
        },
        ordered,
        generated,
        analyzed,
    )
    assert denied is None
    assert "verified_valid" in error

    domain_issue, error = validate_correctness_semantic_output(
        {
            **correctness_semantic_bad(),
            "error_type": "domain_error",
            "reason": "Lời giải thiếu điều kiện xác định cần thiết.",
        },
        ordered,
        generated,
        analyzed,
    )
    assert error == ""
    assert domain_issue["verdict"] == "bad"
    assert domain_issue["first_invalid_transition"]["error_type"] == "domain_error"


def test_redundant_step_becomes_public_warning_not_good_or_bad():
    generated = cubic_question(
        "Ta có 2x^3 = 16, 2x^3 + 4 = 20, x^3 = 8, suy ra x = 2."
    )
    ordered = cubic_ordered_solution(
        "Ta có 2x^3 = 16,",
        "2x^3 + 4 = 20,",
        "x^3 = 8,",
        "suy ra x = 2.",
    )
    analyzed = analyze_transition_stages(build_transition_stages(ordered, generated))

    false_missing, error = validate_process_presentation_judge_output(
        {
            "error_type": "missing_major_step",
            "solution_index": 0,
            "state_order": 1,
            "reason": "Lời giải không viết thành lời thao tác cộng vào hai vế.",
            "suggestion": "Viết thêm câu mô tả thao tác cộng.",
            "verdict": "bad",
        },
        ordered,
        generated,
        analyzed,
    )
    assert error == ""
    assert false_missing["verdict"] == "good"
    assert false_missing["issue"]["error_type"] == "redundant_step"

    compressed, error = validate_process_presentation_judge_output(
        {
            "error_type": "missing_major_step",
            "solution_index": 0,
            "state_order": 2,
            "reason": "Transition gộp thao tác trừ 4 và chia 2 mà không có trạng thái trung gian.",
            "suggestion": "Viết trạng thái 2x^3 = 16 trước khi chia hai vế cho 2.",
            "verdict": "bad",
        },
        ordered,
        generated,
        analyzed,
    )
    assert error == ""
    assert compressed["verdict"] == "bad"
    assert compressed["issue"]["error_type"] == "missing_major_step"

    process, error = validate_process_presentation_judge_output(
        {
            "error_type": "redundant_step",
            "solution_index": 0,
            "state_order": 1,
            "reason": "Bước này đúng nhưng không cần thiết cho tiến trình giải.",
            "suggestion": "Đi trực tiếp từ 2x^3 = 16 đến x^3 = 8.",
            "verdict": "good",
        },
        ordered,
        generated,
    )
    assert error == ""
    assert process["verdict"] == "good"
    assert process["issue"]["error_type"] == "redundant_step"

    public = aggregate_specialized_judge_results(
        {"contract_valid": True, **correctness_good()},
        {"contract_valid": True, **process},
        ordered,
        generated,
        strict_mode=True,
        index=0,
    )
    assert public["is_good"] is False
    assert public["issues"][0]["severity"] == "warning"
    assert public["issues"][0]["repair_intent"] == "clean_solution_reasoning"
    assert "không cần thiết" in public["issues"][0]["reason"]


def test_invalid_transition_remains_a_hard_correctness_error():
    generated = cubic_question("Ta có 2x^3 = 18.")
    ordered = cubic_ordered_solution("Ta có 2x^3 = 18.")
    analyzed = analyze_transition_stages(build_transition_stages(ordered, generated))
    analysis = analyzed["stages"][0]["code_analysis"]

    assert analysis["status"] == "verified_invalid"
    assert analysis["strength"] == "hard"
    assert analysis["issue_type"] == "invalid_equivalence"


@pytest.mark.parametrize(
    ("source", "evidence"),
    [
        ("4 × 3 = 12", r"4 \times 3 = 12"),
        ("x² = 9", "x^2 = 9"),
        ("11 − 2 = 9", "11 - 2 = 9"),
        ("8 ÷ 2 = 4", r"8 \div 2 = 4"),
        ("4  ×\n3 = 12", r"4 \times 3 = 12"),
    ],
)
def test_math_evidence_normalization_accepts_only_notation_variants(source, evidence):
    assert normalize_math_evidence(evidence) in normalize_math_evidence(source)

    ordered = {
        "context_requirements": [],
        "states": [{
            "solution_index": 0,
            "order": 0,
            "source_path": "/solutions/0/solutionContent/0/text",
            "source_text": source,
        }],
    }
    candidate = {
        "first_invalid_transition": {
            "solution_index": 0,
            "from_order": None,
            "to_order": 0,
            "error_type": "calculation_error",
            "evidence_text": evidence,
            "calculation_check": {
                "expected_result": "expected",
                "actual_result": "actual",
                "matches": False,
            },
        },
        "context_issue": None,
        "reason": "Phép tính sai.",
        "suggestion": "Sửa phép tính.",
        "verdict": "bad",
    }

    validated, error = validate_transition_judge_output(
        candidate, ordered, question(source)
    )

    assert error == ""
    assert validated is not None


@pytest.mark.parametrize(
    "evidence",
    [
        "4 * 4 = 16",
        "Nhân bốn với ba được mười hai",
        "Nội dung chỉ xuất hiện trong đề bài",
    ],
)
def test_math_evidence_normalization_rejects_different_or_semantic_text(evidence):
    ordered = {
        "context_requirements": [],
        "states": [{
            "solution_index": 0,
            "order": 0,
            "source_path": "/solutions/0/solutionContent/0/text",
            "source_text": "4 × 3 = 12",
        }],
    }
    candidate = {
        "first_invalid_transition": {
            "solution_index": 0,
            "from_order": None,
            "to_order": 0,
            "error_type": "calculation_error",
            "evidence_text": evidence,
            "calculation_check": None,
        },
        "context_issue": None,
        "reason": "Phép tính sai.",
        "suggestion": "Sửa phép tính.",
        "verdict": "bad",
    }

    validated, error = validate_transition_judge_output(
        candidate, ordered, question("Nội dung chỉ xuất hiện trong đề bài")
    )

    assert validated is None
    assert "evidence_text" in error


def test_process_grounding_derives_path_and_evidence_from_state_anchor():
    ordered = {
        "context_requirements": [],
        "states": [{
            "solution_index": 0,
            "order": 0,
            "source_path": "/solutions/0/solutionContent/0/text",
            "source_text": "Ta tính 4 × 3 = 12.",
        }],
    }
    candidate = process_semantic_bad()

    validated, error = validate_process_presentation_judge_output(
        candidate, ordered, question("Ta tính 4 × 3 = 12.")
    )

    assert error == ""
    assert validated is not None
    assert validated["issue"]["source_path"] == "/solutions/0/solutionContent/0/text"
    assert validated["issue"]["evidence_text"] == "Ta tính 4 × 3 = 12."


def test_good_contract_requires_empty_reason_and_suggestion():
    correctness, error = validate_transition_judge_output(
        correctness_good(), ordered_solution(), question()
    )
    assert error == ""
    assert correctness["verdict"] == "good"

    correctness, error = validate_transition_judge_output(
        {
            **correctness_good(),
            "reason": "Lời giải đúng.",
            "suggestion": "None",
        },
        ordered_solution(),
        question(),
    )
    assert correctness is None
    assert "reason và suggestion rỗng" in error

    presentation, error = validate_process_presentation_judge_output(
        process_semantic_good(), ordered_solution(), question()
    )
    assert error == ""
    assert presentation["verdict"] == "good"

    presentation, error = validate_process_presentation_judge_output(
        {**process_semantic_good(), "reason": "Lời giải trình bày mạch lạc."},
        ordered_solution(),
        question(),
    )
    assert error == ""
    assert presentation == process_good()


def test_json_with_escaped_latex_is_valid():
    parsed, ok, error = parse_json_output(
        r'{"expected_result":"x = \\sqrt[3]{9}"}'
    )
    assert ok is True
    assert error is None
    assert parsed["expected_result"] == r"x = \sqrt[3]{9}"


def test_both_judge_prompts_contain_output_invariants_and_json_escape_rule():
    analyzed = analyze_transition_stages(
        build_transition_stages(ordered_solution(), question())
    )
    prompts = [
        build_generated_question_judge_messages(
            question(), "Correctness rules.", ordered_solution(), analyzed
        )[-1]["content"],
        build_process_presentation_judge_messages(
            question(), "Presentation rules.", ordered_solution()
        )[-1]["content"],
    ]
    assert 'status="good"' in prompts[0]
    assert "reason, suggestion đều là JSON null" in prompts[0]
    assert 'verdict="good"' in prompts[1]
    assert 'reason=""' in prompts[1]
    assert 'suggestion=""' in prompts[1]
    assert "`redundant_step` được phép đi cùng verdict=\"good\"" in prompts[1]
    for prompt in prompts:
        assert "escape dấu gạch chéo ngược" in prompt
    assert "state_order=null" in prompts[1]
    assert "Không trả source_path, evidence_text hoặc scope" in prompts[1]
    assert "QUY TẮC BƯỚC CHUYỂN ĐẦU TIÊN" in prompts[0]
    assert "`from_order=null`" in prompts[0]
    assert "initial_transition" in prompts[0]
    assert "verified_invalid + hard" in prompts[0]
    assert "Không trả `missing_major_step`" in prompts[0]
    assert "compressed_but_equivalent" not in prompts[0]
    assert "missing_major_step" in prompts[1]


def test_process_prompt_receives_compressed_transition_candidate_in_vietnamese():
    generated = cubic_question("Ta có 2x^3 = 16. Suy ra x = 2")
    ordered = cubic_ordered_solution("Ta có 2x^3 = 16.", " Suy ra x = 2")
    analyzed = analyze_transition_stages(build_transition_stages(ordered, generated))

    prompt = build_process_presentation_judge_messages(
        generated,
        "Presentation rules.",
        ordered,
        transition_payload=analyzed,
    )[-1]["content"]

    assert "ỨNG VIÊN THIẾU BƯỚC" in prompt
    assert '"số_thao_tác":2' in prompt
    assert "chia hai vế cho cùng một giá trị" in prompt
    assert "lấy căn bậc lẻ hai vế" in prompt
    assert '"có_trạng_thái_trung_gian":false' in prompt
    assert "Nêu một phần danh sách thao tác vẫn là thiếu bước" in prompt


def test_contracts_use_flat_correctness_schema_and_presentation_verdict_last():
    schema = CorrectnessJudgeOutput.model_json_schema()
    assert "discriminator" not in schema
    assert "oneOf" not in schema
    assert schema["additionalProperties"] is False
    assert list(schema["properties"])[-1] == "status"
    presentation_schema = ProcessPresentationSemanticOutput.model_json_schema()
    assert "oneOf" not in presentation_schema
    assert presentation_schema["additionalProperties"] is False
    assert list(presentation_schema["properties"])[-1] == "verdict"
    assert set(presentation_schema["required"]) == set(presentation_schema["properties"])


def test_correctness_contract_has_no_llm_verification_metadata():
    parsed, error = validate_transition_judge_output(
        correctness_good(),
        ordered_solution(),
        question(),
    )

    assert error == ""
    assert "verification_checks" not in parsed
    assert "verification_payload" not in parsed
    assert "code_analysis_reviews" not in parsed
    assert "code_analysis_reviews" not in json.dumps(
        CorrectnessJudgeOutput.model_json_schema()
    )


def test_splitter_reconstructs_original_text_exactly():
    output = {
        "context_requirements": [],
        "stem": {
            "source_path": "/instruction/0/text",
            "source_text": "Đề bài A.",
        },
        "states": [
            {
                "solution_index": 0,
                "order": order,
                "source_path": "/solutions/0/solutionContent/0/text",
                "source_text": text,
            }
            for order, text in enumerate(("A ⇒ ", "B ⇒ ", "C"))
        ]
    }
    result, error = validate_splitter_output(output, question())
    assert error == ""
    assert "".join(state["source_text"] for state in result["states"]) == "A ⇒ B ⇒ C"


def test_splitter_stem_is_grounded_and_used_as_initial_premise():
    output = ordered_solution()
    output["stem"] = {
        "source_path": "/questionItems/0/stem/0/text",
        "source_text": "Hãy giải.",
    }

    parsed, error = validate_splitter_output(output, question())
    stages = build_transition_stages(parsed, question())["stages"]

    assert error == ""
    assert stages[0]["source_path"] == "/questionItems/0/stem/0/text"
    assert stages[0]["bieu_thuc_truoc"] == "Hãy giải."


@pytest.mark.parametrize(
    ("stem", "replacement"),
    [
        ({"source_path": "/solutions/0/solutionContent/0/text", "source_text": "A ⇒ B ⇒ C"}, None),
        ({"source_path": "/instruction/0/text", "source_text": "Đề bài đã bị sửa."}, None),
        (None, "missing"),
    ],
)
def test_splitter_rejects_missing_or_ungrounded_stem(stem, replacement):
    output = ordered_solution()
    if replacement == "missing":
        output.pop("stem")
    else:
        output["stem"] = stem

    parsed, error = validate_splitter_output(output, question())

    assert parsed is None
    assert error


def _splitter_case(
    solution_text: str,
    pieces: tuple[str, ...],
    *,
    stem_text: str = "Giải phương trình đã cho.",
) -> tuple[dict, dict]:
    generated = question(solution_text)
    generated["instruction"][0]["text"] = stem_text
    output = {
        "context_requirements": [],
        "stem": {
            "source_path": "/instruction/0/text",
            "source_text": stem_text,
        },
        "states": [
            {
                "solution_index": 0,
                "order": order,
                "source_path": "/solutions/0/solutionContent/0/text",
                "source_text": piece,
            }
            for order, piece in enumerate(pieces)
        ],
    }
    return generated, output


def test_splitter_accepts_semantic_comma_step_boundaries_without_mutation():
    text = "Ta có 2x^3 = 18, x^3 = 8, suy ra x = 2."
    pieces = ("Ta có 2x^3 = 18,", " x^3 = 8,", " suy ra x = 2.")
    stem_text = "Giải phương trình 2x^3 + 3 = 19."
    generated, output = _splitter_case(text, pieces, stem_text=stem_text)

    parsed, error = validate_splitter_output(output, generated)
    stages = build_transition_stages(parsed, generated)["stages"]

    assert error == ""
    assert tuple(state["source_text"] for state in parsed["states"]) == pieces
    assert "".join(state["source_text"] for state in parsed["states"]) == text
    assert stages[0]["bieu_thuc_truoc"] == stem_text
    assert [stage["bieu_thuc_sau"] for stage in stages] == list(pieces)


@pytest.mark.parametrize(
    "text",
    [
        "x = 3,12.",
        "A(1, 2), (x,y)=(2,3), f(x,y), x thuộc [1,5].",
    ],
)
def test_splitter_keeps_non_boundary_commas_inside_one_state(text):
    generated, output = _splitter_case(text, (text,))

    parsed, error = validate_splitter_output(output, generated)

    assert error == ""
    assert parsed["states"][0]["source_text"] == text


def test_valid_comma_separated_math_chain_builds_all_adjacent_transitions():
    text = "2x^3 = 16, x^3 = 8, x = 2"
    pieces = ("2x^3 = 16,", " x^3 = 8,", " x = 2")
    generated, output = _splitter_case(
        text,
        pieces,
        stem_text="Giải phương trình 2x^3 = 16.",
    )

    parsed, error = validate_splitter_output(output, generated)
    stages = build_transition_stages(parsed, generated)["stages"]

    assert error == ""
    assert len(stages) == 3
    assert stages[1]["bieu_thuc_truoc"] == pieces[0]
    assert stages[1]["bieu_thuc_sau"] == pieces[1]
    assert stages[2]["bieu_thuc_truoc"] == pieces[1]
    assert stages[2]["bieu_thuc_sau"] == pieces[2]


def test_splitter_prompt_owns_context_requirement_analysis():
    prompt = "\n".join(
        message["content"]
        for message in build_solution_splitter_messages(question())
    )

    assert "Context & Transition Builder" in prompt
    assert "context_requirements" in prompt
    assert "NGỮ CẢNH CÂU HỎI" in prompt
    assert "NGUỒN VĂN BẢN NGỮ CẢNH" in prompt
    assert "Đề bài A." in prompt
    assert "availability" in prompt
    assert "assetId" in prompt
    assert "Không trả verdict" in prompt
    assert "stem là đề bài hoặc dữ liệu trực tiếp" in prompt
    assert "không dùng dạng questionItems[0].stem[0]" in prompt
    assert "3,12" in prompt
    assert "A(1, 2)" in prompt
    assert "không thay dấu phẩy bằng dấu chấm phẩy" in prompt


def test_splitter_prompt_treats_descriptive_markdown_alt_as_textual_context():
    generated = question()
    generated["instruction"][0]["text"] = (
        "Cho đồ thị như hình: "
        "![Graph on [-1, 1] starts at (-1, -2), passes through (0, 0), and ends at (1, 2).]"
        "(https://example.test/graph.webp)"
    )

    content = build_solution_splitter_messages(generated)[-1]["content"]
    assert isinstance(content, list)
    prompt = content[0]["text"]

    assert "MÔ TẢ TRỰC QUAN BẰNG VĂN BẢN" in prompt
    assert "Graph on [-1, 1] starts at (-1, -2)" in prompt
    assert '"asset_url": "https://example.test/graph.webp"' in prompt
    assert "không được trả missing/insufficient chỉ vì cùng Markdown còn có URL" in prompt
    assert "ẢNH ĐƯỢC ĐÍNH KÈM TRỰC TIẾP" in prompt
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.test/graph.webp"},
    }


def test_splitter_visual_description_enriches_correctness_without_resending_image():
    generated = question()
    generated["instruction"][0]["text"] = (
        "Dựa vào đồ thị: ![Đồ thị hàm số](https://example.test/graph.webp)"
    )
    ordered = ordered_solution()
    ordered["visual_descriptions"] = [{
        "source_path": "/instruction/0/text",
        "asset_url": "https://example.test/graph.webp",
        "description": "Đồ thị đi qua O và tăng trên khoảng đang xét.",
    }]
    analyzed = analyze_transition_stages(build_transition_stages(ordered, generated))

    correctness_content = build_generated_question_judge_messages(
        generated, "Correctness rules.", ordered, analyzed
    )[-1]["content"]
    assert isinstance(correctness_content, str)
    assert "Mô tả dữ liệu trực quan do Splitter đọc từ ảnh" in correctness_content
    assert "Đồ thị đi qua O và tăng trên khoảng đang xét." in correctness_content
    assert len(generated["questionItems"][0]["stem"]) == 1
    enriched = enrich_question_stem_with_visual_descriptions(
        compact_generated_question_payload(generated),
        ordered,
    )
    assert enriched["questionItems"][0]["stem"][-1]["id"].startswith(
        "__splitter_visual_context_"
    )
    assert "Đồ thị đi qua O" in enriched["questionItems"][0]["stem"][-1]["text"]

    presentation_content = build_process_presentation_judge_messages(
        generated, "Presentation rules.", ordered
    )[-1]["content"]
    assert isinstance(presentation_content, str)
    assert "Đồ thị đi qua O và tăng trên khoảng đang xét." not in presentation_content


def test_code_splitter_fallback_sends_original_image_to_correctness():
    generated = question()
    generated["instruction"][0]["text"] = (
        "Dựa vào đồ thị: ![Đồ thị hàm số](https://example.test/graph.webp)"
    )

    ordered, error = split_solution_with_code(
        generated,
        allow_direct_visual_fallback=True,
    )

    assert error == ""
    assert ordered is not None
    assert ordered["visual_descriptions"] == []
    assert ordered["_correctness_reads_images_directly"] is True

    analyzed = analyze_transition_stages(build_transition_stages(ordered, generated))
    correctness_content = build_generated_question_judge_messages(
        generated,
        "Correctness rules.",
        ordered,
        analyzed,
    )[-1]["content"]

    assert isinstance(correctness_content, list)
    assert "Splitter không tạo được mô tả ảnh hợp lệ" in correctness_content[0]["text"]
    assert correctness_content[1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.test/graph.webp"},
    }


def test_structured_image_description_becomes_internal_stem_block_without_mutation():
    generated = question()
    generated["questionItems"][0]["stem"] = [{
        "id": "graph-image",
        "type": "image",
        "url": "https://example.test/graph.webp",
    }]
    ordered = ordered_solution()
    ordered["visual_descriptions"] = [{
        "source_path": "/questionItems/0/stem/0/url",
        "asset_url": "https://example.test/graph.webp",
        "description": "Đồ thị cắt trục hoành tại x=1 và x=3.",
    }]
    analyzed = analyze_transition_stages(build_transition_stages(ordered, generated))

    content = build_generated_question_judge_messages(
        generated,
        "Correctness rules.",
        ordered,
        analyzed,
    )[-1]["content"]

    assert isinstance(content, str)
    assert "Mô tả dữ liệu trực quan do Splitter đọc từ ảnh" in content
    assert "Đồ thị cắt trục hoành tại x=1 và x=3." in content
    assert len(generated["questionItems"][0]["stem"]) == 1
    assert generated["questionItems"][0]["stem"][0]["type"] == "image"


def test_splitter_requires_one_grounded_description_for_each_attached_image():
    generated = question()
    generated["instruction"][0]["text"] = (
        "Dựa vào bảng: ![crop](https://example.test/table.webp)"
    )
    split = ordered_solution()
    split["stem"] = {
        "source_path": "/instruction/0/text",
        "source_text": generated["instruction"][0]["text"],
    }

    missing_result, missing_error = validate_splitter_output(split, generated)
    assert missing_result is None
    assert "phải mô tả đúng một lần" in missing_error

    split["visual_descriptions"] = [{
        "source_path": "/instruction/0/text",
        "asset_url": "https://example.test/table.webp",
        "description": "Bảng biến thiên có các mốc x=-1, x=1 và dấu đạo hàm +, -, +.",
    }]
    result, error = validate_splitter_output(split, generated)
    assert error == ""
    assert result["visual_descriptions"][0]["asset_url"].endswith("table.webp")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda output: output.pop("context_requirements"),
        lambda output: output.update(context_requirements=None),
        lambda output: output.update(context_requirements=[{
            "requirement_type": "video",
            "description": "Video minh họa.",
            "availability": "missing",
            "evidence_text": "Đề bài A.",
        }]),
        lambda output: output.update(context_requirements=[{
            "requirement_type": "graph",
            "description": " ",
            "availability": "missing",
            "evidence_text": "Đề bài A.",
        }]),
        lambda output: output.update(context_requirements=[{
            "requirement_type": "graph",
            "description": "Đồ thị bắt buộc.",
            "availability": "unknown",
            "evidence_text": "Đề bài A.",
        }]),
        lambda output: output.update(context_requirements=[{
            "requirement_type": "graph",
            "description": "Đồ thị bắt buộc.",
            "availability": "missing",
            "evidence_text": " ",
        }]),
    ],
)
def test_context_requirement_contract_rejects_invalid_output(mutate):
    output = ordered_solution()
    mutate(output)

    parsed, error = validate_splitter_output(output, question())

    assert parsed is None
    assert error


def test_context_requirement_must_be_grounded_in_question_context():
    output = ordered_solution()
    output["context_requirements"] = [{
        "requirement_type": "graph",
        "description": "Đồ thị bắt buộc.",
        "availability": "missing",
        "evidence_text": "Không tồn tại trong question.",
    }]

    parsed, error = validate_splitter_output(output, question())

    assert parsed is None
    assert "evidence_text" in error


def test_empty_context_requirements_is_valid():
    parsed, error = validate_splitter_output(ordered_solution(), question())

    assert error == ""
    assert parsed["context_requirements"] == []


@pytest.mark.parametrize(
    "output",
    [
        {},
        {"states": [], "extra": True},
        {
            "states": [{
                "solution_index": 0,
                "order": 0,
                "source_path": "/solutions/0/solutionContent/0/text",
                "source_text": "A ⇒ B ⇒ D",
            }]
        },
        {
            "states": [{
                "solution_index": 0,
                "order": 1,
                "source_path": "/solutions/0/solutionContent/0/text",
                "source_text": "A ⇒ B ⇒ C",
            }]
        },
    ],
)
def test_invalid_splitter_output_is_rejected(output):
    parsed, error = validate_splitter_output(output, question())
    assert parsed is None
    assert error


def test_code_splitter_preserves_every_character():
    text = (
        "Ta có bước thứ nhất với nhiều dữ kiện. "
        "Suy ra bước thứ hai cũng có nhiều dữ kiện. Do đó kết luận cuối cùng."
    )
    split, error = split_solution_with_code(question(text), max_tokens=12)
    assert error == ""
    assert len(split["states"]) > 1
    assert split["context_requirements"] == []
    assert split["stem"] == {
        "source_path": "/instruction/0/text",
        "source_text": "Đề bài A.",
    }
    assert "".join(state["source_text"] for state in split["states"]) == text


def test_splitter_normalizes_only_mathematical_division_colons():
    text = "Thay x = -2: tính 1 200 : 24 = 50."

    split, error = split_solution_with_code(question(text))

    assert error == ""
    assert split["states"][0]["source_text"] == text
    assert split["states"][0]["_math_text"] == "Thay x = -2: tính 1 200/24 = 50."


def test_splitter_passes_normalized_division_to_analyzer():
    text = "x = 1 200 : 24 = 50"
    generated = question(text)
    generated["instruction"][0]["text"] = "x × 24 = 1 200"

    split, error = split_solution_with_code(generated)
    analyzed = analyze_transition_stages(build_transition_stages(split, generated))

    assert error == ""
    assert split["states"][0]["source_text"] == text
    assert split["states"][0]["_math_text"] == "x = 1 200/24 = 50"
    assert analyzed["stages"][0]["code_analysis"]["status"] == "verified_valid"
    assert analyzed["stages"][0]["code_analysis"]["strength"] == "hard"


def test_splitter_prevents_assignment_separator_from_becoming_division():
    text = "Thay x = -2: 2 × (-2)² - 3 × (-2) + 1 = 8 + 6 + 1 = 15."
    generated = question(text)
    generated["instruction"][0]["text"] = "Tính 2x² - 3x + 1 tại x = -2."

    split, error = split_solution_with_code(generated)
    analyzed = analyze_transition_stages(build_transition_stages(split, generated))

    assert error == ""
    assert split["states"][0]["_math_text"] == text
    assert analyzed["stages"][0]["code_analysis"]["status"] == "unsupported"
    assert analyzed["stages"][0]["code_analysis"]["strength"] == "none"


def test_code_builds_only_adjacent_stages():
    stages = build_transition_stages(ordered_solution(), question())["stages"]
    assert stages[0]["stage_type"] == "initial_transition"
    assert stages[0]["from_order"] is None
    assert stages[0]["to_order"] == 0
    assert stages[0]["bieu_thuc_sau"] == "A ⇒ "
    assert all(
        stage["to_order"] == stage["from_order"] + 1
        for stage in stages[1:]
    )


def test_single_solution_state_still_gets_initial_transition():
    single_state = {
        "states": [{
            "solution_index": 0,
            "order": 0,
            "source_path": "/solutions/0/solutionContent/0/text",
            "source_text": "Số đối của 5 là -5.",
        }]
    }
    single_question = question("Số đối của 5 là -5.")
    single_question["instruction"][0]["text"] = "Số đối của 5 là bao nhiêu?"

    stages = build_transition_stages(single_state, single_question)["stages"]

    assert len(stages) == 1
    assert stages[0]["stage_type"] == "initial_transition"
    assert stages[0]["from_order"] is None
    assert stages[0]["to_order"] == 0
    assert stages[0]["bieu_thuc_sau"] == "Số đối của 5 là -5."


def test_invalid_splitter_stem_falls_back_to_problem_anchor_for_transition():
    output = ordered_solution()
    output["stem"] = {
        "source_path": "/instruction/0/text",
        "source_text": "Nội dung không grounded.",
    }

    stages = build_transition_stages(output, question())["stages"]

    assert stages[0]["source_path"] == "/instruction/0/text"
    assert stages[0]["bieu_thuc_truoc"] == "Đề bài A."


def test_initial_transition_contract_and_grounding_accept_first_state():
    output = correctness_bad(from_order=None, to_order=0)

    parsed, error = validate_transition_judge_output(
        output,
        ordered_solution(),
        question(),
    )

    assert error == ""
    assert parsed["first_invalid_transition"]["from_order"] is None
    assert parsed["first_invalid_transition"]["to_order"] == 0


def context_uncertain(evidence: str = "Đề bài A.") -> dict:
    return {
        "first_invalid_transition": None,
        "context_issue": {
            "error_type": "insufficient_context",
            "source_path": "/instruction/0/text",
            "evidence_text": evidence,
        },
        "reason": "Builder xác định dữ liệu bắt buộc đang bị thiếu.",
        "suggestion": "Cung cấp dữ liệu được tham chiếu.",
        "verdict": "uncertain",
    }


def ordered_solution_with_context(availability: str = "missing") -> dict:
    output = ordered_solution()
    output["context_requirements"] = [{
        "requirement_type": "graph",
        "description": "Đồ thị cần thiết để kiểm chứng lời giải.",
        "availability": availability,
        "evidence_text": "Đề bài A.",
    }]
    return output


def test_correctness_rejects_context_issue_because_gate_belongs_to_code():
    parsed, error = validate_transition_judge_output(
        context_uncertain(),
        ordered_solution_with_context("missing"),
        question(),
    )

    assert parsed is None
    assert "context gate thuộc code" in error


@pytest.mark.parametrize("availability", ["missing", "insufficient"])
def test_context_requirements_are_not_reinterpreted_by_correctness_validator(availability):
    split = ordered_solution_with_context(availability)

    parsed, error = validate_transition_judge_output(
        correctness_good(), split, question()
    )
    assert error == ""
    assert parsed["verdict"] == "good"

    parsed, error = validate_transition_judge_output(
        correctness_bad(), split, question()
    )
    assert error == ""
    assert parsed["verdict"] == "bad"


def test_correctness_cannot_self_detect_missing_visual_context():
    parsed, error = validate_transition_judge_output(
        context_uncertain(),
        ordered_solution(),
        question(),
    )

    assert parsed is None
    assert "context gate thuộc code" in error


def test_available_context_allows_transition_evaluation():
    parsed, error = validate_transition_judge_output(
        correctness_good(),
        ordered_solution_with_context("available"),
        question(),
    )

    assert error == ""
    assert parsed["verdict"] == "good"


def test_initial_transition_normalizes_to_first_solution_state():
    issue = normalize_correctness_result(
        correctness_bad(from_order=None, to_order=0),
        ordered_solution(),
        question(),
    )

    assert issue.scope == "state"
    assert issue.anchor_order == 0
    assert issue.source_path == "/solutions/0/solutionContent/0/text"


@pytest.mark.parametrize(
    "output",
    [
        correctness_bad(from_order=None, to_order=1),
        {
            **correctness_bad(from_order=None, to_order=0),
            "first_invalid_transition": {
                **correctness_bad(from_order=None, to_order=0)["first_invalid_transition"],
                "evidence_text": "không có trong first state",
            },
        },
        {
            **correctness_bad(from_order=None, to_order=0),
            "first_invalid_transition": {
                **correctness_bad(from_order=None, to_order=0)["first_invalid_transition"],
                "to_order": None,
            },
        },
    ],
)
def test_invalid_initial_transition_is_rejected(output):
    parsed, error = validate_transition_judge_output(
        output,
        ordered_solution(),
        question(),
    )

    assert parsed is None
    assert error


@pytest.mark.parametrize(
    "output",
    [
        {**correctness_good(), "extra": True},
        {**correctness_good(), "first_invalid_transition": correctness_bad()["first_invalid_transition"]},
        {**correctness_bad(), "context_issue": {
            "error_type": "insufficient_context",
            "source_path": "/instruction/0/text",
            "evidence_text": "Đề bài A.",
        }},
        correctness_bad(from_order=0, to_order=2),
        {
            **correctness_bad(),
            "first_invalid_transition": {
                **correctness_bad()["first_invalid_transition"],
                "evidence_text": "không có trong input",
            },
        },
        {
            **correctness_bad(),
            "first_invalid_transition": {
                **correctness_bad()["first_invalid_transition"],
                "calculation_check": {
                    "expected_result": "A",
                    "actual_result": "A",
                    "matches": True,
                },
            },
        },
    ],
)
def test_invalid_correctness_output_never_becomes_canonical(output):
    parsed, error = validate_transition_judge_output(output, ordered_solution(), question())
    assert parsed is None
    assert error


@pytest.mark.parametrize(
    "output",
    [
        {**process_semantic_good(), "extra": True},
        {**process_semantic_bad(), "error_type": None},
        {**process_semantic_bad(), "state_order": 9},
        {**process_semantic_bad(), "solution_index": 9},
    ],
)
def test_invalid_presentation_output_never_becomes_canonical(output):
    parsed, error = validate_process_presentation_judge_output(
        output, ordered_solution(), question()
    )
    assert parsed is None
    assert error


def test_global_presentation_path_is_derived_from_solution_index():
    output = process_semantic_bad(state_order=None)

    parsed, error = validate_process_presentation_judge_output(
        output, ordered_solution(), question()
    )

    assert error == ""
    assert parsed is not None
    assert parsed["issue"]["source_path"] == "/solutions/0"


def test_wrong_enum_and_string_none_fail_schema():
    with pytest.raises(ValidationError):
        CorrectnessJudgeOutput.model_validate(
            {**correctness_semantic_good(), "status": "valid"}
        )
    with pytest.raises(ValidationError):
        CorrectnessJudgeOutput.model_validate({
            **correctness_semantic_good(),
            "solution_index": "None",
        })


@pytest.mark.parametrize(
    ("correctness", "presentation"),
    [
        (None, {"contract_valid": True, **process_good()}),
        ({"contract_valid": True, **correctness_good()}, None),
        ("raw correctness", {"contract_valid": True, **process_good()}),
        ({"contract_valid": False, "contract_error": "bad"}, {"contract_valid": True, **process_good()}),
        ({"contract_valid": True, **correctness_good()}, {"contract_valid": False, "contract_error": "bad"}),
        ({"contract_valid": False, "contract_error": "bad"}, {"contract_valid": False, "contract_error": "bad"}),
        ({"contract_valid": True}, {"contract_valid": True, **process_good()}),
    ],
)
def test_aggregate_fails_closed_before_reading_invalid_result(correctness, presentation):
    result = aggregate_specialized_judge_results(
        correctness,
        presentation,
        ordered_solution(),
        question(),
        strict_mode=True,
        index=0,
    )
    assert result["is_good"] is False
    assert result["issues"][0]["category"] == "runtime"


def test_aggregate_accepts_only_canonical_results():
    result = aggregate_specialized_judge_results(
        {"contract_valid": True, **correctness_good()},
        {"contract_valid": True, **process_bad()},
        ordered_solution(),
        question(),
        strict_mode=True,
        index=0,
    )
    assert result["issues"][0]["reason"] == "Lời giải còn đoạn tự vấn."


def test_two_specialized_gemma_judges_run_in_parallel_without_extra_call():
    split = {
        "context_requirements": [],
        "states": [{
            "solution_index": 0,
            "order": 0,
            "source_path": "/solutions/0/solutionContent/0/text",
            "source_text": "A",
        }]
    }

    class Client:
        def __init__(self):
            self.calls = []
            self.judges_started = Barrier(2)

        def chat_completion(self, **kwargs):
            self.calls.append(kwargs)
            system = kwargs["messages"][0]["content"]
            if "Judge" in system:
                self.judges_started.wait(timeout=2)
                payload = (
                    split
                    if "Solution State Splitter" in system
                    else process_semantic_good()
                    if "Process & Presentation Judge" in system
                else correctness_semantic_good()
            )
            return {
                "content": "```json\n"
                + json.dumps(payload, ensure_ascii=False)
                + "\n```"
            }

    client = Client()
    config = SimpleNamespace(
        primary_judge_model="gemma",
        fallback_judge_model="gemma-4-26b",
        use_fallback_judge=True,
    )
    result = judge_generated_question_object(
        question("A"), {"issues": []}, config, client
    )
    assert result["is_good"] is True
    assert [call["model"] for call in client.calls] == ["gemma", "gemma", "gemma"]
    presentation_call = next(
        call
        for call in client.calls
        if "Process & Presentation Judge" in call["messages"][0]["content"]
    )
    response_format = presentation_call["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    properties = response_format["json_schema"]["schema"]["properties"]
    assert properties["solution_index"]["anyOf"][0]["enum"] == [0]
    assert properties["state_order"]["anyOf"][0]["enum"] == [0]
    assert presentation_call["max_tokens"] == 2048


def test_dedicated_26b_correctness_keeps_12b_on_splitter_and_presentation():
    calls = []
    monkey_split = ordered_solution()

    class Client:
        pass

    config = SimpleNamespace(
        primary_judge_model="gemma-4-12b-it",
        fallback_judge_model="gemma-4-26b",
        solution_splitter_model="gemma-4-12b-it",
        solution_correctness_model="gemma-4-26b",
        process_presentation_model="gemma-4-12b-it",
        use_fallback_judge=True,
    )

    original_splitter = judge_module._call_solution_splitter
    original_correctness = judge_module._call_transition_judge
    original_presentation = judge_module._call_process_presentation_judge
    try:
        judge_module._call_solution_splitter = lambda *a, **k: (
            calls.append(("splitter", a[2])) or monkey_split,
            "",
        )
        judge_module._call_transition_judge = lambda *a, **k: (
            calls.append(("correctness", k["model"]))
            or {"contract_valid": True, **correctness_good()}
        )
        judge_module._call_process_presentation_judge = lambda *a, **k: (
            calls.append(("presentation", k["model"]))
            or {"contract_valid": True, **process_good()}
        )

        result = judge_generated_question_object(
            question(), {"issues": []}, config, Client()
        )
    finally:
        judge_module._call_solution_splitter = original_splitter
        judge_module._call_transition_judge = original_correctness
        judge_module._call_process_presentation_judge = original_presentation

    assert calls[0] == ("splitter", "gemma-4-12b-it")
    assert set(calls[1:]) == {
        ("correctness", "gemma-4-26b"),
        ("presentation", "gemma-4-12b-it"),
    }
    assert result["correctness_primary_model"] == "gemma-4-26b"
    assert result["final_correctness_model"] == "gemma-4-26b"
    assert result["correctness_primary_26b_calls"] == 1
    assert result["primary_12b_calls"] == 0


def test_dedicated_26b_repairs_contract_once_with_same_model(monkeypatch):
    calls = []
    responses = [
        {
            "contract_valid": False,
            "contract_error": "mechanical: missing status",
            "contract_failure_kind": "mechanical",
            "raw_candidate": "{}",
        },
        {"contract_valid": True, **correctness_good()},
    ]
    monkeypatch.setattr(
        judge_module,
        "_call_solution_splitter",
        lambda *a, **k: (ordered_solution(), ""),
    )

    def correctness_call(*args, **kwargs):
        calls.append(kwargs["model"])
        return responses.pop(0)

    monkeypatch.setattr(judge_module, "_call_transition_judge", correctness_call)
    monkeypatch.setattr(
        judge_module,
        "_call_process_presentation_judge",
        lambda *a, **k: {"contract_valid": True, **process_good()},
    )
    config = SimpleNamespace(
        primary_judge_model="gemma-4-12b-it",
        fallback_judge_model="gemma-4-26b",
        solution_splitter_model="gemma-4-12b-it",
        solution_correctness_model="gemma-4-26b",
        process_presentation_model="gemma-4-12b-it",
        use_fallback_judge=True,
    )

    result = judge_generated_question_object(
        question(), {"issues": []}, config, object()
    )

    assert calls == ["gemma-4-26b", "gemma-4-26b"]
    assert result["is_good"] is True
    assert result["correctness_contract_correction_26b_calls"] == 1
    assert result["contract_correction_12b_calls"] == 0
    assert result["correctness_fallback_called"] is False
    assert result["judge_gemma_run_count"] == 2


def test_builder_missing_context_becomes_uncertain_before_transitions():
    split = {
        "context_requirements": [{
            "requirement_type": "graph",
            "description": "Đồ thị cần thiết để đọc kết quả.",
            "availability": "missing",
            "evidence_text": "Đề bài A.",
        }],
        "stem": {
            "source_path": "/instruction/0/text",
            "source_text": "Đề bài A.",
        },
        "states": [{
            "solution_index": 0,
            "order": 0,
            "source_path": "/solutions/0/solutionContent/0/text",
            "source_text": "A",
        }],
    }

    class Client:
        def __init__(self):
            self.calls = []
            self.judges_started = Barrier(2)

        def chat_completion(self, **kwargs):
            self.calls.append(kwargs)
            system = kwargs["messages"][0]["content"]
            if "Judge" in system:
                self.judges_started.wait(timeout=2)
            payload = (
                split
                if "Solution State Splitter" in system
                else process_semantic_good()
                if "Process & Presentation Judge" in system
                else context_uncertain()
            )
            return {"content": json.dumps(payload, ensure_ascii=False)}

    client = Client()
    config = SimpleNamespace(
        primary_judge_model="gemma",
        fallback_judge_model="gemma-4-26b",
        use_fallback_judge=True,
    )

    result = judge_generated_question_object(
        question("A"), {"issues": []}, config, client
    )

    assert result["is_good"] is False
    assert result["issues"][0]["location"] == "/instruction/0/text"
    assert len(client.calls) == 1


def test_builder_available_context_still_calls_both_specialized_judges():
    split = ordered_solution_with_context("available")
    calls = []
    monkey_client = object()
    config = SimpleNamespace(
        primary_judge_model="gemma",
        fallback_judge_model="gemma-4-26b",
        use_fallback_judge=True,
    )

    original_splitter = judge_module._call_solution_splitter
    original_correctness = judge_module._call_transition_judge
    original_presentation = judge_module._call_process_presentation_judge
    try:
        judge_module._call_solution_splitter = lambda *a, **k: (split, "")
        judge_module._call_transition_judge = (
            lambda *a, **k: calls.append("correctness")
            or {"contract_valid": True, **correctness_good()}
        )
        judge_module._call_process_presentation_judge = (
            lambda *a, **k: calls.append("presentation")
            or {"contract_valid": True, **process_good()}
        )
        result = judge_generated_question_object(
            question(), {"issues": []}, config, monkey_client
        )
    finally:
        judge_module._call_solution_splitter = original_splitter
        judge_module._call_transition_judge = original_correctness
        judge_module._call_process_presentation_judge = original_presentation

    assert sorted(calls) == ["correctness", "presentation"]
    assert result["is_good"] is True


@pytest.mark.parametrize(
    ("correctness_result", "presentation_result"),
    [
        (None, {"contract_valid": True, **process_good()}),
        (
            {"contract_valid": True, **correctness_good()},
            {"contract_valid": False, "contract_error": "bad"},
        ),
        (None, None),
    ],
)
def test_invalid_judge_handoff_never_reaches_semantic_aggregate(
    monkeypatch, correctness_result, presentation_result
):
    monkeypatch.setattr(
        judge_module,
        "_call_solution_splitter",
        lambda *a, **k: (ordered_solution(), ""),
    )
    monkeypatch.setattr(
        judge_module,
        "_call_transition_judge",
        lambda *a, **k: correctness_result,
    )
    monkeypatch.setattr(
        judge_module,
        "_call_process_presentation_judge",
        lambda *a, **k: presentation_result,
    )
    monkeypatch.setattr(
        judge_module,
        "aggregate_specialized_judge_results",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("semantic aggregate must not run")
        ),
    )
    monkeypatch.setattr(
        judge_module,
        "_call_combined_contract_correction",
        lambda **kwargs: {
            "contract_valid": False,
            "contract_error": "combined correction invalid",
        },
    )
    config = SimpleNamespace(
        primary_judge_model="gemma",
        fallback_judge_model="gemma",
        use_fallback_judge=True,
    )

    result = judge_generated_question_object(
        question(), {"issues": []}, config, object()
    )

    assert result["is_good"] is False
    assert result["issues"][0]["category"] == "solution_quality"
    assert result["issues"][0]["repair_intent"] == "needs_manual_review"


@pytest.mark.parametrize("invalid_scope", ["correctness", "presentation"])
def test_one_invalid_scope_uses_exactly_one_branch_correction(
    monkeypatch, invalid_scope
):
    calls = []
    monkeypatch.setattr(
        judge_module,
        "_call_solution_splitter",
        lambda *a, **k: (ordered_solution(), ""),
    )

    def correctness_call(*args, **kwargs):
        model = kwargs["model"]
        calls.append(("correctness", model))
        if invalid_scope == "correctness" and len(
            [scope for scope, _ in calls if scope == "correctness"]
        ) == 1:
            return {"contract_valid": False, "contract_error": "bad correctness"}
        return {"contract_valid": True, **correctness_good()}

    def presentation_call(*args, **kwargs):
        model = kwargs["model"]
        calls.append(("presentation", model))
        if invalid_scope == "presentation" and len(
            [scope for scope, _ in calls if scope == "presentation"]
        ) == 1:
            return {"contract_valid": False, "contract_error": "bad presentation"}
        return {"contract_valid": True, **process_good()}

    monkeypatch.setattr(judge_module, "_call_transition_judge", correctness_call)
    monkeypatch.setattr(
        judge_module, "_call_process_presentation_judge", presentation_call
    )
    config = SimpleNamespace(
        primary_judge_model="gemma",
        fallback_judge_model="gemma-4-26b",
        use_fallback_judge=True,
    )

    result = judge_generated_question_object(
        question(), {"issues": []}, config, object()
    )

    assert all(model != "gemma-4-26b" for _, model in calls)
    assert result["is_good"] is True
    assert result["judge_fallback_called"] is False
    assert result["judge_attempt_count"] == 4
    assert len(calls) == 3
    assert result["correctness_contract_correction_26b_calls"] == int(
        invalid_scope == "correctness"
    )
    assert result["contract_correction_12b_calls"] == 1


def test_two_invalid_scopes_share_one_combined_correction_call(monkeypatch):
    combined_calls = []
    monkeypatch.setattr(
        judge_module,
        "_call_solution_splitter",
        lambda *a, **k: (ordered_solution(), ""),
    )
    monkeypatch.setattr(
        judge_module,
        "_call_transition_judge",
        lambda **kwargs: {"contract_valid": False, "contract_error": "bad correctness"},
    )
    monkeypatch.setattr(
        judge_module,
        "_call_process_presentation_judge",
        lambda **kwargs: {"contract_valid": False, "contract_error": "bad process"},
    )
    monkeypatch.setattr(
        judge_module,
        "_call_combined_contract_correction",
        lambda **kwargs: combined_calls.append(kwargs) or {
            "contract_valid": True,
            "correctness_result": correctness_good(),
            "process_presentation_result": process_good(),
        },
    )

    result = judge_generated_question_object(
        question(),
        {"issues": []},
        SimpleNamespace(
            primary_judge_model="gemma-4-12b-it",
            fallback_judge_model="gemma-4-26b",
            solution_correctness_model="gemma-4-26b",
            process_presentation_model="gemma-4-12b-it",
            use_fallback_judge=True,
        ),
        object(),
    )

    assert result["is_good"] is True
    assert len(combined_calls) == 1
    assert result["judge_attempt_count"] == 4
    assert result["correctness_contract_correction_26b_calls"] == 1
    assert result["contract_correction_12b_calls"] == 0


def test_invalid_scope_after_its_fallback_stops_before_aggregate(monkeypatch):
    monkeypatch.setattr(
        judge_module,
        "_call_solution_splitter",
        lambda *a, **k: (ordered_solution(), ""),
    )
    def transition_call(*args, **kwargs):
        if kwargs["model"] == "gemma-4-26b":
            calls.append(kwargs["model"])
            return {"contract_valid": True, **reasoning_candidate}
        return {"contract_valid": True, **correctness_good()}

    monkeypatch.setattr(judge_module, "_call_transition_judge", transition_call)
    monkeypatch.setattr(
        judge_module,
        "_call_process_presentation_judge",
        lambda *a, **k: {
            "contract_valid": False,
            "contract_error": "still invalid",
        },
    )
    monkeypatch.setattr(
        judge_module,
        "aggregate_specialized_judge_results",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("semantic aggregate must not run")
        ),
    )
    config = SimpleNamespace(
        primary_judge_model="gemma",
        fallback_judge_model="gemma-4-26b",
        use_fallback_judge=True,
    )

    result = judge_generated_question_object(
        question(), {"issues": []}, config, object()
    )

    assert result["is_good"] is False
    assert result["issues"][0]["category"] == "solution_quality"
    assert result["issues"][0]["repair_intent"] == "needs_manual_review"
    assert result["judge_attempt_count"] == 4
    assert result["contract_correction_12b_calls"] == 1


def test_splitter_and_fallback_invalid_stop_before_judges(monkeypatch):
    judge_calls = []
    monkeypatch.setattr(
        judge_module,
        "_call_solution_splitter",
        lambda *a, **k: (None, "invalid concatenation"),
    )
    monkeypatch.setattr(
        judge_module,
        "split_solution_with_code",
        lambda *a, **k: (None, "invalid code fallback"),
    )
    monkeypatch.setattr(
        judge_module,
        "_call_transition_judge",
        lambda *a, **k: judge_calls.append("correctness"),
    )
    monkeypatch.setattr(
        judge_module,
        "_call_process_presentation_judge",
        lambda *a, **k: judge_calls.append("presentation"),
    )
    config = SimpleNamespace(
        primary_judge_model="gemma",
        fallback_judge_model="gemma-4-26b",
        use_fallback_judge=True,
    )

    result = judge_generated_question_object(
        question(), {"issues": []}, config, object()
    )

    assert judge_calls == []
    assert result["judge_gemma_run_count"] == 0
    assert result["issues"][0]["category"] == "solution_quality"
    assert result["issues"][0]["repair_intent"] == "needs_manual_review"


def test_splitter_exhausted_runtime_stays_runtime(monkeypatch):
    judge_calls = []
    monkeypatch.setattr(
        judge_module,
        "_call_solution_splitter",
        lambda *a, **k: (
            None,
            judge_module._RuntimeFailureDetail("splitter timeout"),
        ),
    )
    monkeypatch.setattr(
        judge_module,
        "split_solution_with_code",
        lambda *a, **k: (None, "invalid code fallback"),
    )
    monkeypatch.setattr(
        judge_module,
        "_call_transition_judge",
        lambda *a, **k: judge_calls.append("correctness"),
    )
    monkeypatch.setattr(
        judge_module,
        "_call_process_presentation_judge",
        lambda *a, **k: judge_calls.append("presentation"),
    )
    config = SimpleNamespace(
        primary_judge_model="gemma",
        fallback_judge_model="gemma-4-26b",
        use_fallback_judge=True,
    )

    result = judge_generated_question_object(
        question(), {"issues": []}, config, object()
    )

    assert judge_calls == []
    assert result["issues"][0]["category"] == "runtime"
    assert result["judge_fallback_reason"] == "splitter_runtime_failure"
    assert "splitter timeout" in result["failed_reason"][0]


def test_invalid_gemma_splitter_with_valid_code_fallback_runs_judges(monkeypatch):
    calls = []
    monkeypatch.setattr(
        judge_module,
        "_call_solution_splitter",
        lambda *a, **k: (None, "invalid concatenation"),
    )
    monkeypatch.setattr(
        judge_module,
        "split_solution_with_code",
        lambda *a, **k: (ordered_solution(), ""),
    )
    monkeypatch.setattr(
        judge_module,
        "_call_transition_judge",
        lambda *a, **k: calls.append("correctness")
        or {"contract_valid": True, **correctness_good()},
    )
    monkeypatch.setattr(
        judge_module,
        "_call_process_presentation_judge",
        lambda *a, **k: calls.append("presentation")
        or {"contract_valid": True, **process_good()},
    )
    config = SimpleNamespace(
        primary_judge_model="gemma",
        fallback_judge_model="gemma-4-26b",
        use_fallback_judge=True,
    )

    result = judge_generated_question_object(
        question(), {"issues": []}, config, object()
    )

    assert sorted(calls) == ["correctness", "presentation"]
    assert result["is_good"] is True
    assert result["judge_fallback_reason"] == "code_splitter_fallback"


@pytest.mark.parametrize(
    ("before", "after", "status", "transition_type", "issue_type", "count"),
    [
        (
            "Giải phương trình 2x + 3 = 7.",
            "Ta có 2x = 4.",
            "verified_valid",
            "equation_transformation",
            None,
            1,
        ),
        (
            "2x + 3 = 7",
            "x = 2",
            "compressed_but_equivalent",
            "equation_transformation",
            None,
            2,
        ),
        (
            "Giải phương trình 3x + 2 = 11.",
            "Chuyển 2 sang vế phải, ta được 3x = 11 + 2 = 13.",
            "verified_invalid",
            "equality_chain",
            "sign_error",
            None,
        ),
        (
            "S = 1/2 * 6 * 4",
            "S = 12",
            "verified_valid",
            "numeric_calculation",
            None,
            1,
        ),
        (
            "x * 24 = 1 200",
            "x = 1 200 / 24 = 50",
            "verified_valid",
            "equality_chain",
            None,
            1,
        ),
        (
            "5x = -10",
            "x = -10 / 5 = -2",
            "verified_valid",
            "equality_chain",
            None,
            1,
        ),
        (
            "x = 2",
            "Thay x = 2, ta có 2(2) + 3 = 7.",
            "verified_valid",
            "numeric_substitution",
            None,
            1,
        ),
        (
            "Vì OA = OB",
            "tam giác OAB cân",
            "unsupported",
            "semantic_reasoning",
            None,
            None,
        ),
        (
            "Rút gọn (x + 1)^2.",
            "(x + 1)^2 = x^2 + 2x + 1",
            "verified_valid",
            "expression_transformation",
            None,
            1,
        ),
        (
            "Rút gọn 2(x + 1)^2 - 2.",
            "2(x + 1)^2 - 2 = 2x^2 + 4x",
            "compressed_but_equivalent",
            "expression_transformation",
            None,
            3,
        ),
    ],
)
def test_code_transition_analyzer_required_cases(
    before, after, status, transition_type, issue_type, count
):
    result = analyze_code_transition(before=before, after=after)

    assert result["status"] == status
    assert result["transition_type"] == transition_type
    assert result["issue_type"] == issue_type
    assert result["operation_count"] == count
    assert result["strength"] == {
        "verified_valid": "hard",
        "verified_invalid": "hard",
        "compressed_but_equivalent": "soft",
        "unsupported": "none",
    }[status]
    CodeTransitionAnalysis.model_validate(result)


@pytest.mark.parametrize(
    ("source", "normalized"),
    [
        ("x = 1 200 : 24 = 50", "x=1200:24=50"),
        ("70 : (-10) = -7", "70:(-10)=-7"),
        ("Câu 1: Tính", "Câu1:Tính"),
        ("d: y = 2x", "d:y=2*x"),
    ],
)
def test_analyzer_math_normalization_does_not_convert_division_colons(source, normalized):
    assert _normalize_math(source) == normalized


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (
            "Tìm đỉnh của parabol y = x² - 4x + 3.",
            "Hoành độ đỉnh là -b/(2a) = 4/2 = 2.",
        ),
        (
            "Viết phương trình đường thẳng qua A(1; 2), pháp tuyến n=(3; -2).",
            "Phương trình là 3(x - 1) - 2(y - 2) = 0.",
        ),
        (
            "Cấp số cộng có u₁ = 3, công sai d = 4. Tính u₁₀.",
            "u₁₀ = u₁ + (10 - 1)d = 3 + 9 × 4 = 39.",
        ),
        (
            "Tìm nguyên hàm của f(x) = 3x² - 2x + 1.",
            "Lấy nguyên hàm từng hạng tử được F(x) = x³ - x² + x + C.",
        ),
        (
            "Tính I = ∫ từ 0 đến 1 của 2x(x²+1) dx.",
            "Đặt u = x²+1 thì du = 2x dx; khi x=0, u=1 và khi x=1, u=2.",
        ),
        (
            "Tính I=∫ từ 0 đến 1 của (2x+1) dx.",
            "I=[x²+x] từ 0 đến 1=(1²+1)-(0²+0)=2.",
        ),
    ],
)
def test_analyzer_fails_open_when_no_specialized_verifier_owns_transition(
    before, after
):
    result = analyze_code_transition(before=before, after=after)

    assert result["status"] == "unsupported"
    assert result["strength"] == "none"


def test_analyzer_does_not_hard_verify_fragment_of_unsupported_full_claim():
    result = analyze_code_transition(
        before="Cấp số cộng có u₁ = 3, công sai d = 4. Tính u₁₀.",
        after="u₁₀ = u₁ + (10 - 1)d = 3 + 9 × 4 = 39.",
    )

    assert result["status"] == "unsupported"
    assert "fragment" in result["reason"]


def test_every_stage_gets_validated_internal_code_analysis():
    payload = analyze_transition_stages(
        build_transition_stages(algebra_ordered_solution(), algebra_question())
    )

    assert len(payload["stages"]) == 2
    assert payload["stages"][0]["code_analysis"]["status"] == "verified_valid"
    assert payload["stages"][1]["code_analysis"]["status"] == "verified_invalid"
    for stage in payload["stages"]:
        CodeTransitionAnalysis.model_validate(stage["code_analysis"])


def test_standalone_single_operation_error_is_an_internal_mandatory_issue():
    ordered = {
        "context_requirements": [],
        "states": [{
            "solution_index": 0,
            "order": 0,
            "source_path": "/solutions/0/solutionContent/0/text",
            "source_text": "20 / 5 = 5.",
        }],
    }
    analyzed = analyze_transition_stages(build_transition_stages(ordered, question()))

    assert analyzed["stages"][0]["code_analysis"]["status"] == "verified_invalid"
    assert analyzed["stages"][0]["mandatory_code_issue"] == {
        "evidence_text": "20 / 5 = 5",
        "expected_result": "4",
        "actual_result": "5",
    }
    assert "code_alerts" not in analyzed


@pytest.mark.parametrize(
    "expression",
    [
        "48 + 27 = 75.",
        "20 - 5 = 15.",
        "6 × 4 = 24.",
        "20 / 5 = 4.",
    ],
)
def test_standalone_correct_single_operation_is_hard_valid(expression):
    result = analyze_code_transition(before="Tính giá trị.", after=expression)

    assert result["status"] == "verified_valid"
    assert result["strength"] == "hard"
    assert result["transition_type"] == "numeric_calculation"
    assert result["operation_count"] == 1


def test_standalone_correct_operation_does_not_create_mandatory_issue():
    ordered = {
        "context_requirements": [],
        "states": [{
            "solution_index": 0,
            "order": 0,
            "source_path": "/solutions/0/solutionContent/0/text",
            "source_text": "48 + 27 = 75.",
        }],
    }

    analyzed = analyze_transition_stages(build_transition_stages(ordered, question()))

    assert analyzed["stages"][0]["code_analysis"]["status"] == "verified_valid"
    assert analyzed["stages"][0]["code_analysis"]["strength"] == "hard"
    assert "mandatory_code_issue" not in analyzed["stages"][0]


def test_table_fact_prefix_remains_a_narrow_mandatory_assertion():
    payload = {
        "stages": [{
            "solution_index": 0,
            "stage": 0,
            "stage_type": "initial_transition",
            "from_order": None,
            "to_order": 0,
            "source_path": "/instruction/0/text",
            "bieu_thuc_truoc": "Tính 20 : 5.",
            "bieu_thuc_sau": "Theo bảng chia 5, 20 : 5 = 5.",
            "_math_before": "Tính 20 / 5.",
            "_math_after": "Theo bảng chia 5, 20 / 5 = 5.",
        }],
    }

    analyzed = analyze_transition_stages(payload)

    assert analyzed["stages"][0]["mandatory_code_issue"]["expected_result"] == "4"


def test_analyzer_does_not_interpret_raw_colon_as_division():
    result = analyze_code_transition(before="Tính 20 : 5.", after="20 : 5 = 5.")

    assert result["status"] == "unsupported"
    assert result["strength"] == "none"


@pytest.mark.parametrize(
    "text",
    [
        "Cách tính 20 / 5 = 5 là sai.",
        "Nếu giả sử 20 / 5 = 5 thì dẫn đến mâu thuẫn.",
        "Vì 20 / 5 = 5 nên mỗi nhóm có 5 học sinh.",
        "Có phải 20 / 5 = 5 không?",
    ],
)
def test_numeric_claim_inside_reasoning_never_becomes_mandatory(text):
    payload = {
        "stages": [{
            "solution_index": 0,
            "stage": 0,
            "stage_type": "initial_transition",
            "from_order": None,
            "to_order": 0,
            "source_path": "/instruction/0/text",
            "bieu_thuc_truoc": "Hãy nhận xét.",
            "bieu_thuc_sau": text,
        }],
    }

    analyzed = analyze_transition_stages(payload)

    assert "mandatory_code_issue" not in analyzed["stages"][0]
    assert "code_alerts" not in analyzed


def test_verified_valid_stage_has_no_internal_mandatory_issue():
    payload = {
        "stages": [{
            "solution_index": 0,
            "stage": 0,
            "stage_type": "initial_transition",
            "from_order": None,
            "to_order": 0,
            "source_path": "/instruction/0/text",
            "bieu_thuc_truoc": "Tính 20 / 5.",
            "bieu_thuc_sau": "20 / 5 = 4.",
        }],
    }

    analyzed = analyze_transition_stages(payload)

    assert "mandatory_code_issue" not in analyzed["stages"][0]


def test_mandatory_arithmetic_overrides_good_but_not_an_earlier_llm_error():
    generated = question("20 / 5 = 5.")
    ordered = {
        "context_requirements": [],
        "states": [{
            "solution_index": 0,
            "order": 0,
            "source_path": "/solutions/0/solutionContent/0/text",
            "source_text": "20 / 5 = 5.",
        }],
    }
    analyzed = analyze_transition_stages(build_transition_stages(ordered, generated))

    result, applied = judge_module.apply_mandatory_arithmetic_result(
        {
            "contract_valid": True,
            **correctness_good(),
            "semantic_role": "asserted_active",
            "certificate_disposition": "accept",
            "role_evidence": None,
        },
        analyzed,
        ordered,
        generated,
    )

    assert applied is True
    assert result["verdict"] == "bad"
    assert result["first_invalid_transition"]["error_type"] == "calculation_error"
    assert result["first_invalid_transition"]["calculation_check"] == {
        "expected_result": "4",
        "actual_result": "5",
        "matches": False,
    }


@pytest.mark.parametrize(
    "pieces",
    [
        ("Ta có 2x^3 = 18,", " x^3 = 8,", " suy ra x = 2."),
        ("Ta có 2x^3 = 18,", " x^3 = 9,", " suy ra x = 2."),
    ],
)
def test_hard_invalid_equivalence_overrides_good_at_initial_transition(pieces):
    solution_text = "".join(pieces)
    generated, split = _splitter_case(
        solution_text,
        pieces,
        stem_text="Giải phương trình 2x^3 + 3 = 19.",
    )
    analyzed = analyze_transition_stages(build_transition_stages(split, generated))
    semantic = {
        **correctness_semantic_good(),
        "semantic_role": "asserted_active",
        "certificate_disposition": "accept",
        "role_evidence": None,
    }
    canonical, error = validate_correctness_semantic_output(
        semantic,
        split,
        generated,
        analyzed,
    )

    result, applied = judge_module.apply_hard_invalid_equivalence_result(
        {
            "contract_valid": True,
            **canonical,
        },
        analyzed,
        split,
        generated,
    )

    assert analyzed["stages"][0]["code_analysis"]["issue_type"] == "invalid_equivalence"
    assert error == ""
    assert canonical["semantic_role"] == "asserted_active"
    assert applied is True
    assert result["verdict"] == "bad"
    assert result["first_invalid_transition"]["from_order"] is None
    assert result["first_invalid_transition"]["to_order"] == 0
    assert result["first_invalid_transition"]["error_type"] == "non_equivalent_transformation"


def test_active_certificate_with_incomplete_bad_envelope_uses_deterministic_issue():
    pieces = ("Ta có 2x^3 = 18,", " x^3 = 8,", " suy ra x = 2.")
    generated, split = _splitter_case(
        "".join(pieces),
        pieces,
        stem_text="Giải phương trình 2x^3 + 3 = 19.",
    )
    analyzed = analyze_transition_stages(build_transition_stages(split, generated))
    semantic = {
        "error_type": None,
        "solution_index": 0,
        "from_order": None,
        "to_order": 0,
        "reason": None,
        "suggestion": None,
        "semantic_role": "asserted_active",
        "certificate_disposition": "accept",
        "role_evidence": pieces[0],
        "status": "bad",
    }

    canonical, error = validate_correctness_semantic_output(
        semantic,
        split,
        generated,
        analyzed,
    )
    result, applied = judge_module.apply_hard_invalid_equivalence_result(
        {"contract_valid": True, **canonical},
        analyzed,
        split,
        generated,
    )

    assert error == ""
    assert canonical["verdict"] == "good"
    assert applied is True
    assert result["verdict"] == "bad"


def test_hard_equivalence_enforcement_leaves_valid_chain_good():
    pieces = ("Ta có 2x^3 = 16,", " x^3 = 8,", " suy ra x = 2.")
    generated, split = _splitter_case(
        "".join(pieces),
        pieces,
        stem_text="Giải phương trình 2x^3 + 3 = 19.",
    )
    analyzed = analyze_transition_stages(build_transition_stages(split, generated))
    original = {"contract_valid": True, **correctness_good()}

    result, applied = judge_module.apply_hard_invalid_equivalence_result(
        original,
        analyzed,
        split,
        generated,
    )

    assert applied is False
    assert result == original


def test_semantic_connector_does_not_make_math_transition_unsupported():
    result = analyze_code_transition(
        before="2x = 4",
        after="Do đó x = 2.",
    )

    assert result["status"] == "verified_valid"
    assert result["strength"] == "hard"


def test_unicode_superscript_polynomial_expansion_is_verified():
    result = analyze_code_transition(
        before="Tính (x + 2)(x² - 3x + 4).",
        after="Nhân từng hạng tử: x³ - 3x² + 4x + 2x² - 6x + 8.",
    )

    assert result["status"] == "verified_valid"
    assert result["strength"] == "hard"


def test_derivative_transition_stays_outside_simple_algebra_analyzer():
    result = analyze_code_transition(
        before="Xét hàm số y = x³ - 3x.",
        after="y' = 3x² - 3 = 3(x - 1)(x + 1).",
    )

    assert result["status"] == "unsupported"
    assert result["strength"] == "none"


def _hard_certificate_case(solution_text: str) -> tuple[dict, dict, dict]:
    generated, split = _splitter_case(
        solution_text,
        (solution_text,),
        stem_text="Giải phương trình 2x^3 + 3 = 19.",
    )
    analyzed = analyze_transition_stages(build_transition_stages(split, generated))
    assert analyzed["stages"][0]["code_analysis"]["status"] == "verified_invalid"
    assert analyzed["stages"][0]["code_analysis"]["strength"] == "hard"
    return generated, split, analyzed


def test_rejected_hard_certificate_is_grounded_and_not_enforced():
    text = "Ta thử 2x^3 = 18. Kết quả này sai. Ta phải có 2x^3 = 16."
    generated, split, analyzed = _hard_certificate_case(text)
    semantic = {
        **correctness_semantic_good(),
        "semantic_role": "rejected",
        "certificate_disposition": "ignore",
        "role_evidence": "Kết quả này sai.",
    }

    canonical, error = validate_correctness_semantic_output(
        semantic,
        split,
        generated,
        analyzed,
    )
    result, applied = judge_module.apply_hard_invalid_equivalence_result(
        {"contract_valid": True, **canonical},
        analyzed,
        split,
        generated,
    )

    assert error == ""
    assert canonical["semantic_role"] == "rejected"
    assert applied is False
    assert result["verdict"] == "good"


def test_hypothetical_hard_certificate_is_grounded_and_not_enforced():
    text = "Giả sử 2x^3 = 18 để phản chứng. Sau đó ta phải có 2x^3 = 16."
    generated, split, analyzed = _hard_certificate_case(text)
    semantic = {
        **correctness_semantic_good(),
        "semantic_role": "hypothetical",
        "certificate_disposition": "ignore",
        "role_evidence": "Giả sử 2x^3 = 18 để phản chứng.",
    }

    canonical, error = validate_correctness_semantic_output(
        semantic,
        split,
        generated,
        analyzed,
    )
    result, applied = judge_module.apply_hard_invalid_equivalence_result(
        {"contract_valid": True, **canonical},
        analyzed,
        split,
        generated,
    )

    assert error == ""
    assert canonical["semantic_role"] == "hypothetical"
    assert applied is False
    assert result["verdict"] == "good"


def test_uncertain_hard_certificate_fails_closed_to_needs_review():
    text = "Ta có 2x^3 = 18, x^3 = 8, suy ra x = 2."
    pieces = ("Ta có 2x^3 = 18,", " x^3 = 8,", " suy ra x = 2.")
    generated, split = _splitter_case(
        text,
        pieces,
        stem_text="Giải phương trình 2x^3 + 3 = 19.",
    )
    analyzed = analyze_transition_stages(build_transition_stages(split, generated))
    semantic = {
        **correctness_semantic_good(),
        "semantic_role": "uncertain",
        "certificate_disposition": "accept",
        "role_evidence": None,
    }

    canonical, error = validate_correctness_semantic_output(
        semantic,
        split,
        generated,
        analyzed,
    )
    result, applied = judge_module.apply_hard_invalid_equivalence_result(
        {"contract_valid": True, **canonical},
        analyzed,
        split,
        generated,
    )

    assert error == ""
    assert applied is True
    assert result["verdict"] == "uncertain"


@pytest.mark.parametrize(
    "semantic",
    [
        correctness_semantic_good(),
        {
            **correctness_semantic_good(),
            "semantic_role": "rejected",
            "certificate_disposition": "ignore",
            "role_evidence": "Không có trong solution.",
        },
    ],
)
def test_hard_certificate_missing_or_ungrounded_disposition_fails_closed(semantic):
    text = "Ta có 2x^3 = 18, x^3 = 8, suy ra x = 2."
    generated, split = _splitter_case(
        text,
        ("Ta có 2x^3 = 18,", " x^3 = 8,", " suy ra x = 2."),
        stem_text="Giải phương trình 2x^3 + 3 = 19.",
    )
    analyzed = analyze_transition_stages(build_transition_stages(split, generated))

    canonical, error = validate_correctness_semantic_output(
        semantic,
        split,
        generated,
        analyzed,
    )

    assert canonical is None
    assert error.startswith("mechanical:")


def test_bad_reason_wording_is_not_reclassified_by_code(monkeypatch):
    calls = []
    contradictory = correctness_bad(from_order=0, to_order=1)
    contradictory["reason"] = "Phép tính hoàn toàn chính xác và lời giải là đúng."
    contradictory["suggestion"] = "Giữ nguyên, không cần sửa."

    def transition_call(*args, **kwargs):
        calls.append(kwargs["model"])
        return {"contract_valid": True, **contradictory}

    monkeypatch.setattr(
        judge_module,
        "_call_solution_splitter",
        lambda *a, **k: (algebra_ordered_solution(), ""),
    )
    monkeypatch.setattr(judge_module, "_call_transition_judge", transition_call)
    monkeypatch.setattr(
        judge_module,
        "_call_process_presentation_judge",
        lambda *a, **k: {"contract_valid": True, **process_good()},
    )
    config = SimpleNamespace(
        primary_judge_model="gemma",
        fallback_judge_model="gemma-4-26b",
        use_fallback_judge=True,
    )

    result = judge_generated_question_object(
        algebra_question(), {"issues": []}, config, object()
    )

    assert calls == ["gemma"]
    assert result["is_good"] is False
    assert result["contract_correction_12b_calls"] == 0
    assert result["fallback_26b_calls"] == 0


def test_compressed_transition_is_soft_and_missing_step_belongs_to_process_judge():
    generated = algebra_question()
    ordered = {
        "context_requirements": [],
        "states": [{
            "solution_index": 0,
            "order": 0,
            "source_path": "/solutions/0/solutionContent/0/text",
            "source_text": "Suy ra x = 2.",
        }],
    }
    analyzed = analyze_transition_stages(build_transition_stages(ordered, generated))
    assert analyzed["stages"][0]["code_analysis"]["status"] == "compressed_but_equivalent"

    parsed, error = validate_transition_judge_output(
        correctness_good(), ordered, generated, analyzed
    )
    assert error == ""
    assert parsed["verdict"] == "good"

    missing = {
        "error_type": "missing_major_step",
        "solution_index": 0,
        "state_order": 0,
        "reason": "Lời giải bỏ qua các phép biến đổi chính.",
        "suggestion": "Bổ sung các trạng thái trung gian.",
        "verdict": "bad",
    }
    parsed, error = validate_process_presentation_judge_output(
        missing, ordered, generated
    )
    assert error == ""
    assert parsed["issue"]["error_type"] == "missing_major_step"

    assert "missing_major_step" not in json.dumps(
        CorrectnessJudgeOutput.model_json_schema()
    )


def test_ordered_connector_with_explicit_intermediate_steps_is_hard_verified():
    result = analyze_code_transition(
        before="2x + 3 = 7",
        after=(
            "Trừ 3 ở hai vế được 2x = 4, sau đó chia hai vế cho 2 "
            "nên x = 2."
        ),
    )

    assert result["status"] == "verified_valid"
    assert result["strength"] == "hard"
    assert result["operation_count"] == 1
    assert result["explicit_step_count"] == 2
    assert result["intermediate_states_explicit"] is True


def test_explicit_equation_chain_counts_each_written_step_independently():
    result = analyze_code_transition(
        before="2x + 3 = 7",
        after="Ta có 2x = 4. Suy ra x = 2.",
    )

    assert result["status"] == "verified_valid"
    assert result["strength"] == "hard"
    assert result["operation_count"] == 1
    assert result["explicit_step_count"] == 2
    assert result["max_operations_per_step"] == 1
    assert result["total_operation_count"] == 2
    assert result["intermediate_states_explicit"] is True
    assert result["operation_types"] == [
        "subtract_same_value_both_sides",
        "divide_both_sides",
    ]
    CodeTransitionAnalysis.model_validate(result)


def test_direct_jump_remains_soft_after_explicit_chain_support():
    result = analyze_code_transition(
        before="2x + 3 = 7",
        after="Suy ra x = 2.",
    )

    assert result["status"] == "compressed_but_equivalent"
    assert result["operation_count"] == 2
    assert result["max_operations_per_step"] == 2
    assert result["intermediate_states_explicit"] is False


def test_explicit_chain_preserves_the_first_invalid_pair():
    result = analyze_code_transition(
        before="2x + 3 = 7",
        after="Ta có 2x = 5. Suy ra x = 2.5.",
    )

    assert result["status"] == "verified_invalid"
    assert result["strength"] == "hard"
    assert result["intermediate_states_explicit"] is True
    assert "cặp bước 1" in result["reason"]


def test_odd_power_chain_is_verified_step_by_step():
    result = analyze_code_transition(
        before="Giải phương trình 2x^3 + 3 = 19.",
        after="Ta có 2x^3 = 16. Suy ra x^3 = 8. Nên x = 2.",
    )

    assert result["status"] == "verified_valid"
    assert result["strength"] == "hard"
    assert result["explicit_step_count"] == 3
    assert result["max_operations_per_step"] == 1
    assert result["total_operation_count"] == 3
    assert result["intermediate_states_explicit"] is True


@pytest.mark.parametrize(
    ("after", "expected_pair"),
    [
        ("Ta có 2x^3 = 18. Suy ra x^3 = 9. Nên x = 2.", "cặp bước 1"),
        ("Ta có 2x^3 = 18. Suy ra x^3 = 8. Nên x = 2.", "cặp bước 1"),
        ("Ta có 2x^3 = 16. Suy ra x^3 = 9. Nên x = 2.", "cặp bước 2"),
        ("Ta có 2x^3 = 16. Suy ra x^3 = 8. Nên x = 3.", "cặp bước 3"),
    ],
)
def test_odd_power_chain_preserves_earliest_invalid_pair(after, expected_pair):
    result = analyze_code_transition(
        before="Giải phương trình 2x^3 + 3 = 19.",
        after=after,
    )

    assert result["status"] == "verified_invalid"
    assert result["strength"] == "hard"
    assert result["intermediate_states_explicit"] is True
    assert expected_pair in result["reason"]


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("(x - 1)^2 = 0", "x = 1"),
        ("x^3 = 8", "x^5 = 32"),
        ("x^3 - x = 0", "x = 0"),
    ],
)
def test_nonlinear_shapes_outside_safe_odd_monomial_scope_are_not_hard(before, after):
    result = analyze_code_transition(before=before, after=after)

    assert result["status"] in {"unsupported", "ambiguous"}
    assert result["strength"] == "none"


def _odd_power_question_and_ordered_solution():
    generated = question("Ta có 2x^3 = 18. Suy ra x^3 = 8. Nên x = 2.")
    generated["instruction"] = [
        {"type": "text", "text": "Giải phương trình 2x^3 + 3 = 19."}
    ]
    ordered = {
        "context_requirements": [],
        "states": [{
            "solution_index": 0,
            "order": 0,
            "source_path": "/solutions/0/solutionContent/0/text",
            "source_text": generated["solutions"][0]["solutionContent"][0]["text"],
        }],
    }
    return generated, ordered


def _dedicated_models_config():
    return SimpleNamespace(
        primary_judge_model="gemma-4-12b-it",
        fallback_judge_model="gemma-4-26b",
        solution_splitter_model="gemma-4-12b-it",
        solution_correctness_model="gemma-4-26b",
        process_presentation_model="gemma-4-12b-it",
        use_fallback_judge=True,
    )


def test_correctness_owns_used_forward_code_warning_decision(monkeypatch):
    generated, ordered = _odd_power_question_and_ordered_solution()
    monkeypatch.setattr(
        judge_module,
        "_call_solution_splitter",
        lambda *a, **k: (ordered, ""),
    )
    monkeypatch.setattr(
        judge_module,
        "_call_transition_judge",
        lambda **k: {"contract_valid": True, **correctness_bad()},
    )
    monkeypatch.setattr(
        judge_module,
        "_call_process_presentation_judge",
        lambda **k: {"contract_valid": True, **process_good()},
    )

    result = judge_generated_question_object(
        generated,
        {"issues": []},
        _dedicated_models_config(),
        object(),
    )

    assert result["is_good"] is False
    assert result["judge_attempt_count"] == 3


def test_correctness_can_reject_code_warning_from_explicit_refutation(monkeypatch):
    generated, ordered = _odd_power_question_and_ordered_solution()
    generated["solutions"][0]["solutionContent"][0]["text"] = (
        "Ta thử 2x^3 = 18. Kết quả này sai. Do đó 2x^3 = 16."
    )
    ordered["states"][0]["source_text"] = generated["solutions"][0]["solutionContent"][0]["text"]
    candidate_payload = {
        "stages": [{
            "solution_index": 0,
            "stage": 0,
            "stage_type": "initial_transition",
            "from_order": None,
            "to_order": 0,
            "source_path": "/instruction/0/text",
            "bieu_thuc_truoc": generated["instruction"][0]["text"],
            "bieu_thuc_sau": ordered["states"][0]["source_text"],
            "code_analysis": {
                "status": "verified_invalid",
                "strength": "hard",
                "transition_type": "equation_transformation",
                "issue_type": "invalid_equivalence",
                "operation_count": None,
                "operation_types": [],
                "explicit_step_count": 2,
                "max_operations_per_step": None,
                "total_operation_count": None,
                "intermediate_states_explicit": True,
                "failing_pair_index": 1,
                "failing_before": "2x^3 + 3 = 19",
                "failing_after": "2x^3 = 18",
                "reason": "Cặp đầu sai.",
            },
        }],
    }
    monkeypatch.setattr(
        judge_module,
        "_call_solution_splitter",
        lambda *a, **k: (ordered, ""),
    )
    monkeypatch.setattr(
        judge_module,
        "analyze_transition_stages",
        lambda payload: candidate_payload,
    )
    captured = {}

    def correctness_call(**kwargs):
        captured["transitions"] = kwargs["transition_payload"]
        return {
            "contract_valid": True,
            **correctness_good(),
            "semantic_role": "rejected",
            "certificate_disposition": "ignore",
            "role_evidence": "Kết quả này sai.",
        }

    monkeypatch.setattr(
        judge_module,
        "_call_transition_judge",
        correctness_call,
    )
    monkeypatch.setattr(
        judge_module,
        "_call_process_presentation_judge",
        lambda **k: {"contract_valid": True, **process_good()},
    )

    result = judge_generated_question_object(
        generated,
        {"issues": []},
        _dedicated_models_config(),
        object(),
    )

    assert result["is_good"] is True
    assert "claim_role_verification" not in captured["transitions"]["stages"][0]
    assert result["judge_attempt_count"] == 3


def test_verbal_intermediate_reasoning_never_becomes_hard_code_evidence():
    result = analyze_code_transition(
        before="2x^3 = 16",
        after=(
            "Chia cả hai vế cho 2; với phương trình thu được, "
            "lấy căn bậc ba hai vế, suy ra x = 2."
        ),
    )

    assert result["status"] in {"unsupported", "ambiguous", "compressed_but_equivalent"}
    assert result["strength"] != "hard"


def test_pipeline_passes_code_analysis_without_adding_a_model_call(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        judge_module,
        "_call_solution_splitter",
        lambda *a, **k: (algebra_ordered_solution(), ""),
    )

    def call_correctness(*args, **kwargs):
        captured["transitions"] = kwargs["transition_payload"]
        return {"contract_valid": True, **correctness_bad()}

    monkeypatch.setattr(judge_module, "_call_transition_judge", call_correctness)
    monkeypatch.setattr(
        judge_module,
        "_call_process_presentation_judge",
        lambda *a, **k: {"contract_valid": True, **process_good()},
    )
    config = SimpleNamespace(
        primary_judge_model="gemma",
        fallback_judge_model="gemma-4-26b",
        use_fallback_judge=True,
    )

    result = judge_generated_question_object(
        algebra_question(), {"issues": []}, config, object()
    )

    assert len(captured["transitions"]["stages"]) == 2
    assert all("code_analysis" in stage for stage in captured["transitions"]["stages"])
    assert result["judge_attempt_count"] == 3
    assert result["judge_fallback_called"] is False
    assert "code_analysis" not in result

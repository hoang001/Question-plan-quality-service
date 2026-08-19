from types import SimpleNamespace

from src.question_plan.flows.generated_question_service import (
    evaluate_generated_questions_raw,
)
from src.question_plan.logic.generated_question_judge import (
    _candidate_comments,
    aggregate_reviewed_comments,
    build_direct_correctness_messages,
    build_judge_comments_review_messages,
    build_direct_process_presentation_messages,
    build_no_prompt_messages,
    validate_direct_correctness_output,
    validate_direct_process_output,
    validate_judge_comments_review_output,
)
from src.question_plan.shared.utils import parse_json_output


def question(text: str = "Ta co 2x^3 = 16, x^3 = 8, suy ra x = 2.") -> dict:
    return {
        "_id": "direct-judge-test",
        "instruction": [{"type": "text", "text": "Giai phuong trinh 2x^3 + 3 = 19."}],
        "questionItems": [{"stem": [{"type": "text", "text": "Tim x."}]}],
        "solutions": [{"solutionContent": [{"type": "text", "text": text}]}],
    }


def config() -> SimpleNamespace:
    return SimpleNamespace(
        primary_judge_model="fallback-model",
        solution_correctness_model="correctness-model",
        process_presentation_model="process-model",
    )


def test_direct_prompts_receive_original_solution_without_transitions():
    generated = question()
    for messages in (
        build_direct_correctness_messages(generated),
        build_direct_process_presentation_messages(generated),
    ):
        prompt = messages[-1]["content"]
        assert generated["solutions"][0]["solutionContent"][0]["text"] in prompt
        assert "transition_id" not in prompt
        assert "nội dung trước" not in prompt
        assert "nội dung sau" not in prompt

    correctness_prompt = build_direct_correctness_messages(generated)[-1]["content"]
    assert "Kiểm tra solution từ đầu đến cuối" in correctness_prompt
    assert "tự tính biểu thức từ stem" in correctness_prompt
    assert "cách giải khác nhưng hợp lệ" in correctness_prompt
    assert "chỉ ghi lỗi gốc" in correctness_prompt
    assert "Chỉ tạo `comment` khi chỉ ra được một lỗi toán học cụ thể" in correctness_prompt

    process_prompt = build_direct_process_presentation_messages(generated)[-1]["content"]
    assert "không quá 300 ký tự" in process_prompt
    assert "không sao chép cả đoạn lời giải dài" in process_prompt
    assert "Chỉ tạo `missing_major_step`" in process_prompt
    assert process_prompt.count("Chỉ tạo `missing_major_step`") == 1
    assert "chỉ cần một phép biến đổi chính" in process_prompt
    assert "cần ít nhất hai phép biến đổi chính liên tiếp" in process_prompt


def test_contract_retry_omits_oversized_raw_candidate():
    prompt = build_direct_process_presentation_messages(
        question(),
        "Không parse được output.",
        "x" * 5000,
    )[-1]["content"]

    assert '"raw_candidate_omitted":true' in prompt
    assert '"raw_candidate_chars":5000' in prompt
    assert "x" * 100 not in prompt


def test_correctness_contract_does_not_require_verbatim_evidence():
    result, error = validate_direct_correctness_output(
        {
            "opening_checks": [{
                "solution_index": 0,
                "evidence": r"Ta có 2x^3 = 16, rồi x^3 = 8",
                "analysis": "Đã đối chiếu bước mở đầu.",
            }],
            "comments": [{
                "solution_index": 0,
                "evidence": r"Suy ra x = 2 từ \sqrt[3]{8}",
                "reason": "Nhận xét cần được Review kiểm chứng.",
            }],
        },
        question(),
    )

    assert error == ""
    assert result is not None
    assert result["comments"][0]["evidence"] == r"Suy ra x = 2 từ \sqrt[3]{8}"


def test_correctness_coverage_retry_requires_every_question_item():
    prompt = build_direct_correctness_messages(
        question(),
        "mechanical: opening_checks chưa bao phủ tất cả questionItems; cần 3 phần tử nhưng nhận 1.",
        '{"opening_checks":[]}',
    )[-1]["content"]

    assert "mỗi questionItem hoặc câu con được kiểm tra đúng một lượt" in prompt
    assert "vẫn dùng solution_index của solution đó" in prompt


def test_correctness_prompt_stops_when_required_visual_data_is_absent():
    prompt = build_direct_correctness_messages(question())[-1]["content"]

    assert "đề phụ thuộc hình, bảng, đồ thị hoặc dữ liệu trực quan" in prompt
    assert "không kiểm tra tiếp phần đó" in prompt
    assert "không có ảnh, liên kết, mô tả hay dữ liệu chữ đủ thay thế" in prompt


def test_no_prompt_contains_only_problem_solution_and_short_request():
    messages = build_no_prompt_messages(question())

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert content == (
        "De:\nGiai phuong trinh 2x^3 + 3 = 19.\nTim x.\n\n"
        "Loi giai:\nTa co 2x^3 = 16, x^3 = 8, suy ra x = 2.\n\n"
        "Kiem tra loi giai cua bai tren"
    ).replace("De:", "Đề:").replace("Loi giai:", "Lời giải:").replace(
        "Kiem tra loi giai cua bai tren", "Kiểm tra lời giải của bài trên"
    )
    assert "schema" not in content.lower()
    assert "tieu chi" not in content.lower()


def test_direct_contracts_anchor_results_at_solution_level():
    generated = question()
    correctness, correctness_error = validate_direct_correctness_output(
        {
            "opening_checks": [{
                "solution_index": 0,
                "evidence": "Ta co 2x^3 = 16",
                "analysis": "Buoc mo dau dung.",
            }],
            "comments": [{"solution_index": 0, "evidence": "x^3 = 8", "reason": "Sai phep tinh."}],
        },
        generated,
    )
    process, process_error = validate_direct_process_output(
        {
            "comments": [{
                "error_type": "missing_major_step",
                "solution_index": 0,
                "evidence": "x^3 = 9",
                "reason": "Thieu cau noi chinh.",
                "suggestion": "Bo sung lap luan.",
            }],
        },
        generated,
    )

    assert correctness_error == ""
    assert correctness["comments"][0]["solution_index"] == 0
    assert process_error == ""
    assert process["comments"][0]["error_type"] == "missing_major_step"
    assert "conclusions" not in process


def test_process_contract_rejects_any_verdict_field():
    result, error = validate_direct_process_output(
        {
            "comments": [],
            "conclusions": [{"solution_index": 0, "conclusion": "đúng"}],
        },
        question(),
    )

    assert result is None
    assert "conclusions" in error


def test_opening_observation_is_not_promoted_to_correctness_comment():
    generated = question("Ta co 2x^3 = 18, x^3 = 8, suy ra x = 2.")

    result, error = validate_direct_correctness_output(
        {
            "opening_checks": [{
                "solution_index": 0,
                "evidence": "2x^3 = 18",
                "analysis": "Từ đề bài phải được 2x^3 = 16, không phải 18.",
            }],
            "comments": [{
                "solution_index": 0,
                "evidence": "x^3 = 8",
                "reason": "Một nhận xét ở sau.",
            }],
        },
        generated,
    )

    assert error == ""
    assert result["comments"] == [{
        "solution_index": 0,
        "evidence": "x^3 = 8",
        "reason": "Một nhận xét ở sau.",
    }]


def test_correctness_contract_rejects_any_verdict_field():
    result, error = validate_direct_correctness_output(
        {
            "opening_checks": [{
                "solution_index": 0,
                "evidence": "2x^3 = 16",
                "analysis": "Da doi chieu voi de bai.",
            }],
            "comments": [],
            "conclusions": [{"solution_index": 0, "conclusion": "sai"}],
        },
        question(),
    )

    assert result is None
    assert "conclusions" in error


def test_correctness_comments_must_follow_solution_order():
    result, error = validate_direct_correctness_output(
        {
            "opening_checks": [{
                "solution_index": 0,
                "evidence": "Khẳng định A",
                "analysis": "Đã đối chiếu khẳng định đầu tiên.",
            }],
            "comments": [
                {"solution_index": 0, "evidence": "Khẳng định B", "reason": "Lỗi B."},
                {"solution_index": 0, "evidence": "Khẳng định A", "reason": "Lỗi A."},
            ],
        },
        question("Khẳng định A. Khẳng định B."),
    )

    assert result is None
    assert "đúng thứ tự xuất hiện" in error


def test_correctness_evidence_allows_math_delimiters_around_an_inner_fragment():
    result, error = validate_direct_correctness_output(
        {
            "opening_checks": [{
                "solution_index": 0,
                "evidence": "Ta xét biểu thức",
                "analysis": "Đã đối chiếu.",
            }],
            "comments": [{
                "solution_index": 0,
                "evidence": "$15 - 2 \\. 5^2$",
                "reason": "Candidate cần Review kiểm chứng.",
            }],
        },
        question("Ta xét biểu thức $25 + 70 : (15 - 2 \\. 5^2)$.")
    )

    assert error == ""
    assert result["comments"][0]["evidence"] == "$15 - 2 \\. 5^2$"


def test_single_combined_solution_normalizes_subquestion_indexes():
    result, error = validate_direct_correctness_output(
        {
            "opening_checks": [
                {"solution_index": 0, "evidence": "Cau 1", "analysis": "Da kiem tra."},
                {"solution_index": 1, "evidence": "Cau 2", "analysis": "Da kiem tra."},
                {"solution_index": 2, "evidence": "Cau 3", "analysis": "Da kiem tra."},
            ],
            "comments": [{
                "solution_index": 2,
                "evidence": "Cau 3",
                "reason": "Cau con thu ba co loi.",
            }],
        },
        question("Cau 1. Cau 2. Cau 3."),
    )

    assert error == ""
    assert [item["solution_index"] for item in result["opening_checks"]] == [0, 0, 0]
    assert result["comments"][0]["solution_index"] == 0
    assert "conclusions" not in result


def test_single_combined_solution_requires_opening_check_per_question_item():
    generated = question("Cau 1. Cau 2. Cau 3.")
    generated["questionItems"] = [
        {"stem": [{"type": "text", "text": f"De cau {number}."}]}
        for number in (1, 2, 3)
    ]

    incomplete, incomplete_error = validate_direct_correctness_output(
        {
            "opening_checks": [{
                "solution_index": 0,
                "evidence": "Cau 1",
                "analysis": "Da kiem tra cau 1.",
            }],
            "comments": [],
        },
        generated,
    )
    complete, complete_error = validate_direct_correctness_output(
        {
            "opening_checks": [
                {"solution_index": 0, "evidence": f"Cau {number}", "analysis": f"Da kiem tra cau {number}."}
                for number in (1, 2, 3)
            ],
            "comments": [],
        },
        generated,
    )

    assert incomplete is None
    assert "chưa bao phủ tất cả questionItems" in incomplete_error
    assert complete_error == ""
    assert len(complete["opening_checks"]) == 3


def test_correctness_payload_places_problem_before_solutions():
    prompt = build_direct_correctness_messages(question())[-1]["content"]

    assert prompt.index('"instruction"') < prompt.index('"solutions"')


def test_review_prompt_receives_original_solution_and_candidate_comments():
    generated = question()
    comments = [{
        "comment_id": "process-0",
        "source": "process_presentation",
        "solution_index": 0,
        "error_type": "missing_major_step",
        "evidence": "2x^3 = 16",
        "reason": "Thiếu bước.",
        "suggestion": "Bổ sung bước.",
    }]

    prompt = build_judge_comments_review_messages(generated, comments)[-1]["content"]

    assert "2x^3 = 16" in prompt
    assert "process-0" in prompt
    assert "additional_comments" not in prompt
    assert prompt.index("### NHẬN XÉT CẦN KIỂM CHỨNG") < prompt.index("### BẰNG CHỨNG ĐỐI CHIẾU")
    assert "Không tự tìm lỗi mới" in prompt
    assert "không phải bản ghi quá trình cân nhắc" in prompt
    assert "review_reason` tối đa hai câu" in prompt
    assert "review_suggestion" in prompt
    assert "hệ quả của tiền đề sai" in prompt
    assert "Riêng `missing_major_step`" in prompt
    assert "kết luận cuối, không phải bản ghi quá trình cân nhắc" in prompt
    assert "chỉ trình bày lỗi chắc chắn sớm nhất" in prompt
    assert "Giữ nguyên giá trị đúng từ candidate" in prompt
    assert "viết phép nhân bằng ký tự `×`" in prompt
    assert "chỉ nêu phần sai chắc chắn chung cho mọi cách hiểu" in prompt
    assert "Số thập phân dùng dấu phẩy" in prompt


def test_review_prompt_handles_missing_visual_context_comments():
    generated = question()
    generated["instruction"] = [{
        "type": "text",
        "text": "Dựa vào bảng biến thiên sau để xác định cực trị.",
    }]
    comments = [{
        "comment_id": "correctness-0",
        "source": "correctness",
        "solution_index": 0,
        "error_type": None,
        "evidence": "Dựa vào bảng biến thiên sau",
        "reason": "Đề phụ thuộc bảng biến thiên nhưng không cung cấp bảng hoặc dữ liệu thay thế.",
        "independent_checks": [],
        "suggestion": "Kiểm tra và sửa nhận xét toán học đã được xác nhận.",
    }]

    prompt = build_judge_comments_review_messages(generated, comments)[-1]["content"]

    assert "Nếu comment Correctness nêu thiếu hình, bảng, đồ thị hoặc dữ liệu trực quan" in prompt
    assert "việc không thể kiểm chứng vì thiếu nguồn chính là căn cứ chấp nhận comment" in prompt
    assert "Dùng `rejected` khi dữ liệu cần thiết đã được cung cấp" in prompt
    assert "Không tưởng tượng nội dung còn thiếu" in prompt
    assert "Không tự xác minh hay phán quyết đúng sai toán học" not in prompt
    assert "không thay đổi cáo buộc" in prompt


def test_correctness_opening_checks_are_forwarded_to_review():
    candidates = _candidate_comments(
        {
            "opening_checks": [{
                "solution_index": 0,
                "evidence": "2x^3 = 18",
                "analysis": "Từ đề bài phải suy ra 2x^3 = 16.",
            }],
            "comments": [{
                "solution_index": 0,
                "evidence": "2x^3 = 18",
                "reason": "Sai phép chuyển vế.",
            }],
        },
        {"comments": []},
    )

    assert candidates[0]["independent_checks"] == [{
        "evidence": "2x^3 = 18",
        "analysis": "Từ đề bài phải suy ra 2x^3 = 16.",
    }]


def test_process_comments_are_forwarded_without_a_verdict():
    candidates = _candidate_comments(
        {
            "opening_checks": [{
                "solution_index": 0,
                "evidence": "2x^3 = 16",
                "analysis": "Đã đối chiếu.",
            }],
            "comments": [],
        },
        {
            "comments": [{
                "error_type": "redundant_step",
                "solution_index": 0,
                "evidence": "2x^3 + 4 = 20",
                "reason": "Bước này không phục vụ lời giải.",
                "suggestion": "Lược bỏ bước này.",
            }],
        },
    )

    assert [item["comment_id"] for item in candidates] == ["process-0"]


def test_code_uses_only_candidate_comments_accepted_by_review():
    generated = question("Ta co 2x^3 = 18, x^3 = 9, suy ra x = 2.")
    candidates = [{
        "comment_id": "process-0",
        "source": "process_presentation",
        "solution_index": 0,
        "error_type": "missing_major_step",
        "evidence": "2x^3 = 18",
        "reason": "Thiếu bước chuyển vế.",
        "suggestion": "Viết thêm bước.",
    }]
    review = {"reviewed_comments": [{
            "comment_id": "process-0",
            "disposition": "rejected",
            "review_reason": "Chuyển vế thông thường không phải bước lớn.",
            "review_suggestion": "",
        }]}

    result = aggregate_reviewed_comments(
        generated, candidates, review, strict_mode=True, index=0
    )

    assert result["is_good"] is True
    assert result["issues"] == []


def test_rejected_correctness_comments_are_discarded_and_produce_good():
    generated = question("Ta co 2x^3 = 18, x^3 = 9, suy ra x = 2.")
    candidates = [{
        "comment_id": "correctness-0",
        "source": "correctness",
        "solution_index": 0,
        "error_type": None,
        "evidence": "2x^3 = 18",
        "reason": "Sai phép chuyển vế.",
        "independent_checks": [],
        "suggestion": "Sửa phép tính.",
    }]
    review = {"reviewed_comments": [{
        "comment_id": "correctness-0",
        "disposition": "rejected",
        "review_reason": "Không chấp nhận nhận xét.",
        "review_suggestion": "",
    }]}

    result = aggregate_reviewed_comments(
        generated, candidates, review, strict_mode=True, index=0
    )

    assert result["is_good"] is True
    assert result["issues"] == []


def test_aggregate_prioritizes_blocking_correctness_over_earlier_process_issue():
    generated = question("Doan nhap. Ta co 2x^3 = 18.")
    candidates = [
        {
            "comment_id": "process-0",
            "source": "process_presentation",
            "solution_index": 0,
            "error_type": "draft_or_self_questioning",
            "evidence": "Doan nhap",
            "reason": "Loi trinh bay.",
            "suggestion": "Xoa doan nhap.",
        },
        {
            "comment_id": "correctness-0",
            "source": "correctness",
            "solution_index": 0,
            "error_type": None,
            "evidence": "2x^3 = 18",
            "reason": "Sai toan hoc.",
            "suggestion": "Sua phep tinh.",
        },
    ]
    review = {"reviewed_comments": [
        {
            "comment_id": "process-0",
            "disposition": "blocking",
            "review_reason": "Đoạn nháp làm gián đoạn mạch trình bày.",
            "review_suggestion": "Loại bỏ đoạn nháp.",
        },
        {
            "comment_id": "correctness-0",
            "disposition": "blocking",
            "review_reason": "Từ đề bài phải suy ra 2x^3 = 16, không phải 18.",
            "review_suggestion": "Sửa 18 thành 16.",
        },
    ]}

    result = aggregate_reviewed_comments(
        generated, candidates, review, strict_mode=True, index=0
    )

    assert result["issues"][0]["reason"] == (
        "Từ đề bài phải suy ra 2x^3 = 16, không phải 18."
    )
    assert result["issues"][0]["suggestion"] == "Sửa 18 thành 16."


def test_parse_json_output_repairs_common_local_model_json_syntax():
    parsed, ok, error = parse_json_output(
        '{"reviewed_comments":[{"comment_id":"correctness-0",'
        '"disposition":"blocking","review_reason":"Sai tại \\frac{1}{2}.\nDòng hai",}],}'
    )

    assert ok is True
    assert error is None
    assert parsed["reviewed_comments"][0]["review_reason"].startswith("Sai tại \\frac")


def test_review_contract_normalizes_advisory_missing_major_step_to_rejected():
    generated = question()
    candidates = [{
        "comment_id": "process-0",
        "source": "process_presentation",
        "solution_index": 0,
        "error_type": "missing_major_step",
        "evidence": "2x^3 = 16",
        "reason": "Thiếu bước.",
        "suggestion": "Bổ sung bước.",
    }]

    result, error = validate_judge_comments_review_output(
        {
            "reviewed_comments": [{
                "comment_id": "process-0",
                "disposition": "advisory",
                "review_reason": "Chỉ là vấn đề nhẹ.",
                "review_suggestion": "Bổ sung bước nếu cần làm rõ.",
            }],
        },
        generated,
        candidates,
    )

    assert error == ""
    assert result["reviewed_comments"][0]["disposition"] == "rejected"
    assert result["reviewed_comments"][0]["review_suggestion"] == ""


def test_review_contract_keeps_rejection_when_process_mistook_math_error_for_missing_step():
    generated = question("Ta có x^3 = 9, suy ra x = 2.")
    candidates = [{
        "comment_id": "process-0",
        "source": "process_presentation",
        "solution_index": 0,
        "error_type": "missing_major_step",
        "evidence": "x^3 = 9, suy ra x = 2",
        "reason": "Thiếu bước lấy căn bậc ba.",
        "suggestion": "Bổ sung bước trung gian.",
    }]

    result, error = validate_judge_comments_review_output(
        {
            "reviewed_comments": [{
                "comment_id": "process-0",
                "disposition": "rejected",
                "review_reason": "Kết luận x = 2 thực chất sai toán học, không phải lỗi thiếu bước.",
                "review_suggestion": "",
            }],
        },
        generated,
        candidates,
    )

    assert error == ""
    assert result["reviewed_comments"][0]["disposition"] == "rejected"


def test_review_contract_normalizes_accepted_correctness_to_blocking():
    generated = question()
    candidates = [{
        "comment_id": "correctness-0",
        "source": "correctness",
        "solution_index": 0,
        "error_type": None,
        "evidence": "x^3 = 8",
        "reason": "Sai toan hoc.",
        "suggestion": "Sua phep tinh.",
    }]

    result, error = validate_judge_comments_review_output(
        {"reviewed_comments": [{
            "comment_id": "correctness-0",
            "disposition": "advisory",
            "review_reason": "Nhan xet dung.",
            "review_suggestion": "Sua phep tinh.",
        }]},
        generated,
        candidates,
    )

    assert error == ""
    assert result["reviewed_comments"][0]["disposition"] == "blocking"


def test_review_prompt_forbids_independent_self_questioning_detection():
    prompt = build_judge_comments_review_messages(question(), [])[-1]["content"]

    assert "Chỉ kiểm chứng nội dung nháp hoặc thử-sai" in prompt
    assert "khi comment Process nêu đúng loại đó" in prompt


def test_review_prompt_defines_missing_step_once():
    prompt = build_judge_comments_review_messages(question(), [])[-1]["content"]

    assert "ít nhất hai phép biến đổi chính hoặc một ý tưởng bắt buộc" in prompt
    assert "chỉ thiếu một phép biến đổi chính hoặc phép tính con" in prompt
    assert "Không bác chỉ vì thao tác bị thiếu là quen thuộc" in prompt
    assert prompt.count("Riêng `missing_major_step`") == 1


def test_review_prompt_rejects_semantically_equivalent_rewrite():
    prompt = build_judge_comments_review_messages(question(), [])[-1]["content"]

    assert "phủ nhận một biểu thức tương đương" in prompt
    assert "lỗi toán học cụ thể" in prompt


def test_no_prompt_raw_mode_preserves_free_text_without_contract_checks():
    class Client:
        def __init__(self):
            self.calls = []

        def chat_completion(self, **kwargs):
            self.calls.append(kwargs)
            content = (
                "Nhan xet process tu do, khong phai JSON."
                if kwargs["model"] == "process-model"
                else "Nhan xet correctness tu do\nDong thu hai."
            )
            return {"content": content, "latency_seconds": 0}

    client = Client()
    result = evaluate_generated_questions_raw(
        question(), config=config(), client=client
    )

    assert result == {
        "id": "direct-judge-test",
        "correctness_output": "Nhan xet correctness tu do\nDong thu hai.",
        "process_output": "Nhan xet process tu do, khong phai JSON.",
    }
    assert len(client.calls) == 2
    assert {call["model"] for call in client.calls} == {
        "correctness-model",
        "process-model",
    }
    assert all(len(call["messages"]) == 1 for call in client.calls)
    assert all("response_format" not in call for call in client.calls)
    assert all("max_tokens" not in call for call in client.calls)
    assert all(
        call["messages"][0]["content"].endswith("Kiểm tra lời giải của bài trên")
        for call in client.calls
    )

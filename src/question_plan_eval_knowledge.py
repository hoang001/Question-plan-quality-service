"""Gói kiến thức để đánh giá chất lượng question_plan ở cấp source record/câu hỏi.

File này chỉ đóng vai trò thụ động: export policy và tiêu chí cho LLM judge
hiện tại hoặc trong tương lai. File này không tự sửa dữ liệu và không làm thay
đổi hành vi pipeline nếu chưa được import/sử dụng ở nơi khác.
"""

QUESTION_PLAN_EVAL_KNOWLEDGE = {
    "role_definition": {
        "purpose": (
            "Đánh giá `question_plan` như một bản kế hoạch trung gian để chuyển "
            "một bài Toán gốc thành các questionItems và interactions phục vụ "
            "pipeline sinh câu hỏi tương tác."
        ),
        "question_plan_role": (
            "`question_plan` không phải object Question render cuối cùng. Nó cần "
            "giữ được các yêu cầu toán học chính của source và mô tả rõ các "
            "item/interaction đủ điều kiện để bước generation tiếp tục xử lý. "
            "Các thành phần cuối như options, answerSpecs, config chi tiết, "
            "hints và distractors có thể được sinh ở bước sau."
        ),
        "judge_role": (
            "Ưu tiên đánh giá chất lượng ở cấp source record/câu hỏi. Các phát hiện "
            "ở cấp interaction chỉ là bằng chứng hỗ trợ, không phải kết luận chính."
        ),
    },
    "scope_policy": {
        "core_policy": (
            "Plan phải bao phủ toàn bộ các yêu cầu chính có thể trả lời được của "
            "bài toán gốc. Plan có thể tách nhỏ, đổi định dạng hoặc chuyển đổi "
            "source sang các interaction mà hệ thống hỗ trợ, nhưng không được bỏ "
            "sót các ý chính hoặc kỹ năng chính."
        ),
        "required": [
            "Bao phủ toàn bộ yêu cầu chính của source, ví dụ các ý a, b, c.",
            "Map mọi nhiệm vụ chính của source sang plan[], questionItems[] hoặc interactions[].",
            "Giữ nguyên ngữ cảnh và logic toán học cần thiết.",
            "Chuyển đổi/adapt các định dạng chưa hỗ trợ thay vì bỏ qua chúng.",
            "Dùng kết luận ở cấp question/source-record làm kết quả chính.",
        ],
        "allowed": [
            "Thay đổi hình thức trình bày trực quan của bài toán gốc.",
            "Tách bảng, hình, sơ đồ hoặc bài nhiều bước thành các interaction nhỏ hơn.",
            "Chuyển bài tính toán/tự luận mở thành single_choice/multiple_choice nếu bước generation có thể sinh options hợp lệ.",
            "Bỏ qua các chi tiết chỉ là dữ kiện phụ trợ và không phải yêu cầu riêng cần học sinh trả lời.",
        ],
        "not_allowed": [
            "Source có các ý a, b, c nhưng plan chỉ xử lý ý b.",
            "Source có nhiều câu hỏi chính nhưng plan chỉ tạo interaction cho một câu.",
            "Source yêu cầu hoàn thành bảng/hình/nhiều bước nhưng plan chỉ giữ một ô/một bước và bỏ phần còn lại.",
            "Plan làm mất một kỹ năng hoặc thành phần kiến thức chính mà source yêu cầu.",
            "Plan chọn một mảnh rời rạc khiến bài mới không còn tương ứng với bài toán gốc.",
        ],
        "choice_policy": [
            "Ở plan stage, single_choice/multiple_choice chưa cần có options hoặc answerSpec.",
            "Không đánh sai choice interaction chỉ vì source là bài tự luận hoặc bài tính toán.",
            "Chỉ warning hoặc bad khi interactionRequirement quá mơ hồ để sinh options, hoặc số lượng lựa chọn mâu thuẫn với nhiệm vụ.",
        ],
    },
    "schema_description": {
        "question_plan.type": {
            "meaning": "`advanced_question_plan` là marker schema tổng quát, không phải dạng bài.",
            "judge_note": "Không dùng field này làm bằng chứng rằng bài toán có mức độ nâng cao.",
        },
        "plan[].questionOrder": {
            "meaning": "Thứ tự của một question trong question_plan.",
            "judge_note": "Dùng để trace/mapping; order sai định dạng hoặc trùng lặp là lỗi structural.",
        },
        "plan[].questionStatement": {
            "meaning": "Ngữ cảnh chung hoặc phát biểu chung của planned question.",
            "judge_note": "Nên chứa đủ ngữ cảnh từ source cần thiết cho các item bên dưới.",
        },
        "questionItems[].itemOrder": {
            "meaning": "Thứ tự của một item trong planned question.",
            "judge_note": "Dùng để trace/mapping.",
        },
        "questionItems[].requirement": {
            "meaning": "Nhiệm vụ con có thể trả lời được mà học sinh cần thực hiện.",
            "judge_note": "Cần rõ ràng và nên tương ứng với một yêu cầu chính hoặc yêu cầu con rõ ràng từ source.",
        },
        "interactions[].interactionOrder": {
            "meaning": "Thứ tự của một interaction trong item.",
            "judge_note": "Dùng để trace khi một item có nhiều điểm trả lời.",
        },
        "interactions[].interactionType": {
            "meaning": "Loại interaction dự kiến, ví dụ short_answer, single_choice, matching hoặc graph_draw.",
            "judge_note": "Chỉ đánh giá như một chiều chất lượng; không quy toàn bộ chất lượng plan về field này.",
        },
        "interactions[].interactionRequirement": {
            "meaning": "Mô tả ngắn phần học sinh cần trả lời trong interaction đó.",
            "judge_note": "Phải đủ cụ thể để bước generation tạo UI/options/answerSpec.",
        },
    },
    "quality_dimensions": {
        "coverage_issue": {
            "definition": (
                "Plan bỏ sót một hoặc nhiều phần/yêu cầu chính của source, khiến "
                "question_plan không còn bao phủ đầy đủ bài toán gốc."
            ),
            "symptoms": [
                "Source có các ý a, b, c nhưng plan chỉ bao phủ a hoặc b.",
                "Source có nhiều câu hỏi con nhưng plan chỉ tạo interaction cho một câu.",
                "Source yêu cầu hoàn thành bảng nhưng plan chỉ hỏi vài ô và bỏ phần còn lại.",
                "Source có cả phần tính toán và giải thích/chứng minh nhưng plan chỉ giữ phần tính toán.",
                "Raw answer thể hiện nhiều bước/ý tương ứng nhưng plan chỉ dùng một phần nhỏ.",
            ],
            "severity_guidance": {
                "warning": "Thiếu một phần nhỏ, ảnh hưởng hạn chế nhưng vẫn nên bổ sung.",
                "needs_review": "Chưa rõ phần bị thiếu là yêu cầu chính hay chỉ là dữ kiện phụ trợ.",
                "bad": "Bỏ sót rõ ràng một hoặc nhiều yêu cầu chính của source.",
            },
            "examples": [
                "Bad: source yêu cầu a, b, c; plan chỉ có item cho b.",
                "Warning: source có một ghi chú phụ không bắt buộc và plan không biểu diễn ghi chú đó.",
            ],
            "suggested_repair_actions": [
                "add_missing_item",
                "add_missing_interaction",
                "split_item",
                "split_question",
                "adapt_unsupported_format",
                "revise_question_statement",
                "mark_for_human_review",
            ],
        },
        "source_fidelity_issue": {
            "definition": "Nội dung được plan chọn/biểu diễn làm thay đổi ý nghĩa toán học của source.",
            "symptoms": [
                "Sai số liệu, công thức, biến, điều kiện, đơn vị hoặc đối tượng hình học.",
                "Requirement không tương đương với yêu cầu trong source.",
                "Plan tự thêm giả thiết không có trong source.",
            ],
            "severity_guidance": {
                "warning": "Có mơ hồ nhỏ hoặc lệch diễn đạt nhẹ.",
                "needs_review": "Có khả năng mismatch nhưng evidence chưa đủ chắc.",
                "bad": "Sai lệch toán học rõ ràng.",
            },
            "examples": [
                "Source hỏi khoảng cách từ D đến BC, plan hỏi khoảng cách từ A đến BC.",
                "Source có phương trình mx + y = -2, plan đổi thành mx - y = -2.",
            ],
            "suggested_repair_actions": [
                "revise_question_statement",
                "revise_item_requirement",
                "revise_interaction_requirement",
                "mark_for_human_review",
            ],
        },
        "selected_scope_issue": {
            "definition": "Phạm vi của plan không nhất quán với source hoặc với chính requirement mà plan nêu.",
            "symptoms": [
                "Requirement nói giải toàn bộ bài nhưng interactions chỉ bao phủ một phần.",
                "Plan lấy một mảnh rời rạc không tạo thành nhiệm vụ trả lời hoàn chỉnh.",
                "Phạm vi plan vẫn mơ hồ sau khi đọc source và answer.",
            ],
            "severity_guidance": {
                "warning": "Scope hẹp hoặc chưa rõ nhưng phần lớn vẫn generate được.",
                "needs_review": "Không xác định được phần bị bỏ là yêu cầu chính hay dữ kiện phụ trợ.",
                "bad": "Scope khiến câu hỏi sinh ra không còn tương ứng với source.",
            },
            "examples": [
                "Bad: requirement ghi hoàn thành bảng nhưng chỉ hỏi một ô tùy ý.",
                "Needs review: plan bao phủ một công thức nhưng source còn yêu cầu phụ thuộc hình vẽ.",
            ],
            "suggested_repair_actions": [
                "revise_item_requirement",
                "add_missing_item",
                "remove_hallucinated_item",
                "mark_for_human_review",
            ],
        },
        "decomposition_issue": {
            "definition": "questionStatement, questionItems hoặc interactions được tách/gộp chưa hợp lý.",
            "symptoms": [
                "Ngữ cảnh chung bị thiếu khỏi questionStatement.",
                "Một item chứa nhiều nhiệm vụ độc lập nhưng chỉ có một interaction mơ hồ.",
                "Một đáp án đơn giản bị tách thành nhiều interaction dư thừa.",
            ],
            "severity_guidance": {
                "warning": "Chưa tối ưu nhưng vẫn có thể generate.",
                "needs_review": "Có thể cần tách/gộp lại.",
                "bad": "Tách/gộp sai làm cản trở generation hoặc chấm đúng.",
            },
            "examples": [
                "Source hỏi tổng và tích; plan chỉ có một interactionRequirement `Kết quả`.",
                "Item sau phụ thuộc vào m tìm ở item trước, nhưng item tìm m bị thiếu.",
            ],
            "suggested_repair_actions": [
                "split_item",
                "merge_items",
                "split_interaction",
                "merge_interactions",
                "revise_question_statement",
            ],
        },
        "unsupported_format_adaptation_issue": {
            "definition": "Định dạng source như bảng/hình/sơ đồ chưa được chuyển thành interaction được hỗ trợ.",
            "symptoms": [
                "Plan ghi hoàn thành bảng dù hệ thống không hỗ trợ table interaction trực tiếp.",
                "Plan bỏ yêu cầu bảng/hình thay vì phân rã nó thành các interaction phù hợp.",
                "Plan yêu cầu hành vi UI không có interactionType tương ứng.",
            ],
            "severity_guidance": {
                "warning": "Cách adapt có thể cải thiện thêm.",
                "needs_review": "Chưa rõ các interaction hiện tại có biểu diễn được format này không.",
                "bad": "Yêu cầu format chính bị bỏ hoặc không thể render/chấm.",
            },
            "examples": [
                "Good: chuyển các ô trong bảng thành nhiều short_answer/fill_blank interactions.",
                "Bad: `Hoàn thành bảng` nhưng không có interaction rõ đến từng ô/cell.",
            ],
            "suggested_repair_actions": [
                "adapt_unsupported_format",
                "split_interaction",
                "revise_item_requirement",
                "mark_for_human_review",
            ],
        },
        "requirement_clarity_issue": {
            "definition": "requirement hoặc interactionRequirement quá mơ hồ để generation xử lý chính xác.",
            "symptoms": [
                "Cụm chung chung như `Giá trị`, `Đáp án`, `Kết quả` nhưng không nêu đối tượng.",
                "Thiếu đơn vị, yêu cầu làm tròn, biến hoặc đối tượng cần xét.",
                "Không rõ học sinh cần chọn một hay nhiều đáp án.",
            ],
            "severity_guidance": {
                "warning": "Context có thể giúp hiểu nhưng wording nên rõ hơn.",
                "needs_review": "Có nhiều cách hiểu hợp lý.",
                "bad": "Không biết cần generate/chấm cái gì.",
            },
            "examples": [
                "Mơ hồ: `Giá trị (x, y)` với multiple_choice.",
                "Rõ: `Giá trị của m để (1; -2) là nghiệm của phương trình.`",
            ],
            "suggested_repair_actions": [
                "revise_item_requirement",
                "revise_interaction_requirement",
            ],
        },
        "interaction_type_issue": {
            "definition": "interactionType không phù hợp với hành động học sinh cần thực hiện.",
            "symptoms": [
                "Nhiệm vụ vẽ nhưng dùng short_answer mà không có rationale chuyển đổi rõ.",
                "Yêu cầu chọn tất cả nhưng dùng single_choice.",
                "Bài chứng minh/giải thích tự do bị ép thành choice nhưng không có mệnh đề chọn rõ ràng.",
            ],
            "severity_guidance": {
                "warning": "Có thể dùng được nhưng không lý tưởng.",
                "needs_review": "Có thể hợp lệ nếu generation chuyển đổi rất cẩn thận.",
                "bad": "Type rõ ràng không thể render/chấm nhiệm vụ.",
            },
            "examples": [
                "Good: `Chọn nhận định đúng về số lượng nghiệm của phương trình.` -> single_choice.",
                "Bad: `Vẽ đồ thị hàm số` -> single_choice, trừ khi nói rõ là chọn trong các đồ thị cho sẵn.",
            ],
            "suggested_repair_actions": [
                "change_interaction_type",
                "revise_interaction_requirement",
                "mark_for_human_review",
            ],
        },
        "choice_planning_issue": {
            "definition": "Việc lập kế hoạch cho single_choice/multiple_choice chưa rõ hoặc mâu thuẫn với số lượng lựa chọn.",
            "symptoms": [
                "single_choice nhưng requirement yêu cầu chọn tất cả đáp án đúng.",
                "multiple_choice nhưng requirement chỉ hỏi một giá trị duy nhất.",
                "interactionRequirement quá mơ hồ để sinh options.",
            ],
            "severity_guidance": {
                "ok": "Ở plan stage chưa có options/answerSpec là bình thường nếu requirement rõ.",
                "warning": "Choice có thể hợp lý nhưng generation cần làm rõ options.",
                "needs_review": "Không rõ nên chọn một hay nhiều.",
                "bad": "Choice type mâu thuẫn với nhiệm vụ.",
            },
            "examples": [
                "Good single_choice: `Chọn nhận định đúng về số lượng nghiệm của phương trình.`",
                "Good multiple_choice: `Chọn tất cả các cặp (x; y) là nghiệm của phương trình trong các lựa chọn cho sẵn.`",
                "Mơ hồ: `Giá trị (x, y)` với multiple_choice.",
            ],
            "suggested_repair_actions": [
                "revise_interaction_requirement",
                "change_interaction_type",
                "mark_for_human_review",
            ],
        },
        "answerability_issue": {
            "definition": "Plan thiếu dữ kiện cần thiết hoặc không có đáp án/tiêu chí chấm xác định.",
            "symptoms": [
                "Thiếu định nghĩa biến, điểm, yếu tố hình học hoặc điều kiện.",
                "Đại lượng được hỏi không được định nghĩa.",
                "Không có tiêu chí đúng/sai rõ ràng.",
            ],
            "severity_guidance": {
                "warning": "Có vẻ vẫn trả lời được từ source nhưng plan nên thêm context.",
                "needs_review": "Chưa rõ dữ kiện có đủ không.",
                "bad": "Không thể trả lời hoặc chấm từ plan.",
            },
            "examples": [
                "Plan hỏi khoảng cách từ D đến BC nhưng D, B, C không xuất hiện trong statement/context.",
                "Plan hỏi `nhận định đúng` nhưng không có mệnh đề hoặc target rõ.",
            ],
            "suggested_repair_actions": [
                "revise_question_statement",
                "revise_item_requirement",
                "add_missing_interaction",
                "mark_for_human_review",
            ],
        },
        "pedagogical_flow_issue": {
            "definition": "Thứ tự hoặc mạch học tập gây khó hiểu dù từng phần riêng lẻ có thể hợp lệ.",
            "symptoms": [
                "Item sau phụ thuộc vào một bước bị thiếu hoặc xuất hiện sau đó.",
                "Interactions được sắp xếp ngược logic giải bài.",
                "Plan không dẫn dắt học sinh qua các nhiệm vụ phụ thuộc nhau.",
            ],
            "severity_guidance": {
                "warning": "Mạch hơi gượng nhưng vẫn làm được.",
                "needs_review": "Có thể cần reorder/split.",
                "bad": "Mạch làm bài trở nên không thể làm hoặc gây hiểu sai.",
            },
            "examples": [
                "Hỏi nghiệm tổng quát với m đã tìm được trước đó, nhưng item tìm m bị thiếu.",
                "Yêu cầu so sánh hai kết quả trước khi yêu cầu tính hai kết quả đó.",
            ],
            "suggested_repair_actions": [
                "reorder_items",
                "split_item",
                "revise_item_requirement",
                "add_missing_interaction",
            ],
        },
        "other_plan_quality_issue": {
            "definition": "Vấn đề chất lượng plan hợp lý nhưng không map gọn vào taxonomy có sẵn.",
            "symptoms": [
                "Có evidence cụ thể về ảnh hưởng đến generation/render/grading hoặc chất lượng plan.",
                "Không thể phân loại rõ vào các nhóm coverage, fidelity, decomposition, clarity, interaction type, answerability hoặc flow.",
            ],
            "severity_guidance": {
                "warning": "Vấn đề nhỏ nhưng có evidence cụ thể.",
                "needs_review": "Có dấu hiệu hợp lý nhưng cần người kiểm tra thêm.",
                "bad": "Vấn đề rõ ràng làm plan không thể generation/render/grading đúng.",
            },
            "examples": [
                "Chỉ dùng khi issue thật sự không thuộc taxonomy có sẵn.",
                "Không dùng như fallback mơ hồ nếu có thể map vào issue type cụ thể hơn.",
            ],
            "suggested_repair_actions": [
                "mark_for_human_review",
                "no_auto_fix",
            ],
            "usage_constraints": [
                "Phải có evidence cụ thể.",
                "Phải giải thích vì sao ảnh hưởng đến generation/render/grading hoặc chất lượng plan.",
                "Không dùng như fallback mơ hồ.",
                "Nếu có thể map vào issue type có sẵn thì ưu tiên issue type có sẵn.",
            ],
        },
    },
    "severity_policy": {
        "ok": "Không có vấn đề đáng kể; toàn bộ yêu cầu chính của source được bao phủ và generation khả thi.",
        "warning": "Có lỗi nhỏ hoặc điểm mơ hồ; generation vẫn có khả năng thực hiện được.",
        "needs_review": "Evidence chưa đủ hoặc có nhiều cách hiểu; cần người review trước khi kết luận.",
        "bad": "Có lỗi rõ ràng ảnh hưởng đến coverage, ý nghĩa toán học, generation, render hoặc chấm điểm.",
        "structural_error": "Thiếu/sai schema bắt buộc; có thể bỏ qua semantic judge.",
        "skipped_due_to_source_issue": "Bỏ qua đánh giá plan sâu vì source/raw record chưa hợp lệ.",
    },
    "special_cases": {
        "table_adaptation": [
            "Hệ thống không cần giữ nguyên layout bảng trực tiếp.",
            "Bảng nên được chuyển thành các interaction nhỏ hơn được hỗ trợ như short_answer, fill_blank hoặc matching.",
            "Không bỏ yêu cầu chính của bảng chỉ vì định dạng bảng khó xử lý.",
            "Nếu chỉ hỏi một số ô, requirement phải nói rõ phạm vi giới hạn đó.",
            "Không ghi `Hoàn thành bảng` khi plan thực tế chỉ hỏi vài ô.",
        ],
        "choice_interactions_at_plan_stage": [
            "single_choice/multiple_choice chưa cần options trong question_plan.",
            "Options và answerSpec được sinh ở bước sau.",
            "Không đánh sai choice chỉ vì source là bài tự luận hoặc tính toán.",
            "single_choice nên thể hiện việc chọn một đáp án/mệnh đề.",
            "multiple_choice nên thể hiện việc chọn nhiều hoặc tất cả đáp án phù hợp.",
        ],
        "full_main_requirement_coverage": [
            "Các ý a/b/c và những yêu cầu chính khác của source phải được biểu diễn.",
            "Được phép adapt định dạng; không được bỏ yêu cầu chính.",
            "Issue ở cấp interaction chỉ là evidence; final status ở cấp question/source-record.",
        ],
    },
    "good_examples": [
        "Source dạng bảng -> nhiều short_answer interactions cho toàn bộ các ô cần điền.",
        "single_choice: `Chọn nhận định đúng về số lượng nghiệm của phương trình.`",
        "multiple_choice: `Chọn tất cả các cặp (x; y) là nghiệm của phương trình trong các lựa chọn cho sẵn.`",
        "Source có a, b, c -> plan có item/interactions bao phủ đủ a, b, c, có thể đã được đổi định dạng.",
    ],
    "bad_examples": [
        "Source có a, b, c nhưng plan chỉ xử lý b.",
        "Requirement ghi `Hoàn thành bảng` nhưng plan chỉ hỏi một ô.",
        "`multiple_choice` với interactionRequirement mơ hồ `Giá trị (x, y)`.",
        "Plan chọn một dữ kiện rời rạc và bỏ các điều kiện cần để giải nhiệm vụ.",
    ],
    "judge_output_contract": {
        "record_fields": [
            "record_id",
            "record_name",
            "overall_status",
            "coverage_status",
            "source_subparts",
            "covered_subparts",
            "missing_subparts",
            "selected_scope_summary",
            "plan_quality_summary",
            "issues",
            "recommended_actions",
            "confidence",
        ],
        "issue_fields": [
            "issue_id",
            "issue_level",
            "location",
            "severity",
            "issue_type",
            "summary",
            "evidence",
            "impact_on_generation",
            "recommended_action",
            "suggested_fix",
            "requires_human_review",
            "confidence",
        ],
    },
    "evaluation_procedure": [
        "Phân tích raw question để xác định các yêu cầu chính/subparts chính của source.",
        "Phân tích raw answer, nếu có, để xác nhận các phần lời giải cần có.",
        "Map từng yêu cầu chính của source sang question_plan.",
        "Tạo coverage_issue nếu một yêu cầu chính không được map.",
        "Đánh giá source fidelity.",
        "Đánh giá cách decomposition thành questionStatement, questionItems và interactions.",
        "Đánh giá độ rõ của requirement và interactionRequirement.",
        "Đánh giá mức phù hợp của interactionType.",
        "Đánh giá tính khả thi cho generation/render/chấm tự động.",
        "Trả về JSON hợp lệ theo output contract.",
    ],
    "suggested_repair_actions": [
        "add_missing_item",
        "add_missing_interaction",
        "split_item",
        "split_question",
        "adapt_unsupported_format",
        "revise_question_statement",
        "revise_item_requirement",
        "revise_interaction_requirement",
        "change_interaction_type",
        "split_interaction",
        "merge_interactions",
        "reorder_items",
        "mark_for_human_review",
        "no_auto_fix",
    ],
}

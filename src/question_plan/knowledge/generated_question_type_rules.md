# Generated Question Type Rules

File này là knowledge pack cho LLM judge của service generated question.
Các rule ở đây chỉ dùng để đánh giá chất lượng nội bộ của generated question object.
Không so sánh với question_plan, raw question hoặc raw answer.

## single_choice
- Phải có `config.options`.
- Phải có đúng một đáp án đúng.
- `correctOptionId` phải tồn tại trong options.
- Distractors phải hợp lý, cùng kiểu dữ liệu/ngữ nghĩa, không quá vô lý hoặc quá lộ.
- Không có option trùng nghĩa.
- Không có hai option đều có thể đúng.
- Đáp án đúng không nên dễ đoán do luôn dài nhất, chi tiết nhất hoặc khác format bất thường.

## multiple_choice
- Phải có `config.options`.
- Phải có ít nhất 2 đáp án đúng.
- Nếu chỉ có 1 đáp án đúng thì nên dùng `single_choice`.
- Mọi `correctOptionIds` phải tồn tại trong options.
- Các đáp án sai phải là distractors hợp lý.
- Không có option mơ hồ khiến học sinh không biết có nên chọn hay không.
- Không có option trùng nghĩa.

## true_false
- Phải có một mệnh đề rõ ràng.
- Expected phải xác định rõ `true` hoặc `false`.
- Không dùng mệnh đề mơ hồ hoặc phụ thuộc diễn giải.
- `trueLabel`/`falseLabel` nếu có phải phù hợp.

## true_false_multi_statement
- Phải có nhiều statements.
- Mỗi statement phải có expected true/false riêng.
- Statements phải độc lập và rõ nghĩa.
- Không nên có statements quá giống nhau.
- Tránh pattern dễ đoán như tất cả đúng hoặc tất cả sai, trừ khi có lý do sư phạm rõ.

## fill_blank
- Blank phải nằm ở vị trí có ý nghĩa.
- Không để blank quá rộng khiến có nhiều đáp án không kiểm soát được.
- Expected answer phải rõ.
- Nếu có nhiều blank, answerSpecs phải map đúng từng blank.
- Câu còn lại sau khi bỏ blank vẫn phải đủ ngữ cảnh.

## short_answer
- Câu hỏi phải đủ rõ để học sinh biết cần nhập gì.
- Expected nên có `correctValue` hoặc cấu trúc tương đương.
- `inputMode` phải phù hợp: numeric, text, latex...
- Equivalence nếu có phải phù hợp.
- Không dùng short_answer cho câu có quá nhiều cách diễn đạt nếu không có acceptableValues/equivalence phù hợp.
- Nếu yêu cầu biểu thức toán, nên có latex/symbolic equivalence.

## essay
- `essay` được chấp nhận tạm thời trong generated question checker vì data mới có dạng tự luận.
- Không yêu cầu `config.options`, `correctOptionId` hoặc `correctOptionIds`.
- Stem phải nói rõ học sinh cần trình bày gì.
- Nên có `solutions`, `rubric`, `grading`, `modelAnswer`, `sampleAnswer` hoặc answerSpec/rubric tương đương.
- Nếu thiếu toàn bộ solution/rubric/grading/model answer thì đánh `needs_review`, không đánh unsupported type.
- Không đánh lỗi chỉ vì essay không có đáp án đúng dạng option.
- Nếu stem quá mơ hồ hoặc không biết tiêu chí chấm, đánh `needs_review` hoặc `bad` tùy mức độ ảnh hưởng.

## matching
- Phải có hai tập item để ghép.
- Số lượng item phải hợp lý.
- Mỗi cặp đúng phải rõ ràng.
- Không có nhiều cách ghép đúng nếu schema không thiết kế như vậy.
- Distractor nếu có phải hợp lý.
- Không để item hai bên quá lộ vì trùng từ khóa đơn giản.

## ordering
- Phải có danh sách item cần sắp xếp.
- Thứ tự đúng phải xác định rõ.
- Không có nhiều thứ tự đúng nếu không cho phép.
- Các item phải cùng loại.
- Không nên có item quá dễ nhận biết do đánh số sẵn.

## drag_drop
- Phải có draggable items và drop targets.
- Mỗi item/target phải có mapping rõ.
- Không có target/item gây mơ hồ nếu schema không hỗ trợ nhiều đáp án.
- Nội dung kéo-thả phải phù hợp với stem.
- Không dùng drag_drop nếu dạng khác đơn giản hơn và không mất ý nghĩa.

## number_line_range
- Phải có trục số/range config hợp lệ.
- Min/max/tick/step phải phù hợp.
- Expected range/point phải nằm trong trục.
- Đơn vị và điều kiện biến phải rõ.
- Không dùng nếu đáp án không biểu diễn tốt trên trục số.

## image_hotspot
- Phải có image/reference image hợp lệ.
- Hotspot expected phải có vùng tương đối rõ.
- Vùng chọn không quá nhỏ gây khó thao tác.
- Không có nhiều vùng đúng nếu schema chỉ cho một.
- Stem phải nói rõ cần chọn vùng nào.

## coloring_select
- Phải có đối tượng/vùng cụ thể để tô/chọn.
- Expected selected regions phải rõ.
- Không có vùng mơ hồ.
- Màu sắc không được là yếu tố duy nhất nếu gây khó tiếp cận.

## column_arithmetic
- Phải có phép tính dạng cột hợp lệ.
- Các chữ số hàng/cột phải align đúng.
- Carry/borrow nếu có phải được xử lý đúng.
- answerSpecs phải khớp từng ô/slot.
- Không thiếu bước quan trọng trong phép tính.

## choice_blank_fill
- Phải có blanks và danh sách choices.
- Mỗi blank phải có đáp án đúng từ choices.
- Choices sai phải hợp lý.
- Không có choice trùng nghĩa gây nhiều đáp án đúng.
- Không để số choices quá ít khiến đoán dễ.

## expression_transformation_step
- Mỗi bước biến đổi biểu thức phải hợp lệ.
- Expected step phải đúng.
- Không bỏ qua bước quan trọng nếu mục tiêu là kiểm tra quy trình.
- answerSpecs phải kiểm tra đúng phần cần biến đổi.
- Các bước không được mâu thuẫn nhau.

## operation_chain
- Chuỗi phép toán phải có thứ tự rõ.
- Mỗi operation phải hợp lệ.
- Kết quả trung gian/cuối phải đúng theo chain.
- Không để chain quá dài so với độ khó đã gắn.
- answerSpecs phải rõ đang hỏi bước nào.

## chart_draw
- Phải có yêu cầu vẽ chart rõ ràng.
- Dữ liệu đầu vào phải đủ để vẽ.
- Loại chart phải phù hợp với dữ liệu.
- Trục/nhãn/đơn vị phải rõ.
- Expected chart hoặc grading config phải đủ để chấm.

## coordinate_input
- `dimensions` phải phù hợp.
- `dimensionLabels` phải đúng.
- `slots` phải khớp expected coordinates/slotId.
- Nếu chỉ nhập một số đơn giản, cân nhắc short_answer trừ khi product yêu cầu slot.
- Không dùng coordinate_input 2D cho một đại lượng đơn.
- Numeric constraints như allowNegative/allowDecimal phải phù hợp với đáp án.

## graph_draw
- Phải có yêu cầu vẽ đồ thị rõ ràng.
- Loại đồ thị/phương trình/điểm/trục phải đủ thông tin.
- Axis min/max/tick/labels phải hợp lý.
- Expected graph/grading phải có thể chấm.
- Nếu chỉ hỏi công thức hoặc giá trị số thì không nên dùng graph_draw.

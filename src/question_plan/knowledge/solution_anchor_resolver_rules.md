# Quy tắc Solution Resolver

## Mốc canonical

- Không kiểm tra solution đúng hay sai về toán học/chuyên môn.
- Không tự giải lại bài từ instruction/stem.
- Chỉ dùng một kết luận explicit làm canonical khi solution khẳng định rõ đó là đáp án cuối cho đúng đại lượng/interaction được hỏi và cardinality phù hợp interaction type.
- Kết luận explicit không bắt buộc có từ “Vậy”; một câu như “cạnh còn lại bằng 25” vẫn là kết luận explicit vì gắn trực tiếp giá trị với đại lượng được hỏi.
- Không lấy số xuất hiện cuối, số trung gian, giá trị thử, hệ số hoặc kết quả phụ làm canonical chỉ vì nó đứng cuối solution.
- Nếu không xác định chắc kết luận đang trả lời đúng đại lượng/interaction, trả `needs_manual_review`; không ép đối chiếu answerSpec.
- Không dùng answerSpec để phủ định hoặc sửa solution.
- Nếu solution không có kết luận cụ thể, không đoán và trả `needs_manual_review`.

## Interaction type

- `single_choice`: hợp lệ khi có đúng một đáp án cuối. Nhiều số/phương trình/option trung gian không phải nhiều đáp án cuối. Chỉ manual khi kết luận thật sự nói nhiều đáp án như “A hoặc C”, “m=0 hoặc m=1”, “chọn A và B”.
- `multiple_choice`: nhiều đáp án cuối có thể hợp lệ; đối chiếu tập kết luận với `correctOptionIds`.
- `short_answer`, `fill_blank`, `coordinate_input`, `true_false`: dùng kết luận cụ thể làm canonical expected; manual nếu thiếu hoặc có các kết luận cuối mâu thuẫn.
- `essay`: không ép option/đáp án ngắn; chỉ cần model answer/solution đủ rõ theo schema hiện có.

## Map và đối chiếu

- Nếu solution kết luận bằng label, option id hoặc giá trị/nội dung, tự hiểu semantic và map sang option tương ứng.
- Không dùng regex/code rule; không phụ thuộc một mẫu câu tiếng Việt cố định.
- Trả `answer_spec_alignment="matched"` khi answerSpec khớp trực tiếp với kết luận solution.
- Trả `answer_spec_alignment="equivalent"` khi cách biểu diễn khác nhưng tương đương semantic; trường hợp này vẫn đặt `answerSpec_matches_solution=true` và không tạo field fix.
- Trả `answer_spec_alignment="mismatched"` khi answerSpec thực sự lệch kết luận solution; trường hợp này đặt `answerSpec_matches_solution=false` và tạo field fix/issue căn chỉnh theo các invariant hiện có.
- Chỉ enforce mismatch sau khi đã xác định chắc explicit final conclusion theo phần Mốc canonical; không dùng quy tắc “số cuối solution khác answerSpec”.
- Nếu answerSpec lệch canonical, tạo đúng một `solution_anchor_consistency`, intent `align_fields_to_solution`, kèm `fields_to_fix` đến expected hiện có.
- Nếu answerSpec đã khớp, không tạo issue answer mismatch và không tạo field fix.

## Hint alignment

- Chỉ kiểm tra hint khi solution đã resolved.
- Hint chỉ cần dẫn tới cách giải/canonical solution; không cần phù hợp distractor hoặc answerSpec đang sai.
- Nếu answerSpec sai nhưng hint đúng theo solution, chỉ sửa answerSpec.
- Nếu hint mâu thuẫn trực tiếp với solution, tạo `hint_quality` tại hint path, intent `align_hint_to_solution`.
- Nếu solution cần manual review, không emit hint alignment và không sửa hint.

## Solution presentation

- Generic Quality Judge sở hữu lỗi dài dòng, thử-sai, tự vấn hoặc đoạn nháp khi final answer vẫn rõ; resolver không emit trùng.
- Resolver chỉ dùng `solution_quality/needs_manual_review` khi thiếu kết luận hoặc cardinality kết luận không hợp lệ/mâu thuẫn thật sự.
- Khi manual review, không align answerSpec/options/hints và không tự tạo đáp án mới.

# Generated Question Quality Criteria

File này là knowledge pack tĩnh cho LLM judge khi đánh giá một generated question object.
Module này chỉ đánh giá chất lượng nội bộ của generated question đã sinh ra, không so sánh với `question_plan`, đề gốc hay đáp án gốc.

## Vai Trò

`generated question` là object câu hỏi tương tác hoàn chỉnh để render cho học sinh làm bài.
Judge cần kiểm tra object đó có đủ rõ, đúng schema, có interaction/config/answerSpecs nhất quán, có lời giải/hint hợp lý và có thể render/chấm được hay không.

## Scope

- Chỉ dùng generated question object và `schema_validation_result`.
- Không dùng `question_plan`.
- Không dùng raw question/raw answer.
- Không dùng source images/answer images.
- Không đánh giá source fidelity.
- Không tạo category `plan_alignment`.
- Không tạo category `source_fidelity`.
- Không tạo `new_question_plan`.
- Nếu thiếu bằng chứng để kết luận chắc chắn, dùng `needs_review`.

## 1. Answer Internal Consistency

- `answerSpecs` phải map tới interaction tồn tại.
- `answerSpecs[].interactionId` phải trỏ tới đúng interaction.
- Type của answerSpec phải tương thích với type của interaction.
- Expected value/correct option/correct coordinates phải khớp với options/config/solution trong generated question.
- Interaction cần chấm tự động không được thiếu answerSpec.
- Solution nếu có phải nhất quán với answerSpecs và options.

## 2. Interaction Schema / Config Validity

- Mỗi generated question cần có `questionItems`.
- Mỗi questionItem cần có `stem` và `interactions`.
- Mỗi interaction cần có `id`, `type`, `config`, `display`.
- Interaction type phải nằm trong danh sách allowed types của service.
- Config phải tương thích interaction type.
- Content blocks cần có shape hợp lệ: `id`, `type`, và nội dung/display/config phù hợp.
- Không có id duplicate trong cùng phạm vi nếu schema yêu cầu unique.

## 3. Choice Quality

- Options/distractors phải hợp lý theo chính stem và nội dung generated question.
- `single_choice` phải có đúng một đáp án đúng.
- `multiple_choice` phải có ít nhất hai đáp án đúng.
- Không có option trùng nghĩa hoặc nhiều option đều có thể đúng trong `single_choice`.
- Distractors không nên quá vô lý, quá lộ hoặc khác format bất thường.
- Đáp án đúng không nên dễ đoán vì luôn dài nhất, chi tiết nhất, hoặc trùng từ khóa với stem trong khi các option khác không có.

## 4. Hint Quality

- Hints phải đúng hướng với generated question và answerSpecs.
- Hints nên gợi mở từng bước.
- Hint đầu không nên lộ đáp án trực tiếp.
- Hints không được mâu thuẫn với solution/options/answerSpecs.

## 5. Solution Quality

- Solution phải giải đúng nội dung generated question.
- Solution phải khớp answerSpecs.
- Với choice interaction, đáp án nêu trong solution phải khớp `correctOptionId`/`correctOptionIds`.
- Solution không được thêm phần lạ làm lệch bài.
- Nếu solution có visual/graph, visual phải phù hợp với config và yêu cầu sinh ra.

## 6. Difficulty / Bloom Fit

- Độ khó và bloom nên phù hợp với thao tác học sinh phải làm trong generated question.
- Không gắn `remember` cho bài cần nhiều bước suy luận phức tạp.
- Không gắn difficulty quá thấp nếu stem/solution yêu cầu biến đổi dài hoặc nhiều khái niệm.
- Không gắn difficulty quá cao nếu chỉ nhận biết trực tiếp một dữ kiện đơn giản.

## 7. Render / Schema Safety

- Latex/math không có lỗi nghiêm trọng làm không render được.
- Không để null ở field mà schema không cho phép null.
- `display` và `config` phải đủ để UI render interaction.
- Visual block như graph/chart cần display/config đủ thông tin.

## 8. Pedagogical Quality

- Stem rõ ràng, không mơ hồ.
- Học sinh biết cần thao tác gì và trả lời theo định dạng nào.
- Đơn vị/đại lượng/định dạng câu trả lời rõ ràng.
- Câu hỏi không quá dễ đoán do pattern của options hoặc wording.
- Không tạo cognitive load không cần thiết so với mục tiêu câu hỏi.

## Severity Policy

- `bad`: lỗi làm câu hỏi sai, không chấm được, không render được, answerSpec sai rõ ràng, hoặc interaction không hợp lệ.
- `needs_review`: có khả năng sai hoặc mơ hồ cần người xem, nhưng chưa đủ chắc để kết luận bad.
- `warning`: chưa tối ưu nhưng vẫn có thể dùng.
- Không tạo issue nếu không có vấn đề đáng kể.

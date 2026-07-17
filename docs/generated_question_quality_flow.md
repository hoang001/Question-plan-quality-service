# Generated Question Quality Flow

## Scope

Chỉ xử lý generated question object/list/wrapper `generatedQuestions`. Không dùng question_plan, raw question/raw answer, PDF/OCR hoặc pipeline khác.

## Flow

```text
Generated question
→ Structural validator bằng code
→ LLM Solution Quality Judge (dùng Judge hiện có)
→ Solution quality gate
→ LLM Solution Resolver khi solution đạt gate
→ Normalize/deduplicate issues
→ Scoped repair nếu auto_repair=true
→ Compact output
```

Structural validator chỉ kiểm tra shape/reference/cardinality/render schema cơ bản; không đọc semantic solution hoặc hint.

Solution Resolver là nguồn semantic duy nhất cho:

- final answer và cardinality theo interaction type;
- map solution sang option/expected;
- answerSpec mismatch;
- hint alignment khi solution resolved;
- solution thiếu kết luận hoặc có nhiều đáp án cuối không hợp lệ.

Judge hiện có đã được thu hẹp thành Solution Quality Judge. Payload LLM đầu tiên chỉ gồm question ID cần thiết, instruction, stem, interaction type và solutions; không chứa answerSpecs, expected, options, hints, resolver result hoặc schema diagnostics. Judge kiểm tra dữ kiện đầu vào của solution, phép tính và suy luận cục bộ, bước chính, nhánh/điều kiện, độ đầy đủ, dữ liệu ngoài JSON, cùng lỗi dài dòng/thử-sai/tự vấn/đoạn nháp. Judge không đánh giá answerSpec, option, hint, distractor, render hoặc generic pedagogical quality.

Nếu Judge tạo `solution_quality/clean_solution_reasoning`, flow ưu tiên làm sạch rồi check lại solution. Nếu Judge tạo `solution_quality/needs_manual_review`, flow dừng semantic alignment và không gọi Resolver. Trường hợp thiếu bảng/hình/đồ thị cần thiết trong JSON cũng đi theo nhánh manual review này. Chỉ solution vượt qua gate mới được dùng để đối chiếu answerSpec/options/hints.

## Repair

- `align_fields_to_solution`: resolver resolved và có `fields_to_fix` hợp lệ; chỉ sửa answerSpec/expected.
- `align_hint_to_solution`: resolver resolved và hint mâu thuẫn trực tiếp với solution.
- `clean_solution_reasoning`: bỏ thử-sai/tự vấn/đoạn nháp, giữ nguyên final answer.
- `fix_schema`: patch structural/render nhỏ.
- Không full repair, không tự giải lại bài, không tạo đáp án mới.
- Resolver `needs_manual_review` thì không sửa answerSpec/options/hints/solution.
- Thứ tự repair là solution trước, sau đó mới đến `align_fields_to_solution`, `align_hint_to_solution` và các sửa chữa khác.

## Issue/output

Issue được deduplicate theo `category + location + repair_intent + normalized reason`. `issues` là danh sách public duy nhất. Debug chỉ thêm kết quả resolver dạng ngắn, trạng thái repair, patch, số vòng và lý do dừng; không trả `error_snippet`, `required_context_paths`, `fields_to_fix`, danh sách issue lặp trong resolver hoặc `selected_issue`.

Compact item:

```json
{
  "id": "...",
  "is_good": true,
  "issues": [],
  "new_generated_question": null
}
```

Report mặc định chỉ có bảng một dòng mỗi record:

`# | ID | Status | Repair | Issues | Failed Reason | Suggestions`

## Public API/CLI

Generated endpoint chỉ có `strict_mode`, `debug`, `auto_repair`, `max_loop`; `max_loop` clamp 1..3.

```bash
python cli.py --evaluate-generated-questions-service --input data/processed/math_9_bt_test.json
python cli.py --evaluate-generated-questions-service --input data/processed/math_9_bt_test.json --auto-repair --max-loop 3
python cli.py --evaluate-generated-questions-service --input data/processed/math_9_bt_test.json --auto-repair --max-loop 3 --debug
```

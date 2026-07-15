# Generated Question Quality Flow

## Scope

Chỉ xử lý generated question object/list/wrapper `generatedQuestions`. Không dùng question_plan, raw question/raw answer, PDF/OCR hoặc pipeline khác.

## Flow

```text
Generated question
→ Structural validator bằng code
→ LLM Solution Resolver
→ LLM Generic Quality Judge
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

Generic Quality Judge chỉ kiểm tra clarity, presentation, option rỗng/duplicate/malformed, hint leakage, solution dài dòng/thử-sai/tự vấn và chất lượng sư phạm bề mặt. Judge không xác định đáp án đúng, không emit `answer_internal_consistency`, `solution_anchor_consistency` hoặc `fix_correct_option`. Distractor sai số mũ/hệ số/dấu/đơn vị có thể hợp lệ và không bị coi là typo chỉ vì gần canonical answer.

## Repair

- `align_fields_to_solution`: resolver resolved và có `fields_to_fix` hợp lệ; chỉ sửa answerSpec/expected.
- `align_hint_to_solution`: resolver resolved và hint mâu thuẫn trực tiếp với solution.
- `clean_solution_reasoning`: bỏ thử-sai/tự vấn/đoạn nháp, giữ nguyên final answer.
- `fix_schema`: patch structural/render nhỏ.
- Không full repair, không tự giải lại bài, không tạo đáp án mới.
- Resolver `needs_manual_review` thì không sửa answerSpec/options/hints/solution.

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

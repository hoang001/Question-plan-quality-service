# Generated Question Quality Flow

## Mục Tiêu

Module này đánh giá chất lượng nội bộ của generated question object. Nó không so sánh với `question_plan`, đề gốc, đáp án gốc, raw question hoặc raw answer.

Module không:

- gọi `evaluate_question_plan`;
- sửa `question_plan`;
- tạo `new_question_plan`;
- dùng raw question/raw answer;
- kiểm tra source fidelity với đề gốc;
- gom nhiều generated questions vào một LLM call batch.

## Input

Service nhận 3 dạng input:

- một generated question object trực tiếp;
- list generated question object;
- wrapper cũ có field `generatedQuestions`.

Các field ngoài generated question như `question_plan`, `question`, `answer`, `images`, `answer_images` nếu có trong wrapper sẽ không được copy vào prompt LLM.

## Luồng Xử Lý

```text
payload
-> normalize_generated_question_input
-> loop từng generated question object
-> deterministic schema/internal-consistency checks
-> LLM semantic quality judge nếu schema còn đánh giá được
-> optional LLM repair nếu bật auto_repair
-> validate new_generated_question nếu repair trả về candidate
-> normalize result
-> aggregate summary nếu input là list/multi
```

## Static Context

LLM judge load static context từ:

- `src/question_plan/knowledge/generated_question_quality_criteria.md`
- `src/question_plan/knowledge/generated_question_type_rules.md`
- `src/question_plan/knowledge/generated_question_output_schema.md`

Rule theo từng interaction type nằm trong `generated_question_type_rules.md`.
Khi cần cập nhật policy cho một interaction type, sửa file này trước, tránh hard-code rải rác trong prompt.

## Dynamic Context

Mỗi request gửi cho LLM chỉ gồm:

- `id`
- `_id`
- `aiId`
- `difficulty`
- `bloom`
- `interactionTypes`
- `instruction`
- `questionItems`
- `solutions`
- `schema_validation_result` hoặc `check_result`

Không gửi:

- `question_plan`
- raw question
- raw answer
- source images
- answer images

## Service Output

Khi input là list hoặc wrapper có nhiều generated questions, service trả:

```json
{
  "is_good": false,
  "failed_reason": [],
  "suggestions": [],
  "summary": {
    "total": 100,
    "good": 80,
    "bad": 5,
    "needs_review": 10,
    "warning": 5,
    "repaired": 2,
    "repair_failed": 1
  },
  "results": [
    {
      "id": "generated-id",
      "is_good": false,
      "failed_reason": [],
      "suggestions": [],
      "issues": [],
      "repair_status": "repaired",
      "repair_failed_reason": [],
      "repair_suggestions": [],
      "repair_loop_count": 1,
      "new_generated_question": {}
    }
  ]
}
```

Với single object, service có thể trả trực tiếp một result object để giữ compatibility.

## Output Files

### Markdown report

Dành cho người đọc. File này tóm tắt số câu, trạng thái good/bad/needs_review/warning, issue theo từng object, suggestion và repair status.

Ví dụ:

```bash
--report-output results/generated_question_repair_report.md
```

Nếu không truyền `--report-output`, CLI generated checker sẽ ghi report mặc định vào:

```text
results/outputs/generated_question_quality_report.md
```

### Full JSON result

Dành cho debug/tích hợp. File này chứa toàn bộ summary, issues, repair status và `new_generated_question` nếu repair thành công.

Ví dụ:

```bash
--output results/generated_question_check_repair_output.json
```

Nếu không truyền `--output`, CLI generated checker sẽ ghi mặc định vào:

```text
results/generated_question_check_repair_output.json
```

### Repaired objects JSON

Dành cho pipeline tiếp theo. File này chỉ chứa list `new_generated_question` đã sửa thành công, không bọc metadata và không chứa issues/report.

Ví dụ:

```bash
--repaired-output results/generated_question_repaired_objects.json
```

Quy tắc:

- Chỉ ghi object có `new_generated_question != null`.
- Mỗi phần tử là generated question object đầy đủ sau sửa.
- Giữ `id`/`_id` gốc nếu có.
- Không tự tạo file này nếu không truyền `--repaired-output`.
- Nếu có truyền `--repaired-output` nhưng không có object sửa thành công, file vẫn được ghi là `[]`.

### Repair mapping JSON

Dành cho trace/debug object nào được sửa.

Ví dụ:

```bash
--repair-mapping-output results/generated_question_repair_mapping.json
```

Format:

```json
[
  {
    "id": "generated-id",
    "repair_status": "repaired",
    "original_index": 0,
    "issue_count": 2,
    "new_generated_question_included": true
  }
]
```

## CLI

Check + repair mặc định:

```bash
python cli.py --evaluate-generated-questions-service --input data/processed/math_9_bt_test.json
```

Lệnh ngắn phía trên mặc định tương đương:

```bash
python cli.py --evaluate-generated-questions-service ^
  --input data/processed/math_9_bt_test.json ^
  --auto-repair ^
  --is-loop ^
  --max-loop 3 ^
  --output results/generated_question_check_repair_output.json ^
  --report-output results/generated_question_repair_report.md ^
  --repaired-output results/generated_question_repaired_objects.json ^
  --repair-mapping-output results/generated_question_repair_mapping.json
```

Nếu chỉ muốn check, không repair:

```bash
python cli.py --evaluate-generated-questions-service --input data/processed/math_9_bt_test.json --no-auto-repair
```

Non-strict mode:

```bash
python cli.py --evaluate-generated-questions-service --input data/processed/math_9_bt_test.json --no-strict-mode
```

## API

```bash
curl -X POST "http://localhost:8000/evaluate-generated-questions?strict_mode=true&auto_repair=false" ^
  -H "Content-Type: application/json" ^
  -d @data/processed/math_9_bt_test.json
```

## Categories

- `answer_internal_consistency`: answerSpecs không khớp interaction/options/solution/generated content.
- `interaction_schema`: config/type/display/answerSpecs không đúng schema.
- `choice_quality`: options/distractors/correct options không hợp lý.
- `hint_quality`: hints lộ đáp án, sai hướng hoặc không tăng dần.
- `solution_quality`: solution mâu thuẫn với generated question/answerSpecs/options.
- `difficulty_fit`: difficulty/bloom không phù hợp với thao tác cần làm.
- `render_schema`: lỗi id/type/text/latex/display/config.
- `pedagogical_quality`: câu hỏi mơ hồ, quá dễ đoán hoặc không phù hợp học sinh.
- `runtime`: lỗi runtime/parse/LLM.

Không dùng `plan_alignment` hoặc `source_fidelity` trong generated question checker.

## Strict Mode

- `bad` luôn làm `is_good=false`.
- `needs_review` làm `is_good=false` khi `strict_mode=true`.
- `warning` cũng làm `is_good=false` khi `strict_mode=true`.
- Khi `strict_mode=false`, chỉ `bad` làm `is_good=false`; warning/needs_review vẫn được trả trong `issues`.

## Summary Table

| Stage | Input | Static context | Dynamic context | Output |
|---|---|---|---|---|
| Input normalization | single/list/wrapper | input shape rules | generated question candidates | list generated question object |
| Schema validation | one generated question | deterministic schema rules | interactions, answerSpecs, options, content blocks | schema issues |
| Generated question judge | one generated question | criteria, type rules, output schema | generated object + schema_validation_result | LLM issues |
| Optional repair | one failed generated question | criteria, type rules | generated object + check_result | new_generated_question or null |
| Normalize result | schema issues + LLM issues + repair result | severity policy, strict_mode | merged issues | single result |
| Aggregate | list result | status policy | per-object result | summary + results |

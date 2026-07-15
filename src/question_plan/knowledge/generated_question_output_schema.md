# Generic Quality Judge Output Schema

Chỉ trả JSON object hợp lệ:

```json
{
  "id": "",
  "is_good": true,
  "failed_reason": [],
  "suggestions": [],
  "issues": [
    {
      "severity": "warning|needs_review|bad",
      "category": "choice_quality|hint_quality|solution_quality|render_schema|pedagogical_quality|runtime",
      "location": "",
      "reason": "",
      "suggestion": "",
      "error_snippet": "",
      "required_context_paths": [],
      "repair_intent": "improve_distractors|reduce_hint_leakage|clean_solution_reasoning|fix_schema|improve_stem_clarity|needs_manual_review"
    }
  ]
}
```

Quy tắc đường dẫn bắt buộc:

- `location` và mọi phần tử trong `required_context_paths` phải là JSON Pointer bắt đầu trực tiếp bằng `/`.
- Không dùng ký hiệu gốc `$`, `$.` hoặc `/$/`.
- Ví dụ đúng: `/solutions/0/solutionContent/0/text`, `/questionItems/0/hints/1/content/0/text`, `/questionItems/0/interactions/0/config/options/2/content/0/text`.
- Ví dụ sai: `/$/solutions/0`, `$.solutions[0]`, `/generatedQuestion/solutions/0`.

- Không trả `answer_internal_consistency`, `solution_anchor_consistency`, `fix_correct_option` hoặc semantic hint alignment.
- `solution_quality` chỉ dùng cho lỗi trình bày solution và intent `clean_solution_reasoning`.
- `choice_quality` không dùng để phán công thức/đáp án đúng sai.
- Mọi chuỗi diễn giải phải là tiếng Việt có dấu.

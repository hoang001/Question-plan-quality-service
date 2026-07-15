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

- Không trả `answer_internal_consistency`, `solution_anchor_consistency`, `fix_correct_option` hoặc semantic hint alignment.
- `solution_quality` chỉ dùng cho lỗi trình bày solution và intent `clean_solution_reasoning`.
- `choice_quality` không dùng để phán công thức/đáp án đúng sai.
- Mọi chuỗi diễn giải phải là tiếng Việt có dấu.

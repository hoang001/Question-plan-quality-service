# Generated Question Judge Output Schema

LLM phải trả về JSON object hợp lệ, không markdown, không giải thích ngoài JSON.

```json
{
  "is_good": true,
  "failed_reason": [],
  "suggestions": [],
  "issues": [
    {
      "severity": "warning|needs_review|bad",
      "category": "answer_internal_consistency|interaction_schema|choice_quality|hint_quality|solution_quality|difficulty_fit|render_schema|pedagogical_quality|runtime",
      "location": "",
      "reason": "",
      "suggestion": ""
    }
  ]
}
```

## Field Rules

- `is_good` phải là boolean.
- `failed_reason` là list string.
- `suggestions` là list string.
- `issues` là list object, có thể rỗng.
- Mỗi issue phải có đủ `severity`, `category`, `location`, `reason`, `suggestion`.
- Chỉ dùng severity/category trong enum cho phép.
- Không dùng category `plan_alignment`.
- Không dùng category `source_fidelity`.
- Toàn bộ `reason` và `suggestion` viết bằng tiếng Việt.
- Không đề xuất sửa `question_plan` trong output này.
- Không trả `new_question_plan`.

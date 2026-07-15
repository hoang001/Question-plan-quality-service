# Generated Question Scoped Repair Output Schema

Chỉ trả JSON object hợp lệ:

```json
{
  "repair_status": "repaired|failed|needs_manual_review",
  "failed_reason": [],
  "suggestions": [],
  "patches": [
    {
      "op": "replace|add|remove",
      "path": "",
      "value": null
    }
  ]
}
```

- Chỉ trả `repaired` khi patches không rỗng và có thể áp dụng an toàn.
- Path phải là JSON Pointer trong generated question gốc.
- Nếu context không đủ để sửa an toàn, trả `needs_manual_review`; không yêu cầu full repair.
- Không trả full generated question và không viết lại toàn bộ solution.
- Không tự giải bài từ instruction/stem.
- Không sửa raw question, raw answer, question_plan, source, difficulty hoặc bloom.
- Với `align_hint_to_solution`, patch chỉ được nằm trong `questionItems[].hints`.
- Với `clean_solution_reasoning`, chỉ làm sạch wording và giữ nguyên kết luận cuối theo `final_answer` của resolver.
- Chuỗi failed_reason/suggestions phải là tiếng Việt có dấu, trừ tên field/id/path/LaTeX.

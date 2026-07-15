# Generated Question Spelling Output Schema

LLM spelling judge phải trả JSON object hợp lệ, không markdown, không giải thích ngoài JSON.

```json
{
  "issues": [
    {
      "severity": "warning|needs_review|bad",
      "category": "spelling_wording",
      "location": "",
      "reason": "",
      "suggestion": "",
      "error_snippet": "",
      "corrected_text": null,
      "repair_intent": "fix_spelling_wording"
    }
  ]
}
```

## Field Rules

- `issues` là list object, có thể rỗng.
- `location` là JSON Pointer tới text field cần sửa, ví dụ `/questionItems/0/stem/0/text`.
- `corrected_text` chỉ chứa bản sửa chính tả/diễn đạt của chính text node đó.
- Không thay đổi math placeholder hoặc LaTeX trong `corrected_text`.
- Không dùng category khác ngoài `spelling_wording`.
- Toàn bộ chuỗi người đọc phải viết bằng tiếng Việt có dấu.
- Không dùng câu tiếng Anh trong output JSON, trừ tên field, id, enum, JSON Pointer, code hoặc LaTeX.

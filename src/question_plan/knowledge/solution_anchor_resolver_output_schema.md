# Solution Resolver Output Schema

Chỉ trả JSON object hợp lệ:

```json
{
  "resolver_status": "resolved|needs_manual_review",
  "final_answer": {
    "text": null,
    "matched_option_id": null,
    "correctOptionIds": [],
    "expected": null,
    "evidence_from_solution": ""
  },
  "answerSpec_matches_solution": true,
  "fields_to_fix": [
    {
      "path": "/questionItems/0/answerSpecs/0/expected/correctOptionId",
      "value": "opt_d",
      "reason": "",
      "suggestion": ""
    }
  ],
  "issues": [
    {
      "severity": "warning|needs_review|bad",
      "category": "solution_anchor_consistency|solution_quality|hint_quality",
      "location": "",
      "reason": "",
      "suggestion": "",
      "repair_intent": "align_fields_to_solution|align_hint_to_solution|clean_solution_reasoning|needs_manual_review"
    }
  ]
}
```

- `resolved`: solution có kết luận cuối cụ thể và cardinality phù hợp interaction type.
- `needs_manual_review`: solution thiếu kết luận hoặc có số đáp án cuối không phù hợp interaction type.
- `resolved` phải có ít nhất một giá trị trong `final_answer`.
- `evidence_from_solution` là trích đoạn ngắn từ solution.
- `fields_to_fix` chỉ dùng cho answerSpec/expected hiện có và bắt buộc có `value`.
- Chỉ kiểm tra/sửa hint khi `resolver_status=resolved`.
- Khi `needs_manual_review`, chỉ trả một issue `solution_quality` với intent `needs_manual_review`; để trống `fields_to_fix` và không emit mismatch/hint alignment.
- Mọi reason/suggestion/evidence phải là tiếng Việt có dấu, trừ field/id/path/LaTeX.

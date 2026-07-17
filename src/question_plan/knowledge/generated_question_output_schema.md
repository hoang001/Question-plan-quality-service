# Solution Quality Judge Output Schema

Chỉ trả JSON object hợp lệ:

```json
{
  "id": "",
  "is_good": true,
  "failed_reason": [],
  "suggestions": [],
  "issues": [
    {
      "severity": "warning|needs_review",
      "category": "solution_quality",
      "location": "/solutions/0/solutionContent/0/text",
      "reason": "",
      "suggestion": "",
      "error_snippet": "",
      "required_context_paths": ["/solutions/0/solutionContent/0/text"],
      "repair_intent": "clean_solution_reasoning|needs_manual_review"
    }
  ]
}
```

Quy tắc:

- `location` và mọi `required_context_paths` phải là JSON Pointer bắt đầu bằng `/` và trỏ tới solution block liên quan.
- Không dùng `$`, `$.`, `/$/` hoặc đường dẫn ngoài generated question.
- Dùng `clean_solution_reasoning` khi chỉ cần làm gọn trình bày mà vẫn giữ nguyên toàn bộ logic toán học và final answer.
- Dùng severity `needs_review` và `needs_manual_review` cho phép tính/biến đổi sai, thiếu bước chính, thiếu quá trình giải, thiếu nhánh/điều kiện, thiếu dữ liệu ngoài JSON hoặc solution không đủ để kiểm chứng.
- Nếu một lỗi gây sai dây chuyền, chỉ trả issue cho bước sai rõ ràng đầu tiên.
- Không trả category hoặc repair intent nào khác.
- Không tự tạo phép sửa toán học, bước giải hoặc final answer thay thế.
- Mọi reason/suggestion phải viết bằng tiếng Việt có dấu.

# Generated Question Repair Rules

- Chỉ sửa generated question object được cung cấp.
- Không dùng question_plan, raw question, raw answer, PDF, OCR hoặc dữ liệu ngoài.
- Chỉ dùng scoped patch; không full-object fallback và không tạo lời giải mới.
- Giữ nguyên id/_id; không sửa difficulty/bloom hoặc thêm metadata report vào object.
- `align_fields_to_solution` chỉ sửa answerSpec theo `fields_to_fix` đã được resolver trả và code xác thực. Không sửa solution hoặc nội dung option đã tồn tại.
- `align_hint_to_solution` chỉ sửa hint khi resolver đã resolved và issue xác nhận hint mâu thuẫn trực tiếp với solution.
- `clean_solution_reasoning` chỉ làm sạch wording thử-sai/tự vấn trong solution, giữ nguyên kết luận cuối theo `solution_anchor_result.final_answer`; không đổi answerSpec/options nếu chúng đã khớp.
- Nếu resolver là `needs_manual_review`, không đoán đáp án và không repair answerSpec/options/hints/solution.
- Nếu scoped context/path không đủ an toàn, trả `needs_manual_review`.
- Patch phải là JSON Patch áp dụng trên generated question gốc và chỉ chạm đúng phạm vi issue.
- Không tự giải lại bài từ instruction/stem để quyết định đáp án.
- Chuỗi diễn giải phải viết bằng tiếng Việt có dấu, trừ tên field/id/path/code/LaTeX.

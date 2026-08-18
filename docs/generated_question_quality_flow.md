# Luồng kiểm tra lời giải

```text
Generated Question
→ Structural Validator
→ Correctness Judge và Process/Presentation Judge chạy song song
→ mỗi output sai cấu trúc được gọi lại đúng Judge một lần để sửa hợp đồng
→ Code gom các nhận xét ứng viên
→ Review Judge kiểm chứng nhận xét nếu có
→ Review sai cấu trúc được gọi lại một lần để sửa hợp đồng
→ Code tổng hợp blocking/advisory và bỏ rejected
→ Quality Gate
→ Solution Resolver nếu gate mở và có dữ liệu interaction
→ Public output
```

Correctness và Process nhận trực tiếp đề bài cùng nguyên văn solution; không có Splitter, Transition Builder hoặc Code Analyzer. Correctness trả `opening_checks` và nhận xét toán học. Process chỉ trả nhận xét về quá trình, mức đầy đủ và cách trình bày. Hai Judge này không trả `good/bad/uncertain`.

Review Judge chỉ kiểm chứng từng nhận xét do Correctness và Process tạo ra; đề bài và lời giải là bằng chứng đối chiếu. Review không tự tìm hoặc bổ sung lỗi mới. Code ưu tiên nhận xét Correctness dạng `blocking`, chuyển `advisory` thành cảnh báo và bỏ `rejected`.

Không có bước chia state hoặc dựng transition.

## Chế độ so sánh prompt

Thêm `--no-prompt` khi chạy CLI để cả hai Judge chỉ nhận:

```text
Đề:
...

Lời giải:
...

Kiểm tra lời giải của bài trên
```

Chế độ này không truyền JSON schema, không parse output, không chạy Review Judge, validator, Aggregate, Resolver hoặc repair. Chuỗi `content` của mỗi model được ghi nguyên vào `correctness_output` và `process_output`.

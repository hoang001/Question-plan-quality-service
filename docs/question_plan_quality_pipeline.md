# Luồng đánh giá chất lượng câu hỏi hiện tại

```mermaid
flowchart TB
    A["Câu hỏi sinh ra"] --> B["Kiểm tra cấu trúc"]
    B --> C0["Code đánh dấu dấu câu, ký hiệu và dấu phẩy sát số"]
    C0 --> C["Gemma 12B chỉ chọn nối hoặc tách"]
    C --> E["Code dựng state, chọn stem và ghép các bước liền kề"]
    E --> F1["Gemma 26B tự kiểm tra tính đúng đắn toán học"]
    E --> F2["Gemma 12B kiểm tra độ đầy đủ và trình bày"]
    F1 --> G{"Hai output đúng cấu trúc?"}
    F2 --> G
    G -->|Không| H["Gọi LLM sửa cấu trúc một lần"]
    H --> I{"Output đã hợp lệ?"}
    I -->|Không| N
    G -->|Có| J["Tổng hợp kết quả"]
    I -->|Có| J
    J --> K{"Solution đạt quality gate?"}
    K -->|Không| O["Trả kết quả"]
    K -->|Có| L["Gemma 26B đối chiếu đáp án"]
    L --> O
```

Correctness nhận ngữ cảnh đề và các transition theo thứ tự, bắt buộc điền checklist cho từng cặp `Đề → A`, `A → B`, ... (reason trước, `is_valid` sau), tự kiểm tra context
Process nhận cùng chuỗi transition, bao gồm cặp đầu từ đề đến state 0, để kiểm tra thiếu cầu nối, bước thừa và chất lượng trình bày; Process không xác minh lại phép toán.
cùng toàn bộ toán học. Process/Presentation nhận nguyên văn các semantic state và tự quyết định lời
giải có thiếu bước chính hoặc có vấn đề trình bày hay không. Transition Builder
chỉ tổ chức dữ liệu, không sinh nhận định toán học cho hai model.

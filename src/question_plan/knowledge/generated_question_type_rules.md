# Generic Judge Type Rules

- Với choice interactions, chỉ đánh giá option ở mức trình bày: rỗng, duplicate/tương đương hoàn toàn, malformed hoặc không thể sử dụng.
- Không xác định option đúng và không đánh giá distractor chỉ vì nó sai/gần canonical answer.
- Với hint, chỉ đánh giá leakage, độ rõ và tiến trình gợi mở; không kiểm tra alignment đáp án.
- Với solution, chỉ đánh giá dài dòng, thử-sai, tự vấn hoặc đoạn nháp; không kiểm tra final answer.
- Với essay, chỉ đánh giá yêu cầu/rubric/model answer ở mức đủ rõ và có thể sử dụng; không ép option hay đáp án ngắn.
- Lỗi cấu trúc interaction/answerSpec do structural validator xử lý; Generic Judge không emit lại.

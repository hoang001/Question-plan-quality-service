1. Đánh giá chất lượng của một lời giải mẫu cho học sinh:
   - thiếu bước biến đổi chính hoặc thiếu diễn giải khiến chuỗi lời giải không thể theo dõi;
   - đoạn nháp hoặc tự vấn;
   - thử-sai chưa làm sạch;
   - mâu thuẫn trình bày;
   - lặp lại hoặc dài dòng nghiêm trọng;
   - wording khó hiểu, chưa hoàn chỉnh hoặc không phù hợp với học sinh.

2. Với `missing_major_step`:
   - Chỉ báo khi lời giải nhảy qua một hay nhiều biến đổi chính cần thiết mà không viết trạng thái trung gian và cũng không mô tả bằng văn bản đầy đủ thao tác theo đúng thứ tự.
   - Một bước có thể được thể hiện bằng công thức hoặc bằng lời giải thích. Không bắt buộc mỗi bước nằm ở content block hay dòng riêng.
   - Phải đọc toàn bộ nguyên văn state. Nếu biểu thức trung gian đã xuất hiện trong state thì không được nói rằng biểu thức đó bị thiếu.
   - Các từ nối chung chung như “ta có”, “biến đổi”, “suy ra”, “hay”, “nên” không tự thay thế cho một phép biến đổi bị lược bỏ.
   - Ví dụ thiếu bước: `2x^3 = 16 → x = 2`; `4x + 6 = 26 → x = 5`; hoặc khẳng định kết quả giới hạn sau nhân liên hợp nhưng không viết phép nhân liên hợp hay biểu thức rút gọn.
   - Ví dụ đủ bước: `2x^3 = 16. x^3 = 8. x = 2`; `2x = 16 nên x = 8`; hoặc thay trạng thái trung gian bằng câu “chia hai vế cho 2, rồi lấy căn bậc ba hai vế”.

3. Không kiểm tra lại phép tính, tính tương đương, điều kiện, nghiệm, định lý hoặc tính đúng sai của kết quả. Nếu bước đã được viết nhưng phép tính trong bước sai, đó thuộc Correctness Judge. Thiếu nghiệm, thiếu trường hợp, thiếu điều kiện hoặc áp dụng sai định lý cũng thuộc Correctness Judge, không phải `missing_major_step` ở đây.

4. Không đánh giá answerSpec, expected, options, hints hoặc metadata.

5. Không báo lỗi chỉ vì solution không có câu kết luận riêng như “Vậy ...”. Nếu đáp số hoặc kết quả đã xuất hiện trong các bước thì không được yêu cầu lặp lại thành câu kết luận.

6. Lỗi cục bộ dùng `scope=state`, đúng state và `evidence_text` nguyên văn. Lỗi toàn bộ lời giải dùng `scope=global`.

7. Chỉ trả lỗi process/presentation sớm nhất và rõ nhất. `good`: không có lỗi. `bad`: có lỗi chắc chắn. `uncertain`: thiếu dữ liệu đánh giá.

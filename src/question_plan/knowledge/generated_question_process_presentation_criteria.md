1. Đánh giá chất lượng của một lời giải mẫu cho học sinh:
   - thiếu bước biến đổi chính hoặc thiếu diễn giải khiến chuỗi lời giải không thể theo dõi;
   - đoạn nháp hoặc tự vấn;
   - thử-sai chưa làm sạch;
   - mâu thuẫn trình bày;
   - lặp lại hoặc dài dòng nghiêm trọng;
   - wording khó hiểu, chưa hoàn chỉnh hoặc không phù hợp với học sinh.

2. Với `missing_major_step`:

   - Báo khi giữa các đẳng thức sử dụng cùng một lúc từ 2 phép biến đổi trở lên, mỗi phép biến đổi CHỈ dùng một phép toán cơ bản: cộng, trừ, nhân, chia, lấy căn, mũ, liên hợp,....
   - Nếu lời giải chỉ nêu một phần các phép biến đổi cần thiết thì vẫn là `missing_major_step`. Việc viết trạng thái cuối hoặc dùng từ “suy ra” không thay thế cho phép biến đổi còn thiếu.
   - Khi báo lỗi, phải nêu chính xác suy luận nào bị thiếu và vì sao state sau không thể suy ra từ nội dung đã có. Nếu không chỉ ra được thì không báo lỗi.
   - Một bước có thể được thể hiện bằng công thức hoặc bằng lời giải thích. Không bắt buộc mỗi bước nằm ở content block hay dòng riêng.
   - Phải đọc toàn bộ nguyên văn state. Nếu biểu thức trung gian đã xuất hiện trong state thì không được nói rằng biểu thức đó bị thiếu.
   - Không báo lỗi chỉ vì thiếu công thức tổng quát, thiếu câu diễn giải sư phạm, thiếu phép tính nhẩm, gộp một chuyển vế/rút gọn thông thường hoặc không tách mỗi thao tác thành một dòng riêng, miễn là tiến trình vẫn kiểm chứng được.
   - Các từ nối chung chung như “ta có”, “biến đổi”, “suy ra”, “hay”, “nên” không tự thay thế cho một phép biến đổi bị lược bỏ.
   - Ví dụ thiếu bước: `2x^3 = 16 → x = 2`; `4x + 6 = 26 → x = 5`; hoặc khẳng định kết quả giới hạn sau nhân liên hợp nhưng không viết phép nhân liên hợp hay biểu thức rút gọn.
   - Ví dụ đủ bước: `2x^3 = 16. x^3 = 8. x = 2`; `2x = 16 nên x = 8`; hoặc thay trạng thái trung gian bằng câu “chia hai vế cho 2, rồi lấy căn bậc ba hai vế”.

3. Không kiểm tra lại phép tính, tính tương đương, điều kiện, nghiệm, định lý hoặc tính đúng sai của kết quả. Nếu bước đã được viết nhưng phép tính trong bước sai, đó thuộc Correctness Judge. Thiếu nghiệm, thiếu trường hợp, thiếu điều kiện hoặc áp dụng sai định lý cũng thuộc Correctness Judge, không phải `missing_major_step` ở đây.

4. Không đánh giá answerSpec, expected, options, hints hoặc metadata.

5. Không báo lỗi chỉ vì solution không có câu kết luận riêng như “Vậy ...”. Nếu đáp số hoặc kết quả đã xuất hiện trong các bước thì không được yêu cầu lặp lại thành câu kết luận.

6. Lỗi cục bộ dùng `scope=state`, đúng state và `evidence_text` nguyên văn. Lỗi toàn bộ lời giải dùng `scope=global`.

7. Chỉ trả lỗi process/presentation sớm nhất và rõ nhất. `good`: không có lỗi. `bad`: có lỗi chắc chắn. `uncertain`: thiếu dữ liệu đánh giá.

8. Với `redundant_step`:
   - Chỉ đánh giá bước có cần thiết, bị lặp hoặc làm luồng reasoning khó theo dõi; không tự xác minh lại phép toán.
   - Nếu transition đã có certificate `verified_valid + hard`, xem tính hợp lệ toán học của transition đó là dữ kiện đầu vào.
   - Không báo `missing_major_step` cho chính transition `verified_valid + hard` chỉ vì không có câu văn mô tả thao tác; khi biểu thức đích đã xuất hiện, đó là trạng thái trung gian tường minh.
   - Nếu nhận định state đích không cần thiết, không liên quan đến tiến trình hoặc có thể xóa mà hai state còn lại vẫn nối được, bắt buộc dùng `redundant_step + good`; không dùng `missing_major_step`.
   - Một bước đúng nhưng không cần thiết là advisory nhẹ: trả `error_type=redundant_step` cùng `verdict=good` và neo đúng state.
   - Không dùng `redundant_step` cho phép biến đổi sai. Chỉ trả `bad` khi nhiều bước thừa tạo vòng lặp, mâu thuẫn hoặc làm lời giải rất khó theo dõi; khi đó dùng loại lỗi nghiêm trọng phù hợp như `severe_repetition` hoặc `excessive_verbosity`.

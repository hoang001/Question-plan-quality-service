1. Các stage đã được tạo sẵn từ ordered states. Không chia lại, thêm, bớt, sửa hoặc đổi thứ tự stage. Stage `0` là đề bài; stage `1` có thể chỉ viết lại đề bài và vẫn được chấp nhận.

2. Từ stage `1`, kiểm tra từng `bieu_thuc_truoc → bieu_thuc_sau` theo thứ tự:
   - tự tính lại kết quả số học;
   - kiểm tra quan hệ suy luận và tính tương đương;
   - kiểm tra điều kiện áp dụng;
   - kiểm tra mất nghiệm, thêm nghiệm, thiếu nhánh hoặc thiếu trường hợp;
   - kiểm tra bước biến đổi cốt lõi bị lược bỏ.

3. Với mỗi stage, điền `kiem_tra_so_hoc` khi có phép tính; nếu không có phép tính thì đặt `null`. Luôn điền `kiem_tra_lap_luan` và `trang_thai_buoc`.

4. Chỉ kiểm tra stage tiếp theo khi `trang_thai_buoc=true`. Khi gặp `false` đầu tiên, trả stage đó rồi dừng; không trả hoặc nhận xét stage phía sau.

5. Không báo lỗi khi chỉ lược bỏ phép tính tiểu tiết có thể kiểm chứng trực tiếp, ví dụ:
   - `2x^3 + 3 = 19 → 2x^3 = 16`;
   - `3x = 12 → x = 4`;
   - không bắt buộc ghi riêng phép tính `19 - 3`.

6. Đặt `trang_thai_buoc=false` khi thiếu một trạng thái toán học cốt lõi cần thiết để thể hiện lập luận, che mất điều kiện hoặc nhánh nghiệm. Ví dụ, `2x^3 = 16 → x = 2` thiếu trạng thái `x^3 = 8`.

7. Báo lỗi tại chính stage làm mất nghiệm hoặc vi phạm điều kiện. Ví dụ:
   - `x^2 = 4 → x = 2` làm mất nghiệm `x = -2`;
   - `log_2(x^2) = 2log_2(x)` không hợp lệ nếu chưa có điều kiện `x > 0`.

8. Chỉ khi mọi stage hợp lệ mới đánh giá chất lượng tổng thể: thiếu kết luận hoặc nhánh quan trọng; đoạn nháp; tự vấn; thử-sai chưa làm sạch; lặp lại hoặc dài dòng nghiêm trọng; cách diễn đạt không phù hợp với học sinh.

9. Kết luận:
   - `good`: trả đủ mọi stage với `trang_thai_buoc=true`, `reason=""`, `suggestion=""`.
   - `bad`: có lỗi chắc chắn về toán học, logic, điều kiện, tính đầy đủ hoặc chất lượng.
   - `uncertain`: dữ liệu không đủ để xác nhận đúng hoặc sai.

Với `bad` hoặc `uncertain`, `reason` và `suggestion` phải có nội dung. Nếu một stage sai, chính stage đó phải có `trang_thai_buoc=false` và `reason`; nếu chỉ có lỗi chất lượng tổng thể thì trả đủ các stage với trạng thái `true`.

1. Context bắt buộc đã được Builder và code xử lý trước khi gọi Correctness. Không tự phát hiện hình, bảng, đồ thị hoặc dữ liệu tham chiếu bị thiếu. Không đánh giá answerSpec, expected, options, hints hoặc metadata.

2. Đọc các transition đúng thứ tự. Kiểm tra `initial premise → first state` trước, sau đó mới kiểm tra các state liền kề. Dừng tại lỗi correctness đầu tiên. Không dùng state phía sau hoặc đáp án cuối để hợp thức hóa lỗi đứng trước.

3. Một số transition có thể kèm `code_analysis` ở trạng thái `verified_invalid + hard`:
   - Đây chỉ là cảnh báo số học hoặc tương đương đại số chắc chắn để ưu tiên kiểm tra, không phải verdict.
   - Correctness Judge phải đọc nguyên văn toàn bộ state, tự kiểm chứng và tự xác định biểu thức bị cảnh báo có được dùng làm tiền đề hoặc kết luận hay không.
   - Chỉ được bác bỏ cảnh báo khi có bằng chứng rõ ràng trong lời giải rằng biểu thức bị sửa/bác bỏ, hoặc chỉ được trích dẫn/đặt làm giả thiết phản ví dụ và không được dùng cho bước sau.
   - Nếu không có bằng chứng rõ ràng hoặc biểu thức được dùng tiếp, phải xử lý nó như lỗi correctness.
   - Transition không có cảnh báo hard phải được đánh giá trực tiếp từ đề bài và lời giải; không suy ra rằng code đã xác nhận đúng.

4. Correctness chỉ đánh giá tính đúng đắn toán học:
   - phép tính số học, dấu, hệ số, đẳng thức và tính tương đương của phép biến đổi;
   - công thức, định lý, quy tắc và điều kiện áp dụng;
   - điều kiện xác định, mất hoặc sinh nghiệm, nghiệm ngoại lai;
   - thiếu nghiệm, nhánh, trường hợp hoặc điều kiện làm kết luận toán học không đầy đủ;
   - suy luận logic và việc kết quả cuối có trả lời đúng yêu cầu toán học ban đầu hay không.

5. Không đánh giá mức độ chi tiết sư phạm của lời giải:
   - Không báo `missing_major_step`; lỗi thiếu bước biến đổi, thiếu diễn giải hoặc khó theo dõi thuộc Process & Presentation Judge.
   - Không dùng `operation_count`, `max_operations_per_step`, `total_operation_count`, quy tắc one-operation-per-transition hoặc nhãn compressed để chọn verdict.
   - Không báo lỗi chỉ vì nhiều phép biến đổi nằm trong cùng một content block hoặc cùng một state.
   - Nếu một biểu thức hoặc trạng thái trung gian đã xuất hiện nguyên văn trong `bieu_thuc_sau`, không được nói rằng biểu thức hoặc trạng thái đó bị thiếu.
   - Vẫn phải báo lỗi nếu một biểu thức đã viết ra nhưng bản thân phép tính, phép biến đổi hoặc kết luận của nó sai.

6. Trước khi chọn trạng thái, bắt buộc tự kiểm chứng độc lập:
   - Tính lại từng phép tính số học, từng đẳng thức và từng kết quả số xuất hiện trong lời giải; không mặc định đúng chỉ vì lời giải viết trôi chảy hoặc tự nhất quán.
   - Kiểm tra công thức, định lý và quy tắc được chọn có đúng và có đủ điều kiện áp dụng hay không.
   - Kiểm tra trạng thái cuối có trả lời đầy đủ đúng yêu cầu ban đầu hay không, kể cả điều kiện xác định, nghiệm, trường hợp, đơn vị và phạm vi kết luận.
   - Chỉ sau các bước kiểm chứng trên mới chọn `good`, `bad` hoặc `uncertain`. Không dùng answerSpec, expected, options hoặc hints để suy ra kết luận.

7. `good` khi không có lỗi correctness. `bad` khi có lỗi toán học, logic, điều kiện, nghiệm hoặc nhánh chắc chắn. `uncertain` chỉ dùng khi nội dung hiện có không đủ để quyết định một transition mà Builder không coi là context dependency bên ngoài.

8. Không đánh giá độ dài, lặp lại, đoạn nháp, tự vấn, thử-sai, wording, ký hiệu trình bày, mức độ gộp bước hoặc yêu cầu phải có một câu kết luận riêng. Các vấn đề này thuộc Process & Presentation hoặc Resolver.

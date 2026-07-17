# Solution Quality Criteria

Judge chỉ đánh giá solution dựa trên instruction, stem và interaction type trong payload.

## Solution nhiều bước trong một text

- Một text block có thể chứa toàn bộ solution. Không giả định mỗi object, mỗi dòng hoặc mỗi block ID tương ứng với một bước.
- Đọc các text block theo `contentIndex` tăng dần, sau đó tự nhận diện các đơn vị lời giải bên trong từng text theo đúng thứ tự xuất hiện.
- Xuống dòng, bullet, số thứ tự, dấu suy ra/tương đương, chuỗi phương trình và các từ nối chỉ là tín hiệu ngữ nghĩa để LLM hiểu thứ tự; không phải quy tắc tách máy móc.
- Việc phân đoạn chỉ thực hiện nội bộ trong LLM call hiện có. Không yêu cầu code tách bước và không trả step index, line index, internal trace hoặc danh sách bước mới.
- Map lỗi bằng `path` của text block gốc trong payload, không dùng content ID và không tạo JSON Pointer giả ở mức dòng hoặc step.

## Trình tự kiểm tra bắt buộc

1. Xem dữ kiện và yêu cầu trong instruction/stem là đơn vị đứng ngay trước đơn vị lời giải đầu tiên, rồi áp dụng cùng checklist chuyển tiếp cho `instruction/stem → đơn vị đầu tiên`. Không yêu cầu solution chép lại đề bài, nhưng đơn vị đầu tiên phải trực tiếp suy ra từ dữ kiện, có phép tính chính xác và không gộp quá một phép biến đổi chính. Nếu cần từ hai phép biến đổi chính độc lập trở lên thì báo thiếu bước trung gian. Câu hỏi nhận biết trực tiếp không bị yêu cầu thêm bước khi không có phép biến đổi cần trình bày.
2. Với mỗi đơn vị lời giải tiếp theo, chỉ kiểm tra sau khi đơn vị trước đã hợp lệ và thực hiện checklist nội bộ theo đúng thứ tự:
   - Tính lại phép toán hoặc quan hệ trong đơn vị mới.
   - Xác nhận đơn vị mới thực sự suy ra từ đơn vị ngay trước; không dùng final answer hay bước phía sau để hợp thức hóa.
   - Xác định số phép biến đổi chính của chuyển tiếp. Chấp nhận tối đa một phép biến đổi chính; tính trực tiếp, rút gọn hệ số, chuẩn hóa ký hiệu hoặc viết lại tương đương trực tiếp trong chính phép biến đổi đó là thao tác vi mô.
   - Nếu phép tính sai, quan hệ không suy ra được hoặc cần từ hai phép biến đổi chính độc lập trở lên thì dừng toàn bộ kiểm tra toán học, chỉ trả issue cho chuyển tiếp đầu tiên này và không nhận xét bước sau/final answer.
   - Chỉ khi chuyển tiếp đúng mới đánh giá cách trình bày có đủ rõ để người học hiểu quan hệ suy ra hay không.
3. Với bài lập luận, mỗi chuyển tiếp phải áp dụng đúng quan hệ, định nghĩa, định lý hoặc quy tắc chuyên môn và có đủ tiền đề/điều kiện cần thiết.
4. Chỉ khi toàn bộ chuỗi toán học hợp lệ mới kiểm tra độ đầy đủ, nhánh nghiệm, điều kiện, trường hợp quan trọng và chất lượng trình bày toàn cục.
5. Trước khi trả issue, kiểm tra lại nhận định lỗi một lần. Nếu không đứng vững thì loại bỏ issue; reason không được khẳng định cùng một nội dung vừa sai vừa đúng.
6. Đây là nhiệm vụ xác minh trung lập. Nếu toàn bộ nội dung đạt thì trả `is_good=true` và `issues=[]`; câu hỏi nhận biết trực tiếp không bị bắt buộc có bước trung gian không tồn tại.

## Chất lượng trình bày

- Báo `solution_quality/clean_solution_reasoning` khi logic toán học và kết luận vẫn đầy đủ, rõ ràng nhưng solution dài dòng, lặp ý, tự vấn, thử-sai, chứa đoạn nháp hoặc wording không phù hợp để hiển thị cho người học.
- `clean_solution_reasoning` chỉ làm gọn trình bày; phải giữ nguyên các bước toán học cần thiết và final answer.
- Báo `solution_quality/needs_manual_review` với severity `needs_review` khi có phép tính/biến đổi sai, bước sau không suy ra từ bước trước, thiếu bước chính, chỉ nêu đáp án trong bài cần quá trình giải, thiếu nhánh/điều kiện hoặc solution không đủ thông tin để kiểm chứng.
- Không tự giải lại toàn bộ bài, không tạo final answer mới và không tự repair lỗi toán học hoặc bước còn thiếu.

## Dữ liệu ngoài JSON

- Nếu dữ liệu bảng, hình, sơ đồ hoặc đồ thị cần thiết đã được mô tả đầy đủ trong instruction/stem, tiếp tục kiểm tra bình thường.
- Nếu solution phụ thuộc dữ liệu ngoài JSON mà instruction/stem không cung cấp, không tự tưởng tượng dữ liệu; trả `solution_quality/needs_manual_review` và yêu cầu review theo dữ liệu gốc.

## Ngoài phạm vi

- Không đánh giá answerSpecs, expected, correctOptionId/correctOptionIds, options, distractors hoặc hints.
- Không kiểm tra solution có khớp answerSpec hay không.
- Không tạo `answer_internal_consistency`, `solution_anchor_consistency`, `fix_correct_option`, hint leakage, distractor quality hoặc generic pedagogical/render issue.

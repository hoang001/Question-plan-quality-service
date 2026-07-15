# Generic Generated Question Quality Criteria

Generic Quality Judge chỉ đánh giá chất lượng trình bày và sư phạm bề mặt của generated question.

## Không thuộc quyền Generic Judge

- Không xác định đáp án đúng và không tự giải bài.
- Không kiểm tra solution đúng/sai chuyên môn.
- Không tạo `answer_internal_consistency` hoặc `solution_anchor_consistency`.
- Không tạo `fix_correct_option`, không sửa hint theo answerSpec và không phủ định canonical answer từ solution.
- Hint alignment với solution thuộc Solution Resolver.

## Instruction và stem

- Yêu cầu phải rõ ràng, đủ dữ kiện hiển thị trong generated object và học sinh biết cần làm gì.
- Không báo lỗi difficulty/bloom.

## Choice presentation

- Báo `choice_quality` khi option rỗng, trùng nội dung, hai option tương đương hoàn toàn, hỏng render, vô nghĩa hoặc không thể dùng.
- Distractor được phép sai có chủ đích. Sai số mũ, hệ số, dấu, đơn vị hoặc phép biến đổi có thể là distractor tốt.
- Không coi distractor là typo chỉ vì gần giống canonical answer.
- Không dùng kiến thức toán để kết luận distractor “quá sai”, “công thức không chuẩn” hoặc yêu cầu đổi correct option.

## Hint presentation

- Chỉ kiểm tra hint có quá lộ đáp án, quá chung chung, khó hiểu hoặc không tạo tiến trình gợi mở hay không.
- Không kết luận hint sai theo answerSpec/options. Semantic hint alignment thuộc Solution Resolver và chỉ chạy khi solution resolved.

## Solution presentation

- Có thể báo `solution_quality` khi solution dài dòng, thử-sai, tự vấn, chứa đoạn nháp hoặc không phù hợp để hiển thị cho học sinh.
- Chỉ dùng intent `clean_solution_reasoning`; giữ nguyên final answer và không tự viết lời giải mới.
- Thiếu kết luận/cardinality đáp án thuộc Solution Resolver, không emit trùng.

## Render và sư phạm

- Báo lỗi render/schema rõ, content rỗng hoặc cấu hình không thể hiển thị.
- Báo lỗi wording/pedagogical surface đáng kể; không kiểm tra chính tả input như một flow riêng.
- Nếu thiếu bằng chứng, dùng `needs_review`; không tạo issue nếu object dùng được.

## Severity

- `bad`: không thể dùng/render hoặc thiếu dữ kiện hiển thị quan trọng.
- `needs_review`: mơ hồ đáng kể nhưng chưa đủ chắc.
- `warning`: chưa tối ưu nhưng vẫn dùng được.

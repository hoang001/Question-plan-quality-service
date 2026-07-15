# Generated Question Spelling/Wording Rules

LLM spelling judge chỉ kiểm tra lỗi chính tả và diễn đạt trong text node của generated question.

## Scope

- Chỉ dùng các text node được cung cấp.
- Không dùng question_plan, raw question, raw answer, source answer, source PDF/OCR.
- Không sửa đáp án, logic toán, answerSpecs hoặc interaction config.
- Không sửa id, JSON key, code, JSON Pointer.
- Không sửa nội dung bên trong math placeholder hoặc LaTeX segment.

## What To Check

- Lỗi chính tả tiếng Việt rõ ràng.
- Lỗi gõ, lặp từ, lặp dấu câu bất thường.
- Khoảng trắng sai trước/sau dấu câu.
- Câu quá khó hiểu do diễn đạt.
- Mojibake/encoding lỗi rõ ràng.

## What Not To Flag

- Thuật ngữ toán học đúng.
- Biến toán như x, y, m, n.
- LaTeX command hoặc biểu thức toán.
- Tên interaction type, id, enum.
- Cách viết hơi khác nhưng vẫn rõ nghĩa.

## Correction Policy

- `corrected_text` chỉ sửa chính tả/diễn đạt.
- Không đổi nghĩa chuyên môn.
- Không đổi đáp án.
- Không đổi math segment/LaTeX.
- Nếu không chắc, chỉ báo `needs_review` và để `corrected_text=null`.

## Language

- Toàn bộ `reason`, `suggestion`, `corrected_text` nếu có phải viết bằng tiếng Việt có dấu.
- Không dùng câu tiếng Anh, trừ tên field, id, enum, JSON Pointer, code hoặc LaTeX.

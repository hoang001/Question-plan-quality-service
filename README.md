# LLM Model Tester

Công cụ đánh giá chất lượng `question_plan` từ source record Toán.

Đường chạy chính hiện tại là **Question Plan Service**: nhận một record có `question_plan`, trả về đúng 4 field để tích hợp vào hệ thống chính. Các pipeline ghi report/CSV cũ vẫn còn trong repo để debug hoặc đối chiếu, nhưng không phải đường chạy chính.

## Cài Đặt

```bash
cd llm_model_tester
python -m pip install -r requirements.txt
copy .env.example .env
```

Cấu hình `.env`:

```env
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODELS_ENDPOINT=
LLM_CHAT_COMPLETIONS_ENDPOINT=

PRIMARY_JUDGE_MODEL=gemma-4-12b-it
FALLBACK_JUDGE_MODEL=qwen3.6-35b
USE_JUDGE_FALLBACK=true

REQUEST_TIMEOUT_SECONDS=60
```

## CLI Chính

File CLI chính là:

```bash
python cli.py ...
```

## Question Plan Service

Chạy với một record JSON:

```bash
python cli.py --evaluate-question-plan-service --input data/processed/one_record.json
```

Chạy với list record:

```bash
python cli.py --evaluate-question-plan-service --input data/processed/math_9_bt_test.json
```

Ghi output ra file:

```bash
python cli.py --evaluate-question-plan-service --input data/processed/math_9_bt_test.json --output results/question_plan_service_output.json
```


## Input

Input là một source record object:

```json
{
  "_id": "...",
  "name": "...",
  "question": "...",
  "images": [],
  "answer": "...",
  "answer_images": [],
  "question_plan": {
    "type": "advanced_question_plan",
    "plan": []
  },
  "start_page": 4,
  "end_page": 9
}
```

CLI service nhận được cả object đơn lẻ hoặc list object.

## Output

Output luôn có đúng 4 field:

```json
{
  "is_good": true,
  "failed_reason": [],
  "suggestions": [],
  "new_question_plan": null
}
```

Nếu `is_good = false`:

```json
{
  "is_good": false,
  "failed_reason": [
    "Hệ phương trình b vô nghiệm nhưng question_plan yêu cầu học sinh nhập giá trị x và y cụ thể."
  ],
  "suggestions": [
    "Sửa questionOrder 2 thành yêu cầu học sinh kết luận hệ phương trình vô nghiệm."
  ],
  "new_question_plan": {
    "type": "advanced_question_plan",
    "plan": []
  }
}
```

`new_question_plan` là object `question_plan` sau sửa. Service không trả full source record, không trả patch, không trả before/after, và không tự sửa dữ liệu gốc.

## Logic Chính

Service đánh giá `question_plan` theo các tiêu chí:

- Bao phủ đủ yêu cầu chính của đề gốc.
- Không làm sai ý nghĩa toán học.
- Không yêu cầu học sinh nhập đáp án không tồn tại hoặc không chấm được.
- `interactionRequirement` đủ rõ để bước generation tiếp tục sinh câu hỏi/options/answerSpec.
- `interactionType` phù hợp với hành động học sinh cần làm.
- Không dùng field ngoài schema `question_plan`.

Điểm quan trọng: nếu một phần đã xuất hiện trong plan nhưng hỏi sai cách, đó là lỗi chất lượng/answerability, không phải thiếu coverage. Ví dụ hệ c đã có ở `questionOrder 3` nhưng nghiệm tổng quát mà plan yêu cầu nhập một cặp `x/y` cụ thể thì chỉ sửa `questionOrder 3`, không thêm `questionOrder 4`.

## Code Chính

```text
src/question_plan_service.py
```

Entry point để tích hợp:

```python
from src.question_plan_service import evaluate_question_plan, evaluate_question_plans

result = evaluate_question_plan(record)
results = evaluate_question_plans(records)
```

Các module liên quan:

```text
src/question_plan_judge.py       # gọi LLM judge chất lượng plan
src/question_plan_repair.py      # wrapper tạo new_question_plan
src/question_plan_schema.py      # validate question_plan và normalize service output
src/question_plan_eval_schema.py # schema/normalize nội bộ của judge
src/question_plan_rule_validator.py
src/interaction_type_knowledge.py
src/llm_client.py
src/config.py
src/utils.py
```

## Legacy / Debug Pipelines

Các lệnh dưới đây vẫn còn để debug hoặc đối chiếu kết quả cũ.

Chạy report đánh giá `question_plan`:

```bash
python cli.py --evaluate-question-plan-service --input data/processed/math_9_bt_test.json
```

Output legacy nằm trong:

```text
results/outputs/plan_quality/
```

Chạy quick check source/raw không OCR:

```bash
python cli.py --run-quick-check --input data/processed/math_9_bt_test.json --skip-pdf
```

Chạy quick check có OCR:

```bash
python cli.py --run-source-record-pipeline --input data/processed/math_9_bt_test.json --pdf "path/to/book.pdf" --refresh-ocr-cache
```

Các output này phục vụ debug, không phải contract service.

## Kiểm Tra Model

```bash
python cli.py --list-models
python cli.py --ping
python cli.py --ping --model qwen3.6-35b
```

## Dữ Liệu

```text
data/processed/
```

JSON record/list record đã chuẩn bị để chạy service.

```text
data/raw/mentor/
```

File gốc mentor gửi, ví dụ PDF.

```text
data/ocr-raw/
```

Cache OCR, không nên commit.

## Git Hygiene

Không commit:

```text
.env
results/outputs/
results/overview/
results/quick_check/
data/ocr-raw/
.cache/
__pycache__/
```

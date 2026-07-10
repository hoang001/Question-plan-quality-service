# Question Plan Quality Service

Service đánh giá chất lượng `question_plan` từ một source record Toán.

Luồng chính hiện tại chỉ tập trung vào `question_plan`: nhận raw question, raw answer nếu có, và `question_plan`; sau đó trả về kết quả JSON gọn để hệ thống khác tích hợp.

## Cài đặt

```bash
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

## Chạy Service

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

Kiểm tra kết nối model:

```bash
python cli.py --list-models
python cli.py --ping
python cli.py --ping --model qwen3.6-35b
```

## Input

Input là một source record object hoặc list source record:

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

## Output

Output luôn có 4 field:

```json
{
  "is_good": true,
  "failed_reason": [],
  "suggestions": [],
  "new_question_plan": null
}
```

Nếu `is_good = false`, service trả lý do lỗi, gợi ý sửa, và `new_question_plan` nếu đủ an toàn để viết lại toàn bộ plan.

Service không tự sửa dữ liệu gốc, không trả patch, không trả before/after, và không ghi report/CSV.

## Luồng Xử Lý

1. Structural validation: kiểm tra `question_plan` có đúng shape/schema cơ bản không.
2. LLM judge: đánh giá chất lượng tổng thể của plan ở cấp source record.
3. Mapping-first evaluation: mapping từng yêu cầu chính của source sang vị trí trong plan trước khi kết luận coverage.
4. Repair suggestion: nếu có lỗi, LLM đề xuất cách sửa và có thể trả `new_question_plan`.
5. Service output: chuẩn hóa về 4 field `is_good`, `failed_reason`, `suggestions`, `new_question_plan`.

Điểm quan trọng: nếu một phần đã xuất hiện trong plan nhưng hỏi sai cách, đó là lỗi chất lượng/answerability, không phải thiếu coverage. Khi đó repair chỉ sửa đúng location đã match, không thêm questionOrder mới.

## Cấu Trúc Code

```text
cli.py
src/
  __init__.py
  question_plan/
    __init__.py
    flows/
      service.py
    infra/
      config.py
      llm_client.py
    logic/
      judge.py
      repair.py
      repair_suggester.py
      rule_validator.py
    knowledge/
      plan_knowledge.py
      interaction_type_knowledge.py
      prompts.py
      alias.py
    schemas/
      service_schema.py
      eval_schema.py
    shared/
      real_schema.py
      utils.py
```

## Entry Point Tích Hợp

```python
from src.question_plan.flows.service import evaluate_question_plan, evaluate_question_plans

result = evaluate_question_plan(record)
results = evaluate_question_plans(records)
```

## Git Hygiene

Không commit:

```text
.env
cache/
data/ocr-raw/
results/
__pycache__/
```

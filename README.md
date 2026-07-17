# Question Plan Quality Service

Service kiểm tra chất lượng `question_plan` và `generated question` bằng rule validation kết hợp LLM judge/repair.

Tài liệu này ưu tiên hướng dẫn chạy local để mentor hoặc service khác có thể gọi API/CLI.

## Cài Đặt

```bash
python -m pip install -r requirements.txt
copy .env.example .env
```

Cấu hình `.env` tối thiểu:

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

Không commit `.env`.

## Chạy API Local

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Kiểm tra service:

```bash
curl http://localhost:8000/health
```

Swagger UI:

```text
http://localhost:8000/docs
```

Nếu `.env` có `QUESTION_PLAN_API_KEY`, mọi request cần thêm header:

```bash
X-API-Key: <key>
```

## Generated Question Checker

Luồng này chỉ kiểm tra chất lượng nội bộ của generated question object. Nó không so sánh với `question_plan`, raw question, raw answer hoặc source PDF.

Input có thể là:

- một generated question object trực tiếp;
- list generated question object;
- wrapper có field `generatedQuestions`.

Các field ngoài generated question như `question`, `answer`, `question_plan`, `images`, `answer_images` nếu có trong wrapper sẽ không được gửi sang LLM.

### Chạy Bằng CLI

Lệnh ngắn mặc định:

```bash
python cli.py --evaluate-generated-questions-service --input data/processed/math_9_bt_test.json
```

Nếu muốn chỉ đánh giá một số object đầu tiên:

```bash
python cli.py --evaluate-generated-questions-service --input data/processed/math_9_bt_test.json --amount 50
```

Nếu bỏ trống `--amount`, CLI sẽ đánh giá toàn bộ input.

Lệnh trên mặc định chỉ check, không tự repair. Output JSON mặc định là compact: chỉ gồm `id`, `is_good`, `issues`, `new_generated_question`.

Các file output mặc định:

```text
results/generated_question_check_repair_output.json
results/generated_question_repair_report.md
results/generated_question_repaired_objects.json
```

Nếu muốn bật repair tự động:

```bash
python cli.py --evaluate-generated-questions-service --input data/processed/math_9_bt_test.json --auto-repair
```

Nếu muốn repair rồi check lại tối đa 3 vòng:

```bash
python cli.py --evaluate-generated-questions-service --input data/processed/math_9_bt_test.json --auto-repair --max-loop 3
```

Nếu muốn xem diagnostics nội bộ:

```bash
python cli.py --evaluate-generated-questions-service --input data/processed/math_9_bt_test.json --debug
```

Nếu muốn truyền output path riêng:

```bash
python cli.py --evaluate-generated-questions-service ^
  --input data/processed/math_9_bt_test.json ^
  --auto-repair ^
  --max-loop 3 ^
  --output results/generated_question_check_repair_output.json ^
  --report-output results/generated_question_repair_report.md ^
  --repaired-output results/generated_question_repaired_objects.json
```

### Ý Nghĩa Output

`generated_question_check_repair_output.json`

Compact result để tích hợp, gồm summary, issues đã dedupe/prioritize và `new_generated_question` nếu sửa được. Chạy thêm `--debug` nếu cần diagnostics nội bộ.

`generated_question_repair_report.md`

Report cho người đọc, tóm tắt good/bad/needs_review/warning, lỗi từng object, suggestion và trạng thái repair.

`generated_question_repaired_objects.json`

Chỉ chứa list generated question object đã sửa thành công. File này không có issues/report metadata, phù hợp để đưa sang pipeline tiếp theo.

### Chạy Bằng API/Swagger

Endpoint:

```text
POST /evaluate-generated-questions
```

Ví dụ curl chỉ check, không repair:

```bash
curl -X POST "http://localhost:8000/evaluate-generated-questions?strict_mode=true" ^
  -H "Content-Type: application/json" ^
  -d @data/processed/math_9_bt_test.json
```

API mặc định `auto_repair=false` để tránh sửa ngoài ý muốn. Muốn test repair trên Swagger hoặc curl thì bật:

```bash
curl -X POST "http://localhost:8000/evaluate-generated-questions?strict_mode=true&auto_repair=true&max_loop=3" ^
  -H "Content-Type: application/json" ^
  -d @data/processed/math_9_bt_test.json
```

Trong Swagger, endpoint generated question chỉ còn các query params public: `strict_mode`, `debug`, `auto_repair`, `max_loop`.

Các policy đã được internal hóa:

- Flow generated question chạy theo thứ tự structural validator → Solution Quality Judge (dùng Judge hiện có) → solution quality gate → Solution Resolver → normalize/repair.
- Nếu solution cần làm sạch hoặc review, flow xử lý solution trước và chưa căn chỉnh `answerSpecs`/options/hints; Resolver chỉ chạy khi solution vượt qua quality gate.
- Solution Resolver là nguồn semantic duy nhất khi đối chiếu `solutions` với `answerSpecs/options/hints`.
- Payload của Judge chỉ chứa ID cần thiết, instruction, stem, interaction type và solutions; không gửi answerSpecs, expected, options hoặc hints.
- Judge chỉ kiểm tra chất lượng, độ đầy đủ và tính nhất quán nội bộ của solution; không đánh giá distractor, hint leakage, option semantic quality hoặc generic pedagogical/render quality.
- Không tự giải lại bài từ `instruction/stem`.
- Không check spelling/wording của input ban đầu.
- Chỉ dùng spelling/render guardrail cho text do repair sinh ra.

## Question Plan Service

Endpoint:

```text
POST /evaluate-question-plan
POST /evaluate-question-plans
```

CLI:

```bash
python cli.py --evaluate-question-plan-service --input data/processed/math_9_bt_test.json
```

Với input dạng list, có thể thêm `--amount 50` để chỉ đánh giá 50 record đầu.

Output chuẩn:

```json
{
  "is_good": true,
  "failed_reason": [],
  "suggestions": [],
  "new_question_plan": null,
  "is_loop": false,
  "loop_count": 0
}
```

## Kiểm Tra Model

```bash
python cli.py --list-models
python cli.py --ping
python cli.py --ping --model qwen3.6-35b
```

## Cấu Trúc Chính

```text
cli.py
src/
  api.py
  question_plan/
    flows/
      service.py
      generated_question_service.py
    infra/
      config.py
      llm_client.py
      debug.py
    logic/
      judge.py
      repair.py
      rule_validator.py
      generated_question_judge.py
      generated_question_repair.py
      generated_question_schema.py
      generated_question_schema_inspector.py
    knowledge/
      plan_knowledge.py
      interaction_type_knowledge.py
      prompts.py
      generated_question_quality_criteria.md
      generated_question_output_schema.md
    schemas/
      service_schema.py
      eval_schema.py
    shared/
      real_schema.py
      utils.py
docs/
  generated_question_quality_flow.md
tests/
  test_generated_question_service.py
```

## Git Hygiene

Không commit:

```text
.env
cache/
results/
__pycache__/
.pytest_cache/
```

Khi chỉ push luồng generated question checker, ưu tiên stage các file:

```text
README.md
.gitignore
cli.py
src/api.py
src/question_plan/flows/generated_question_service.py
src/question_plan/logic/generated_question_judge.py
src/question_plan/logic/generated_question_repair.py
src/question_plan/logic/generated_question_schema.py
src/question_plan/logic/generated_question_schema_inspector.py
src/question_plan/knowledge/generated_question_quality_criteria.md
src/question_plan/knowledge/generated_question_output_schema.md
docs/generated_question_quality_flow.md
tests/test_generated_question_service.py
```

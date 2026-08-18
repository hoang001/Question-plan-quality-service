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
FALLBACK_JUDGE_MODEL=gemma-4-26b
SOLUTION_CORRECTNESS_MODEL=gemma-4-26b
PROCESS_PRESENTATION_MODEL=gemma-4-12b-it
SOLUTION_RESOLVER_MODEL=gemma-4-26b
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

CLI mặc định xử lý 2 generated question song song. Giá trị hiệu lực được giới hạn trong khoảng 1..2; dùng `--workers 1` để chạy tuần tự.

Nếu muốn chỉ đánh giá một số object đầu tiên:

```bash
python cli.py --evaluate-generated-questions-service --input data/processed/math_9_bt_test.json --amount 50
```

Nếu bỏ trống `--amount`, CLI sẽ đánh giá toàn bộ input.

Lệnh trên mặc định chỉ check, không tự repair. Output JSON mặc định là compact: chỉ gồm `id`, `is_good`, `issues`, `new_generated_question`.

Các file output mặc định:

```text
results/generated_question_check_repair_output.json
results/<input-stem>_report.md
results/generated_question_repaired_objects.json
```

Lệnh đầy đủ để chạy toàn bộ file mẫu, bật diagnostics và chỉ định rõ output:

```powershell
python cli.py --evaluate-generated-questions-service `
  --input data/processed/math_9_bt_test.json `
  --output results/math_9_bt_test_results.json `
  --report-output results/math_9_bt_test_report.md `
  --repaired-output results/math_9_bt_test_repaired.json `
  --workers 2 `
  --debug
```

Không truyền `--amount` nghĩa là xử lý toàn bộ object trong input.

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

Để chạy check-only và lưu toàn bộ input/output của từng stage (Structural Validator,
Correctness, Process, contract correction nếu có,
Aggregate, Quality Gate và Resolver) vào một file JSONL dễ đối chiếu, dùng một dòng
PowerShell sau. `--workers 1` giữ log của các object theo đúng thứ tự; luồng repair
không chạy và không tạo file repaired:

```powershell
python cli.py --evaluate-generated-questions-service --input tests/judge_problem_solution_test_cases.json --output results/judge_problem_results.json --report-output results/judge_problem_report.md --trace-output results/judge_problem_pipeline_trace.jsonl --workers 1 --no-auto-repair
```

Mỗi dòng trong file trace là một JSON object đầy đủ, có `object_id`, `stage`, `event`
và `payload`. Terminal vẫn chỉ hiển thị tiến độ và đường dẫn output thông thường.

Nếu muốn truyền output path riêng:

```powershell
python cli.py --evaluate-generated-questions-service `
  --input data/processed/math_9_bt_test.json `
  --auto-repair `
  --max-loop 3 `
  --output results/generated_question_check_repair_output.json `
  --report-output results/math_9_bt_test_report.md `
  --repaired-output results/generated_question_repaired_objects.json
```

### Ý Nghĩa Output

`generated_question_check_repair_output.json`

Compact result để tích hợp, gồm summary, issues đã dedupe/prioritize và `new_generated_question` nếu sửa được. Chạy thêm `--debug` nếu cần diagnostics nội bộ.

`<input-stem>_report.md`

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

Trong Swagger, endpoint generated question có các query params public: `strict_mode`, `debug`, `auto_repair`, `max_loop` và `workers`.

Các policy đã được internal hóa:

- Flow generated question chạy theo thứ tự structural validator → Correctness và Process chạy song song trên nguyên văn solution → aggregate → solution quality gate → Solution Resolver → normalize/repair.
- Correctness và Process nhận trực tiếp đề bài cùng toàn bộ solution; không chia state và không dựng transition.
- Transition Builder chỉ ghép đề bài và các trạng thái lời giải liền kề; không chấm đúng sai bằng code.
- Gemma 26B Correctness trả checklist theo đúng chuỗi `Đề → A`, `A → B`, ...; mỗi phần tử chỉ gồm `transition_id`, `is_valid`, `reason`. Code kiểm tra đủ ID, đúng thứ tự và tự dựng verdict từ transition sai đầu tiên; LLM không trả error type, anchor tổng hợp hoặc suggestion. Gemma 12B Process & Presentation nhận cùng chuỗi transition (bao gồm `Đề → A`) để chấm độ đầy đủ và chất lượng trình bày nhưng không phán đúng/sai toán học.
- Chuỗi dài được code chia thành các lô liên tiếp: tối đa 20 transition cho Correctness và 24 transition cho Process & Presentation. Correctness dừng sau lô chứa lỗi đầu tiên; Process dừng sau lô chứa issue `bad/uncertain`. Mỗi lô sau giữ nguyên `transition_id` và không lặp lại `initial_transition`.
- Với interaction `essay`, hai Judge nhận thêm config và rubric/yêu cầu tự luận: Correctness kiểm tra nội dung bắt buộc, Process kiểm tra mạch lập luận và trình bày; Resolver không ép essay thành đáp án ngắn.
- Hai Judge dùng chung tối đa một contract-correction call cho mỗi object. Handoff còn non-canonical hoặc runtime sau correction sẽ fail closed trước Resolver.
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
python cli.py --ping 
```

Chạy chế độ không chèn prompt tiêu chí để so sánh:

```powershell
python cli.py --evaluate-generated-questions-service --input tests/test.json --output results/test_no_prompt.json --no-prompt
```

Trong chế độ này, message gửi Correctness và Process chỉ gồm đề, lời giải và câu `Kiểm tra lời giải của bài trên`. Service không truyền schema, không parse, không aggregate và không chạy Resolver; chuỗi model trả về được ghi nguyên vào `correctness_output` và `process_output`.

chạy đầy đủ:
 "python cli.py --evaluate-generated-questions-service --input tests/test.json --output results/test_results.json --report-output results/test_report.md --repaired-output results/test_repaired.json --debug"

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.question_plan.flows.service import evaluate_question_plan, evaluate_question_plans
from src.question_plan.infra.config import ConfigError, load_config
from src.question_plan.infra.llm_client import LLMClient


ROOT_DIR = Path(__file__).resolve().parent


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def resolve_input_path(input_path: str | None) -> Path:
    if not input_path:
        raise ValueError("Cần truyền --input là đường dẫn tới file JSON record hoặc list record.")
    path = Path(input_path)
    return path if path.is_absolute() else ROOT_DIR / path


def load_json_payload(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def write_json_output(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Service đánh giá chất lượng question_plan.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--evaluate-question-plan-service", action="store_true", help="Đánh giá question_plan và trả JSON 4 field.")
    action.add_argument("--list-models", action="store_true", help="Liệt kê các model có sẵn từ LLM endpoint.")
    action.add_argument("--ping", action="store_true", help="Gửi một request nhỏ để kiểm tra kết nối model.")
    parser.add_argument("--input", default=None, help="Đường dẫn tới file JSON object hoặc list object source record.")
    parser.add_argument("--output", default=None, help="Ghi JSON output của service ra file thay vì stdout.")
    parser.add_argument("--model", default=None, help="Model dùng cho --ping. Nếu bỏ trống sẽ dùng PRIMARY_JUDGE_MODEL.")
    return parser


def ping_model(client: LLMClient, model: str) -> None:
    response = client.chat_completion(
        model=model,
        messages=[{"role": "user", "content": "Xin chào, hãy trả lời ngắn gọn bằng tiếng Việt."}],
        temperature=0,
    )
    preview = response["content"].strip().replace("\n", " ")[:200]
    print(f"[OK] {model} ({response['latency_seconds']}s): {preview}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = load_config(ROOT_DIR)
        client = LLMClient(config)

        if args.list_models:
            for model_id in client.list_models():
                print(model_id)
            return 0

        if args.ping:
            ping_model(client, args.model or config.primary_judge_model)
            return 0

        if args.evaluate_question_plan_service:
            input_path = resolve_input_path(args.input)
            payload = load_json_payload(input_path)
            if isinstance(payload, list):
                result = evaluate_question_plans(payload, config=config, client=client)
            elif isinstance(payload, dict):
                result = evaluate_question_plan(payload, config=config, client=client)
            else:
                raise ValueError("--input phải là JSON object hoặc list object.")

            if args.output:
                output_path = Path(args.output)
                if not output_path.is_absolute():
                    output_path = ROOT_DIR / output_path
                write_json_output(output_path, result)
                print(f"Đã ghi service output: {output_path}")
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

    except ConfigError as exc:
        print(f"Lỗi cấu hình: {exc}")
        return 2
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

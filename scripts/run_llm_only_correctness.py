"""Gọi trực tiếp một LLM để chấm solution, không chạy pipeline của service.

Runner này cố ý KHÔNG import bất kỳ logic nào trong ``src/question_plan`` và
không chạy Process /
Presentation Judge, Aggregate, Resolver, parser hay Pydantic contract.

Mỗi object tạo đúng một HTTP request. Response được lưu nguyên bản để phục vụ
so sánh khả năng semantic thuần của model với pipeline đầy đủ.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ENDPOINT = "/v1/chat/completions"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SYSTEM_PROMPT = """Bạn là một chuyên gia kiểm định chất lượng giáo dục và là giáo
viên Toán học có tính cách cực kỳ cẩn thận, nghiêm túc. Nhiệm vụ của bạn là kiểm tra
tính chính xác của lời giải toán do học sinh nộp trực tiếp từng dòng một.

Nhiệm vụ:
- Xác định solution là good, bad hay uncertain.
- Nếu bad, chỉ ra lỗi toán học hoặc lỗi lập luận thực sự xuất hiện sớm nhất.
- Lời giải mẫu phải trình bày đủ các biến đổi chính cần thiết để học sinh theo dõi.
- Không coi ranh giới content block hoặc câu văn là ranh giới bắt buộc của một bước.
- Một đoạn có thể chứa nhiều phép biến đổi nếu các trạng thái trung gian hoặc thao
  tác đã được viết rõ và chuỗi suy luận có thể kiểm chứng.
- Không báo missing step nếu biểu thức trung gian cần thiết đã xuất hiện nguyên văn.
- Chỉ báo thiếu bước khi solution thật sự nhảy qua một biến đổi chính mà không ghi
  trạng thái trung gian và cũng không mô tả thao tác tương đương bằng lời.
- Không bắt lỗi về nội dung nháp, tự vấn, lặp, wording hoặc hình thức trình bày.

Chỉ trả một JSON object, không dùng Markdown và không viết thêm văn bản:
{
  "status": "good | bad | uncertain",
  "reason": [],
  "suggestion": []
}

Nếu status=good thì reason và suggestion phải là null. Nếu status=bad hoặc
uncertain thì reason phải giải thích ngắn gọn; suggestion chỉ bắt buộc khi bad.
"""

BARE_SYSTEM_PROMPT = """Bạn là người đánh giá lời giải toán. Dựa duy nhất vào đề bài và lời giải được cung cấp, hãy xác định lời giải đúng, sai hoặc chưa đủ chắc chắn. Nếu sai, nêu lỗi đầu tiên và cách sửa ngắn gọn.

Chỉ trả một JSON object, không dùng Markdown và không viết thêm văn bản:
{
  "status": "good | bad | uncertain",
  "reason": [],
  "suggestion": []
}

Nếu status=good thì reason và suggestion là null."""


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def _objects(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("generatedQuestions"), list):
        payload = payload["generatedQuestions"]
    elif isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        payload = payload["cases"]
    elif isinstance(payload, dict):
        payload = [payload]

    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(
            "Input phải là generated-question object, list object hoặc wrapper "
            "generatedQuestions/cases."
        )
    return payload


def _object_id(item: dict[str, Any]) -> str:
    value = item.get("_id") or item.get("id")
    return str(value) if value is not None else "unknown"


def _content_blocks_text(blocks: Any) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    if not isinstance(blocks, list):
        return rendered
    for block in blocks:
        if not isinstance(block, dict):
            continue
        content: dict[str, Any] = {
            "id": block.get("id"),
            "type": block.get("type"),
        }
        for key in ("text", "html", "latex", "description", "alt", "url", "src"):
            value = block.get(key)
            if value not in (None, "", [], {}):
                content[key] = value
        # Giữ content không-text để model vẫn nhận mô tả bảng/đồ thị/hình nếu có.
        for key in ("data", "config", "columns", "rows", "series", "points"):
            value = block.get(key)
            if value not in (None, "", [], {}):
                content[key] = value
        rendered.append(content)
    return rendered


def _semantic_input(item: dict[str, Any]) -> dict[str, Any]:
    question_items: list[dict[str, Any]] = []
    for question_item in item.get("questionItems") or []:
        if not isinstance(question_item, dict):
            continue
        question_items.append(
            {
                "id": question_item.get("id"),
                "stem": _content_blocks_text(question_item.get("stem")),
            }
        )

    solutions: list[dict[str, Any]] = []
    for solution in item.get("solutions") or []:
        if not isinstance(solution, dict):
            continue
        solutions.append(
            {
                "solverName": solution.get("solverName"),
                "solutionContent": _content_blocks_text(solution.get("solutionContent")),
            }
        )

    # answerSpecs, interactions và expected cố ý không được đưa vào request.
    return {
        "id": _object_id(item),
        "instruction": _content_blocks_text(item.get("instruction")),
        "questionItems": question_items,
        "solutions": solutions,
    }


def _chat_url(base_url: str, configured_endpoint: str | None) -> str:
    endpoint = configured_endpoint or DEFAULT_ENDPOINT
    normalized_base = base_url.rstrip("/") + "/"
    return urljoin(normalized_base, endpoint.lstrip("/"))


def _short_response(response: requests.Response) -> str:
    text = response.text.strip().replace("\r", " ").replace("\n", " ")
    return text[:500]


def _extract_content(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    return message.get("content") if isinstance(message, dict) else None


def _call_once(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    started_at = time.perf_counter()
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"{url}: {exc}") from exc

    latency = round(time.perf_counter() - started_at, 3)
    if not response.ok:
        raise RuntimeError(
            f"{url}: HTTP {response.status_code}: {_short_response(response)}"
        )
    try:
        raw_response = response.json()
    except ValueError:
        raw_response = response.text
    return {
        "endpoint": url,
        "http_status": response.status_code,
        "latency_seconds": latency,
        "content": _extract_content(raw_response),
        "raw_response": raw_response,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chấm solution trực tiếp bằng một LLM; không chạy bất kỳ bước phân tích "
            "hoặc hậu xử lý nghiệp vụ nào của service."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="File generated-question JSON.")
    parser.add_argument(
        "--id",
        action="append",
        dest="ids",
        help="Chỉ chạy ID này; có thể lặp --id nhiều lần. Bỏ qua để chạy toàn file.",
    )
    parser.add_argument("--output", type=Path, help="File JSON lưu request và raw response.")
    parser.add_argument("--model", help="Model cần gọi; mặc định SOLUTION_CORRECTNESS_MODEL.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, help="Timeout HTTP; mặc định REQUEST_TIMEOUT_SECONDS.")
    parser.add_argument(
        "--prompt-mode",
        choices=("simple", "bare"),
        default="simple",
        help="bare chỉ gửi yêu cầu chấm chung, không kèm tiêu chí nghiệp vụ.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ dựng payload để kiểm tra; không gọi LLM.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    load_dotenv(ROOT_DIR / ".env")

    input_path = args.input if args.input.is_absolute() else ROOT_DIR / args.input
    selected = _objects(_load_json(input_path))
    if args.ids:
        selected_ids = set(args.ids)
        selected = [item for item in selected if _object_id(item) in selected_ids]
        found_ids = {_object_id(item) for item in selected}
        missing_ids = selected_ids - found_ids
        if missing_ids:
            raise ValueError("Không tìm thấy ID: " + ", ".join(sorted(missing_ids)))

    base_url = os.getenv("LLM_BASE_URL", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = (
        args.model
        or os.getenv("SOLUTION_CORRECTNESS_MODEL", "").strip()
        or os.getenv("FALLBACK_JUDGE_MODEL", "").strip()
        or "gemma-4-26b"
    )
    endpoint = os.getenv("LLM_CHAT_COMPLETIONS_ENDPOINT", "").strip() or None
    top_p = float(os.getenv("LLM_TOP_P", "0.1"))
    timeout_seconds = args.timeout or int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))

    if not args.dry_run and (not base_url or not api_key):
        raise ValueError(".env phải có LLM_BASE_URL và LLM_API_KEY.")

    url = _chat_url(base_url, endpoint) if base_url else ""
    results: list[dict[str, Any]] = []
    system_prompt = BARE_SYSTEM_PROMPT if args.prompt_mode == "bare" else SYSTEM_PROMPT
    for item in selected:
        semantic_input = _semantic_input(item)
        request_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(semantic_input, ensure_ascii=False, indent=2),
                },
            ],
            "temperature": 0,
            "top_p": top_p,
            "max_tokens": max(1, args.max_tokens),
        }
        record: dict[str, Any] = {
            "id": _object_id(item),
            "llm_only": True,
            "pipeline_steps_called": [],
            "request": request_payload,
        }
        if args.dry_run:
            record["response"] = None
        else:
            try:
                record["response"] = _call_once(
                    url=url,
                    api_key=api_key,
                    payload=request_payload,
                    timeout_seconds=timeout_seconds,
                )
            except RuntimeError as exc:
                # Không retry/fallback: lưu nguyên lỗi của đúng một lần gọi logic.
                record["response"] = {"runtime_error": str(exc)}
        results.append(record)

    output: dict[str, Any] = {
        "runner": "llm_only_correctness",
        "input": str(input_path),
        "model": model,
        "prompt_mode": args.prompt_mode,
        "dry_run": args.dry_run,
        "object_count": len(results),
        "results": results,
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2) + "\n"

    if args.output:
        output_path = args.output if args.output.is_absolute() else ROOT_DIR / args.output
    else:
        suffix = "_dry_run" if args.dry_run else ""
        output_path = ROOT_DIR / "results" / f"llm_only_{input_path.stem}{suffix}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

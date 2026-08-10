"""Chạy Correctness theo prompt production nhưng không chạy Code Analyzer.

Luồng của runner:
    generated question
    -> Gemma 12B Solution Splitter production
    -> Transition Builder production
    -> Gemma 26B Correctness với prompt/schema/validator production

Không chạy Code Transition Analyzer, Claim Role Verifier, Process & Presentation
Judge, Aggregate hoặc Resolver. Transition payload gửi cho Correctness không có
field ``code_analysis``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.question_plan.infra.config import (  # noqa: E402
    generated_question_correctness_model,
    generated_question_splitter_model,
    load_config,
)
from src.question_plan.infra.llm_client import LLMClient  # noqa: E402
from src.question_plan.logic.generated_question_judge import (  # noqa: E402
    CORRECTNESS_OUTPUT_INVARIANTS,
    CRITERIA_PATH,
    FIRST_TRANSITION_RULE,
    JSON_SERIALIZATION_CONSTRAINTS,
    _call_solution_splitter,
    _prompt_json,
    build_transition_stages,
    compact_generated_question_payload,
    correctness_response_format,
    load_text,
    validate_correctness_semantic_output,
)
from src.question_plan.shared.utils import parse_json_output  # noqa: E402


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


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
        raise ValueError("Input phải là generated-question object hoặc list object.")
    return payload


def _object_id(item: dict[str, Any]) -> str:
    return str(item.get("_id") or item.get("id") or "unknown")


def _criteria_without_analyzer(criteria_text: str) -> str:
    """Giữ policy production, chỉ thay các đoạn phụ thuộc Code Analyzer."""

    sections = re.split(r"(?m)(?=^\d+\. )", criteria_text.strip())
    rendered: list[str] = []
    for section in sections:
        if section.startswith("3. "):
            rendered.append(
                "3. Tự kiểm tra độc lập từng transition theo đúng thứ tự và tự "
                "quyết định toàn bộ tính đúng đắn semantic và cơ học.\n\n"
            )
            continue
        if section.startswith("4. "):
            section = section.replace(
                "Correctness tự đánh giá những phần code không kết luận:",
                "Correctness tự đánh giá đầy đủ:",
            )
        elif section.startswith("5. "):
            section = section.replace(
                "Với transition `unsupported/ambiguous/parse_error`, vẫn áp dụng",
                "Với mọi transition, áp dụng",
            )
        elif section.startswith("6. "):
            section = (
                "6. Với phép tính số học nằm trong bài tự luận, tự tính lại từng "
                "phép và chỉ trả verdict semantic cuối cùng.\n\n"
            )
        rendered.append(section)
    return "".join(rendered).strip()


def _first_transition_rule_without_analyzer() -> str:
    return "\n".join(
        line
        for line in FIRST_TRANSITION_RULE.splitlines()
        if "code_analysis" not in line
    )


def build_full_correctness_messages_without_analyzer(
    generated_question: dict[str, Any],
    transition_payload: dict[str, Any],
) -> list[dict[str, str]]:
    """Bản production prompt chỉ bỏ mọi chỉ dẫn/input về Code Analyzer."""

    payload = compact_generated_question_payload(generated_question)
    payload.pop("solutions", None)
    stages = {
        "stages": [
            {
                name: value
                for name, value in stage.items()
                if name not in {
                    "code_analysis",
                    "claim_role_verification",
                    "mandatory_code_issue",
                }
            }
            for stage in transition_payload.get("stages") or []
        ]
    }
    criteria_text = _criteria_without_analyzer(load_text(CRITERIA_PATH))

    return [
        {
            "role": "system",
            "content": (
                "Bạn là Gemma Correctness Judge. Chỉ kiểm tra tính đúng đắn của đề và lời giải; "
                "context bắt buộc đã được code xác nhận đầy đủ trước khi gọi bạn. "
                "Không đánh giá context availability hoặc chất lượng trình bày tổng thể. "
                "Bạn phải tự kiểm tra và quyết định toàn bộ tính đúng đắn semantic và cơ học. "
                "Chỉ trả JSON đúng schema, không sửa dữ liệu."
            ),
        },
        {
            "role": "user",
            "content": (
                f"### TIÊU CHÍ\n{criteria_text}\n\n"
                "### HỢP ĐỒNG ĐẦU RA\n"
                "Chỉ trả 7 field semantic theo thứ tự: error_type, solution_index, "
                "from_order, to_order, reason, suggestion, status.\n\n"
                f"### NGỮ CẢNH CÂU HỎI\n{_prompt_json(payload)}\n\n"
                f"### LỜI GIẢI THEO THỨ TỰ\n{_prompt_json(stages)}\n\n"
                "### RÀNG BUỘC CUỐI\n"
                "- Chỉ trả lỗi correctness đầu tiên.\n"
                "- Đọc nội dung nguyên văn của mỗi state. Không bắt buộc tách mỗi bước thành "
                "content block hoặc dòng công thức riêng.\n"
                "- Nếu một state chứa nhiều phép biến đổi nhưng đã viết rõ các biểu thức trung gian "
                "hoặc mô tả đầy đủ từng thao tác theo đúng thứ tự thì có thể trả good.\n"
                "- Nếu chỉ dùng từ nối chung chung rồi nhảy qua từ hai phép biến đổi độc lập trở lên "
                "thì báo missing_major_step.\n"
                "- Khi trả anchor, phải sao chép đúng solution_index/from_order/to_order của một stage "
                "đã cung cấp; không tự tạo cặp order, không bỏ qua state ở giữa và chỉ dùng "
                "from_order=null cho chính initial_transition.\n"
                "- Không báo lỗi nháp, tự vấn, lặp lại, wording, ký hiệu `^` hoặc ký tự trình bày; "
                "các lỗi này không thuộc Correctness Judge.\n"
                "- Trước khi chọn trạng thái, bắt buộc tự tính lại từng phép tính và đẳng thức, kiểm tra "
                "điều kiện áp dụng của công thức hoặc định lý, rồi kiểm tra trạng thái cuối có trả lời "
                "đầy đủ yêu cầu ban đầu hay không.\n"
                "- Reason và suggestion viết bằng tiếng Việt.\n"
                "- reason tối đa 3 câu; suggestion tối đa 2 câu.\n"
                "- Chỉ ghi status sau khi đã tự kiểm tra xong quyết định semantic; status phải là field cuối.\n"
                "- Chỉ trả một JSON object đúng schema; không thêm field, không bỏ field bắt buộc.\n"
                "- Dùng null thay vì chuỗi \"None\"; enum phải đúng giá trị trong schema; không markdown.\n\n"
                f"{_first_transition_rule_without_analyzer()}\n\n"
                f"{CORRECTNESS_OUTPUT_INVARIANTS.replace(' hoặc code_analysis review', '')}\n\n"
                f"{JSON_SERIALIZATION_CONSTRAINTS}"
            ),
        },
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy full Correctness production nhưng loại bỏ Code Analyzer."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--id",
        action="append",
        dest="ids",
        help="Có thể lặp --id; bỏ qua để chạy toàn file.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = args.input if args.input.is_absolute() else ROOT_DIR / args.input
    output_path = args.output if args.output.is_absolute() else ROOT_DIR / args.output
    selected = _objects(_load_json(input_path))
    if args.ids:
        requested = set(args.ids)
        selected = [item for item in selected if _object_id(item) in requested]
        missing = requested - {_object_id(item) for item in selected}
        if missing:
            raise ValueError("Không tìm thấy ID: " + ", ".join(sorted(missing)))

    config = load_config(ROOT_DIR)
    client = LLMClient(config)
    splitter_model = generated_question_splitter_model(config)
    correctness_model = generated_question_correctness_model(config)
    records: list[dict[str, Any]] = []

    for item in selected:
        split, split_error = _call_solution_splitter(
            item,
            client,
            splitter_model,
            debug=args.debug,
        )
        if split is None:
            records.append(
                {
                    "id": _object_id(item),
                    "splitter_model": splitter_model,
                    "splitter_error": str(split_error),
                    "correctness_called": False,
                    "code_analyzer_called": False,
                }
            )
            continue

        transition_payload = build_transition_stages(split, item)
        messages = build_full_correctness_messages_without_analyzer(
            item,
            transition_payload,
        )
        response = client.chat_completion(
            model=correctness_model,
            messages=messages,
            temperature=0,
            response_format=correctness_response_format(),
            max_tokens=2048,
            retry_transient_once=False,
        )
        parsed, parse_ok, parse_error = parse_json_output(
            str(response.get("content") or "")
        )
        canonical = None
        validation_error = parse_error
        if parse_ok and parsed is not None:
            canonical, validation_error = validate_correctness_semantic_output(
                parsed,
                split,
                item,
                transition_payload,
            )

        records.append(
            {
                "id": _object_id(item),
                "splitter_model": splitter_model,
                "correctness_model": correctness_model,
                "code_analyzer_called": False,
                "claim_role_verifier_called": False,
                "process_presentation_called": False,
                "aggregate_called": False,
                "resolver_called": False,
                "ordered_solution": split,
                "transition_payload": transition_payload,
                "messages": messages,
                "response_format": correctness_response_format(),
                "raw_content": response.get("content"),
                "raw_response": response.get("raw_response"),
                "latency_seconds": response.get("latency_seconds"),
                "parse_ok": parse_ok,
                "canonical_result": canonical,
                "validation_error": validation_error or None,
            }
        )

    result = {
        "runner": "full_correctness_without_code_analyzer",
        "input": str(input_path),
        "object_count": len(records),
        "splitter_model": splitter_model,
        "correctness_model": correctness_model,
        "results": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

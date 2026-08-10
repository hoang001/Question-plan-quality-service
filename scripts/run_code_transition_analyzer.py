"""Chạy Code Transition Analyzer độc lập, không gọi bất kỳ model LLM nào."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.question_plan.logic.code_transition_analyzer import (  # noqa: E402
    analyze_code_transition,
    analyze_transition_stages,
)
from src.question_plan.logic.generated_question_judge import (  # noqa: E402
    build_transition_stages,
)


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
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("Input phải là generated-question object, list object hoặc generatedQuestions wrapper.")
    return payload


def _ordered_solution_from_blocks(generated_question: dict[str, Any]) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    for solution_index, solution in enumerate(generated_question.get("solutions") or []):
        order = 0
        for block_index, block in enumerate(solution.get("solutionContent") or []):
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            source_text = block.get("text")
            if not isinstance(source_text, str) or not source_text.strip():
                continue
            states.append(
                {
                    "solution_index": solution_index,
                    "order": order,
                    "source_path": (
                        f"/solutions/{solution_index}/solutionContent/{block_index}/text"
                    ),
                    "source_text": source_text,
                }
            )
            order += 1
    if not states:
        raise ValueError("Object không có solutionContent text để Analyzer kiểm tra.")
    return {"context_requirements": [], "states": states}


def _analyze_object(generated_question: dict[str, Any]) -> dict[str, Any]:
    ordered_solution = _ordered_solution_from_blocks(generated_question)
    stages = build_transition_stages(ordered_solution, generated_question)
    analyzed = analyze_transition_stages(stages)
    return {
        "id": generated_question.get("_id") or generated_question.get("id"),
        "llm_called": False,
        "state_source": "solutionContent_text_blocks",
        "analysis": analyzed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy riêng Code Transition Analyzer, không gọi Splitter/Judge/Resolver."
    )
    parser.add_argument("--before", help="Biểu thức hoặc trạng thái trước transition.")
    parser.add_argument("--after", help="Biểu thức hoặc trạng thái sau transition.")
    parser.add_argument("--input", type=Path, help="File generated-question JSON.")
    parser.add_argument("--id", help="Chỉ chạy object có _id/id tương ứng trong file list.")
    parser.add_argument("--output", type=Path, help="Ghi kết quả JSON thay vì chỉ in terminal.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    direct_mode = args.before is not None or args.after is not None
    if direct_mode:
        if args.before is None or args.after is None or args.input is not None:
            raise ValueError("Dùng đồng thời --before và --after, hoặc chỉ dùng --input.")
        result: Any = {
            "llm_called": False,
            "before": args.before,
            "after": args.after,
            "code_analysis": analyze_code_transition(
                before=args.before,
                after=args.after,
            ),
        }
    else:
        if args.input is None:
            raise ValueError("Cần --before/--after hoặc --input.")
        input_path = args.input if args.input.is_absolute() else ROOT_DIR / args.input
        selected = _objects(_load_json(input_path))
        if args.id:
            selected = [
                item
                for item in selected
                if str(item.get("_id") or item.get("id")) == args.id
            ]
            if not selected:
                raise ValueError(f"Không tìm thấy object id={args.id!r}.")
        object_results = [_analyze_object(item) for item in selected]
        result = object_results[0] if len(object_results) == 1 else object_results

    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output_path = args.output if args.output.is_absolute() else ROOT_DIR / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(str(output_path))
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

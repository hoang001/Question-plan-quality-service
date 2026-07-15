import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.question_plan.flows.generated_question_service import evaluate_generated_questions
from src.question_plan.flows.service import evaluate_question_plan, evaluate_question_plans
from src.question_plan.infra.config import ConfigError, load_config
from src.question_plan.infra.llm_client import LLMClient
from src.question_plan.logic.generated_question_schema_inspector import (
    format_generated_question_schema_summary,
    inspect_generated_question_records,
)


ROOT_DIR = Path(__file__).resolve().parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def resolve_input_path(input_path: str | None) -> Path:
    if not input_path:
        raise ValueError("Cần truyền --input là đường dẫn tới file JSON.")
    path = Path(input_path)
    return path if path.is_absolute() else ROOT_DIR / path


def load_json_payload(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def limit_payload_amount(payload: Any, amount: int | None) -> Any:
    if amount is None:
        return payload
    if amount <= 0:
        raise ValueError("--amount phải là số nguyên dương.")
    if isinstance(payload, list):
        return payload[:amount]
    if isinstance(payload, dict) and isinstance(payload.get("generatedQuestions"), list):
        limited = dict(payload)
        limited["generatedQuestions"] = payload["generatedQuestions"][:amount]
        return limited
    return payload


def write_json_output(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def markdown_text(value: Any) -> str:
    return str(value or "").strip().replace("|", "\\|").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def generated_result_status(result: dict[str, Any]) -> str:
    severities = {issue.get("severity") for issue in result.get("issues") or [] if isinstance(issue, dict)}
    if "bad" in severities:
        return "bad"
    if "needs_review" in severities:
        return "needs_review"
    if "warning" in severities:
        return "warning"
    return "good" if result.get("is_good") else "bad"


def generated_repair_status(result: dict[str, Any]) -> str:
    if isinstance(result.get("new_generated_question"), dict):
        return "repaired"
    status = str(result.get("repair_status") or "").strip()
    if status in {"failed", "needs_manual_review"}:
        return status
    intents = {
        str(issue.get("repair_intent") or "")
        for issue in result.get("issues") or []
        if isinstance(issue, dict)
    }
    return "needs_manual_review" if "needs_manual_review" in intents else "skipped"


def generated_result_list(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = result.get("results")
    if isinstance(rows, list):
        return [item for item in rows if isinstance(item, dict)]
    return [result]


def generated_report_summary(result: dict[str, Any]) -> dict[str, int]:
    rows = generated_result_list(result)
    status_counts = {"good": 0, "bad": 0, "needs_review": 0, "warning": 0}
    repair_counts = {"repaired": 0, "needs_manual_review": 0, "failed": 0, "skipped": 0}
    for item in rows:
        status = generated_result_status(item)
        status_counts[status] = status_counts.get(status, 0) + 1
        repair = generated_repair_status(item)
        repair_counts[repair] = repair_counts.get(repair, 0) + 1
    return {
        "total": len(rows),
        "good": status_counts["good"],
        "with_issues": len(rows) - status_counts["good"],
        "bad": status_counts["bad"],
        "needs_review": status_counts["needs_review"],
        "warning": status_counts["warning"],
        "repaired": repair_counts["repaired"],
        "needs_manual_review": repair_counts["needs_manual_review"],
        "repair_failed": repair_counts["failed"],
        "skipped": repair_counts["skipped"],
    }


def format_generated_question_markdown(result: dict[str, Any], *, source_name: str = "") -> str:
    lines: list[str] = ["# Generated Question Quality Report", ""]
    if source_name:
        lines.extend([f"Source file: `{source_name}`", ""])
    summary = generated_report_summary(result)
    lines.extend(
        [
            "## Tổng quan",
            "",
            "| Tổng object | Good | Có vấn đề | Bad | Needs review | Warning | Repaired | Manual review | Repair failed | Skipped |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {summary['total']} | {summary['good']} | {summary['with_issues']} | {summary['bad']} | "
                f"{summary['needs_review']} | {summary['warning']} | {summary['repaired']} | "
                f"{summary['needs_manual_review']} | {summary['repair_failed']} | {summary['skipped']} |"
            ),
            "",
            "## Chi tiết",
            "",
        ]
    )
    lines.extend(
        [
            "| # | ID | Status | Repair | Issues | Failed Reason | Suggestions |",
            "|---:|---|---|---|---|---|---|",
        ]
    )
    for index, item in enumerate(generated_result_list(result), start=1):
        issues = [issue for issue in item.get("issues") or [] if isinstance(issue, dict)]
        issue_text = "; ".join(
            f"{issue.get('category', 'unknown')}({issue.get('severity', 'warning')})"
            for issue in issues
        )
        reasons = list(dict.fromkeys(str(issue.get("reason") or "").strip() for issue in issues if issue.get("reason")))[:2]
        suggestions = list(
            dict.fromkeys(str(issue.get("suggestion") or "").strip() for issue in issues if issue.get("suggestion"))
        )[:2]
        lines.append(
            "| {index} | {id} | {status} | {repair} | {issues} | {reasons} | {suggestions} |".format(
                index=index,
                id=markdown_text(item.get("id") or f"item[{index - 1}]"),
                status=generated_result_status(item),
                repair=generated_repair_status(item),
                issues=markdown_text(issue_text),
                reasons=markdown_text("; ".join(reasons)),
                suggestions=markdown_text("; ".join(suggestions)),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def extract_repaired_generated_questions(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item["new_generated_question"]
        for item in generated_result_list(result)
        if isinstance(item.get("new_generated_question"), dict)
    ]


def write_generated_question_cli_outputs(
    *,
    result: dict[str, Any],
    source_name: str,
    output_path: Path,
    report_output_path: Path | None,
    repaired_output_path: Path | None,
) -> list[Path]:
    written = [output_path]
    write_json_output(output_path, result)
    if report_output_path:
        report_output_path.parent.mkdir(parents=True, exist_ok=True)
        report_output_path.write_text(format_generated_question_markdown(result, source_name=source_name), encoding="utf-8")
        written.append(report_output_path)
    if repaired_output_path:
        write_json_output(repaired_output_path, extract_repaired_generated_questions(result))
        written.append(repaired_output_path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Service đánh giá question_plan và generated questions.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--evaluate-question-plan-service", action="store_true")
    action.add_argument("--evaluate-generated-questions-service", action="store_true")
    action.add_argument("--inspect-generated-question-schema", action="store_true")
    action.add_argument("--list-models", action="store_true")
    action.add_argument("--ping", action="store_true")
    parser.add_argument("--input", default=None)
    parser.add_argument("--amount", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report-output", default=None)
    parser.add_argument("--repaired-output", default=None)
    parser.add_argument("--model", default=None, help="Model dùng riêng cho --ping.")
    parser.add_argument("--max-loop", type=int, default=1, help="Số vòng repair/check, clamp 1..3.")
    parser.add_argument("--strict-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--auto-repair", action=argparse.BooleanOptionalAction, default=False)
    return parser


def ping_model(client: LLMClient, model: str) -> None:
    response = client.chat_completion(
        model=model,
        messages=[{"role": "user", "content": "Xin chào, hãy trả lời ngắn gọn bằng tiếng Việt."}],
        temperature=0,
    )
    preview = response["content"].strip().replace("\n", " ")[:200]
    print(f"[OK] {model} ({response['latency_seconds']}s): {preview}")


def run_generated_question_schema_inspection(args: argparse.Namespace) -> None:
    input_path = resolve_input_path(args.input)
    payload = load_json_payload(input_path)
    records = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("--input phải là JSON object hoặc list object.")
    report = inspect_generated_question_records(records)
    if not args.output:
        print(format_generated_question_schema_summary(report, source_name=str(input_path)))
        return
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".md":
        output_path.write_text(format_generated_question_schema_summary(report, source_name=str(input_path)), encoding="utf-8")
    else:
        write_json_output(output_path, report)
    print(f"Đã ghi generatedQuestions schema summary: {output_path}")


def print_generated_question_progress(current: int, total: int, generated_question: dict[str, Any]) -> None:
    if total > 1:
        question_id = generated_question.get("id") or generated_question.get("_id") or f"item[{current - 1}]"
        print(f"Đang đánh giá generated question {current}/{total}: {question_id}", file=sys.stderr, flush=True)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.inspect_generated_question_schema:
            run_generated_question_schema_inspection(args)
            return 0

        config = load_config(ROOT_DIR)
        client = LLMClient(config)
        if args.list_models:
            for model_id in client.list_models():
                print(model_id)
            return 0
        if args.ping:
            ping_model(client, args.model or config.primary_judge_model)
            return 0

        input_path = resolve_input_path(args.input)
        payload = limit_payload_amount(load_json_payload(input_path), args.amount)
        if args.evaluate_question_plan_service:
            if isinstance(payload, list):
                result = evaluate_question_plans(payload, config=config, client=client, is_loop=False, max_loop=args.max_loop)
            elif isinstance(payload, dict):
                result = evaluate_question_plan(payload, config=config, client=client, is_loop=False, max_loop=args.max_loop)
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

        if args.evaluate_generated_questions_service:
            if not isinstance(payload, (dict, list)):
                raise ValueError("Generated checker nhận object, list hoặc wrapper generatedQuestions.")
            result = evaluate_generated_questions(
                payload,
                config=config,
                client=client,
                strict_mode=args.strict_mode,
                debug=args.debug,
                progress_callback=print_generated_question_progress,
                auto_repair=args.auto_repair,
                max_loop=args.max_loop,
            )
            output_path = Path(args.output or "results/generated_question_check_repair_output.json")
            report_path = Path(args.report_output or "results/generated_question_repair_report.md")
            repaired_path = Path(args.repaired_output or "results/generated_question_repaired_objects.json")
            output_path = output_path if output_path.is_absolute() else ROOT_DIR / output_path
            report_path = report_path if report_path.is_absolute() else ROOT_DIR / report_path
            repaired_path = repaired_path if repaired_path.is_absolute() else ROOT_DIR / repaired_path
            for path in write_generated_question_cli_outputs(
                result=result,
                source_name=str(input_path),
                output_path=output_path,
                report_output_path=report_path,
                repaired_output_path=repaired_path,
            ):
                print(f"Đã ghi generatedQuestions output: {path}")
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

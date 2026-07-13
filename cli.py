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
        raise ValueError("Cần truyền --input là đường dẫn tới file JSON record hoặc list record.")
    path = Path(input_path)
    return path if path.is_absolute() else ROOT_DIR / path


def load_json_payload(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def write_json_output(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def markdown_text(value: Any) -> str:
    text = str(value or "").strip()
    return text.replace("|", "\\|").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def generated_result_status(result: dict[str, Any]) -> str:
    severities = {issue.get("severity") for issue in result.get("issues") or [] if isinstance(issue, dict)}
    if "bad" in severities:
        return "bad"
    if "needs_review" in severities:
        return "needs_review"
    if "warning" in severities:
        return "warning"
    return "good" if result.get("is_good") else "bad"


def generated_result_list(result: dict[str, Any]) -> list[dict[str, Any]]:
    results = result.get("results")
    if isinstance(results, list):
        return [item for item in results if isinstance(item, dict)]
    return [result]


def format_generated_question_markdown(result: dict[str, Any], *, source_name: str = "") -> str:
    rows = generated_result_list(result)
    lines: list[str] = ["# Generated Question Quality Report", ""]
    if source_name:
        lines.extend([f"Source file: `{source_name}`", ""])

    summary = result.get("summary") if isinstance(result.get("summary"), dict) else None
    lines.append("## Summary")
    if summary:
        for key in ("total", "good", "bad", "needs_review", "warning", "repaired", "repair_failed"):
            lines.append(f"- {key}: {summary.get(key, 0)}")
    else:
        lines.append(f"- total: {len(rows)}")
        if rows:
            lines.append(f"- status: {generated_result_status(rows[0])}")
    lines.append(f"- is_good: {result.get('is_good')}")
    lines.append("")

    lines.extend(
        [
            "## Result Table",
            "",
            "| # | ID | Status | Repair | Issues | Failed Reason | Suggestions |",
            "|---|---|---|---|---:|---|---|",
        ]
    )
    for index, item in enumerate(rows, start=1):
        issues = item.get("issues") if isinstance(item.get("issues"), list) else []
        failed_reason = "; ".join(str(reason) for reason in item.get("failed_reason") or [])
        suggestions = "; ".join(str(suggestion) for suggestion in item.get("suggestions") or [])
        lines.append(
            "| {index} | {id} | {status} | {repair_status} | {issue_count} | {failed_reason} | {suggestions} |".format(
                index=index,
                id=markdown_text(item.get("id") or f"item[{index - 1}]"),
                status=markdown_text(generated_result_status(item)),
                repair_status=markdown_text(item.get("repair_status") or "skipped"),
                issue_count=len(issues),
                failed_reason=markdown_text(failed_reason),
                suggestions=markdown_text(suggestions),
            )
        )

    issue_rows = [item for item in rows if item.get("issues")]
    lines.extend(["", "## Issue Details", ""])
    if not issue_rows:
        lines.append("Không có issue.")
    for index, item in enumerate(issue_rows, start=1):
        lines.extend(
            [
                f"### {index}. {markdown_text(item.get('id') or 'unknown')}",
                "",
                f"- Status: `{generated_result_status(item)}`",
                f"- is_good: `{item.get('is_good')}`",
                f"- Repair status: `{item.get('repair_status', 'skipped')}`",
                f"- Has new_generated_question: `{item.get('new_generated_question') is not None}`",
                "",
                "| Severity | Category | Location | Reason | Suggestion |",
                "|---|---|---|---|---|",
            ]
        )
        for issue in item.get("issues") or []:
            if not isinstance(issue, dict):
                continue
            lines.append(
                "| {severity} | {category} | {location} | {reason} | {suggestion} |".format(
                    severity=markdown_text(issue.get("severity")),
                    category=markdown_text(issue.get("category")),
                    location=markdown_text(issue.get("location")),
                    reason=markdown_text(issue.get("reason")),
                    suggestion=markdown_text(issue.get("suggestion")),
                )
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def extract_repaired_generated_questions(result: dict[str, Any]) -> list[dict[str, Any]]:
    repaired: list[dict[str, Any]] = []
    for item in generated_result_list(result):
        new_generated_question = item.get("new_generated_question")
        if isinstance(new_generated_question, dict):
            repaired.append(new_generated_question)
    return repaired


def build_generated_question_repair_mapping(result: dict[str, Any]) -> list[dict[str, Any]]:
    mapping: list[dict[str, Any]] = []
    for index, item in enumerate(generated_result_list(result)):
        new_generated_question = item.get("new_generated_question")
        repair_status = item.get("repair_status") or "skipped"
        if repair_status not in {"repaired", "failed", "skipped"}:
            repair_status = "failed" if item.get("repair_failed_reason") else "skipped"
        issues = item.get("issues") if isinstance(item.get("issues"), list) else []
        mapping.append(
            {
                "id": item.get("id") or f"item[{index}]",
                "repair_status": repair_status,
                "original_index": index,
                "issue_count": len(issues),
                "new_generated_question_included": isinstance(new_generated_question, dict),
            }
        )
    return mapping


def write_generated_question_output(path: Path, result: dict[str, Any], *, source_name: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".md":
        path.write_text(format_generated_question_markdown(result, source_name=source_name), encoding="utf-8")
    else:
        write_json_output(path, result)


def write_generated_question_cli_outputs(
    *,
    result: dict[str, Any],
    source_name: str,
    output_path: Path,
    report_output_path: Path | None,
    repaired_output_path: Path | None,
    repair_mapping_output_path: Path | None,
) -> list[Path]:
    written: list[Path] = []
    write_json_output(output_path, result)
    written.append(output_path)

    if report_output_path:
        report_output_path.parent.mkdir(parents=True, exist_ok=True)
        report_output_path.write_text(format_generated_question_markdown(result, source_name=source_name), encoding="utf-8")
        written.append(report_output_path)

    if repaired_output_path:
        write_json_output(repaired_output_path, extract_repaired_generated_questions(result))
        written.append(repaired_output_path)

    if repair_mapping_output_path:
        write_json_output(repair_mapping_output_path, build_generated_question_repair_mapping(result))
        written.append(repair_mapping_output_path)

    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Service đánh giá question_plan và generatedQuestions.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--evaluate-question-plan-service", action="store_true", help="Đánh giá question_plan và trả JSON service output.")
    action.add_argument(
        "--evaluate-generated-questions-service",
        action="store_true",
        help="Đánh giá generated question object/list/wrapper.",
    )
    action.add_argument("--inspect-generated-question-schema", action="store_true", help="Tổng hợp field/schema generatedQuestions từ file JSON.")
    action.add_argument("--list-models", action="store_true", help="Liệt kê các model có sẵn từ LLM endpoint.")
    action.add_argument("--ping", action="store_true", help="Gửi một request nhỏ để kiểm tra kết nối model.")
    parser.add_argument("--input", default=None, help="Đường dẫn tới file JSON object hoặc list object source record.")
    parser.add_argument("--output", default=None, help="Ghi output ra file. Với generated questions, .md sẽ ghi report Markdown.")
    parser.add_argument("--report-output", default=None, help="Ghi Markdown report cho generated question checker.")
    parser.add_argument("--repaired-output", default=None, help="Ghi list new_generated_question đã repair thành công.")
    parser.add_argument("--repair-mapping-output", default=None, help="Ghi mapping/tracing kết quả repair generated question.")
    parser.add_argument("--model", default=None, help="Model dùng cho --ping. Nếu bỏ trống sẽ dùng PRIMARY_JUDGE_MODEL.")
    parser.add_argument("--is-loop", action="store_true", help="Bật loop/refinement sau khi repair tạo candidate mới.")
    parser.add_argument("--max-loop", type=int, default=3, help="Số vòng loop tối đa, được clamp trong khoảng 1..3.")
    parser.add_argument("--strict-mode", action=argparse.BooleanOptionalAction, default=True, help="Bật/tắt strict mode cho generatedQuestions.")
    parser.add_argument("--debug", action="store_true", help="Bật debug metadata an toàn cho LLM prompt/call.")
    parser.add_argument(
        "--auto-repair",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Bật/tắt LLM repair cho generated questions. Mặc định bật với generated checker.",
    )
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
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT_DIR / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".md":
            output_path.write_text(
                format_generated_question_schema_summary(report, source_name=str(input_path)),
                encoding="utf-8",
            )
        else:
            write_json_output(output_path, report)
        print(f"Đã ghi generatedQuestions schema summary: {output_path}")
    else:
        print(format_generated_question_schema_summary(report, source_name=str(input_path)))


def print_generated_question_progress(current: int, total: int, generated_question: dict[str, Any]) -> None:
    if total <= 1:
        return
    question_id = str(generated_question.get("id") or generated_question.get("_id") or f"item[{current - 1}]")
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

        if args.evaluate_question_plan_service:
            input_path = resolve_input_path(args.input)
            payload = load_json_payload(input_path)
            if isinstance(payload, list):
                result = evaluate_question_plans(
                    payload,
                    config=config,
                    client=client,
                    is_loop=args.is_loop,
                    max_loop=args.max_loop,
                )
            elif isinstance(payload, dict):
                result = evaluate_question_plan(
                    payload,
                    config=config,
                    client=client,
                    is_loop=args.is_loop,
                    max_loop=args.max_loop,
                )
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
            input_path = resolve_input_path(args.input)
            payload = load_json_payload(input_path)
            if not isinstance(payload, (dict, list)):
                raise ValueError(
                    "--evaluate-generated-questions-service nhận JSON object, list object, "
                    "hoặc wrapper có generatedQuestions."
                )

            generated_auto_repair = True if args.auto_repair is None else args.auto_repair
            generated_is_loop = True if generated_auto_repair else args.is_loop

            result = evaluate_generated_questions(
                payload,
                config=config,
                client=client,
                strict_mode=args.strict_mode,
                debug=args.debug,
                progress_callback=print_generated_question_progress,
                auto_repair=generated_auto_repair,
                is_loop=generated_is_loop,
                max_loop=args.max_loop,
            )

            output_path = Path(args.output) if args.output else Path("results/generated_question_check_repair_output.json")
            if not output_path.is_absolute():
                output_path = ROOT_DIR / output_path

            report_output_path = Path(args.report_output) if args.report_output else Path("results/generated_question_repair_report.md")
            if not report_output_path.is_absolute():
                report_output_path = ROOT_DIR / report_output_path

            repaired_output_path = Path(args.repaired_output) if args.repaired_output else Path("results/generated_question_repaired_objects.json")
            if repaired_output_path and not repaired_output_path.is_absolute():
                repaired_output_path = ROOT_DIR / repaired_output_path

            repair_mapping_output_path = Path(args.repair_mapping_output) if args.repair_mapping_output else Path("results/generated_question_repair_mapping.json")
            if repair_mapping_output_path and not repair_mapping_output_path.is_absolute():
                repair_mapping_output_path = ROOT_DIR / repair_mapping_output_path

            written_paths = write_generated_question_cli_outputs(
                result=result,
                source_name=str(input_path),
                output_path=output_path,
                report_output_path=report_output_path,
                repaired_output_path=repaired_output_path,
                repair_mapping_output_path=repair_mapping_output_path,
            )
            for written_path in written_paths:
                print(f"Đã ghi generatedQuestions output: {written_path}")
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

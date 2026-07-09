import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.config import ConfigError, load_config
from src.label_assignment_pipeline import build_record_label_assignment_summary, run_label_assignment_pipeline
from src.label_assignment_rule_validator import inspect_label_assignment
from src.llm_client import LLMClient
from src.question_plan_eval_pipeline import run_question_plan_eval_pipeline
from src.question_plan_rule_validator import inspect_question_plan_quality
from src.question_plan_service import evaluate_question_plan, evaluate_question_plans
from src.real_data_flattener import flatten_real_data
from src.real_interaction_classifier import classify_real_interaction, validate_real_interaction_type
from src.real_pipeline import run_real_question_pipeline, validate_plan_alignment
from src.real_rule_validator import validate_real_question_case
from src.real_schema_inspector import inspect_real_schema, write_schema_inspection
from src.result_writer import (
    write_csv,
    write_json,
    write_plan_interaction_type_output_files,
    write_question_plan_eval_output_files,
    write_real_question_result_files,
)
from src.result_writer import write_source_record_result_files
from src.source_record_pipeline import build_source_record_problem_report, run_source_record_pipeline
from src.source_record_rule_validator import flatten_source_records, validate_source_record_rules
from src.utils import load_json_list


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_REAL_SAMPLE_PATH = ROOT_DIR / "data" / "processed" / "math_9_bt_2_questions.json"
RESULTS_DIR = ROOT_DIR / "results"
OVERVIEW_DIR = RESULTS_DIR / "overview"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def resolve_input_path(input_path: str | None) -> Path:
    path = Path(input_path) if input_path else DEFAULT_REAL_SAMPLE_PATH
    return path if path.is_absolute() else ROOT_DIR / path


def load_json_payload(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def ping_model(client: LLMClient, model: str) -> None:
    try:
        response = client.chat_completion(model=model, messages=[{"role": "user", "content": "Xin chào, hãy mô tả ngắn về bản thân bạn."}], temperature=0)
        preview = response["content"].strip().replace("\n", " ")[:100]
        print(f"[THÀNH CÔNG] {model} ({response['latency_seconds']}s): {preview}")
    except Exception as exc:
        print(f"[LỖI] {model}: {exc}")


def validate_real_schema(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for case in flatten_real_data(records):
        rule = validate_real_question_case(case)
        results.append(
            {
                "case_id": case.get("case_id"),
                "record_id": case.get("record_id"),
                "record_name": case.get("record_name"),
                "generated_question_id": case.get("generated_question_id"),
                "question_item_id": case.get("question_item_id"),
                "interaction_id": case.get("interaction_id"),
                "interaction_index": case.get("interaction_index"),
                "start_page": case.get("start_page"),
                "end_page": case.get("end_page"),
                "declared_interaction_type": case.get("declared_interaction_type"),
                "planned_interaction_type": case.get("planned_interaction_type"),
                "schema_valid": rule.get("schema_valid"),
                "rule_passed": rule.get("rule_passed"),
                "object_structure_valid": rule.get("object_structure_valid"),
                "interaction_schema_valid": rule.get("interaction_schema_valid"),
                "question_text_valid": rule.get("question_text_valid"),
                "answer_specs_present": rule.get("answer_specs_present"),
                "answer_check_skipped": True,
                "error_type": ((rule.get("issues") or [{}])[0]).get("error_type", ""),
                "reason": rule.get("reason", ""),
                "rule_result": rule,
                "case": case,
            }
        )
    return results


def classify_real_interactions(records: list[dict[str, Any]], config, client: LLMClient) -> list[dict[str, Any]]:
    rows = []
    for case in flatten_real_data(records):
        print(f"Classifying real interaction: {case.get('case_id')}")
        classifier = classify_real_interaction(case, config, client)
        semantic = validate_real_interaction_type(case, classifier)
        plan_alignment = validate_plan_alignment(case, classifier)
        rows.append(
            {
                "case_id": case.get("case_id"),
                "record_id": case.get("record_id"),
                "record_name": case.get("record_name"),
                "generated_question_id": case.get("generated_question_id"),
                "question_item_id": case.get("question_item_id"),
                "interaction_id": case.get("interaction_id"),
                "interaction_index": case.get("interaction_index"),
                "declared_interaction_type": case.get("declared_interaction_type"),
                "planned_interaction_type": case.get("planned_interaction_type"),
                "detected_interaction_type": classifier.get("detected_interaction_type"),
                "suggested_interaction_type": semantic.get("suggested_interaction_type", ""),
                "interaction_type_valid": semantic.get("interaction_type_valid"),
                "interaction_type_mismatch": semantic.get("interaction_type_mismatch"),
                "soft_mismatch": semantic.get("soft_mismatch"),
                "warning_type": semantic.get("warning_type"),
                "suggested_normalized_type": semantic.get("suggested_normalized_type"),
                "policy_override_applied": semantic.get("policy_override_applied"),
                "plan_alignment_valid": plan_alignment.get("plan_alignment_valid"),
                "classifier_model": classifier.get("model"),
                "classifier_confidence": classifier.get("confidence"),
                "classifier_json_parse_ok": classifier.get("json_parse_ok"),
                "classifier_fallback_called": classifier.get("fallback_called"),
                "classifier_error": classifier.get("error"),
                "need_human_review": semantic.get("need_human_review") or plan_alignment.get("need_human_review"),
                "error_type": semantic.get("error_type") or plan_alignment.get("error_type"),
                "reason": semantic.get("reason") or plan_alignment.get("reason") or classifier.get("reason"),
                "classifier_result": classifier,
                "case": case,
            }
        )
    return rows


def validate_source_records(records: list[dict[str, Any]], pdf_path: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for case in flatten_source_records(records, pdf_path=pdf_path):
        rule = validate_source_record_rules(case)
        rows.append(
            {
                "case_id": case.get("case_id"),
                "record_id": case.get("record_id"),
                "record_name": case.get("record_name"),
                "start_page": case.get("start_page"),
                "end_page": case.get("end_page"),
                "record_structure_valid": rule.get("record_structure_valid"),
                "record_id_valid": rule.get("record_id_valid"),
                "record_name_valid": rule.get("record_name_valid"),
                "raw_question_present": rule.get("raw_question_present"),
                "raw_answer_present": rule.get("raw_answer_present"),
                "page_range_valid": rule.get("page_range_valid"),
                "question_plan_present": rule.get("question_plan_present"),
                "raw_question_text_valid": rule.get("raw_question_text_valid"),
                "raw_answer_text_valid": rule.get("raw_answer_text_valid"),
                "planned_interaction_types_valid": rule.get("planned_interaction_types_valid"),
                "error_type": ((rule.get("issues") or [{}])[0]).get("error_type", ""),
                "reason": rule.get("reason", ""),
                "rule_result": rule,
                "case": case,
            }
        )
    return rows


def run_quick_check(records: list[dict[str, Any]], config, args) -> tuple[Path, Path, Path]:
    results = run_source_record_pipeline(
        records,
        config,
        skip_pdf=args.skip_pdf,
        pdf_path=args.pdf,
        pdf_page_offset=args.pdf_page_offset,
        refresh_ocr_cache=args.refresh_ocr_cache,
    )
    report = build_source_record_problem_report(results)
    json_path, csv_path, report_path = write_source_record_result_files(RESULTS_DIR, results, report)
    print(f"Wrote quick_check JSON: {json_path}")
    print(f"Wrote quick_check CSV: {csv_path}")
    print(f"Wrote quick_check problem report: {report_path}")
    return json_path, csv_path, report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kiểm tra dữ liệu mentor và các pipeline đánh giá bằng LLM.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list-models", action="store_true", help="Liệt kê các id model có sẵn.")
    action.add_argument("--ping", action="store_true", help="Gửi một yêu cầu trò chuyện nhỏ tới một model.")
    action.add_argument("--inspect-real-schema", action="store_true", help="Khảo sát schema dữ liệu thật của mentor mà không dùng LLM.")
    action.add_argument("--validate-real-schema", action="store_true", help="Kiểm tra schema dữ liệu thật đã được làm phẳng mà không dùng LLM.")
    action.add_argument("--classify-real-interactions", action="store_true", help="Phân loại kiểu tương tác trong schema thật.")
    action.add_argument("--run-real-question-pipeline", action="store_true", help="Chạy pipeline đánh giá câu hỏi theo schema thật.")
    action.add_argument("--inspect-source-records", action="store_true", help="Khảo sát các bản ghi nguồn cấp cao mà không dùng LLM.")
    action.add_argument("--validate-source-records", action="store_true", help="Kiểm tra các bản ghi nguồn bằng quy tắc thuần túy.")
    action.add_argument("--run-source-record-pipeline", action="store_true", help="Chạy pipeline kiểm tra bản ghi nguồn.")
    action.add_argument(
        "--run-quick-check",
        dest="run_source_record_pipeline",
        action="store_true",
        help="Tên thay thế cho --run-source-record-pipeline; ghi kết quả vào thư mục results/quick_check.",
    )
    action.add_argument("--inspect-label-assignment", action="store_true", help="Khảo sát các nhãn interactionType của question_plan mà không dùng LLM.")
    action.add_argument("--run-label-assignment-pipeline", action="store_true", help="Chạy pipeline gán nhãn theo hướng mentor.")
    action.add_argument(
        "--inspect-question-plan-quality",
        action="store_true",
        help="Khảo sát cấu trúc question_plan để đánh giá chất lượng kế hoạch mà không dùng LLM.",
    )
    action.add_argument(
        "--run-question-plan-quality-pipeline",
        action="store_true",
        help="Chạy pipeline đánh giá chất lượng question_plan ở cấp bản ghi nguồn.",
    )
    action.add_argument(
        "--evaluate-question-plan-service",
        action="store_true",
        help="Service gọn: đánh giá question_plan và trả JSON 4 field, không ghi report/CSV.",
    )
    parser.add_argument("--input", default=None, help="Đường dẫn tới file JSON dữ liệu mentor.")
    parser.add_argument("--output", default=None, help="Ghi JSON output của service ra file thay vì stdout.")
    parser.add_argument("--model", default=None, help="Ghi đè model cho --ping.")
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Chế độ pipeline thực: không OCR PDF, chỉ dùng raw_question và question_plan.",
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="Chế độ pipeline thực: đường dẫn PDF dùng cho OCR từ start_page đến end_page.",
    )
    parser.add_argument(
        "--refresh-ocr-cache",
        action="store_true",
        help="Bỏ qua cache OCR đã lưu và gọi OCR lại cho các trường hợp PDF/trang.",
    )
    parser.add_argument("--pdf-page-offset", type=int, default=0, help="Độ lệch áp dụng cho start_page/end_page khi OCR PDF.")
    parser.add_argument(
        "--include-debug",
        action="store_true",
        help="Chế độ gán nhãn: đồng thời ghi các file debug vào results/outputs/debug.",
    )
    parser.add_argument(
        "--legacy-outputs",
        action="store_true",
        help="Chế độ gán nhãn: đồng thời ghi các file cũ ở cấp gốc như label_assignment_* và plan_interaction_type_*.",
    )
    parser.add_argument(
        "--with-quick-check",
        action="store_true",
        help="Chế độ gán nhãn: chạy quick_check cấp bản ghi nguồn trước khi đánh giá interactionType.",
    )
    parser.add_argument(
        "--source-check-summary",
        default=None,
        help="Question-plan-quality mode: optional source/OCR summary JSON or CSV used to skip records with source issues.",
    )
    return parser


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

        input_path = resolve_input_path(args.input)
        if args.evaluate_question_plan_service:
            payload = load_json_payload(input_path)
            if isinstance(payload, list):
                result = evaluate_question_plans(payload, config=config, client=client)
            elif isinstance(payload, dict):
                result = evaluate_question_plan(payload, config=config, client=client)
            else:
                raise ValueError("--evaluate-question-plan-service yêu cầu input là JSON object hoặc list object.")
            output_text = json.dumps(result, ensure_ascii=False, indent=2)
            if args.output:
                output_path = Path(args.output)
                if not output_path.is_absolute():
                    output_path = ROOT_DIR / output_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(output_text + "\n", encoding="utf-8")
                print(f"Đã ghi service output: {output_path}")
            else:
                print(output_text)
            return 0

        records = load_json_list(input_path)

        if args.inspect_real_schema:
            report = inspect_real_schema(records)
            json_path, txt_path = write_schema_inspection(OVERVIEW_DIR, report)
            print(f"Đã ghi file khảo sát schema JSON: {json_path}")
            print(f"Đã ghi file khảo sát schema dạng text: {txt_path}")
            return 0

        if args.inspect_source_records:
            rows = flatten_source_records(records, pdf_path=args.pdf)
            report = {
                "record_count": len(rows),
                "records_with_question": sum(1 for row in rows if row.get("raw_question")),
                "records_with_answer": sum(1 for row in rows if row.get("raw_answer")),
                "records_with_question_plan": sum(1 for row in rows if row.get("question_plan")),
                "records_with_images": sum(1 for row in rows if row.get("images")),
                "records_with_answer_images": sum(1 for row in rows if row.get("answer_images")),
                "page_ranges": sorted(
                    {
                        (row.get("start_page"), row.get("end_page"))
                        for row in rows
                    },
                    key=lambda item: (item[0] or 0, item[1] or 0),
                ),
            }
            json_path = OVERVIEW_DIR / "source_record_inspection.json"
            write_json(json_path, report)
            print(f"Đã ghi file khảo sát bản ghi nguồn: {json_path}")
            return 0

        if args.inspect_label_assignment:
            report = inspect_label_assignment(records)
            json_path = OVERVIEW_DIR / "label_assignment_inspection.json"
            write_json(json_path, report)
            print(f"Đã ghi file khảo sát gán nhãn: {json_path}")
            print(f"Tổng số bản ghi: {report['total_records']}")
            print(f"Tổng số case nhãn: {report['total_label_cases']}")
            print(f"question_plan.type: {report['question_plan_type_distribution']}")
            print(f"interactionType trong question_plan: {report['question_plan_interaction_type_distribution']}")
            print(f"Kiểu tương tác được sinh ra: {report['generated_interaction_type_distribution']}")
            print(f"Số trường hợp không khớp giữa plan và generated: {report['plan_generated_type_mismatch_count']}")
            return 0

        if args.inspect_question_plan_quality:
            report = inspect_question_plan_quality(records)
            json_path = OVERVIEW_DIR / "question_plan_quality_inspection.json"
            write_json(json_path, report)
            print(f"Đã ghi file khảo sát chất lượng question_plan: {json_path}")
            print(f"Tổng số bản ghi: {report['total_records']}")
            print(f"Cấu trúc OK: {report['structural_ok_count']}")
            print(f"Lỗi cấu trúc: {report['structural_error_count']}")
            print(f"Các loại lỗi cấu trúc: {report['count_by_structural_error_type']}")
            return 0

        if args.validate_source_records:
            results = validate_source_records(records, pdf_path=args.pdf)
            json_path = RESULTS_DIR / "source_record_validation_results.json"
            csv_path = RESULTS_DIR / "source_record_validation_summary.csv"
            write_json(json_path, results)
            write_csv(
                csv_path,
                results,
                [
                    "case_id",
                    "record_id",
                    "record_name",
                    "start_page",
                    "end_page",
                    "record_structure_valid",
                    "record_id_valid",
                    "record_name_valid",
                    "raw_question_present",
                    "raw_answer_present",
                    "page_range_valid",
                    "question_plan_present",
                    "raw_question_text_valid",
                    "raw_answer_text_valid",
                    "planned_interaction_types_valid",
                    "error_type",
                    "reason",
                ],
            )
            print(f"Đã ghi file kiểm tra bản ghi nguồn dạng JSON: {json_path}")
            print(f"Đã ghi file kiểm tra bản ghi nguồn dạng CSV: {csv_path}")
            return 0

        if args.validate_real_schema:
            results = validate_real_schema(records)
            json_path = RESULTS_DIR / "real_schema_validation_results.json"
            csv_path = RESULTS_DIR / "real_schema_validation_summary.csv"
            write_json(json_path, results)
            write_csv(
                csv_path,
                results,
                [
                    "case_id",
                    "record_id",
                    "record_name",
                    "generated_question_id",
                    "question_item_id",
                    "interaction_id",
                    "interaction_index",
                    "start_page",
                    "end_page",
                    "declared_interaction_type",
                    "planned_interaction_type",
                    "schema_valid",
                    "rule_passed",
                    "object_structure_valid",
                    "interaction_schema_valid",
                    "question_text_valid",
                    "answer_specs_present",
                    "answer_check_skipped",
                    "error_type",
                    "reason",
                ],
            )
            print(f"Đã ghi file kiểm tra schema thật dạng JSON: {json_path}")
            print(f"Đã ghi file kiểm tra schema thật dạng CSV: {csv_path}")
            return 0

        if args.classify_real_interactions:
            results = classify_real_interactions(records, config, client)
            json_path = RESULTS_DIR / "real_interaction_classification_results.json"
            csv_path = RESULTS_DIR / "real_interaction_classification_summary.csv"
            write_json(json_path, results)
            write_csv(
                csv_path,
                results,
                [
                    "case_id",
                    "record_id",
                    "record_name",
                    "generated_question_id",
                    "question_item_id",
                    "interaction_id",
                    "interaction_index",
                    "declared_interaction_type",
                    "planned_interaction_type",
                    "detected_interaction_type",
                    "suggested_interaction_type",
                    "interaction_type_valid",
                    "interaction_type_mismatch",
                    "soft_mismatch",
                    "warning_type",
                    "suggested_normalized_type",
                    "policy_override_applied",
                    "plan_alignment_valid",
                    "classifier_model",
                    "classifier_confidence",
                    "classifier_json_parse_ok",
                    "classifier_fallback_called",
                    "classifier_error",
                    "need_human_review",
                    "error_type",
                    "reason",
                ],
            )
            print(f"Đã ghi file phân loại tương tác thật dạng JSON: {json_path}")
            print(f"Đã ghi file phân loại tương tác thật dạng CSV: {csv_path}")
            return 0

        if args.run_real_question_pipeline:
            results = run_real_question_pipeline(
                records,
                config,
                skip_pdf=args.skip_pdf,
                refresh_ocr_cache=args.refresh_ocr_cache,
                pdf_path=args.pdf,
            )
            json_path, csv_path = write_real_question_result_files(RESULTS_DIR, results)
            print(f"Đã ghi file pipeline câu hỏi thật dạng JSON: {json_path}")
            print(f"Đã ghi file pipeline câu hỏi thật dạng CSV: {csv_path}")
            return 0

        if args.run_source_record_pipeline:
            run_quick_check(records, config, args)
            return 0

        if args.run_label_assignment_pipeline:
            if args.with_quick_check:
                print("Running quick_check before label assignment...")
                run_quick_check(records, config, args)
            payload = run_label_assignment_pipeline(records, config)
            results = payload["results"]
            record_summary = build_record_label_assignment_summary(results, payload["record_coverage"])
            repair_suggestions = payload.get("repair_suggestions") or []
            repair_summary = payload.get("repair_summary") or {}
            output_paths = write_plan_interaction_type_output_files(
                RESULTS_DIR,
                results,
                payload["problem_report"],
                record_summary,
                repair_suggestions,
                repair_summary,
                include_debug=args.include_debug,
                legacy_outputs=args.legacy_outputs,
            )
            print(f"Đã ghi báo cáo tổng quát: {output_paths['general_report']}")
            print(f"Đã ghi chi tiết vấn đề câu hỏi dạng JSON: {output_paths['question_issue_details']}")
            print(f"Đã ghi file CSV câu hỏi lỗi: {output_paths['question_bad']}")
            print(f"Đã ghi file CSV câu hỏi cảnh báo/cần xem lại: {output_paths['question_warning_needs_review']}")
            print(f"Đã ghi file CSV tương tác lỗi: {output_paths['interaction_bad']}")
            print(f"Đã ghi file CSV tương tác cảnh báo/cần xem lại: {output_paths['interaction_warning_needs_review']}")
            print(f"Đã ghi tóm tắt chạy: {output_paths['run_summary']}")
            if args.include_debug:
                print(f"Đã ghi kết quả pipeline debug thô: {output_paths['raw_pipeline_results']}")
                print(f"Đã ghi báo cáo vấn đề debug thô: {output_paths['raw_problem_report']}")
                print(f"Đã ghi báo cáo đánh giá debug: {output_paths['eval_debug_report']}")
                print(f"Đã ghi danh sách vấn đề tương tác debug: {output_paths['debug_interaction_issues']}")
                print(f"Đã ghi tóm tắt sửa lỗi debug: {output_paths['debug_repair_summary']}")
                print(f"Đã ghi tóm tắt câu hỏi debug: {output_paths['question_summary']}")
                print(f"Đã ghi gợi ý sửa debug: {output_paths['repair_suggestions']}")
                print(f"Đã ghi bằng chứng tương tác debug: {output_paths['interaction_evidence']}")
            if args.legacy_outputs:
                print("Đã ghi các file đầu ra cũ ở cấp gốc.")
            print(
                "Tóm tắt kiểm tra nhãn: "
                f"{payload['problem_report'].get('final_ok_count')} OK, "
                f"{payload['problem_report'].get('final_bad_count')} lỗi, "
                f"{payload['problem_report'].get('final_warning_count')} cảnh báo, "
                f"{payload['problem_report'].get('final_needs_review_count')} cần xem lại, "
                f"{payload['problem_report'].get('final_structural_error_count')} lỗi cấu trúc."
            )
            print(
                "Tóm tắt gợi ý sửa: "
                f"{repair_summary.get('repair_suggestion_count')} gợi ý, "
                f"{repair_summary.get('repair_needed_count')} cần sửa, "
                f"{repair_summary.get('needs_human_review_count')} cần xem lại thủ công."
            )
            return 0

        if args.run_question_plan_quality_pipeline:
            payload = run_question_plan_eval_pipeline(records, config, source_check_summary=args.source_check_summary)
            output_paths = write_question_plan_eval_output_files(
                RESULTS_DIR,
                payload["results"],
                payload["summary"],
                source_records=records,
            )
            summary = payload["summary"]
            print(f"Đã ghi báo cáo tổng quát chất lượng question_plan: {output_paths['general_report']}")
            print(f"Đã ghi chi tiết vấn đề question_plan dạng JSON: {output_paths['question_plan_issue_details']}")
            print(f"Đã ghi file CSV câu hỏi lỗi: {output_paths['question_bad']}")
            print(f"Đã ghi file CSV câu hỏi cảnh báo/cần xem lại: {output_paths['question_warning_needs_review']}")
            print(
                "Đã ghi full records replacement hợp lệ: "
                f"{output_paths['question_plan_replacement_records_all']}"
            )
            print(
                "Tóm tắt chất lượng question_plan: "
                f"{summary.get('ok_count')} OK, "
                f"{summary.get('bad_count')} lỗi, "
                f"{summary.get('warning_count')} cảnh báo, "
                f"{summary.get('needs_review_count')} cần xem lại, "
                f"{summary.get('structural_error_count')} lỗi cấu trúc."
            )
            print(f"Skipped due to source/raw issue: {summary.get('skipped_due_to_source_issue_count')}")
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

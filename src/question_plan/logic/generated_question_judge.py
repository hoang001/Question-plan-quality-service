"""LLM judge đánh giá chất lượng solution của generated question object."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from ..infra.config import AppConfig, generated_question_fast_model, generated_question_reasoning_model
from ..infra.debug import debug_llm_messages, llm_prompt_debug_enabled
from ..infra.llm_client import LLMClient
from ..shared.utils import parse_json_output
from .generated_question_schema import (
    fail_closed_output,
    generated_question_id,
    normalize_generated_question_result,
)


KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"
CRITERIA_PATH = KNOWLEDGE_DIR / "generated_question_quality_criteria.md"
OUTPUT_SCHEMA_PATH = KNOWLEDGE_DIR / "generated_question_output_schema.md"


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def compact_generated_question_payload(
    generated_question: dict[str, Any],
) -> dict[str, Any]:
    """Chỉ gửi ngữ cảnh cần thiết để đánh giá solution, không gửi đáp án."""

    question_items = []
    for item_index, item in enumerate(generated_question.get("questionItems") or []):
        if not isinstance(item, dict):
            continue
        interaction_types = [
            str(interaction.get("type") or "")
            for interaction in item.get("interactions") or []
            if isinstance(interaction, dict) and str(interaction.get("type") or "")
        ]
        question_items.append(
            {
                "index": item_index,
                "id": item.get("id"),
                "stem": item.get("stem"),
                "interactionTypes": interaction_types,
            }
        )

    solutions = []
    for solution_index, solution in enumerate(generated_question.get("solutions") or []):
        if not isinstance(solution, dict):
            continue
        text_blocks = []
        for content_index, block in enumerate(solution.get("solutionContent") or []):
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text_value = block.get("text")
            if not isinstance(text_value, str) or not text_value.strip():
                continue
            text_blocks.append(
                {
                    "contentIndex": content_index,
                    "path": f"/solutions/{solution_index}/solutionContent/{content_index}/text",
                    "text": text_value,
                }
            )
        solutions.append(
            {
                "index": solution_index,
                "solverName": solution.get("solverName"),
                "textBlocks": text_blocks,
            }
        )

    return {
        "id": generated_question.get("id") or generated_question.get("_id") or "",
        "interactionTypes": generated_question.get("interactionTypes"),
        "instruction": generated_question.get("instruction"),
        "questionItems": question_items,
        "solutions": solutions,
    }


def build_generated_question_judge_messages(
    generated_question: dict[str, Any],
    criteria_text: str,
    output_schema_text: str,
) -> list[dict[str, str]]:
    payload = compact_generated_question_payload(generated_question)
    vietnamese_output_policy = (
        "Chính sách ngôn ngữ bắt buộc: mọi chuỗi người đọc trong output JSON phải viết bằng tiếng Việt có dấu. "
        "Áp dụng cho failed_reason, suggestions, issues[].reason, issues[].suggestion và mọi notes nếu có. "
        "Không dùng câu tiếng Anh. Chỉ giữ nguyên tiếng Anh/ký hiệu khi đó là tên field, enum, id, JSON Pointer, "
        "code, LaTeX hoặc nội dung trích nguyên văn từ generated question."
    )
    return [
        {
            "role": "system",
            "content": (
                "Bạn là Solution Quality Judge cho generated question. "
                "Nhiệm vụ duy nhất là đánh giá chất lượng, tính đầy đủ và tính nhất quán nội bộ của solution "
                "dựa trên instruction, stem và interaction type được cung cấp. "
                "Không dùng question_plan, raw question, raw answer, PDF, OCR hoặc dữ liệu ngoài payload. "
                "Không tạo generated question mới và không trực tiếp repair dữ liệu. "
                "Không tự giải lại toàn bộ bài toán để thiết lập một đáp án chuẩn mới. "
                "Không tạo final answer mới và không thay đổi answerSpec hoặc options. "
                "Đây là nhiệm vụ xác minh trung lập, không phải nhiệm vụ cố gắng tìm lỗi; không được giả định solution phải có lỗi. "
                "Một solution có thể chứa toàn bộ lời giải trong một text block duy nhất. Không giả định mỗi bước nằm trong một object, "
                "mỗi dòng là một bước hoặc mỗi block có ID riêng. Đọc các textBlocks theo contentIndex tăng dần. "
                "Trong từng text block, phải tự phân tách nội bộ các đơn vị lời giải theo đúng thứ tự đọc dựa trên ngữ nghĩa, "
                "có thể nhận biết qua xuống dòng, bullet, số thứ tự, dấu suy ra/ tương đương, chuỗi phương trình hoặc bất đẳng thức "
                "và các từ nối như Ta có, Suy ra, Do đó, Vậy, Thay vào, Rút gọn, Chia hai vế, Theo giả thiết. "
                "Các tín hiệu này không phải quy tắc máy móc: phải hiểu nội dung để xác định đơn vị lời giải. "
                "Việc phân tách chỉ dùng nội bộ trong lần đánh giá này; không trả danh sách step, step index, line index, internal trace "
                "hoặc chain-of-thought và không thay đổi dữ liệu đầu vào. "
                "Được phép và bắt buộc sử dụng kiến thức toán học/chuyên môn để thực hiện QUY TRÌNH TUẦN TỰ BẮT BUỘC sau: "
                "(1) Xem dữ kiện và yêu cầu trong instruction/stem là đơn vị đứng ngay trước đơn vị lời giải đầu tiên, "
                "rồi áp dụng đầy đủ checklist chuyển tiếp cho instruction/stem → đơn vị đầu tiên. Không yêu cầu solution chép lại đề bài, "
                "nhưng đơn vị đầu tiên phải trực tiếp suy ra từ dữ kiện, có phép tính chính xác và không gộp quá một phép biến đổi chính. "
                "Nếu để đi từ đề bài tới đơn vị đầu tiên cần từ hai phép biến đổi chính độc lập trở lên thì báo thiếu bước trung gian. "
                "Câu hỏi nhận biết trực tiếp không bị yêu cầu thêm bước trung gian khi thực tế không có phép biến đổi cần trình bày. "
                "(2) Với mỗi đơn vị lời giải tiếp theo, chỉ bắt đầu sau khi đơn vị trước đã được xác nhận hợp lệ và phải tự trả lời nội bộ theo đúng thứ tự: "
                "Thứ nhất, tính lại các phép toán hoặc quan hệ xuất hiện trong đơn vị mới. "
                "Thứ hai, xác nhận đơn vị mới có thực sự suy ra từ đúng đơn vị ngay trước hay không; không dùng final answer hoặc bước phía sau để hợp thức hóa. "
                "Thứ ba, xác định chuyển tiếp cần bao nhiêu phép biến đổi chính. "
                "Phép biến đổi tương đương không thay đổi tập nghiệm, gồm: chuyển vế đổi dấu (là cách viết tắt của cộng hoặc trừ cùng một biểu thức ở hai vế); "
                "cộng hoặc trừ cùng một biểu thức xác định; nhân hoặc chia hai vế cho cùng một hằng số khác 0; "
                "áp dụng lũy thừa bậc lẻ hoặc căn bậc lẻ tương ứng trên miền số thực. "
                "Phép biến đổi có nguy cơ thay đổi tập nghiệm gồm: lũy thừa bậc chẵn; nhân hoặc chia với biểu thức chứa ẩn; "
                "và phép biến đổi có điều kiện xác định như căn, logarit hoặc lượng giác. Với nhóm này phải giữ điều kiện cần thiết và kiểm tra nghiệm ngoại lai hoặc nghiệm bị mất. "
                "Mỗi lần áp dụng một phép biến đổi độc lập được tính là một phép biến đổi chính; các cách diễn đạt tương đương của cùng một thao tác không được đếm lặp. "
                "Chỉ chấp nhận tối đa một phép biến đổi chính trong mỗi chuyển tiếp. "
                "Các phép tính trực tiếp, rút gọn hệ số, chuẩn hóa ký hiệu hoặc viết lại tương đương trực tiếp trong chính phép biến đổi đó là thao tác vi mô. "
                "Thứ tư, nếu phép tính sai, quan hệ không suy ra được hoặc cần từ hai phép biến đổi chính độc lập trở lên thì "
                "DỪNG TOÀN BỘ kiểm tra toán học, chỉ tạo một issue cho chuyển tiếp đầu tiên này, không phân tích bước phía sau và không nhận xét final answer. "
                "Thứ năm, chỉ khi chuyển tiếp đúng mới đánh giá cách trình bày có đủ rõ để người học hiểu vì sao đơn vị mới suy ra từ đơn vị trước hay không. "
                "Với bài lập luận, bước xác nhận phải kiểm tra đúng quan hệ/quy tắc cùng đầy đủ tiền đề và điều kiện cần thiết. "
                "(3) Chỉ khi toàn bộ các chuyển tiếp đã hợp lệ mới kiểm tra thiếu nhánh nghiệm, điều kiện, trường hợp quan trọng, "
                "thiếu quá trình cần thiết, dài dòng, lặp ý, thử-sai, tự vấn hoặc đoạn nháp. "
                "(4) Trước khi trả issue, kiểm tra lại chính nhận định lỗi một lần; nếu nhận định không đứng vững thì hủy hoàn toàn issue đã dự kiến. "
                "Reason phải nhất quán và chỉ mô tả lỗi đầu tiên, không được đồng thời nói cùng một nội dung vừa sai vừa đúng. "
                "Nếu tất cả bước đều hợp lệ thì trả is_good=true và issues=[]. "
                "Nếu solution phụ thuộc bảng, hình, sơ đồ hoặc đồ thị nhưng dữ liệu cần thiết không có trong instruction/stem, "
                "không tự tưởng tượng dữ liệu và phải trả solution_quality/needs_manual_review. "
                "Không đánh giá answerSpec, option hoặc hint trong bước này. "
                f"{vietnamese_output_policy}"
            ),
        },
        {
            "role": "user",
            "content": (
                "Hãy chỉ đánh giá solution theo criteria và output schema bên dưới.\n\n"
                "Yêu cầu bắt buộc:\n"
                "- Chỉ dùng SOLUTION QUALITY PAYLOAD.\n"
                "- Không hard-code theo id/_id/aiId cụ thể.\n"
                "- Không so sánh với question_plan hoặc source/raw question/raw answer.\n"
                f"- {vietnamese_output_policy}\n"
                "- Chỉ tạo category solution_quality.\n"
                "- Chỉ dùng clean_solution_reasoning cho lỗi trình bày có thể làm sạch mà không đổi logic toán học hoặc final answer.\n"
                "- Dùng severity needs_review và repair_intent needs_manual_review cho phép tính/biến đổi sai, thiếu bước chính, "
                "thiếu nhánh/điều kiện, chỉ nêu đáp án, thiếu dữ liệu ngoài JSON hoặc solution không đủ để kiểm chứng.\n"
                "- Không tự repair lỗi toán học và không đề xuất final answer mới.\n"
                "- Solution có thể có nhiều bước trong cùng một text; phải tự phân đoạn nội bộ, không coi mỗi object là một bước và không dựa vào block ID.\n"
                "- Đọc các textBlocks theo contentIndex rồi đọc nội dung mỗi block theo thứ tự xuất hiện. Chỉ kiểm tra chuyển tiếp sau khi chuyển tiếp trước đã hợp lệ.\n"
                "- Khi gặp lỗi toán học đầu tiên, chỉ trả một issue cho solution đó; không bỏ qua lỗi trước để chọn lỗi nổi bật hơn phía sau.\n"
                "- location và required_context_paths phải dùng path thật của textBlocks trong payload, trỏ tới text block gốc chứa chuyển tiếp sai. "
                "Không tạo path step/line/statement không tồn tại.\n"
                "- Reason phải mô tả rõ đơn vị trước, đơn vị sai và phép tính hoặc quan hệ đúng chứng minh lỗi; không trả step index, line index, "
                "internal trace, chain-of-thought hoặc danh sách bước đã tách.\n"
                "- Chỉ trả JSON object hợp lệ, không markdown, không giải thích ngoài JSON.\n"
                "QUALITY CRITERIA:\n"
                f"{criteria_text}\n\n"
                "OUTPUT SCHEMA:\n"
                f"{output_schema_text}\n\n"
                "SOLUTION QUALITY PAYLOAD:\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def debug_generated_question_call(
    *,
    step: str,
    model: str,
    generated_question: dict[str, Any],
    schema_validation_result: dict[str, Any],
    prompt_chars: int,
    elapsed_seconds: float = 0,
    debug: bool = False,
) -> None:
    if not llm_prompt_debug_enabled(debug):
        return

    question_items = generated_question.get("questionItems")
    question_items = question_items if isinstance(question_items, list) else []
    interaction_count = 0
    for item in question_items:
        if isinstance(item, dict) and isinstance(item.get("interactions"), list):
            interaction_count += len(item["interactions"])

    payload = {
        "step": step,
        "model": model,
        "id": generated_question.get("id") or generated_question.get("_id"),
        "question_item_count": len(question_items),
        "interaction_count": interaction_count,
        "schema_issue_count": len(schema_validation_result.get("issues") or []),
        "prompt_chars": prompt_chars,
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    print("[DEBUG_GENERATED_QUESTION_JUDGE] " + json.dumps(payload, ensure_ascii=False), file=sys.stderr)


def normalize_llm_response(
    *,
    response: dict[str, Any] | None,
    strict_mode: bool,
    generated_question: dict[str, Any],
    index: int = 0,
    error: str = "",
) -> dict[str, Any]:
    if error:
        return fail_closed_output(error, generated_question=generated_question, index=index)
    content = str((response or {}).get("content") or "")
    parsed, parse_ok, parse_error = parse_json_output(content)
    if not parse_ok or parsed is None:
        return fail_closed_output(
            parse_error or "LLM output không parse được thành JSON.",
            generated_question=generated_question,
            index=index,
        )
    parsed.setdefault("id", generated_question_id(generated_question, index))
    parsed["issues"] = [
        issue
        for issue in parsed.get("issues") or []
        if isinstance(issue, dict) and issue.get("category") == "solution_quality"
    ]
    for issue in parsed["issues"]:
        intent = str(issue.get("repair_intent") or "")
        if intent not in {"clean_solution_reasoning", "needs_manual_review"}:
            issue["repair_intent"] = "needs_manual_review"
        if issue.get("repair_intent") == "needs_manual_review":
            issue["severity"] = "needs_review"
        else:
            issue["severity"] = "warning"
    first_blocking_by_solution: set[str] = set()
    retained_issues = []
    for issue in parsed["issues"]:
        location_parts = str(issue.get("location") or "").split("/")
        solution_owner = (
            location_parts[2]
            if len(location_parts) > 2
            and location_parts[0] == ""
            and location_parts[1] == "solutions"
            and location_parts[2].isdigit()
            else None
        )
        if solution_owner in first_blocking_by_solution:
            continue
        retained_issues.append(issue)
        if solution_owner is not None and issue.get("repair_intent") == "needs_manual_review":
            first_blocking_by_solution.add(solution_owner)
    parsed["issues"] = retained_issues
    return normalize_generated_question_result(
        parsed,
        strict_mode=strict_mode,
        generated_question=generated_question,
        index=index,
    )


def call_generated_question_judge(
    *,
    generated_question: dict[str, Any],
    schema_validation_result: dict[str, Any],
    config: AppConfig,
    client: LLMClient,
    model: str,
    strict_mode: bool,
    index: int = 0,
    debug: bool = False,
) -> dict[str, Any]:
    criteria_text = load_text(CRITERIA_PATH)
    output_schema_text = load_text(OUTPUT_SCHEMA_PATH)
    messages = build_generated_question_judge_messages(
        generated_question,
        criteria_text,
        output_schema_text,
    )
    prompt_chars = sum(len(message.get("content") or "") for message in messages)
    start = time.perf_counter()
    try:
        debug_llm_messages(step="generated_question_judge", model=model, messages=messages, debug=debug)
        response = client.chat_completion(
            model=model,
            messages=messages,
            temperature=0,
        )
        elapsed = float(response.get("latency_seconds") or (time.perf_counter() - start))
        debug_generated_question_call(
            step="generated_question_judge",
            model=model,
            generated_question=generated_question,
            schema_validation_result=schema_validation_result,
            prompt_chars=prompt_chars,
            elapsed_seconds=elapsed,
            debug=debug,
        )
        return normalize_llm_response(
            response=response,
            strict_mode=strict_mode,
            generated_question=generated_question,
            index=index,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        debug_generated_question_call(
            step="generated_question_judge_error",
            model=model,
            generated_question=generated_question,
            schema_validation_result=schema_validation_result,
            prompt_chars=prompt_chars,
            elapsed_seconds=elapsed,
            debug=debug,
        )
        return normalize_llm_response(
            response=None,
            strict_mode=strict_mode,
            generated_question=generated_question,
            index=index,
            error=str(exc),
        )


def judge_generated_question_object(
    generated_question: dict[str, Any],
    schema_validation_result: dict[str, Any],
    config: AppConfig,
    client: LLMClient,
    *,
    strict_mode: bool = True,
    index: int = 0,
    debug: bool = False,
) -> dict[str, Any]:
    reasoning_model = generated_question_reasoning_model(config)
    primary = call_generated_question_judge(
        generated_question=generated_question,
        schema_validation_result=schema_validation_result,
        config=config,
        client=client,
        model=reasoning_model,
        strict_mode=strict_mode,
        index=index,
        debug=debug,
    )
    if primary.get("issues") and any(issue.get("category") == "runtime" for issue in primary["issues"]):
        fallback_model = generated_question_fast_model(config)
        if config.use_fallback_judge and fallback_model and fallback_model != reasoning_model:
            fallback = call_generated_question_judge(
                generated_question=generated_question,
                schema_validation_result=schema_validation_result,
                config=config,
                client=client,
                model=fallback_model,
                strict_mode=strict_mode,
                index=index,
                debug=debug,
            )
            if not any(issue.get("category") == "runtime" for issue in fallback.get("issues") or []):
                return fallback
    return primary

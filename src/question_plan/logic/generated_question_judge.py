"""LLM judge riêng cho chất lượng generated question object."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from ..infra.config import AppConfig
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
TYPE_RULES_PATH = KNOWLEDGE_DIR / "generated_question_type_rules.md"
OUTPUT_SCHEMA_PATH = KNOWLEDGE_DIR / "generated_question_output_schema.md"


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def compact_generated_question_payload(
    generated_question: dict[str, Any],
    schema_validation_result: dict[str, Any],
) -> dict[str, Any]:
    """Chỉ giữ phần cần thiết để LLM đánh giá nội bộ generated question."""

    return {
        "id": generated_question.get("id") or generated_question.get("_id") or "",
        "_id": generated_question.get("_id"),
        "aiId": generated_question.get("aiId"),
        "difficulty": generated_question.get("difficulty"),
        "bloom": generated_question.get("bloom"),
        "interactionTypes": generated_question.get("interactionTypes"),
        "instruction": generated_question.get("instruction"),
        "questionItems": generated_question.get("questionItems"),
        "solutions": generated_question.get("solutions"),
        "schema_validation_result": schema_validation_result,
    }


def build_generated_question_judge_messages(
    generated_question: dict[str, Any],
    schema_validation_result: dict[str, Any],
    criteria_text: str,
    type_rules_text: str,
    output_schema_text: str,
) -> list[dict[str, str]]:
    payload = compact_generated_question_payload(generated_question, schema_validation_result)
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
                "Bạn là LLM judge đánh giá chất lượng nội bộ của một generated question object "
                "cho câu hỏi Toán tương tác. Chỉ đánh giá generated question được cung cấp. "
                "Không dùng question_plan, raw question, raw answer, source images hoặc answer images. "
                "Không tạo new_question_plan và không repair dữ liệu. Không xác định đáp án đúng, không tự giải bài "
                "và không dùng kiến thức chuyên môn để phủ định solution hoặc answerSpec. "
                f"{vietnamese_output_policy}"
            ),
        },
        {
            "role": "user",
            "content": (
                "Hãy đánh giá generated question object theo criteria, type rules và output schema bên dưới.\n\n"
                "Yêu cầu bắt buộc:\n"
                "- Chỉ dùng DYNAMIC PAYLOAD và schema_validation_result.\n"
                "- Không hard-code theo id/_id/aiId cụ thể.\n"
                "- Không so sánh với question_plan hoặc source/raw question/raw answer.\n"
                "- Không tạo category plan_alignment hoặc source_fidelity.\n"
                "- Bỏ qua hoàn toàn difficulty và bloom; không tạo issue, failed_reason hoặc suggestion liên quan đến hai field này.\n"
                f"- {vietnamese_output_policy}\n"
                "- Nếu criteria/type rules/schema/payload có tiếng Anh, vẫn phải diễn giải lỗi và gợi ý bằng tiếng Việt có dấu.\n"
                "- Với mỗi issue có thể sửa, trả location dạng JSON Pointer, error_snippet ngắn, required_context_paths đủ hẹp và repair_intent.\n"
                "- Không tạo answer_internal_consistency, solution_anchor_consistency hoặc repair_intent fix_correct_option.\n"
                "- Không kiểm tra hint đúng/sai theo đáp án; hint alignment thuộc Solution Resolver. Chỉ kiểm tra hint quá lộ đáp án hoặc chất lượng trình bày bề mặt.\n"
                "- Distractor được phép sai có chủ đích về số mũ, hệ số, dấu, đơn vị hoặc phép biến đổi. Sự gần giống đáp án không phải bằng chứng typo.\n"
                "- Chỉ báo choice_quality khi option rỗng, trùng/tương đương hoàn toàn, hỏng render, vô nghĩa hoặc không thể dùng.\n"
                "- Với lỗi trình bày solution, required_context_paths cần gồm solution block liên quan.\n"
                "- Với lỗi hint, required_context_paths cần gồm stem, answerSpecs và hints liên quan.\n"
                "- Nếu không chắc context đủ để sửa scoped an toàn, đặt repair_intent='needs_manual_review'.\n"
                "- Chỉ trả JSON object hợp lệ, không markdown, không giải thích ngoài JSON.\n"
                "- Nếu thiếu bằng chứng để kết luận chắc chắn, dùng severity needs_review.\n\n"
                "- Đánh giá chất lượng trình bày solution từ góc nhìn một học sinh chưa biết cách giải bài: solution phải có cầu nối lập luận đủ rõ từ dữ kiện ban đầu đến kết luận để học sinh hiểu vì sao kết luận được suy ra. "
                "Không coi việc chỉ lặp lại đề bài, chỉ nêu công thức chưa áp dụng hoặc nhảy thẳng đến đáp án là một lời giải đầy đủ. "
                "Không yêu cầu trình bày các phép biến đổi hiển nhiên hoặc mọi bước tính vi mô; solution ngắn vẫn đạt nếu mạch suy luận đủ hiểu. "
                "Chỉ đánh giá mức độ đầy đủ và chất lượng trình bày, không kiểm tra đúng sai chuyên môn, không tự giải lại bài và không thay đổi hoặc tự kết luận lại final answer.\n\n"
                "QUALITY CRITERIA:\n"
                f"{criteria_text}\n\n"
                "GENERATED QUESTION TYPE RULES:\n"
                f"{type_rules_text}\n\n"
                "OUTPUT SCHEMA:\n"
                f"{output_schema_text}\n\n"
                "DYNAMIC PAYLOAD:\n"
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
    type_rules_text = load_text(TYPE_RULES_PATH)
    output_schema_text = load_text(OUTPUT_SCHEMA_PATH)
    messages = build_generated_question_judge_messages(
        generated_question,
        schema_validation_result,
        criteria_text,
        type_rules_text,
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
    primary = call_generated_question_judge(
        generated_question=generated_question,
        schema_validation_result=schema_validation_result,
        config=config,
        client=client,
        model=config.primary_judge_model,
        strict_mode=strict_mode,
        index=index,
        debug=debug,
    )
    if primary.get("issues") and any(issue.get("category") == "runtime" for issue in primary["issues"]):
        if config.use_fallback_judge and config.fallback_judge_model:
            fallback = call_generated_question_judge(
                generated_question=generated_question,
                schema_validation_result=schema_validation_result,
                config=config,
                client=client,
                model=config.fallback_judge_model,
                strict_mode=strict_mode,
                index=index,
                debug=debug,
            )
            if not any(issue.get("category") == "runtime" for issue in fallback.get("issues") or []):
                return fallback
    return primary


def judge_generated_questions(
    generated_question: dict[str, Any],
    schema_validation_result: dict[str, Any],
    config: AppConfig,
    client: LLMClient,
    *,
    strict_mode: bool = True,
    index: int = 0,
    debug: bool = False,
) -> dict[str, Any]:
    """Backward-compatible alias cho code cũ."""

    return judge_generated_question_object(
        generated_question,
        schema_validation_result,
        config,
        client,
        strict_mode=strict_mode,
        index=index,
        debug=debug,
    )

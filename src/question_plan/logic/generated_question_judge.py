"""LLM judge đánh giá chất lượng solution của generated question object."""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ..infra.config import (
    AppConfig,
    generated_question_correctness_model,
    generated_question_presentation_model,
)
from ..infra.debug import (
    debug_llm_messages,
    llm_prompt_debug_enabled,
    trace_pipeline_event,
)
from ..infra.llm_client import LLMClient
from ..shared.utils import parse_json_output
from ..vision.visual_context import attach_remote_visual_assets, extract_asset_references
from ..schemas.generated_question_contracts import (
    DirectSolutionCorrectnessOutput,
    JudgeCommentsReviewOutput,
    ProcessPresentationSemanticOutput,
    validation_error_text,
)
from .generated_question_schema import (
    fail_closed_output,
    normalize_generated_question_result,
)


MARKDOWN_IMAGE_PATTERN = re.compile(
    r"!\[(?P<description>.*?)\]\((?P<asset_url>https?://[^)\s]+)\)",
    re.DOTALL,
)
class _RuntimeFailureDetail(str):
    """Internal marker that preserves the existing `(result, error)` handoff."""


_SUPERSCRIPT_CHARACTERS = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁺": "+",
        "⁻": "-",
    }
)
_SUPERSCRIPT_RUN = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+")


def normalize_math_evidence(text: str) -> str:
    """Normalize harmless math notation differences without semantic matching."""

    normalized = str(text or "")
    normalized = _SUPERSCRIPT_RUN.sub(
        lambda match: "^" + match.group(0).translate(_SUPERSCRIPT_CHARACTERS),
        normalized,
    )
    normalized = re.sub(r"\\+(?:times|cdot)\b", "*", normalized)
    normalized = re.sub(r"\\+div\b", "/", normalized)
    normalized = re.sub(r"\\+(?:left|right)\b", "", normalized)
    normalized = normalized.replace("$", "")
    normalized = re.sub(r"\\+[,;! ]", "", normalized)
    normalized = re.sub(r"\\+(?=[A-Za-z])", r"\\", normalized)
    normalized = re.sub(r"\^\s*\{\s*([+-]?\d+)\s*\}", r"^\1", normalized)
    normalized = normalized.translate(
        str.maketrans(
            {
                "×": "*",
                "⋅": "*",
                "·": "*",
                "÷": "/",
                "−": "-",
                "–": "-",
                "—": "-",
                "‑": "-",
            }
        )
    )
    return re.sub(r"\s+", "", normalized)


def _normalized_evidence_position(evidence: str, source: str) -> int:
    normalized_evidence = normalize_math_evidence(evidence)
    if not normalized_evidence:
        return -1
    return normalize_math_evidence(source).find(normalized_evidence)


def _prompt_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _contract_retry_candidate_payload(candidate: str | None) -> dict[str, Any]:
    raw_candidate = str(candidate or "")
    if len(raw_candidate) <= 2000:
        return {"raw_candidate": raw_candidate}
    return {
        "raw_candidate_omitted": True,
        "raw_candidate_chars": len(raw_candidate),
        "note": "Output trước quá dài và đã bị cắt; hãy tạo lại JSON ngắn từ input gốc.",
    }


def direct_correctness_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "direct_solution_correctness_decision",
            "strict": True,
            "schema": DirectSolutionCorrectnessOutput.model_json_schema(),
        },
    }


def judge_comments_review_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "judge_comments_review",
            "strict": True,
            "schema": JudgeCommentsReviewOutput.model_json_schema(),
        },
    }


def process_presentation_response_format() -> dict[str, Any]:
    """Schema for candidate process/presentation comments."""

    schema = ProcessPresentationSemanticOutput.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "solution_process_presentation_decision",
            "strict": True,
            "schema": schema,
        },
    }


def compact_generated_question_payload(
    generated_question: dict[str, Any],
) -> dict[str, Any]:
    """Chỉ gửi ngữ cảnh cần thiết để đánh giá solution, không gửi đáp án."""

    question_items = []
    for item_index, item in enumerate(generated_question.get("questionItems") or []):
        if not isinstance(item, dict):
            continue
        essay_interactions = []
        for interaction_index, interaction in enumerate(item.get("interactions") or []):
            if not isinstance(interaction, dict) or str(interaction.get("type") or "") != "essay":
                continue
            essay_context = {
                "index": interaction_index,
                "id": interaction.get("id"),
                "type": "essay",
                "config": interaction.get("config") or {},
            }
            for field_name in (
                "prompt",
                "instruction",
                "instructions",
                "requirements",
                "rubric",
                "criteria",
                "evaluationCriteria",
            ):
                if interaction.get(field_name) not in (None, "", [], {}):
                    essay_context[field_name] = interaction[field_name]
            essay_interactions.append(essay_context)
        interaction_types = [
            str(interaction.get("type") or "")
            for interaction in item.get("interactions") or []
            if isinstance(interaction, dict) and str(interaction.get("type") or "")
        ]
        compact_item = {
            "index": item_index,
            "id": item.get("id"),
            "stem": item.get("stem"),
            "interactionTypes": interaction_types,
        }
        if essay_interactions:
            compact_item["essayInteractions"] = essay_interactions
            for field_name in (
                "rubric",
                "criteria",
                "evaluationCriteria",
                "requirements",
            ):
                if item.get(field_name) not in (None, "", [], {}):
                    compact_item[field_name] = item[field_name]
        question_items.append(compact_item)

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
                "path": f"/solutions/{solution_index}",
                "textBlocks": text_blocks,
            }
        )

    payload = {
        "id": generated_question.get("id") or generated_question.get("_id") or "",
    }
    if generated_question.get("instruction"):
        payload["instruction"] = generated_question["instruction"]
    payload["questionItems"] = question_items
    payload["solutions"] = solutions
    return payload

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

def _call_structured_judge(
    *,
    generated_question: dict[str, Any],
    schema_validation_result: dict[str, Any],
    client: LLMClient,
    model: str,
    debug: bool,
    step: str,
    messages: list[dict[str, Any]],
    validator: Callable[[Any], tuple[dict[str, Any] | None, str]],
    invalid_json_message: str,
    response_format: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    retry_transient_once: bool = False,
) -> dict[str, Any]:
    prompt_chars = sum(
        sum(
            len(str(part.get("text") or ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
        if isinstance((content := message.get("content")), list)
        else len(str(content or ""))
        for message in messages
    )
    start = time.perf_counter()
    try:
        trace_pipeline_event(
            stage=step,
            event="input",
            payload={
                "model": model,
                "messages": messages,
                "response_format": response_format,
                "max_tokens": max_tokens,
            },
            object_id=generated_question.get("id") or generated_question.get("_id"),
        )
        debug_llm_messages(step=step, model=model, messages=messages, debug=debug)
        response = client.chat_completion(
            model=model,
            messages=messages,
            temperature=0,
            response_format=response_format,
            max_tokens=max_tokens,
            retry_transient_once=retry_transient_once,
        )
        http_retry_count = int(response.get("http_retry_count") or 0)
        trace_pipeline_event(
            stage=step,
            event="output",
            payload=response,
            object_id=generated_question.get("id") or generated_question.get("_id"),
        )
        elapsed = float(response.get("latency_seconds") or (time.perf_counter() - start))
        debug_generated_question_call(
            step=step,
            model=model,
            generated_question=generated_question,
            schema_validation_result=schema_validation_result,
            prompt_chars=prompt_chars,
            elapsed_seconds=elapsed,
            debug=debug,
        )
        parsed, parse_ok, parse_error = parse_json_output(str(response.get("content") or ""))
        if not parse_ok or parsed is None:
            if debug:
                print(
                    "[DEBUG_GENERATED_QUESTION_CONTRACT_INVALID] "
                    + json.dumps(
                        {
                            "step": step,
                            "model": model,
                            "error": parse_error or invalid_json_message,
                            "raw_candidate": str(response.get("content") or ""),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                )
            return {
                "contract_valid": False,
                "contract_error": parse_error or invalid_json_message,
                "contract_failure_kind": "mechanical",
                "raw_candidate": str(response.get("content") or ""),
                "http_retry_count": http_retry_count,
            }
        output, error = validator(parsed)
        if output is None:
            if debug:
                print(
                    "[DEBUG_GENERATED_QUESTION_CONTRACT_INVALID] "
                    + json.dumps(
                        {
                            "step": step,
                            "model": model,
                            "error": error,
                            "raw_candidate": str(response.get("content") or ""),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                )
            return {
                "contract_valid": False,
                "contract_error": error,
                "contract_failure_kind": (
                    "mechanical"
                    if str(error or "").startswith("mechanical:")
                    else "semantic"
                ),
                "raw_candidate": str(response.get("content") or ""),
                "http_retry_count": http_retry_count,
            }
        return {
            "contract_valid": True,
            "http_retry_count": http_retry_count,
            **output,
        }
    except Exception as exc:
        trace_pipeline_event(
            stage=step,
            event="error",
            payload={"type": type(exc).__name__, "message": str(exc)},
            object_id=generated_question.get("id") or generated_question.get("_id"),
        )
        debug_generated_question_call(
            step=f"{step}_error",
            model=model,
            generated_question=generated_question,
            schema_validation_result=schema_validation_result,
            prompt_chars=prompt_chars,
            elapsed_seconds=time.perf_counter() - start,
            debug=debug,
        )
        return {
            "contract_valid": False,
            "contract_error": str(exc),
            "failure_kind": "runtime",
            "http_retry_count": int(getattr(exc, "retry_count", 0) or 0),
        }

def _direct_solution_payload(generated_question: dict[str, Any]) -> dict[str, Any]:
    """Return question context and original solution blocks without segmentation."""

    return compact_generated_question_payload(generated_question)


def build_no_prompt_messages(
    generated_question: dict[str, Any],
) -> list[dict[str, Any]]:
    def text_from_blocks(blocks: Any) -> list[str]:
        return [
            str(block["text"])
            for block in blocks or []
            if isinstance(block, dict)
            and isinstance(block.get("text"), str)
            and block["text"].strip()
        ]

    problem_parts = text_from_blocks(generated_question.get("instruction"))
    for item in generated_question.get("questionItems") or []:
        if isinstance(item, dict):
            problem_parts.extend(text_from_blocks(item.get("stem")))
    solution_parts = []
    for solution in generated_question.get("solutions") or []:
        if isinstance(solution, dict):
            solution_parts.extend(text_from_blocks(solution.get("solutionContent")))
    return [{
        "role": "user",
        "content": (
            f"Đề:\n{'\n'.join(problem_parts)}\n\n"
            f"Lời giải:\n{'\n'.join(solution_parts)}\n\n"
            "Kiểm tra lời giải của bài trên"
        ),
    }]


JSON_SERIALIZATION_CONSTRAINTS = """### QUY TẮC GHI JSON
- Chỉ trả một JSON object hợp lệ, không dùng markdown hoặc code fence.
- Yêu cầu kết quả trả về phải viết ngắn gọn bằng tiếng Việt, rõ nghĩa.
- Phải escape dấu gạch chéo ngược `\\` bên trong JSON string thành `\\\\`.
- Các biểu thức LaTeX như `\\sqrt`, `\\frac`, `\\pi` phải được ghi thành `\\\\sqrt`, `\\\\frac`, `\\\\pi`.
- Không viết nội dung bên ngoài JSON object."""

CORRECTNESS_CRITERIA = """1. Chỉ dùng instruction, stem, dữ liệu trực quan và nguyên văn solution. Không dùng answerSpec, expected, options, hints hoặc metadata để suy ra lời giải đúng.

2. Kiểm tra solution từ đầu đến cuối. Trước khi chuyển sang một khẳng định, phải kiểm chứng xong nội dung đứng trước và quan hệ của nó với ngữ cảnh đang có. Kiểm tra lần lượt phép tính, phép biến đổi, công thức, điều kiện áp dụng, miền xác định, các trường hợp bắt buộc và kết luận. Giữ nguyên thứ tự đó trong `comments`; không dùng kết quả đúng ở cuối để bỏ qua lỗi sớm hơn. Phân biệt nội dung đang được khẳng định với phần thử sai hoặc đã tự bác bỏ.

3. Với mỗi questionItem hoặc câu con được solution trả lời, tạo một `opening_check` theo đúng thứ tự và đối chiếu khẳng định toán học đầu tiên của phần lời giải tương ứng với stem của phần đó. Nếu một solution trả lời nhiều questionItems, các opening checks vẫn dùng cùng `solution_index`. Với phép tính số học đơn giản, tự tính biểu thức từ stem trước rồi ghi phép tính và kết quả độc lập vào `analysis`. Nếu đề phụ thuộc hình, bảng, đồ thị hoặc dữ liệu trực quan nhưng input không có ảnh, liên kết, mô tả hay dữ liệu chữ đủ thay thế, ghi nhận thiếu dữ kiện trong cả `opening_checks` và `comments`, rồi không kiểm tra tiếp phần đó. Không tưởng tượng dữ liệu còn thiếu.

4. Chỉ tạo `comment` khi chỉ ra được một lỗi toán học cụ thể trong solution: khẳng định hoặc phép tính sai, suy luận không hợp lệ, vi phạm điều kiện áp dụng, hoặc thiếu điều kiện, trường hợp hay kết luận bắt buộc. Tự kiểm chứng phép tính thay vì lấy kết quả trong solution làm mốc. Trích `evidence` là đoạn nguyên văn ngắn nhất đủ định vị lỗi. Nếu một bước chỉ kế thừa tiền đề sai nhưng phép suy ra tại bước đó vẫn hợp lệ, chỉ ghi lỗi gốc; chỉ ghi thêm khi bước sau có lỗi độc lập.

5. Không tạo lỗi Correctness cho cách giải khác nhưng hợp lệ, cách viết ngắn hoặc gộp bước vẫn kiểm chứng được, câu dẫn, mức độ chi tiết, bước thừa, nội dung nháp hay độ mạch lạc. Những vấn đề này thuộc kiểm tra quá trình và trình bày.

6. Với interaction `essay`, coi solution là bài trả lời mẫu và kiểm tra tính đúng đắn, dẫn chứng, trường hợp cùng kết luận theo instruction, stem và rubric được cung cấp. Không dùng `minWords` hoặc `maxWords` làm chuẩn độ dài của solution mẫu."""

def build_direct_correctness_messages(
    generated_question: dict[str, Any],
    contract_retry_error: str | None = None,
    contract_retry_candidate: str | None = None,
) -> list[dict[str, Any]]:
    payload = _direct_solution_payload(generated_question)
    retry = ""
    if contract_retry_error:
        retry_instruction = "Chỉ sửa lỗi hợp đồng; không đổi nhận định toán học."
        if "opening_checks chưa bao phủ tất cả questionItems" in contract_retry_error:
            retry_instruction = (
                "Giữ các nhận định đã có và bổ sung opening_check còn thiếu để mỗi questionItem "
                "hoặc câu con được kiểm tra đúng một lượt theo thứ tự xuất hiện. Mọi opening_check "
                "của solution kết hợp vẫn dùng solution_index của solution đó."
            )
        retry = (
            "\n\n### SỬA HỢP ĐỒNG ĐẦU RA\n"
            f"Lỗi hợp đồng: {contract_retry_error}\n"
            f"Kết quả thô: {_prompt_json(_contract_retry_candidate_payload(contract_retry_candidate))}\n"
            f"{retry_instruction}"
        )
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Bạn là giáo viên toán chịu trách nhiệm kiểm tra tính đúng đắn của lời giải học sinh. "
                "Đọc và kiểm chứng nội dung theo các quy tắc được cung cấp. Chỉ trả JSON đúng schema."
            ),
        },
        {
            "role": "user",
            "content": (
                f"### TIÊU CHÍ VÀ YÊU CẦU KIỂM TRA\n{CORRECTNESS_CRITERIA}\n\n"
                f"### ĐỀ BÀI VÀ LỜI GIẢI NGUYÊN VĂN\n{_prompt_json(payload)}\n\n"
                "### ĐẦU RA\n"
                "opening_checks phải bao phủ từng questionItem hoặc câu con được solution trả lời; các phần trong cùng solution vẫn dùng index của solution đó. "
                "Mọi vấn đề cần kiểm chứng phải nằm trong comments; nếu không có thì trả comments=[]. "
                "Mỗi evidence phải được sao chép từ một đoạn liên tục trong solution; chỉ đặt dấu `$` bao quanh khi chính đoạn nguồn cũng có dấu `$` tại hai biên đó. "
                "Không tự phản biện comments và không trả kết luận tổng hợp, verdict, error type hoặc suggestion.\n"
                f"{_prompt_json(DirectSolutionCorrectnessOutput.model_json_schema())}\n\n"
                f"{JSON_SERIALIZATION_CONSTRAINTS}{retry}"
            ),
        },
    ]
    return attach_remote_visual_assets(messages, payload)


PROCESS_PRESENTATION_CRITERIA = """1. Đọc toàn bộ solution theo thứ tự và chỉ đánh giá mức đầy đủ, khả năng theo dõi cùng chất lượng trình bày. Không tự xác minh hay phán quyết đúng sai toán học. Không dùng answerSpec, expected, options, hints hoặc metadata.

2. Chỉ báo `missing_major_step` khi một cầu nối bắt buộc hoàn toàn không xuất hiện trong lời giải, kể cả dưới dạng công thức hoặc câu chữ.

    Trước khi kết luận, phải đọc toàn bộ `bieu_thuc_truoc` và toàn bộ `bieu_thuc_sau`, rồi khôi phục chuỗi các bước đã được viết theo đúng thứ tự. Các biểu thức ngăn cách bằng dấu phẩy, chấm phẩy, dấu chấm hoặc xuống dòng vẫn được tính là các bước tường minh, kể cả khi nằm trong cùng một state hoặc cùng một câu.

    Nếu biểu thức trung gian đã xuất hiện ở bất kỳ vị trí nào trong nội dung đang kiểm tra thì tuyệt đối không được báo biểu thức đó bị thiếu.

    Không yêu cầu viết riêng cụ thể các phép tính số học phụ như:
    `5x^5 = 200 - 40, 5x^5 = 160`, hay `x^5 = 160 : 5, x^5 = 32`,
    nếu kết quả của chúng đã được thể hiện trong bước biến đổi chính.

    Ví dụ: Giải phương trình 5x^5 + 40 = 200. Ta có: 5x^5 = 160, x^5 = 32, x = 2
    Chuỗi tường minh là:
    I. 5x^5 = 160
    II. x^5 = 32
    III. x = 2
4. Dùng `redundant_step` cho nội dung không cần thiết và mô tả mức ảnh hưởng. Các loại còn lại dùng cho nội dung nháp hoặc tự hỏi, thử-sai chưa làm sạch, tự mâu thuẫn trong trình bày, lặp nghiêm trọng, quá dài dòng hoặc diễn đạt không phù hợp. Không báo lỗi chỉ vì thiếu một câu kết luận riêng khi kết quả đã xuất hiện rõ. Chỉ trả vấn đề rõ ràng nhất và sớm nhất của mỗi solution.

5. Với interaction `essay`, dùng instruction, stem và rubric được cung cấp để xem xét các phần trình bày chính và khả năng theo dõi mạch lập luận. Không dùng giới hạn ô nhập làm chuẩn độ dài của solution mẫu.

6. Mỗi comment phải có `solution_index`, một đoạn `evidence` nguyên văn đủ định vị, `reason` và `suggestion`. Nếu không có vấn đề, trả `comments=[]`. Không kết luận toàn bộ lời giải tốt hay xấu."""

def build_direct_process_presentation_messages(
    generated_question: dict[str, Any],
    contract_retry_error: str | None = None,
    contract_retry_candidate: str | None = None,
) -> list[dict[str, Any]]:
    payload = _direct_solution_payload(generated_question)
    retry = ""
    if contract_retry_error:
        retry = (
            "\n\n### SỬA HỢP ĐỒNG ĐẦU RA\n"
            f"Lỗi hợp đồng: {contract_retry_error}\n"
            f"Kết quả thô: {_prompt_json(_contract_retry_candidate_payload(contract_retry_candidate))}\n"
            "Chỉ sửa cấu trúc JSON; không đánh giá lại nội dung."
        )
    return [
        {
            "role": "system",
            "content": (
                "Bạn là giáo viên toán chịu trách nhiệm xem xét quá trình và cách trình bày lời giải của bài toán để học sinh có thể theo dõi. "
                "Đọc và đánh giá theo các quy tắc được cung cấp. Chỉ trả JSON đúng schema."
            ),
        },
        {
            "role": "user",
            "content": (
                f"### QUY TẮC\n{PROCESS_PRESENTATION_CRITERIA}\n\n"
                f"### ĐỀ BÀI VÀ LỜI GIẢI NGUYÊN VĂN\n{_prompt_json(payload)}\n\n"
                "### YÊU CẦU\n"
                "- Nếu không thấy vấn đề, trả comments là mảng rỗng. Chỉ trả tối đa một nhận xét rõ ràng nhất và sớm nhất cho mỗi solution.\n"
                "- evidence phải là một đoạn trích nguyên văn liên tục, ngắn nhất đủ định vị lỗi và không quá 300 ký tự; tuyệt đối không sao chép cả đoạn lời giải dài.\n"
                "- reason tối đa ba câu và suggestion một câu.\n"
                "- Chỉ trả mảng comments theo schema; không trả kết luận tổng hợp hoặc verdict.\n"
                f"### LƯỢC ĐỒ ĐẦU RA\n{_prompt_json(ProcessPresentationSemanticOutput.model_json_schema())}\n\n"
                f"{JSON_SERIALIZATION_CONSTRAINTS}{retry}"
            ),
        },
    ]


def validate_direct_correctness_output(
    parsed: Any,
    generated_question: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    try:
        decision = DirectSolutionCorrectnessOutput.model_validate(parsed).model_dump()
    except ValidationError as exc:
        return None, "mechanical: " + validation_error_text(exc)
    direct_payload = _direct_solution_payload(generated_question)
    solutions = direct_payload.get("solutions") or []
    question_items = direct_payload.get("questionItems") or []
    valid_indexes = {int(solution["index"]) for solution in solutions}
    opening_checks = decision["opening_checks"]
    if valid_indexes == {0}:
        for collection_name in ("opening_checks", "comments"):
            for item in decision[collection_name]:
                item["solution_index"] = 0
    opening_indexes = [check["solution_index"] for check in opening_checks]
    if set(opening_indexes) != valid_indexes or opening_indexes != sorted(opening_indexes):
        return None, "mechanical: opening_checks phải bao phủ mỗi solution và đúng thứ tự."
    if len(solutions) == 1 and len(question_items) > 1 and len(opening_checks) != len(question_items):
        return None, (
            "mechanical: opening_checks chưa bao phủ tất cả questionItems; "
            f"cần {len(question_items)} phần tử nhưng nhận {len(opening_checks)}."
        )
    opening_evidence_keys: set[str] = set()
    for check in opening_checks:
        if not check["evidence"].strip() or not check["analysis"].strip():
            return None, "mechanical: opening check phải có evidence và analysis."
        evidence_key = normalize_math_evidence(check["evidence"])
        if evidence_key in opening_evidence_keys:
            return None, "mechanical: opening_checks không được lặp lại cùng một evidence."
        opening_evidence_keys.add(evidence_key)
    comments = decision["comments"]
    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    solution_texts = {
        int(solution["index"]): "\n".join(
            str(block.get("text") or "") for block in solution.get("textBlocks") or []
        )
        for solution in solutions
    }
    last_position_by_solution: dict[int, int] = {}
    for comment in comments:
        if comment["solution_index"] not in valid_indexes:
            return None, "mechanical: nhận xét Correctness phải trỏ tới solution_index hợp lệ."
        if not comment["evidence"].strip() or not comment["reason"].strip():
            return None, "mechanical: nhận xét Correctness phải có evidence và reason."
        solution_index = int(comment["solution_index"])
        evidence_position = _normalized_evidence_position(
            str(comment["evidence"]),
            solution_texts.get(solution_index, ""),
        )
        if (
            evidence_position >= 0
            and evidence_position < last_position_by_solution.get(solution_index, -1)
        ):
            return None, "mechanical: comments Correctness phải theo đúng thứ tự xuất hiện trong solution."
        if evidence_position >= 0:
            last_position_by_solution[solution_index] = evidence_position
        key = (comment["solution_index"], normalize_math_evidence(comment["evidence"]))
        if key not in seen:
            seen.add(key)
            deduplicated.append(comment)
    return {
        "opening_checks": opening_checks,
        "comments": deduplicated,
    }, ""


def validate_direct_process_output(
    parsed: Any,
    generated_question: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    try:
        semantic = ProcessPresentationSemanticOutput.model_validate(parsed).model_dump()
    except ValidationError as exc:
        return None, "mechanical: " + validation_error_text(exc)
    valid_indexes = {
        int(solution["index"])
        for solution in _direct_solution_payload(generated_question).get("solutions") or []
    }
    if valid_indexes == {0}:
        for item in semantic["comments"]:
            item["solution_index"] = 0
    comments = semantic["comments"]
    for comment in comments:
        if comment["solution_index"] not in valid_indexes:
            return None, "mechanical: nhận xét Process phải trỏ tới solution_index hợp lệ."
        if comment["error_type"] is None:
            return None, "mechanical: nhận xét Process phải có error_type."
        if not comment["evidence"].strip() or not comment["reason"].strip() or not comment["suggestion"].strip():
            return None, "mechanical: nhận xét Process phải có evidence, reason và suggestion."
    return {"comments": comments}, ""


COMMENTS_REVIEW_CRITERIA = """1. Chỉ kiểm chứng từng comment được cung cấp. Tìm `evidence` trong solution, đọc ngữ cảnh cần thiết và đối chiếu cáo buộc trong `reason` với instruction, stem cùng các `independent_checks` đi kèm. Không tự tìm lỗi mới, không thay đổi cáo buộc và không đánh giá lại toàn bộ solution. Nếu cáo buộc không được chứng minh thì bác bỏ.

2. Trả đúng một disposition cho mỗi `comment_id`:
   - `blocking`: nhận xét đúng và chỉ ra lỗi chắc chắn làm lời giải sai, không đầy đủ về toán học, hoặc hỏng nghiêm trọng mạch trình bày;
   - `advisory`: nhận xét đúng nhưng chỉ là vấn đề trình bày nhẹ, không làm lời giải sai;
   - `rejected`: nhận xét sai, không đủ căn cứ hoặc quá khắt khe.

3. Với comment từ Correctness, chỉ dùng `blocking` khi `reason` chứng minh một lỗi toán học cụ thể. Dùng `rejected` nếu lý do vòng vo, tự mâu thuẫn, chỉ nói về tính cần thiết hay cách trình bày, phủ nhận một biểu thức tương đương, hoặc biến hệ quả của tiền đề sai thành lỗi mới dù phép suy ra hiện tại vẫn hợp lệ. Khi chấp nhận, lấy phép tính hoặc quan hệ đúng trực tiếp từ candidate `reason`; chỉ rút gọn câu chữ, không tự tính lại hay tạo giá trị trung gian mới. Không dùng `advisory` cho comment Correctness.

   Nếu comment Correctness nêu thiếu hình, bảng, đồ thị hoặc dữ liệu trực quan, đối chiếu instruction, stem, dữ liệu chữ và danh sách ảnh đính kèm. Dùng `blocking` khi đề thực sự phụ thuộc dữ liệu đó nhưng không có ảnh, liên kết, mô tả hoặc dữ liệu chữ đủ thay thế; việc không thể kiểm chứng vì thiếu nguồn chính là căn cứ chấp nhận comment, không phải lý do bác bỏ. Dùng `rejected` khi dữ liệu cần thiết đã được cung cấp hoặc đề không phụ thuộc nó. Không tưởng tượng nội dung còn thiếu.

4. Với comment từ Process, kiểm chứng đúng loại lỗi đã nêu. Riêng `missing_major_step`, dùng `blocking` khi giữa hai nội dung liền kề thực sự chứa ít nhất nhất hai phép biến đổi chính (cộng, trừ, nhân, chia, khai căn, số mũ, liên hợp,...) với cùng một toán tử hoặc một ý tưởng bắt buộc. 
    - Tuy nhiên, không yêu cầu viết riêng cụ thể các phép tính số học đơn giản như sau: 
    `5x^5 = 200 - 40, 5x^5 = 160`, hay `x^5 = 160 : 5, x^5 = 32`. 
    Ví dụ sau là lời giải tốt: Giải phương trình 5x^5 + 40 = 200. Ta có: 5x^5 = 160, x^5 = 32, x = 2
    Chuỗi tường minh là:
    I. 5x^5 = 160
    II. x^5 = 32
    III. x = 2
    - Dùng `rejected` nếu định vị sai, cầu nối đã xuất hiện qua các phương thức khác như lập luận,... hay nội dung sau thực chất sai toán học. Khi chấp nhận, viết lại ngắn gọn đúng nội dung trước, nội dung sau và cầu nối bị thiếu. Với `redundant_step`, dùng `advisory`. Chỉ kiểm chứng nội dung nháp hoặc thử-sai khi comment Process nêu đúng loại đó.

5. `review_reason` là kết luận cuối, không phải bản ghi quá trình cân nhắc. Mọi khẳng định phải khớp evidence, không tự mâu thuẫn và không thêm lỗi mới. Nếu comment chứa nhiều cáo buộc, chỉ trình bày lỗi chắc chắn sớm nhất. Với `blocking` hoặc `advisory`, viết một `review_suggestion` trực tiếp sửa đúng lỗi đó; với `rejected`, trả `review_suggestion=""`. Giữ nguyên giá trị đúng từ candidate; viết phép nhân bằng ký tự `×`, không dùng dấu chấm, chữ `x` hoặc lệnh LaTeX. Số thập phân dùng dấu phẩy. Nếu ký hiệu nguồn còn nhiều cách hiểu, chỉ nêu phần sai chắc chắn chung cho mọi cách hiểu, không tự chọn hoặc liệt kê nhiều đáp án. Không nhắc tên các thành phần pipeline. `review_reason` tối đa hai câu và 400 ký tự; `review_suggestion` tối đa một câu và 300 ký tự.

6. Không trả verdict tổng hợp cho solution."""


def _candidate_comments(
    correctness_result: dict[str, Any],
    process_result: dict[str, Any],
) -> list[dict[str, Any]]:
    comments: list[dict[str, Any]] = []
    opening_checks_by_solution: dict[int, list[dict[str, str]]] = {}
    for check in correctness_result.get("opening_checks") or []:
        solution_index = int(check["solution_index"])
        opening_checks_by_solution.setdefault(solution_index, []).append({
            "evidence": str(check.get("evidence") or ""),
            "analysis": str(check.get("analysis") or ""),
        })
    for position, comment in enumerate(correctness_result.get("comments") or []):
        comments.append({
            "comment_id": f"correctness-{position}",
            "source": "correctness",
            "solution_index": comment["solution_index"],
            "error_type": None,
            "evidence": comment["evidence"],
            "reason": comment["reason"],
            "independent_checks": opening_checks_by_solution.get(
                int(comment["solution_index"]),
                [],
            ),
            "suggestion": "Kiểm tra và sửa nhận xét toán học đã được xác nhận.",
        })
    for position, comment in enumerate(process_result.get("comments") or []):
        comments.append({
            "comment_id": f"process-{position}",
            "source": "process_presentation",
            **comment,
        })
    return comments


def build_judge_comments_review_messages(
    generated_question: dict[str, Any],
    candidate_comments: list[dict[str, Any]],
    contract_retry_error: str | None = None,
    contract_retry_candidate: str | None = None,
) -> list[dict[str, Any]]:
    payload = _direct_solution_payload(generated_question)
    retry = ""
    if contract_retry_error:
        retry = (
            "\n\n### SỬA HỢP ĐỒNG ĐẦU RA\n"
            f"Lỗi hợp đồng: {contract_retry_error}\n"
            f"Kết quả thô: {_prompt_json(_contract_retry_candidate_payload(contract_retry_candidate))}\n"
            "Chỉ sửa cấu trúc JSON; không thay đổi nội dung kiểm chứng."
        )
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Bạn là giáo viên toán chịu trách nhiệm kiểm chứng các nhận xét đã có về lời giải học sinh. "
                "Đối chiếu từng nhận xét theo các quy tắc được cung cấp. Chỉ trả JSON đúng schema."
            ),
        },
        {
            "role": "user",
            "content": (
                f"### QUY TẮC KIỂM CHỨNG\n{COMMENTS_REVIEW_CRITERIA}\n\n"
                f"### NHẬN XÉT CẦN KIỂM CHỨNG\n{_prompt_json(candidate_comments)}\n\n"
                f"### BẰNG CHỨNG ĐỐI CHIẾU\n{_prompt_json(payload)}\n\n"
                "### ĐẦU RA\n"
                "reviewed_comments phải chứa đúng một phần tử cho mỗi comment_id đầu vào, không thiếu và không lặp. "
                "Không được trả nhận xét cho nội dung không có comment_id đầu vào.\n"
                f"{_prompt_json(JudgeCommentsReviewOutput.model_json_schema())}\n\n"
                f"{JSON_SERIALIZATION_CONSTRAINTS}{retry}"
            ),
        },
    ]
    return attach_remote_visual_assets(messages, payload)


def validate_judge_comments_review_output(
    parsed: Any,
    generated_question: dict[str, Any],
    candidate_comments: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    try:
        review = JudgeCommentsReviewOutput.model_validate(parsed).model_dump()
    except ValidationError as exc:
        return None, "mechanical: " + validation_error_text(exc)
    expected_ids = [comment["comment_id"] for comment in candidate_comments]
    candidates_by_id = {
        comment["comment_id"]: comment for comment in candidate_comments
    }
    actual_ids = [comment["comment_id"] for comment in review["reviewed_comments"]]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
        return None, "mechanical: reviewed_comments phải chứa đúng một lần mỗi comment_id đầu vào."
    for comment in review["reviewed_comments"]:
        if not comment["review_reason"].strip():
            return None, "mechanical: mỗi nhận xét đã duyệt phải có review_reason."
        if comment["disposition"] in {"blocking", "advisory"} and not comment["review_suggestion"].strip():
            return None, "mechanical: nhận xét được chấp nhận phải có review_suggestion."
        if comment["disposition"] == "rejected":
            comment["review_suggestion"] = ""
        candidate = candidates_by_id[comment["comment_id"]]
        if (
            candidate.get("error_type") == "missing_major_step"
            and comment["disposition"] == "advisory"
        ):
            comment["disposition"] = "rejected"
            comment["review_suggestion"] = ""
        if candidate.get("source") == "correctness" and comment["disposition"] == "advisory":
            comment["disposition"] = "blocking"
    return review, ""


def aggregate_reviewed_comments(
    generated_question: dict[str, Any],
    candidate_comments: list[dict[str, Any]],
    review_result: dict[str, Any],
    *,
    strict_mode: bool,
    index: int,
) -> dict[str, Any]:
    reviewed_by_id = {
        item["comment_id"]: item
        for item in review_result.get("reviewed_comments") or []
    }
    accepted = [
        {
            **comment,
            "disposition": reviewed_by_id[comment["comment_id"]]["disposition"],
            "reason": reviewed_by_id[comment["comment_id"]]["review_reason"],
            "suggestion": reviewed_by_id[comment["comment_id"]]["review_suggestion"],
        }
        for comment in candidate_comments
        if reviewed_by_id.get(comment["comment_id"], {}).get("disposition")
        in {"blocking", "advisory"}
    ]
    blocking = [comment for comment in accepted if comment["disposition"] == "blocking"]
    selectable = blocking or accepted
    if not selectable:
        payload = {"is_good": True, "issues": []}
    else:
        solution_texts = {
            int(solution["index"]): "\n".join(
                str(block.get("text") or "") for block in solution.get("textBlocks") or []
            )
            for solution in _direct_solution_payload(generated_question).get("solutions") or []
        }
        selected = min(
            selectable,
            key=lambda comment: (
                0 if comment["source"] == "correctness" else 1,
                int(comment["solution_index"]),
                _normalized_evidence_position(
                    str(comment.get("evidence") or ""),
                    solution_texts.get(int(comment["solution_index"]), ""),
                )
                if _normalized_evidence_position(
                    str(comment.get("evidence") or ""),
                    solution_texts.get(int(comment["solution_index"]), ""),
                ) >= 0
                else 10**9,
            ),
        )
        advisory = selected["disposition"] == "advisory"
        payload = {
            "is_good": False,
            "issues": [{
                "severity": "warning" if advisory else "needs_review",
                "category": "solution_quality",
                "location": f"/solutions/{int(selected['solution_index'])}",
                "reason": selected["reason"],
                "suggestion": selected["suggestion"],
                "repair_intent": "clean_solution_reasoning" if advisory else "needs_manual_review",
            }],
        }
    return normalize_generated_question_result(
        payload,
        strict_mode=strict_mode,
        generated_question=generated_question,
        index=index,
    )


def _judge_generated_question_direct(
    generated_question: dict[str, Any],
    schema_validation_result: dict[str, Any],
    config: AppConfig,
    client: LLMClient,
    *,
    strict_mode: bool,
    index: int,
    debug: bool,
) -> dict[str, Any]:
    correctness_model = generated_question_correctness_model(config)
    presentation_model = generated_question_presentation_model(config)

    def call_correctness(error: str | None = None, candidate: str | None = None) -> dict[str, Any]:
        return _call_structured_judge(
            generated_question=generated_question,
            schema_validation_result=schema_validation_result,
            client=client,
            model=correctness_model,
            debug=debug,
            step="solution_correctness_judge",
            messages=build_direct_correctness_messages(
                generated_question,
                error,
                candidate,
            ),
            validator=lambda parsed: validate_direct_correctness_output(parsed, generated_question),
            invalid_json_message="Gemma Correctness không trả JSON hợp lệ.",
            response_format=direct_correctness_response_format(),
            max_tokens=3072,
            retry_transient_once=True,
        )

    def call_process(error: str | None = None, candidate: str | None = None) -> dict[str, Any]:
        return _call_structured_judge(
            generated_question=generated_question,
            schema_validation_result=schema_validation_result,
            client=client,
            model=presentation_model,
            debug=debug,
            step="solution_process_presentation_judge",
            messages=build_direct_process_presentation_messages(
                generated_question,
                error,
                candidate,
            ),
            validator=lambda parsed: validate_direct_process_output(parsed, generated_question),
            invalid_json_message="Gemma Process & Presentation không trả JSON hợp lệ.",
            response_format=process_presentation_response_format(),
            max_tokens=3072,
            retry_transient_once=True,
        )

    def call_review(
        candidate_comments: list[dict[str, Any]],
        error: str | None = None,
        candidate: str | None = None,
    ) -> dict[str, Any]:
        return _call_structured_judge(
            generated_question=generated_question,
            schema_validation_result=schema_validation_result,
            client=client,
            model=correctness_model,
            debug=debug,
            step="solution_judge_comments_review",
            messages=build_judge_comments_review_messages(
                generated_question,
                candidate_comments,
                error,
                candidate,
            ),
            validator=lambda parsed: validate_judge_comments_review_output(
                parsed,
                generated_question,
                candidate_comments,
            ),
            invalid_json_message="Judge kiểm chứng nhận xét không trả JSON hợp lệ.",
            response_format=judge_comments_review_response_format(),
            max_tokens=3072,
            retry_transient_once=True,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        correctness_future = executor.submit(call_correctness)
        process_future = executor.submit(call_process)
        correctness_result = correctness_future.result()
        process_result = process_future.result()

    correctness_correction_calls = 0
    process_correction_calls = 0
    review_correction_calls = 0
    review_result: dict[str, Any] | None = None
    candidate_comments: list[dict[str, Any]] = []
    if correctness_result.get("contract_valid") is not True and correctness_result.get("failure_kind") != "runtime":
        correctness_correction_calls += 1
        correctness_result = call_correctness(
            str(correctness_result.get("contract_error") or "Output không canonical."),
            str(correctness_result.get("raw_candidate") or ""),
        )
    if process_result.get("contract_valid") is not True and process_result.get("failure_kind") != "runtime":
        process_correction_calls += 1
        process_result = call_process(
            str(process_result.get("contract_error") or "Output không canonical."),
            str(process_result.get("raw_candidate") or ""),
        )

    if correctness_result.get("contract_valid") is True and process_result.get("contract_valid") is True:
        candidate_comments = _candidate_comments(correctness_result, process_result)
        if candidate_comments:
            review_result = call_review(candidate_comments)
            if review_result.get("contract_valid") is not True and review_result.get("failure_kind") != "runtime":
                review_correction_calls += 1
                review_result = call_review(
                    candidate_comments,
                    str(review_result.get("contract_error") or "Output không canonical."),
                    str(review_result.get("raw_candidate") or ""),
                )
        else:
            review_result = {
                "contract_valid": True,
                "reviewed_comments": [],
                "skipped": True,
                "skip_reason": "Hai Judge đều chốt đúng; không có nhận xét sai cần kiểm chứng.",
            }

    if (
        correctness_result.get("contract_valid") is not True
        or process_result.get("contract_valid") is not True
        or not isinstance(review_result, dict)
        or review_result.get("contract_valid") is not True
    ):
        invalid = []
        if correctness_result.get("contract_valid") is not True:
            invalid.append("correctness: " + str(correctness_result.get("contract_error") or "output không canonical"))
        if process_result.get("contract_valid") is not True:
            invalid.append("process_presentation: " + str(process_result.get("contract_error") or "output không canonical"))
        if review_result is not None and (
            not isinstance(review_result, dict)
            or review_result.get("contract_valid") is not True
        ):
            invalid.append(
                "comments_review: "
                + str((review_result or {}).get("contract_error") or "output không canonical")
            )
        detail = "Không aggregate semantic vì Judge handoff invalid: " + " | ".join(invalid)
        runtime_failure = (
            correctness_result.get("failure_kind") == "runtime"
            or process_result.get("failure_kind") == "runtime"
            or (review_result or {}).get("failure_kind") == "runtime"
        )
        result = (
            _judge_runtime_needs_review_output(
                detail,
                generated_question=generated_question,
                strict_mode=strict_mode,
                index=index,
            )
            if runtime_failure
            else _judge_contract_needs_review_output(
                detail,
                generated_question=generated_question,
                strict_mode=strict_mode,
                index=index,
            )
        )
    else:
        result = aggregate_reviewed_comments(
            generated_question,
            candidate_comments,
            review_result,
            strict_mode=strict_mode,
            index=index,
        )

    trace_pipeline_event(
        stage="aggregate",
        event="output",
        payload={
            "correctness": correctness_result,
            "process_presentation": process_result,
            "candidate_comments": candidate_comments,
            "comments_review": review_result,
            "aggregate": result,
        },
        object_id=generated_question.get("id") or generated_question.get("_id"),
    )
    correction_calls = correctness_correction_calls + process_correction_calls + review_correction_calls
    review_calls = int(
        review_result is not None and not review_result.get("skipped")
    )
    reviewed_by_id = {
        item["comment_id"]: item
        for item in (review_result or {}).get("reviewed_comments") or []
    }

    def comments_with_review(
        comments: list[dict[str, Any]],
        source: str,
    ) -> list[dict[str, Any]]:
        annotated: list[dict[str, Any]] = []
        for position, comment in enumerate(comments):
            item = dict(comment)
            reviewed = reviewed_by_id.get(f"{source}-{position}")
            if reviewed:
                item.update(
                    review_disposition=reviewed["disposition"],
                    review_reason=reviewed["review_reason"],
                    review_suggestion=reviewed["review_suggestion"],
                )
            annotated.append(item)
        return annotated

    dedicated_correctness = bool(getattr(config, "solution_correctness_model", None))
    result.update(
        correctness_comments=comments_with_review(
            list(correctness_result.get("comments") or []),
            "correctness",
        ),
        process_comments=comments_with_review(
            list(process_result.get("comments") or []),
            "process",
        ),
        judge_model=correctness_model,
        correctness_primary_model=correctness_model,
        process_presentation_model=presentation_model,
        final_correctness_model=correctness_model,
        judge_attempt_count=2 + review_calls + correction_calls,
        judge_fallback_called=False,
        judge_gemma_run_count=2 + review_calls,
        judge_fallback_reason=None,
        correctness_fallback_called=False,
        correctness_primary_26b_calls=1 + review_calls,
        correctness_contract_correction_26b_calls=(
            correctness_correction_calls + review_correction_calls if dedicated_correctness else 0
        ),
        correctness_rejudge_26b_calls=0,
        primary_12b_calls=1 + (2 * int(not dedicated_correctness)),
        contract_correction_12b_calls=(
            process_correction_calls
            + ((correctness_correction_calls + review_correction_calls) if not dedicated_correctness else 0)
        ),
        fallback_26b_calls=0,
        fallback_reason_contract=0,
        fallback_reason_resolver_mismatch=0,
        http_retry_count=int(correctness_result.get("http_retry_count") or 0)
        + int(process_result.get("http_retry_count") or 0)
        + int((review_result or {}).get("http_retry_count") or 0),
        non_canonical_count=correction_calls,
        runtime_count=int(correctness_result.get("failure_kind") == "runtime")
        + int(process_result.get("failure_kind") == "runtime")
        + int((review_result or {}).get("failure_kind") == "runtime"),
        correctness_error_type_normalized_count=0,
    )
    return result


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
    """Run both Judges directly on the original solution text."""

    return _judge_generated_question_direct(
        generated_question,
        schema_validation_result,
        config,
        client,
        strict_mode=strict_mode,
        index=index,
        debug=debug,
    )

def _judge_contract_needs_review_output(
    contract_detail: str,
    *,
    generated_question: dict[str, Any],
    strict_mode: bool,
    index: int,
) -> dict[str, Any]:
    """Return a canonical terminal blocker when Judge contract retries are exhausted."""

    reason = (
        "Không thể hoàn tất đánh giá lời giải vì ít nhất một Judge không trả "
        "kết quả canonical."
    )
    suggestion = (
        "Giữ nguyên generated question và chuyển lời giải sang review thủ công; "
        "không chạy Resolver hoặc tự động sửa answerSpec."
    )
    return normalize_generated_question_result(
        {
            "id": generated_question.get("id")
            or generated_question.get("_id")
            or f"item[{index}]",
            "is_good": False,
            "failed_reason": [contract_detail],
            "suggestions": [suggestion],
            "issues": [
                {
                    "severity": "needs_review",
                    "category": "solution_quality",
                    "location": "/solutions",
                    "reason": reason,
                    "suggestion": suggestion,
                    "repair_intent": "needs_manual_review",
                }
            ],
        },
        strict_mode=strict_mode,
        generated_question=generated_question,
        index=index,
    )


def _judge_runtime_needs_review_output(
    runtime_detail: str,
    *,
    generated_question: dict[str, Any],
    strict_mode: bool,
    index: int,
) -> dict[str, Any]:
    """Keep exhausted model/API exceptions distinguishable from contract errors."""

    return normalize_generated_question_result(
        fail_closed_output(
            runtime_detail,
            generated_question=generated_question,
            index=index,
        ),
        strict_mode=strict_mode,
        generated_question=generated_question,
        index=index,
    )

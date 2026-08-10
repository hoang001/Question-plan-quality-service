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
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from ..infra.config import (
    AppConfig,
    generated_question_correctness_model,
    generated_question_presentation_model,
    generated_question_splitter_model,
)
from ..infra.debug import debug_llm_messages, llm_prompt_debug_enabled
from ..infra.llm_client import LLMClient
from ..shared.utils import parse_json_output
from ..vision.visual_context import attach_remote_visual_assets, extract_asset_references
from ..schemas.generated_question_contracts import (
    CanonicalCorrectnessResult,
    CombinedJudgeCorrectionOutput,
    CorrectnessJudgeOutput,
    ProcessPresentationJudgeOutput,
    ProcessPresentationSemanticOutput,
    SolutionSplitOutput,
    contract_schema_text,
    validation_error_text,
)
from .generated_question_schema import (
    fail_closed_output,
    normalize_generated_question_result,
)
from .code_transition_analyzer import analyze_code_transition, analyze_transition_stages


KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"
CRITERIA_PATH = KNOWLEDGE_DIR / "generated_question_quality_criteria.md"
PROCESS_PRESENTATION_CRITERIA_PATH = KNOWLEDGE_DIR / "generated_question_process_presentation_criteria.md"
MARKDOWN_IMAGE_PATTERN = re.compile(
    r"!\[(?P<description>.*?)\]\((?P<asset_url>https?://[^)\s]+)\)",
    re.DOTALL,
)
JSON_SERIALIZATION_CONSTRAINTS = """### QUY TẮC GHI JSON
- Chỉ trả một JSON object hợp lệ, không dùng markdown hoặc code fence.
- Phải escape dấu gạch chéo ngược `\\` bên trong JSON string thành `\\\\`.
- Các biểu thức LaTeX như `\\sqrt`, `\\frac`, `\\pi` phải được ghi thành `\\\\sqrt`, `\\\\frac`, `\\\\pi`.
- Không viết nội dung bên ngoài JSON object."""
CORRECTNESS_OUTPUT_INVARIANTS = """### ĐIỀU KIỆN BẤT BIẾN CỦA ĐẦU RA
- status="good": error_type, solution_index, from_order, to_order, reason, suggestion đều là JSON null.
- status="bad": error_type, solution_index, to_order, reason, suggestion bắt buộc; from_order chỉ null cho initial_transition.
- status="uncertain": reason bắt buộc; các field lỗi/anchor khác có thể null.
- Không có `verified_invalid + hard`: semantic_role, certificate_disposition, role_evidence đều null.
- Có `verified_invalid + hard`: semantic_role và certificate_disposition bắt buộc mô tả certificate sớm nhất.
- asserted_active hoặc uncertain dùng certificate_disposition="accept".
- hypothetical, rejected hoặc self_corrected dùng certificate_disposition="ignore" và role_evidence nguyên văn bắt buộc.
- Chỉ trả quyết định semantic; không trả location, source_path, evidence_text, before, after hoặc code_analysis review.
- Không dùng chuỗi "None", "null" hoặc "N/A" thay cho JSON null."""
PROCESS_PRESENTATION_OUTPUT_INVARIANTS = """### ĐIỀU KIỆN BẤT BIẾN CỦA ĐẦU RA
- Nếu verdict="good" và không có advisory: error_type=null, solution_index=null, state_order=null, reason="", suggestion="".
- Chỉ `redundant_step` được phép đi cùng verdict="good": phải có solution_index, state_order, reason và suggestion để code ghi nhận advisory nội bộ nhưng không làm lời giải thành bad.
- Nếu verdict="bad" hoặc verdict="uncertain": error_type, solution_index, reason và suggestion bắt buộc; state_order=null chỉ khi lỗi áp dụng cho toàn solution.
- Không trả source_path, evidence_text hoặc scope; code tự dựng các metadata này từ anchor đã validate.
- Không dùng chuỗi "None", "null" hoặc "N/A" thay cho giá trị JSON; dùng JSON null hoặc "" đúng theo schema."""
FIRST_TRANSITION_RULE = """### QUY TẮC BƯỚC CHUYỂN ĐẦU TIÊN
- Bắt buộc đọc `initial_transition` trước các transition `loi_giai`.
- Nếu lỗi đầu tiên ở initial transition, trả `from_order=null` và `to_order` của first state.
- Evidence phải nằm nguyên văn trong first state.
- Không dùng state phía sau để hợp thức hóa lỗi đứng trước.
- `code_analysis` là certificate toán học; chỉ xác định vai trò ngữ nghĩa của bước bị cảnh báo, không phản biện lại phép toán."""
CODE_SPLITTER_MAX_TOKENS = 160
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_BOUNDARY_PATTERNS = (
    re.compile(r"\\Rightarrow|\\implies|⇒|=>"),
    re.compile(r"\r?\n+"),
    re.compile(r"[.!?;:,]+(?:[\"'»”)\]]*)\s+"),
    re.compile(r"\b(?:Suy ra|Do đó|Vậy|Khi đó|Ta có|Tiếp theo|Mặt khác)\b", re.IGNORECASE),
)
_SPLITTER_DIVISION_COLON = re.compile(
    r"(?<=[0-9)\]}])\s*:\s*(?=[+\-]?\s*(?:\d|[({\[]|\\(?:d?frac|sqrt)))"
    r"|(?<=[A-Za-z])\s*:\s*(?=[+\-]?\s*(?:\d|[({\[]))"
    r"|(?<=[)\]}])\s*:\s*(?=[A-Za-z])"
)
_ASSIGNMENT_BEFORE_COLON = re.compile(
    r"[A-Za-z_][A-Za-z_0-9]*\s*=\s*[+\-]?\s*(?:\d+(?:[.,]\d+)?|\.\d+)\s*$"
)
_NUMBERED_LABEL_BEFORE_COLON = re.compile(
    r"\b(?:bước|câu|ý|lần|ví\s+dụ)\s+\d+\s*$",
    re.IGNORECASE,
)


def _normalize_splitter_division(text: str) -> str:
    """Chuẩn hóa dấu chia trong bản phân tích, không sửa văn bản nguồn."""

    source = str(text or "")

    def replace_colon(match: re.Match[str]) -> str:
        prefix = source[: match.start()]
        if _ASSIGNMENT_BEFORE_COLON.search(prefix):
            return match.group(0)
        if _NUMBERED_LABEL_BEFORE_COLON.search(prefix):
            return match.group(0)
        return "/"

    return _SPLITTER_DIVISION_COLON.sub(replace_colon, source)


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


def _normalized_evidence_in_source(evidence: str, source: str) -> bool:
    return _normalized_evidence_position(evidence, source) >= 0


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _prompt_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def correctness_response_format() -> dict[str, Any]:
    """OpenAI-compatible strict schema verified against the configured backend."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "solution_correctness_decision",
            "strict": True,
            "schema": CorrectnessJudgeOutput.model_json_schema(),
        },
    }


def process_presentation_response_format(
    ordered_solution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strict flat envelope; presentation policy remains entirely prompt-driven."""

    schema = ProcessPresentationSemanticOutput.model_json_schema()
    if ordered_solution is not None:
        states = ordered_solution.get("states") or []
        solution_indexes = sorted({int(state["solution_index"]) for state in states})
        state_orders = sorted({int(state["order"]) for state in states})
        if solution_indexes:
            schema["properties"]["solution_index"]["anyOf"][0]["enum"] = solution_indexes
        if state_orders:
            schema["properties"]["state_order"]["anyOf"][0]["enum"] = state_orders
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "solution_process_presentation_decision",
            "strict": True,
            "schema": schema,
        },
    }


def combined_correction_response_format(
    ordered_solution: dict[str, Any],
) -> dict[str, Any]:
    """Strict flat schema for the rare case where both parallel outputs fail."""

    schema = CombinedJudgeCorrectionOutput.model_json_schema()
    states = ordered_solution.get("states") or []
    solution_indexes = sorted({int(state["solution_index"]) for state in states})
    state_orders = sorted({int(state["order"]) for state in states})
    if solution_indexes:
        for field_name in (
            "correctness_solution_index",
            "process_solution_index",
        ):
            schema["properties"][field_name]["anyOf"][0]["enum"] = solution_indexes
    if state_orders:
        for field_name in (
            "correctness_from_order",
            "correctness_to_order",
            "process_state_order",
        ):
            schema["properties"][field_name]["anyOf"][0]["enum"] = state_orders
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "combined_solution_judge_contract_correction",
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
                "path": f"/solutions/{solution_index}",
                "textBlocks": text_blocks,
            }
        )

    payload = {
        "id": generated_question.get("id") or generated_question.get("_id") or "",
        "questionItems": question_items,
        "solutions": solutions,
    }
    if generated_question.get("instruction"):
        payload["instruction"] = generated_question["instruction"]
    return payload


def _context_text_sources(payload: dict[str, Any]) -> dict[str, str]:
    sources = {
        f"/instruction/{index}/text": str(block["text"])
        for index, block in enumerate(payload.get("instruction") or [])
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    }
    for item in payload.get("questionItems") or []:
        if not isinstance(item, dict):
            continue
        item_index = int(item.get("index") or 0)
        for block_index, block in enumerate(item.get("stem") or []):
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                sources[f"/questionItems/{item_index}/stem/{block_index}/text"] = str(block["text"])
    return sources


def validate_splitter_output(
    parsed: Any,
    generated_question: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    try:
        split = SolutionSplitOutput.model_validate(parsed).model_dump()
    except ValidationError as exc:
        return None, validation_error_text(exc)

    payload = compact_generated_question_payload(generated_question)
    question_context = {
        "id": payload.get("id"),
        "instruction": payload.get("instruction") or [],
        "questionItems": payload.get("questionItems") or [],
    }
    context_sources = _context_text_sources(payload)
    stem = split["stem"]
    if stem["source_path"] not in context_sources:
        return None, "stem.source_path does not exist in instruction or questionItems[].stem."
    if stem["source_text"] != context_sources[stem["source_path"]]:
        return None, "stem.source_text does not exactly match the question context."
    expected_visuals = {
        (reference.source_path, reference.source)
        for reference in extract_asset_references(question_context)
        if reference.source.startswith(("http://", "https://"))
    }
    described_visuals: set[tuple[str, str]] = set()
    for visual in split.get("visual_descriptions") or []:
        key = (str(visual["source_path"]), str(visual["asset_url"]))
        if key not in expected_visuals:
            return None, "visual_descriptions không trỏ tới ảnh trong question context."
        if key in described_visuals:
            return None, "visual_descriptions chứa ảnh trùng lặp."
        described_visuals.add(key)
    if described_visuals != expected_visuals:
        return None, (
            "Splitter phải mô tả đúng một lần mọi image_url được đính kèm "
            "trong question context."
        )

    context_texts = list(context_sources.values())
    for requirement in split["context_requirements"]:
        if not any(
            _normalized_evidence_in_source(
                requirement["evidence_text"], source_text
            )
            for source_text in context_texts
        ):
            return None, (
                "context_requirements.evidence_text không nằm nguyên văn "
                "trong question context."
            )

    sources: dict[str, str] = {}
    path_ranks: dict[str, tuple[int, int]] = {}
    for solution in payload.get("solutions") or []:
        solution_index = int(solution["index"])
        for block_rank, block in enumerate(solution.get("textBlocks") or []):
            path = str(block["path"])
            sources[path] = str(block["text"])
            path_ranks[path] = (solution_index, block_rank)

    states_by_path: dict[str, list[str]] = {}
    states_by_solution: dict[int, list[dict[str, Any]]] = {}
    for state in split["states"]:
        path = str(state["source_path"])
        if path not in sources:
            return None, f"source_path không tồn tại: {path}"
        expected_solution_index, _block_rank = path_ranks[path]
        if int(state["solution_index"]) != expected_solution_index:
            return None, f"solution_index không khớp source_path: {path}"
        if not state["source_text"]:
            return None, f"Splitter trả state rỗng tại {path}"
        states_by_path.setdefault(path, []).append(state["source_text"])
        states_by_solution.setdefault(expected_solution_index, []).append(state)

    if set(states_by_path) != set(sources):
        return None, "Splitter đã bỏ qua một hoặc nhiều solution block."
    for path, source in sources.items():
        if "".join(states_by_path[path]) != source:
            return None, f"Splitter đã thêm, bớt hoặc thay đổi ký tự tại {path}"

    for solution_index, states in sorted(states_by_solution.items()):
        if [state["order"] for state in states] != list(range(len(states))):
            return None, f"order của solution {solution_index} không liên tục hoặc bị trùng."
        ranks = [path_ranks[state["source_path"]][1] for state in states]
        if ranks != sorted(ranks):
            return None, f"Splitter đã thay đổi thứ tự solution block của solution {solution_index}."
    for state in split["states"]:
        state["_math_text"] = _normalize_splitter_division(state["source_text"])
    return split, ""


def _split_text_by_token_window(text: str, max_tokens: int) -> list[str]:
    """Cắt gần giới hạn token tại ranh giới lập luận gần nhất và giữ nguyên ký tự."""

    tokens = list(_TOKEN_PATTERN.finditer(text))
    if len(tokens) <= max_tokens:
        return [text]
    boundaries = sorted({
        match.end() if pattern is not _BOUNDARY_PATTERNS[-1] else match.start()
        for pattern in _BOUNDARY_PATTERNS
        for match in pattern.finditer(text)
        if 0 < (match.end() if pattern is not _BOUNDARY_PATTERNS[-1] else match.start()) < len(text)
    })
    parts: list[str] = []
    token_start = 0
    char_start = 0
    while len(tokens) - token_start > max_tokens:
        token_end = token_start + max_tokens
        hard_end = tokens[token_end - 1].end()
        soft_start = tokens[token_start + max(1, max_tokens // 2) - 1].end()
        nearby = [position for position in boundaries if soft_start <= position <= hard_end]
        earlier = [position for position in boundaries if char_start < position <= hard_end]
        char_end = max(nearby or earlier, default=hard_end)
        parts.append(text[char_start:char_end])
        char_start = char_end
        while token_start < len(tokens) and tokens[token_start].end() <= char_start:
            token_start += 1
    if char_start < len(text):
        parts.append(text[char_start:])
    return [part for part in parts if part]


def split_solution_with_code(
    generated_question: dict[str, Any],
    *,
    max_tokens: int = CODE_SPLITTER_MAX_TOKENS,
) -> tuple[dict[str, Any] | None, str]:
    """Fallback deterministic: cắt text gốc theo cửa sổ token, không sửa nội dung."""

    if max_tokens < 2:
        return None, "Giới hạn token của code splitter phải lớn hơn 1."
    payload = compact_generated_question_payload(generated_question)
    states: list[dict[str, Any]] = []
    for solution in payload.get("solutions") or []:
        order = 0
        for block in solution.get("textBlocks") or []:
            for source_text in _split_text_by_token_window(str(block["text"]), max_tokens):
                states.append({
                    "solution_index": int(solution["index"]),
                    "order": order,
                    "source_path": str(block["path"]),
                    "source_text": source_text,
                })
                order += 1
    if not states:
        return None, "Code splitter không tìm thấy solution text."
    stem = _problem_anchor(payload)
    if stem is None:
        return None, "Code splitter không tìm thấy instruction hoặc question stem."
    stem["source_text"] = _context_text_sources(payload)[stem["source_path"]]
    return validate_splitter_output(
        {"context_requirements": [], "stem": stem, "states": states},
        generated_question,
    )


def extract_textual_visual_descriptions(
    question_context: dict[str, Any],
) -> list[dict[str, str]]:
    """Expose Markdown alt text separately from its non-readable asset URL."""

    sources: list[tuple[str, str]] = []
    for block_index, block in enumerate(question_context.get("instruction") or []):
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            sources.append((f"/instruction/{block_index}/text", block["text"]))
    for item in question_context.get("questionItems") or []:
        if not isinstance(item, dict):
            continue
        item_index = int(item.get("index") or 0)
        for block_index, block in enumerate(item.get("stem") or []):
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                sources.append(
                    (f"/questionItems/{item_index}/stem/{block_index}/text", block["text"])
                )

    descriptions: list[dict[str, str]] = []
    for source_path, text in sources:
        for match in MARKDOWN_IMAGE_PATTERN.finditer(text):
            description = match.group("description").strip()
            if description:
                descriptions.append(
                    {
                        "source_path": source_path,
                        "visual_description": description,
                        "asset_url": match.group("asset_url"),
                    }
                )
    return descriptions


def build_solution_splitter_messages(generated_question: dict[str, Any]) -> list[dict[str, Any]]:
    payload = compact_generated_question_payload(generated_question)
    question_context = {
        "id": payload.get("id"),
        "instruction": payload.get("instruction") or [],
        "questionItems": payload.get("questionItems") or [],
    }
    context_text_sources = _context_text_sources(payload)
    visual_descriptions = extract_textual_visual_descriptions(question_context)
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là Solution State Splitter / Context & Transition Builder. Xác định context requirements rồi chia solution text "
                "thành ordered states; không đánh giá đúng sai. "
                "Tuyệt đối không thêm, bớt, sửa, chuẩn hóa hoặc diễn giải bất kỳ chữ, khoảng trắng, xuống dòng, "
                "dấu câu, ký hiệu hay LaTeX nào. Chỉ trả một JSON object hợp lệ, không markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                "### PHÂN TÍCH YÊU CẦU NGỮ CẢNH\n"
                "Trước khi chia solution, xác định đề có phụ thuộc dữ liệu bổ sung bắt buộc để kiểm chứng lời giải "
                "hay không. Dữ liệu có thể là hình vẽ, hình học, đồ thị, biểu đồ, bảng số liệu, bảng biến thiên, "
                "trục số, hệ trục tọa độ, sơ đồ, vùng tô màu, dữ liệu trực quan hoặc dữ liệu tham chiếu khác.\n"
                "Với mỗi dữ liệu thực sự cần thiết, trả một phần tử trong context_requirements:\n"
                "- available: nội dung cần thiết đã được cung cấp đủ để kiểm chứng;\n"
                "- missing: đề tham chiếu dữ liệu cần thiết nhưng dữ liệu không xuất hiện;\n"
                "- insufficient: có tham chiếu/object như assetId, imageId, fileName, URL hoặc placeholder nhưng "
                "model không nhận được nội dung đủ để kiểm chứng.\n"
                "Ảnh trong mục ẢNH ĐƯỢC ĐÍNH KÈM TRỰC TIẾP là dữ liệu thật model phải đọc và được coi là available "
                "nếu ảnh chứa đủ dữ kiện. Không đánh dấu available chỉ vì có identifier, URL hoặc placeholder mà "
                "không có phần image_url đính kèm. Bảng đầy đủ dưới dạng text "
                "có thể là available. Markdown alt text trong cú pháp ![mô tả](URL) là dữ liệu text thực sự, không phải "
                "nội dung tải từ URL. Nếu alt text đã nêu đủ điểm, giá trị, nhãn, quan hệ hoặc xu hướng cần dùng để "
                "kiểm chứng thì phải coi dữ liệu là available (hoặc trả context_requirements=[] nếu không còn dependency); "
                "không được trả missing/insufficient chỉ vì cùng Markdown còn có URL. Chỉ trả insufficient khi mô tả text "
                "vẫn thiếu dữ kiện bắt buộc và không có image_url đính kèm có thể đọc (hoặc ảnh đính kèm thực sự không đọc được). "
                "Bài hình học đủ dữ kiện bằng text và không phụ thuộc hình minh họa nên trả "
                "context_requirements=[]. evidence_text phải là đoạn nguyên văn trong NGỮ CẢNH CÂU HỎI thể hiện "
                "sự phụ thuộc. Nếu không có dependency bổ sung bắt buộc, trả context_requirements=[].\n"
                "Không trả verdict, context_issue, is_good, error_type hoặc suggestion; không đánh giá solution.\n\n"
                "### MÔ TẢ DỮ LIỆU TRỰC QUAN\n"
                "Với mỗi phần image_url được đính kèm, bắt buộc trả đúng một phần tử visual_descriptions. "
                "Sao chép chính xác source_path và asset_url từ manifest ảnh. description phải mô tả bằng tiếng Việt "
                "những gì thực sự nhìn thấy và đủ chi tiết để Judge phía sau kiểm chứng bài: loại hình/bảng/đồ thị, "
                "mọi nhãn, hàng/cột, mốc, giá trị, dấu, chiều biến thiên, điểm và quan hệ có liên quan. "
                "Không tự giải bài, không suy đoán phần không nhìn rõ. Nếu không đọc được ảnh, đặt description đúng bằng "
                "'Không thể đọc nội dung ảnh.' và trả context requirement insufficient tương ứng; flow sẽ dừng trước Correctness. "
                "Nếu không có ảnh đính kèm, trả visual_descriptions=[].\n\n"
                "### TÁCH TRẠNG THÁI\n"
                "Chỉ trả context_requirements, visual_descriptions, stem và states. "
                "stem là đề bài hoặc dữ liệu trực tiếp mà lời giải đang giải, lấy nguyên văn từ instruction hoặc "
                "questionItems[].stem; source_path phải tồn tại và source_text phải khớp chính xác. "
                "Sao chép nguyên cặp source_path/source_text từ NGUỒN VĂN BẢN NGỮ CẢNH; source_path phải là JSON Pointer "
                "bắt đầu bằng '/instruction/' hoặc '/questionItems/', không dùng dạng questionItems[0].stem[0]. "
                "Nếu instruction chứa phương trình/dữ kiện đầy đủ còn questionItems[].stem chỉ yêu cầu nhập hoặc chọn đáp án, "
                "chọn instruction làm stem. "
                "stem không phải là một solution state. Mỗi states.source_text phải là một lát cắt nguyên văn của đúng solution source_path. "
                "Trong từng solution, order bắt đầu từ 0 và liên tục qua các text block theo thứ tự gốc. "
                "Khi ghép source_text của các state thuộc cùng source_path theo order, kết quả phải giống tuyệt đối "
                "text gốc từng ký tự. Không trả transitions và không bỏ qua solution block.\n"
                "Dấu phẩy có thể là ranh giới state khi nó phân cách các bước lập luận độc lập. Ví dụ "
                "'Ta có 2x^3 = 18, x^3 = 8, suy ra x = 2.' nên được tách thành "
                "'Ta có 2x^3 = 18,', ' x^3 = 8,', ' suy ra x = 2.'. "
                "Phải giữ nguyên dấu phẩy và khoảng trắng; không thay dấu phẩy bằng dấu chấm phẩy. "
                "Không tách dấu phẩy trong số thập phân như 3,12; tọa độ/biểu thức như A(1, 2), "
                "(x,y)=(2,3), f(x,y); hoặc khoảng như [1,5]. Chỉ tách khi ngữ nghĩa xác nhận đó là ranh giới chuyển bước.\n\n"
                f"LƯỢC ĐỒ JSON:\n{contract_schema_text(SolutionSplitOutput)}\n\n"
                f"NGỮ CẢNH CÂU HỎI:\n{json.dumps(question_context, ensure_ascii=False, indent=2)}\n\n"
                f"NGUỒN VĂN BẢN NGỮ CẢNH:\n{json.dumps(context_text_sources, ensure_ascii=False, indent=2)}\n\n"
                f"MÔ TẢ TRỰC QUAN BẰNG VĂN BẢN:\n{json.dumps(visual_descriptions, ensure_ascii=False, indent=2)}\n\n"
                f"DỮ LIỆU LỜI GIẢI:\n{json.dumps(payload.get('solutions') or [], ensure_ascii=False, indent=2)}\n\n"
                "Chỉ trả một JSON object đúng schema: không thêm field, không bỏ field bắt buộc, "
                "dùng null thay vì chuỗi \"None\", và không trả nội dung ngoài JSON."
            ),
        },
    ]
    return attach_remote_visual_assets(messages, question_context)


def _problem_anchor(payload: dict[str, Any]) -> dict[str, str] | None:
    candidates: list[dict[str, str]] = []
    for index, block in enumerate(payload.get("instruction") or []):
        if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"].strip():
            candidates.append({"source_path": f"/instruction/{index}/text", "source_text": block["text"].strip()})
    for item in payload.get("questionItems") or []:
        item_index = int(item["index"])
        for block_index, block in enumerate(item.get("stem") or []):
            if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"].strip():
                candidates.append({
                    "source_path": f"/questionItems/{item_index}/stem/{block_index}/text",
                    "source_text": block["text"].strip(),
                })
    return max(
        candidates,
        key=lambda item: (item["source_text"].count("="), len(item["source_text"])),
        default=None,
    )


def _transition_premise(
    ordered_solution: dict[str, Any],
    generated_question: dict[str, Any],
) -> dict[str, str] | None:
    payload = compact_generated_question_payload(generated_question)
    sources = _context_text_sources(payload)
    stem = ordered_solution.get("stem")
    if isinstance(stem, dict):
        source_path = stem.get("source_path")
        source_text = stem.get("source_text")
        if (
            isinstance(source_path, str)
            and isinstance(source_text, str)
            and sources.get(source_path) == source_text
        ):
            return {
                "source_path": source_path,
                "source_text": source_text,
                "_math_text": _normalize_splitter_division(source_text),
            }
    fallback = _problem_anchor(payload)
    if fallback is not None:
        fallback["_math_text"] = _normalize_splitter_division(fallback["source_text"])
    return fallback


def build_context_issue_from_requirements(
    ordered_solution: dict[str, Any],
    generated_question: dict[str, Any],
    *,
    strict_mode: bool,
    index: int,
) -> dict[str, Any] | None:
    """Convert a Builder-validated blocking requirement into one canonical issue."""

    blocking = next(
        (
            requirement
            for requirement in ordered_solution.get("context_requirements") or []
            if requirement.get("availability") in {"missing", "insufficient"}
        ),
        None,
    )
    if blocking is None:
        return None

    evidence = str(blocking["evidence_text"])
    payload = compact_generated_question_payload(generated_question)
    context_sources = [
        *[
            (f"/instruction/{block_index}/text", str(block["text"]))
            for block_index, block in enumerate(payload.get("instruction") or [])
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ],
        *[
            (
                f"/questionItems/{int(item['index'])}/stem/{block_index}/text",
                str(block["text"]),
            )
            for item in payload.get("questionItems") or []
            for block_index, block in enumerate(item.get("stem") or [])
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ],
    ]
    source_path = next(
        (
            path
            for path, source_text in context_sources
            if _normalized_evidence_in_source(evidence, source_text)
        ),
        None,
    )
    if source_path is None:
        return _judge_contract_needs_review_output(
            "Builder context requirement không còn grounding trong question context.",
            generated_question=generated_question,
            strict_mode=strict_mode,
            index=index,
        )

    description = str(blocking.get("description") or evidence).strip()
    return normalize_generated_question_result(
        {
            "is_good": False,
            "issues": [{
                "severity": "needs_review",
                "category": "solution_quality",
                "location": source_path,
                "reason": (
                    "Không đủ dữ liệu ngữ cảnh bắt buộc để kiểm chứng lời giải: "
                    + description
                ),
                "suggestion": (
                    "Cung cấp đầy đủ dữ liệu ngữ cảnh được đề bài tham chiếu "
                    "trước khi đánh giá lời giải."
                ),
                "repair_intent": "needs_manual_review",
            }],
        },
        strict_mode=strict_mode,
        generated_question=generated_question,
        index=index,
    )


_CORRECTNESS_HARD_ANALYSIS_FIELDS = (
    "status",
    "strength",
    "transition_type",
    "issue_type",
    "reason",
    "failing_pair_index",
    "failing_before",
    "failing_after",
    "expected_result",
    "actual_result",
)


def _correctness_prompt_stage(stage: dict[str, Any]) -> dict[str, Any]:
    """Expose hard deterministic math certificates to Correctness.

    Operation counts and compression annotations are intentionally withheld: they
    belong to pedagogical completeness, which is judged independently by Process.
    """

    prompt_stage = {
        name: value
        for name, value in stage.items()
        if name not in {"mandatory_code_issue", "code_analysis"}
    }
    analysis = stage.get("code_analysis")
    if (
        isinstance(analysis, dict)
        and analysis.get("status") in {"verified_valid", "verified_invalid"}
        and analysis.get("strength") == "hard"
    ):
        prompt_stage["code_analysis"] = {
            name: analysis[name]
            for name in _CORRECTNESS_HARD_ANALYSIS_FIELDS
            if analysis.get(name) is not None
        }
    return prompt_stage


def _hard_valid_process_certificates(
    transition_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Chỉ đưa metadata cần thiết cho Process, không yêu cầu model tính lại toán."""

    return [
        {
            "solution_index": int(stage["solution_index"]),
            "from_order": stage["from_order"],
            "to_order": int(stage["to_order"]),
            "status": "verified_valid",
            "strength": "hard",
        }
        for stage in (transition_payload or {}).get("stages") or []
        if isinstance(stage, dict)
        and isinstance(stage.get("code_analysis"), dict)
        and stage["code_analysis"].get("status") == "verified_valid"
        and stage["code_analysis"].get("strength") == "hard"
    ]


_PROCESS_OPERATION_LABELS = {
    "add_same_value_both_sides": "cộng cùng một giá trị vào hai vế",
    "subtract_same_value_both_sides": "trừ cùng một giá trị ở hai vế",
    "multiply_both_sides": "nhân hai vế với cùng một giá trị",
    "divide_both_sides": "chia hai vế cho cùng một giá trị",
    "take_odd_root_both_sides": "lấy căn bậc lẻ hai vế",
    "expand_power": "khai triển lũy thừa",
    "distribute_coefficient": "phân phối hệ số",
    "combine_constant_terms": "thu gọn các hạng tử hằng",
    "simplify_expression": "rút gọn biểu thức",
}


def _compressed_process_candidates(
    transition_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Đưa bằng chứng bước bị nén cho Process mà không tự kết luận thiếu bước."""

    candidates: list[dict[str, Any]] = []
    for stage in (transition_payload or {}).get("stages") or []:
        if not isinstance(stage, dict):
            continue
        analysis = stage.get("code_analysis")
        if not isinstance(analysis, dict):
            continue
        if (
            analysis.get("status") != "compressed_but_equivalent"
            or analysis.get("strength") != "soft"
        ):
            continue
        operation_types = analysis.get("operation_types") or []
        candidates.append(
            {
                "chỉ_số_lời_giải": int(stage["solution_index"]),
                "bước_trước": stage.get("from_order"),
                "bước_sau": int(stage["to_order"]),
                "số_thao_tác": analysis.get("operation_count"),
                "các_thao_tác": [
                    _PROCESS_OPERATION_LABELS.get(str(name), str(name))
                    for name in operation_types
                ],
                "có_trạng_thái_trung_gian": bool(
                    analysis.get("intermediate_states_explicit")
                ),
                "nhận_định": str(analysis.get("reason") or ""),
            }
        )
    return candidates


def _redundant_state_candidates(
    ordered_solution: dict[str, Any],
    transition_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Tìm state giữa có thể bỏ bằng một bypass hard-valid của Analyzer."""

    stages = (transition_payload or {}).get("stages") or []
    states = [state for state in ordered_solution.get("states") or [] if isinstance(state, dict)]
    candidates: list[dict[str, Any]] = []
    for current in states:
        solution_index = int(current["solution_index"])
        order = int(current["order"])
        if order <= 0:
            continue
        previous = next(
            (
                state
                for state in states
                if int(state["solution_index"]) == solution_index
                and int(state["order"]) == order - 1
            ),
            None,
        )
        following = next(
            (
                state
                for state in states
                if int(state["solution_index"]) == solution_index
                and int(state["order"]) == order + 1
            ),
            None,
        )
        incoming = next(
            (
                stage
                for stage in stages
                if int(stage["solution_index"]) == solution_index
                and stage["from_order"] == order - 1
                and int(stage["to_order"]) == order
            ),
            None,
        )
        outgoing = next(
            (
                stage
                for stage in stages
                if int(stage["solution_index"]) == solution_index
                and stage["from_order"] == order
                and int(stage["to_order"]) == order + 1
            ),
            None,
        )
        if previous is None or following is None or incoming is None or outgoing is None:
            continue
        incoming_analysis = incoming.get("code_analysis") or {}
        outgoing_analysis = outgoing.get("code_analysis") or {}
        if not (
            incoming_analysis.get("status") == "verified_valid"
            and incoming_analysis.get("strength") == "hard"
        ):
            continue
        if (
            outgoing_analysis.get("status") == "verified_invalid"
            and outgoing_analysis.get("strength") == "hard"
        ):
            continue
        bypass = analyze_code_transition(
            before=str(previous.get("_math_text", previous["source_text"])),
            after=str(following.get("_math_text", following["source_text"])),
        )
        if bypass.get("status") == "verified_valid" and bypass.get("strength") == "hard":
            candidates.append(
                {
                    "solution_index": solution_index,
                    "state_order": order,
                    "source_text": str(current["source_text"]),
                    "bypass_from_order": order - 1,
                    "bypass_to_order": order + 1,
                    "bypass_status": "verified_valid",
                    "bypass_strength": "hard",
                }
            )
    return candidates


def _hard_invalid_stages(
    transition_payload: dict[str, Any],
) -> list[tuple[int, dict[str, Any]]]:
    return [
        (stage_index, stage)
        for stage_index, stage in enumerate(transition_payload.get("stages") or [])
        if isinstance(stage, dict)
        and isinstance(stage.get("code_analysis"), dict)
        and stage["code_analysis"].get("status") == "verified_invalid"
        and stage["code_analysis"].get("strength") == "hard"
    ]


_HARD_VALID_MATH_ERROR_TYPES = {
    "calculation_error",
    "sign_error",
    "coefficient_error",
    "incorrect_transformation",
    "non_equivalent_transformation",
}
_HARD_VALID_DENIAL_PHRASES = (
    "không hợp lệ",
    "không bảo toàn",
    "phép biến đổi sai",
    "sai phép biến đổi",
    "sai về mặt số học",
    "sai về mặt đại số",
    "mathematically invalid",
    "algebraically invalid",
    "does not preserve",
    "not equivalent",
)


def _conflicts_with_hard_valid_math(
    decision: dict[str, Any],
    stage: dict[str, Any],
) -> bool:
    """Chỉ bắt conflict phủ nhận validity tại đúng transition hard-valid."""

    analysis = stage.get("code_analysis") or {}
    if not (
        analysis.get("status") == "verified_valid"
        and analysis.get("strength") == "hard"
    ):
        return False
    if decision.get("error_type") in _HARD_VALID_MATH_ERROR_TYPES:
        return True
    reason = str(decision.get("reason") or "").casefold()
    return any(phrase in reason for phrase in _HARD_VALID_DENIAL_PHRASES)


def _validate_certificate_semantics(
    decision: dict[str, Any],
    transition_payload: dict[str, Any],
    ordered_solution: dict[str, Any],
) -> str:
    hard_stages = _hard_invalid_stages(transition_payload)
    role = decision.get("semantic_role")
    disposition = decision.get("certificate_disposition")
    evidence = decision.get("role_evidence")
    if not hard_stages:
        if any(value is not None for value in (role, disposition, evidence)):
            return "mechanical: Không có hard-invalid certificate nên semantic disposition phải là null."
        return ""
    if role is None or disposition is None:
        return "mechanical: Hard-invalid certificate thiếu semantic_role hoặc certificate_disposition."

    ignored_roles = {"hypothetical", "rejected", "self_corrected"}
    expected_disposition = "ignore" if role in ignored_roles else "accept"
    if disposition != expected_disposition:
        return "mechanical: certificate_disposition không nhất quán với semantic_role."
    if role in ignored_roles and (
        not isinstance(evidence, str) or not evidence.strip()
    ):
        return "mechanical: Certificate chỉ được ignore khi có role_evidence nguyên văn."
    if isinstance(evidence, str) and evidence.strip() and not any(
        _normalized_evidence_in_source(evidence, str(state.get("source_text") or ""))
        for state in ordered_solution.get("states") or []
        if isinstance(state, dict)
    ):
        return "mechanical: role_evidence không grounded trong solution states."
    return ""


def enrich_question_stem_with_visual_descriptions(
    payload: dict[str, Any],
    ordered_solution: dict[str, Any],
) -> dict[str, Any]:
    """Append Splitter-produced visual facts to an internal question payload only."""

    enriched = deepcopy(payload)
    descriptions_by_path: dict[str, list[str]] = {}
    for visual in ordered_solution.get("visual_descriptions") or []:
        source_path = str(visual.get("source_path") or "")
        description = str(visual.get("description") or "").strip()
        if source_path and description:
            descriptions_by_path.setdefault(source_path, []).append(description)

    for visual_index, (source_path, descriptions) in enumerate(
        descriptions_by_path.items(),
        start=1,
    ):
        instruction_match = re.fullmatch(
            r"/instruction/(\d+)(?:/.*)?",
            source_path,
        )
        stem_match = re.fullmatch(
            r"/questionItems/(\d+)/stem/(\d+)(?:/.*)?",
            source_path,
        )
        block: dict[str, Any] | None = None
        container: list[Any] | None = None
        try:
            if instruction_match:
                first_item = next(
                    (
                        item
                        for item in enriched.get("questionItems") or []
                        if isinstance(item, dict) and isinstance(item.get("stem"), list)
                    ),
                    None,
                )
                if first_item is not None:
                    container = first_item["stem"]
                    block = None
                else:
                    container = enriched["instruction"]
                    block = container[int(instruction_match.group(1))]
            elif stem_match:
                container = enriched["questionItems"][int(stem_match.group(1))]["stem"]
                block = container[int(stem_match.group(2))]
        except (KeyError, IndexError, TypeError):
            block = None
            container = None
        if not isinstance(container, list):
            continue
        visual_text = "\n".join(
            f"- {description}" for description in descriptions
        )
        enriched_text = (
            "[Mô tả dữ liệu trực quan do Splitter đọc từ ảnh]\n" + visual_text
        )
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            block["text"] += "\n\n" + enriched_text
        else:
            container.append(
                {
                    "id": f"__splitter_visual_context_{visual_index}",
                    "type": "text",
                    "text": enriched_text,
                }
            )
    return enriched


def build_generated_question_judge_messages(
    generated_question: dict[str, Any],
    criteria_text: str,
    ordered_solution: dict[str, Any],
    transition_payload: dict[str, Any],
    contract_retry_error: str | None = None,
    contract_retry_candidate: str | None = None,
) -> list[dict[str, Any]]:
    """Prompt chuyên kiểm tra tính đúng đắn của đề và lời giải."""

    payload = enrich_question_stem_with_visual_descriptions(
        compact_generated_question_payload(generated_question),
        ordered_solution,
    )
    payload.pop("solutions", None)
    stages = {
        "stages": [
            _correctness_prompt_stage(stage)
            for stage in transition_payload.get("stages") or []
        ]
    }
    contract_retry_instructions = ""
    if contract_retry_error:
        contract_retry_instructions = (
            "\n\n### SỬA HỢP ĐỒNG ĐẦU RA\n"
            f"Lỗi hợp đồng cụ thể: {contract_retry_error}\n"
            f"### KẾT QUẢ THÔ CẦN SỬA\n{_prompt_json({'raw_candidate': contract_retry_candidate or ''})}\n"
            "Chỉ sửa lỗi contract của candidate và trả lại đúng 10 field. Không đổi quyết định semantic "
            "hoặc tự đánh giá lại bài toán nếu không cần để sửa anchor/invariant."
        )
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là Gemma Correctness Judge. Chỉ kiểm tra tính đúng đắn của đề và lời giải; "
                "context bắt buộc đã được code xác nhận đầy đủ trước khi gọi bạn. "
                "Không đánh giá context availability, mức độ gộp bước hoặc chất lượng trình bày. "
                "Một transition chỉ có `code_analysis` khi code phát hiện nghi vấn số học hard-invalid; "
                "certificate này quyết định tính đúng sai toán học; bạn chỉ xác định vai trò ngữ nghĩa của bước trong lời giải. "
                "Chỉ trả JSON đúng schema, không sửa dữ liệu."
            ),
        },
        {
            "role": "user",
            "content": (
                f"### TIÊU CHÍ\n{criteria_text}\n\n"
                "### HỢP ĐỒNG ĐẦU RA\n"
                "Chỉ trả 10 field semantic theo thứ tự: error_type, solution_index, from_order, to_order, reason, suggestion, "
                "semantic_role, certificate_disposition, role_evidence, status.\n\n"
                f"### NGỮ CẢNH CÂU HỎI\n{_prompt_json(payload)}\n\n"
                f"### LỜI GIẢI THEO THỨ TỰ\n{_prompt_json(stages)}\n\n"
                "### RÀNG BUỘC CUỐI\n"
                "- Chỉ trả lỗi correctness đầu tiên.\n"
                "- `verified_invalid + hard` là certificate toán học đã được code xác minh; không được kết luận phép toán đó thực ra đúng. "
                "Chỉ phân loại vai trò của certificate sớm nhất bằng semantic_role.\n"
                "- Transition có `status=verified_valid` và `strength=hard` đã được xác minh deterministic là phép biến đổi toán học hợp lệ. "
                "Không được kết luận chính transition đó sai phép tính, sai đại số hoặc không bảo toàn tương đương. "
                "Vẫn phải đánh giá các yêu cầu certificate không chứng minh như điều kiện xác định, thiếu trường hợp, tính đầy đủ hoặc mức độ liên quan.\n"
                "- asserted_active: bước đang được chấp nhận và dùng tiếp; trả certificate_disposition=accept.\n"
                "- hypothetical: bước chỉ là giả sử/thử; rejected: bước bị bác bỏ; self_corrected: lời giải tự sửa trước khi kết luận. "
                "Ba role này chỉ được trả certificate_disposition=ignore khi role_evidence là đoạn nguyên văn trong solution chứng minh cách dùng đó.\n"
                "- uncertain: không đủ bằng chứng phân loại; trả certificate_disposition=accept để code fail-closed thành needs_review.\n"
                "- Khi semantic_role=asserted_active, vẫn phải trả status=bad và đầy đủ error_type, anchor, reason, suggestion theo contract; "
                "không để các field lỗi bằng null.\n"
                "- Nếu không có `verified_invalid + hard`, trả cả ba field semantic_role, certificate_disposition, role_evidence bằng null.\n"
                "- Không trả `missing_major_step`; không đánh giá operation count, bước gộp, độ ngắn hoặc mức độ diễn giải. "
                "Những vấn đề đó thuộc Process & Presentation Judge.\n"
                "- Ưu tiên nguyên văn `bieu_thuc_sau` và các trạng thái tường minh. Nếu một biểu thức đã xuất hiện trong state, "
                "không được nói rằng biểu thức đó bị thiếu.\n"
                "- Tự đánh giá phép tính, phép biến đổi, lập luận, định lý, hình học, xác suất, điều kiện, nghiệm và trường hợp; "
                "không tự đánh giá wording hoặc độ chi tiết sư phạm.\n"
                "- Các đoạn `[Mô tả dữ liệu trực quan do Splitter đọc từ ảnh]` trong stem/instruction là bản chép nội bộ "
                "từ ảnh đã được Splitter đọc. Phải dùng phần text này như dữ liệu đề bài; không tải lại URL và không tự báo thiếu ảnh.\n"
                "- Khi trả anchor, phải sao chép đúng solution_index/from_order/to_order của một stage đã cung cấp; "
                "không tự tạo cặp order, không bỏ qua state ở giữa và chỉ dùng from_order=null cho chính initial_transition.\n"
                "- Không báo lỗi nháp, tự vấn, lặp lại, wording, ký hiệu `^` hoặc ký tự trình bày; các lỗi này không thuộc Correctness Judge.\n"
                "- Trước khi chọn trạng thái, bắt buộc tự tính lại từng phép tính và đẳng thức, kiểm tra điều kiện áp dụng của công thức hoặc định lý, "
                "rồi kiểm tra trạng thái cuối có trả lời đầy đủ yêu cầu ban đầu hay không. Không được kết luận đúng chỉ vì lời giải viết trôi chảy hoặc tự nhất quán.\n"
                "- Nếu solution khẳng định một đáp án cho đại lượng được hỏi, phải kiểm tra kết luận là chính đại lượng đó, không phải bình phương, lũy thừa, biểu thức trung gian hoặc một đại lượng biến đổi chưa được hoàn nguyên.\n"
                "- Reason và suggestion viết bằng tiếng Việt.\n"
                "- reason tối đa 3 câu; suggestion tối đa 2 câu.\n"
                "- Chỉ ghi status sau khi đã tự kiểm tra xong quyết định semantic; status phải là field cuối.\n"
                "- Chỉ trả một JSON object đúng schema; không thêm field, không bỏ field bắt buộc.\n"
                "- Dùng null thay vì chuỗi \"None\"; enum phải đúng giá trị trong schema; không markdown.\n\n"
                f"{FIRST_TRANSITION_RULE}\n\n"
                f"{CORRECTNESS_OUTPUT_INVARIANTS}\n\n"
                f"{JSON_SERIALIZATION_CONSTRAINTS}"
                f"{contract_retry_instructions}"
            ),
        },
    ]
    return messages


def build_process_presentation_judge_messages(
    generated_question: dict[str, Any],
    criteria_text: str,
    ordered_solution: dict[str, Any],
    contract_retry_error: str | None = None,
    contract_retry_candidate: str | None = None,
    transition_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Prompt chuyên kiểm tra process và chất lượng trình bày."""

    payload = compact_generated_question_payload(generated_question)
    context = {
        "id": payload.get("id"),
        "instruction": payload.get("instruction") or [],
        "questionItems": [
            {"index": item.get("index"), "id": item.get("id"), "stem": item.get("stem") or []}
            for item in payload.get("questionItems") or []
        ],
    }
    solution_states = [
        state
        for state in ordered_solution.get("states") or []
        if str(state.get("source_path") or "").startswith("/solutions/")
    ]
    solution_paths = [
        str(solution["path"])
        for solution in payload.get("solutions") or []
    ]
    hard_valid_certificates = _hard_valid_process_certificates(transition_payload)
    compressed_candidates = _compressed_process_candidates(transition_payload)
    redundant_candidates = _redundant_state_candidates(
        ordered_solution,
        transition_payload,
    )
    contract_retry_instructions = ""
    if contract_retry_error:
        contract_retry_instructions = (
            "\n\n### SỬA HỢP ĐỒNG ĐẦU RA\n"
            f"Lỗi hợp đồng cụ thể: {contract_retry_error}\n"
            f"### KẾT QUẢ THÔ CẦN SỬA\n{_prompt_json({'raw_candidate': contract_retry_candidate or ''})}\n"
            "Giữ nguyên verdict, error_type, reason và suggestion của candidate. "
            "Chỉ sửa field contract hoặc anchor để khớp đúng solution_index/state_order có trong ORDERED SOLUTION STATES. "
            "Không đánh giá lại nội dung và chỉ trả đúng 6 field."
        )
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là Gemma Process & Presentation Judge. Kiểm tra mức độ đầy đủ của một lời giải mẫu, "
                "process và chất lượng trình bày; không tính lại hay phán quyết đúng/sai toán học. "
                "Chỉ trả JSON đúng schema, không sửa dữ liệu."
            ),
        },
        {
            "role": "user",
            "content": (
                f"### QUY TẮC\n{criteria_text}\n\n"
                f"### LƯỢC ĐỒ ĐẦU RA\n{_prompt_json(ProcessPresentationSemanticOutput.model_json_schema())}\n\n"
                f"### NGỮ CẢNH CÂU HỎI\n{_prompt_json(context)}\n\n"
                f"### CÁC TRẠNG THÁI LỜI GIẢI THEO THỨ TỰ\n{_prompt_json(solution_states)}\n\n"
                f"### CHỨNG NHẬN PHÉP BIẾN ĐỔI HỢP LỆ\n{_prompt_json(hard_valid_certificates)}\n\n"
                f"### ỨNG VIÊN THIẾU BƯỚC\n{_prompt_json(compressed_candidates)}\n\n"
                f"### ỨNG VIÊN BƯỚC THỪA\n{_prompt_json(redundant_candidates)}\n\n"
                f"### ĐƯỜNG DẪN LỜI GIẢI\n{_prompt_json(solution_paths)}\n\n"
                "### RÀNG BUỘC CUỐI\n"
                "- Chỉ trả 6 field theo thứ tự: error_type, solution_index, state_order, reason, suggestion, verdict.\n"
                "- Lỗi tại state phải trả đúng solution_index và state_order; code tự dựng source_path và evidence_text.\n"
                "- Lỗi toàn bộ solution trả solution_index và state_order=null.\n"
                "- Chỉ dùng missing_major_step khi thiếu một suy luận chính làm state sau không thể kiểm chứng hoặc khôi phục trực tiếp từ các state đã viết, khiến chuỗi reasoning thực sự bị đứt; "
                "phải nêu chính xác suy luận bị thiếu và vì sao không thể suy ra state sau.\n"
                "- Viết ngắn nhưng vẫn kiểm chứng trực tiếp được là good. Không báo missing_major_step chỉ vì thiếu công thức tổng quát, câu diễn giải sư phạm, phép tính nhẩm, một bước chuyển vế/rút gọn thông thường hoặc dòng trình bày riêng.\n"
                "- Không dùng missing_major_step cho phép tính sai, thiếu nghiệm, thiếu trường hợp hoặc thiếu điều kiện toán học.\n"
                "- Đọc toàn bộ state: nếu trạng thái trung gian đã xuất hiện nguyên văn thì không được báo nó bị thiếu.\n"
                "- Với mỗi ỨNG VIÊN THIẾU BƯỚC, code đã xác nhận hai đầu tương đương nhưng cần nhiều thao tác và chưa thấy trạng thái trung gian. Đối chiếu toàn bộ lời của state đích với danh sách thao tác: nếu công thức hoặc lời giải chưa thể hiện đủ từng thao tác theo đúng thứ tự thì báo missing_major_step; nếu đã mô tả đủ thì trả good.\n"
                "- Nêu một phần danh sách thao tác vẫn là thiếu bước. Trạng thái đích và các từ nối như 'suy ra', 'nên', 'do đó' không được tính là mô tả cho thao tác còn thiếu.\n"
                "- ỨNG VIÊN THIẾU BƯỚC chỉ là bằng chứng về độ nén, không tự động là lỗi; bạn vẫn phải đọc phần trình bày để quyết định.\n"
                "- Không kiểm tra lại hay sửa kết quả toán học.\n"
                "- Certificate verified_valid với strength=hard xác nhận transition tương ứng hợp lệ toán học; không được biến redundant_step thành lỗi toán học.\n"
                "- Không báo missing_major_step cho chính transition verified_valid với strength=hard chỉ vì lời giải không viết thành lời thao tác cộng, trừ, nhân hoặc chia; biểu thức đích đã xuất hiện là trạng thái trung gian tường minh.\n"
                "- Nếu transition verified_valid với strength=hard bị nhận định là không cần thiết, không liên quan đến tiến trình hoặc có thể xóa mà lời giải vẫn nối được, bắt buộc dùng error_type=redundant_step và verdict=good; tuyệt đối không dùng missing_major_step.\n"
                "- Mỗi state trong ỨNG VIÊN BƯỚC THỪA đã có bypass verified_valid với strength=hard. Nếu state chỉ chứa phép biến đổi trung gian và không bổ sung điều kiện hay giải thích cần thiết, trả redundant_step tại đúng state đó.\n"
                "- Chỉ đánh giá bước có cần thiết, bị lặp, làm reasoning khó theo dõi hay là nội dung nháp/tự sửa hay không.\n"
                "- Một bước đúng nhưng không cần thiết có thể dùng error_type=redundant_step với verdict=good; đây là advisory nhẹ và không làm toàn bộ lời giải thành bad.\n"
                "- Chỉ dùng verdict=bad cho redundancy khi lời giải vòng lặp, mâu thuẫn, rất khó theo dõi hoặc có nhiều bước thừa làm hỏng cách trình bày reasoning.\n"
                "- verdict=good không có advisory: code bỏ qua các field nhận xét còn lại và canonical hóa thành không có issue.\n"
                "- Reason và suggestion viết bằng tiếng Việt.\n"
                "- Chỉ trả một JSON object đúng schema; không thêm field, không bỏ field bắt buộc.\n"
                "- Dùng null thay vì chuỗi \"None\"; enum phải đúng giá trị trong schema; không markdown.\n\n"
                f"{PROCESS_PRESENTATION_OUTPUT_INVARIANTS}\n\n"
                f"{JSON_SERIALIZATION_CONSTRAINTS}"
                f"{contract_retry_instructions}"
            ),
        },
    ]
    return messages


def build_combined_contract_correction_messages(
    correctness_candidate: Any,
    process_candidate: Any,
    ordered_solution: dict[str, Any],
) -> list[dict[str, str]]:
    """Repair two invalid envelopes in one call without merging Judge duties."""

    states = ordered_solution.get("states") or []
    anchors = {
        "solution_indexes": sorted({int(state["solution_index"]) for state in states}),
        "state_orders": sorted({int(state["order"]) for state in states}),
    }
    correction_input = {
        "correctness": {
            "contract_error": (
                correctness_candidate.get("contract_error")
                if isinstance(correctness_candidate, dict)
                else "Output không phải object."
            ),
            "raw_candidate": (
                correctness_candidate.get("raw_candidate")
                if isinstance(correctness_candidate, dict)
                else str(correctness_candidate or "")
            ),
        },
        "process_presentation": {
            "contract_error": (
                process_candidate.get("contract_error")
                if isinstance(process_candidate, dict)
                else "Output không phải object."
            ),
            "raw_candidate": (
                process_candidate.get("raw_candidate")
                if isinstance(process_candidate, dict)
                else str(process_candidate or "")
            ),
        },
        "canonical_anchors": anchors,
    }
    return [
        {
            "role": "system",
            "content": (
                "Bạn là bộ sửa hợp đồng JSON cho hai Judge độc lập. Chỉ sửa lỗi cấu trúc, kiểu, enum, invariant "
                "hoặc anchor được chỉ rõ; không đánh giá lại bài toán và không thay đổi quyết định semantic trong raw candidate. "
                "Chỉ trả JSON đúng strict flat schema."
            ),
        },
        {
            "role": "user",
            "content": (
                "Sửa đồng thời candidate Correctness và Process & Presentation thành 16 field phẳng có prefix. "
                "Mọi field đều bắt buộc; dùng null đúng kiểu. Chỉ dùng solution_index/state order trong canonical_anchors.\n\n"
                f"### DỮ LIỆU CẦN SỬA\n{_prompt_json(correction_input)}\n\n"
                "### ĐIỀU KIỆN BẤT BIẾN CỦA CORRECTNESS\n"
                f"{CORRECTNESS_OUTPUT_INVARIANTS}\n\n"
                "### ĐIỀU KIỆN BẤT BIẾN CỦA PROCESS & PRESENTATION\n"
                f"{PROCESS_PRESENTATION_OUTPUT_INVARIANTS}\n\n"
                f"{JSON_SERIALIZATION_CONSTRAINTS}"
            ),
        },
    ]


def build_transition_stages(
    ordered_solution: dict[str, Any],
    generated_question: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ghép các state liền kề thành stage cố định để Judge chỉ việc kiểm tra."""

    stages: list[dict[str, Any]] = []
    premise = (
        _transition_premise(ordered_solution, generated_question)
        if generated_question is not None
        else None
    )
    solution_indexes = sorted({int(state["solution_index"]) for state in ordered_solution.get("states") or []})
    for solution_index in solution_indexes:
        states = sorted(
            (
                state
                for state in ordered_solution.get("states") or []
                if int(state["solution_index"]) == solution_index
            ),
            key=lambda state: int(state["order"]),
        )
        if not states:
            continue
        if premise is not None:
            stages.append({
                "solution_index": solution_index,
                "stage": 0,
                "stage_type": "initial_transition",
                "from_order": None,
                "to_order": int(states[0]["order"]),
                "source_path": premise["source_path"],
                "bieu_thuc_truoc": premise["source_text"],
                "bieu_thuc_sau": states[0]["source_text"],
                "_math_before": premise.get("_math_text", premise["source_text"]),
                "_math_after": states[0].get("_math_text", states[0]["source_text"]),
            })
        for previous, current in zip(states, states[1:]):
            stages.append({
                "solution_index": solution_index,
                "stage": int(current["order"]),
                "stage_type": "loi_giai",
                "from_order": int(previous["order"]),
                "to_order": int(current["order"]),
                "bieu_thuc_truoc": previous["source_text"],
                "bieu_thuc_sau": current["source_text"],
                "_math_before": previous.get("_math_text", previous["source_text"]),
                "_math_after": current.get("_math_text", current["source_text"]),
            })
    return {"stages": stages}


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


def validate_transition_judge_output(
    parsed: Any,
    ordered_solution: dict[str, Any],
    generated_question: dict[str, Any] | None = None,
    transition_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Validate Correctness contract, invariants và grounding."""

    try:
        output = CanonicalCorrectnessResult.model_validate(parsed).model_dump()
    except ValidationError as exc:
        return None, validation_error_text(exc)

    verdict = output["verdict"]
    transition = output["first_invalid_transition"]
    context_issue = output["context_issue"]
    stage_payload = transition_payload or build_transition_stages(
        ordered_solution, generated_question
    )
    stages = stage_payload.get("stages") or []
    if verdict == "good":
        if transition is not None or context_issue is not None:
            return None, "Correctness good không được có issue."
        if output["reason"].strip() or output["suggestion"].strip():
            return None, "Correctness good phải có reason và suggestion rỗng."
        return output, ""
    if not output["reason"].strip():
        return None, "Correctness bad/uncertain phải có reason."
    if verdict == "bad" and not output["suggestion"].strip():
        return None, "Correctness bad phải có suggestion."
    if context_issue is not None:
        return None, "Correctness không được trả context_issue; context gate thuộc code."
    if transition is None:
        return None, "Correctness bad/uncertain phải có first_invalid_transition."

    states = {
        (int(state["solution_index"]), int(state["order"])): state
        for state in ordered_solution.get("states") or []
    }
    if transition is not None:
        solution_index = int(transition["solution_index"])
        raw_from_order = transition["from_order"]
        to_order = int(transition["to_order"])
        current = states.get((solution_index, to_order))
        evidence = transition["evidence_text"]
        if raw_from_order is None:
            solution_orders = sorted(
                order
                for current_solution_index, order in states
                if current_solution_index == solution_index
            )
            if (
                not solution_orders
                or current is None
                or to_order != solution_orders[0]
            ):
                return None, "Correctness initial transition không trỏ tới first state."
            if not evidence or not _normalized_evidence_in_source(
                evidence, current["source_text"]
            ):
                return None, "Correctness initial evidence_text không nằm nguyên văn trong first state."
        else:
            from_order = int(raw_from_order)
            previous = states.get((solution_index, from_order))
            if previous is None or current is None or to_order != from_order + 1:
                return None, "Correctness transition không trỏ tới hai state liền kề."
            if not evidence or (
                not _normalized_evidence_in_source(evidence, current["source_text"])
                and not _normalized_evidence_in_source(
                    evidence, previous["source_text"]
                )
            ):
                return None, "Correctness evidence_text không nằm nguyên văn trong transition."
        candidate_index = next(
            (
                stage_index
                for stage_index, stage in enumerate(stages)
                if int(stage["solution_index"]) == solution_index
                and stage["from_order"] == raw_from_order
                and int(stage["to_order"]) == to_order
            ),
            None,
        )
        if candidate_index is None:
            return None, "Correctness candidate không trỏ tới ordered transition đã annotate."
        analysis = stages[candidate_index].get("code_analysis") or {}
        error_type = str(transition["error_type"])
        calculation = transition.get("calculation_check")
        if (
            error_type == "calculation_error"
            and calculation
            and calculation["matches"]
            and not (
                analysis.get("status") == "verified_invalid"
                and analysis.get("strength") == "hard"
            )
        ):
            return None, "calculation_error không được có calculation_check.matches=true."
        return output, ""
    return None, "Correctness output không có transition canonical."


def validate_correctness_semantic_output(
    parsed: Any,
    ordered_solution: dict[str, Any],
    generated_question: dict[str, Any] | None = None,
    transition_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Validate the small LLM decision, then ground it entirely from Builder data."""

    error_type_normalized = False
    stage_payload = transition_payload or build_transition_stages(
        ordered_solution, generated_question
    )

    # The model's only new responsibility is semantic disposition. If it accepts
    # a hard certificate but leaves the ordinary bad envelope incomplete, keep
    # the grounded disposition and let deterministic enforcement build the issue.
    if (
        isinstance(parsed, dict)
        and _hard_invalid_stages(stage_payload)
        and parsed.get("semantic_role") in {"asserted_active", "uncertain"}
        and parsed.get("certificate_disposition") == "accept"
        and parsed.get("status") in {"bad", "uncertain"}
    ):
        bad_envelope_complete = (
            parsed.get("status") == "bad"
            and parsed.get("error_type") is not None
            and isinstance(parsed.get("solution_index"), int)
            and isinstance(parsed.get("to_order"), int)
            and bool(str(parsed.get("reason") or "").strip())
            and bool(str(parsed.get("suggestion") or "").strip())
        )
        uncertain_envelope_complete = (
            parsed.get("status") == "uncertain"
            and bool(str(parsed.get("reason") or "").strip())
        )
        if not bad_envelope_complete and not uncertain_envelope_complete:
            parsed = {
                "error_type": None,
                "solution_index": None,
                "from_order": None,
                "to_order": None,
                "reason": None,
                "suggestion": None,
                "semantic_role": parsed.get("semantic_role"),
                "certificate_disposition": parsed.get("certificate_disposition"),
                "role_evidence": parsed.get("role_evidence"),
                "status": "good",
            }

    # `status` is the sole semantic decision for good. Any stale error metadata
    # emitted before the final status is deliberately discarded, not repaired.
    if isinstance(parsed, dict) and parsed.get("status") == "good":
        parsed = {
            "error_type": None,
            "solution_index": None,
            "from_order": None,
            "to_order": None,
            "reason": None,
            "suggestion": None,
            "semantic_role": parsed.get("semantic_role"),
            "certificate_disposition": parsed.get("certificate_disposition"),
            "role_evidence": parsed.get("role_evidence"),
            "status": "good",
        }

    # Normalize only a missing label. The model must already have selected bad,
    # supplied a valid canonical anchor and written a substantive correction.
    # This never infers or changes the semantic verdict.
    if (
        isinstance(parsed, dict)
        and parsed.get("status") == "bad"
        and parsed.get("error_type") is None
    ):
        reason = str(parsed.get("reason") or "").strip()
        suggestion = str(parsed.get("suggestion") or "").strip()
        reason_lower = reason.casefold()
        suggestion_lower = suggestion.casefold()
        contrast_markers = ("nhưng", "tuy nhiên", "song", "however", "but")
        positive_only = (
            (
                any(
                    phrase in reason_lower
                    for phrase in (
                        "lời giải hoàn toàn đúng",
                        "lời giải đúng",
                        "không có lỗi",
                        "solution is correct",
                    )
                )
                and not any(marker in reason_lower for marker in contrast_markers)
            )
            or any(
                phrase in suggestion_lower
                for phrase in ("giữ nguyên", "không cần sửa", "keep unchanged")
            )
        )
        raw_solution_index = parsed.get("solution_index")
        raw_from_order = parsed.get("from_order")
        raw_to_order = parsed.get("to_order")
        typed_anchor = (
            isinstance(raw_solution_index, int)
            and not isinstance(raw_solution_index, bool)
            and (
                raw_from_order is None
                or (
                    isinstance(raw_from_order, int)
                    and not isinstance(raw_from_order, bool)
                )
            )
            and isinstance(raw_to_order, int)
            and not isinstance(raw_to_order, bool)
        )
        stage_payload = transition_payload or build_transition_stages(
            ordered_solution, generated_question
        )
        anchored = typed_anchor and any(
            int(stage["solution_index"]) == raw_solution_index
            and stage["from_order"] == raw_from_order
            and int(stage["to_order"]) == raw_to_order
            for stage in stage_payload.get("stages") or []
        )
        if reason and suggestion and anchored and not positive_only:
            parsed = {**parsed, "error_type": "other_correctness_error"}
            error_type_normalized = True

    try:
        decision = CorrectnessJudgeOutput.model_validate(parsed).model_dump()
    except ValidationError as exc:
        return None, "mechanical: " + validation_error_text(exc)

    certificate_error = _validate_certificate_semantics(
        decision,
        stage_payload,
        ordered_solution,
    )
    if certificate_error:
        return None, certificate_error
    certificate_fields = {
        "semantic_role": decision["semantic_role"],
        "certificate_disposition": decision["certificate_disposition"],
        "role_evidence": decision["role_evidence"],
    }

    status = decision["status"]
    if status == "good":
        canonical = {
            "first_invalid_transition": None,
            "context_issue": None,
            "reason": "",
            "suggestion": "",
            "verdict": "good",
        }
        validated, error = validate_transition_judge_output(
            canonical, ordered_solution, generated_question, transition_payload
        )
        if validated is not None:
            validated.update(certificate_fields)
        return validated, error

    anchor_values = (
        decision.get("solution_index"),
        decision.get("from_order"),
        decision.get("to_order"),
    )
    if status == "uncertain" and all(value is None for value in anchor_values):
        return {
            "first_invalid_transition": None,
            "context_issue": None,
            "reason": decision["reason"],
            "suggestion": decision.get("suggestion") or "",
            "verdict": "uncertain",
            **certificate_fields,
        }, ""
    if decision.get("solution_index") is None or decision.get("to_order") is None:
        return None, "mechanical: Correctness anchor chỉ được bỏ hoàn toàn cho status=uncertain."

    stage = next(
        (
            candidate
            for candidate in stage_payload.get("stages") or []
            if int(candidate["solution_index"]) == int(decision["solution_index"])
            and candidate["from_order"] == decision["from_order"]
            and int(candidate["to_order"]) == int(decision["to_order"])
        ),
        None,
    )
    if stage is None:
        return None, "mechanical: Correctness anchor không trỏ tới ordered transition canonical."

    if _conflicts_with_hard_valid_math(decision, stage):
        return None, (
            "semantic: Correctness phủ nhận tính hợp lệ toán học của transition "
            "đã được Code Analyzer xác nhận verified_valid với strength=hard."
        )

    evidence_text = str(stage.get("bieu_thuc_sau") or "").strip()
    if not evidence_text:
        return None, "mechanical: Correctness anchor không có state đích để code dựng grounding."
    canonical = {
        "first_invalid_transition": {
            "solution_index": int(decision["solution_index"]),
            "from_order": decision["from_order"],
            "to_order": int(decision["to_order"]),
            "error_type": decision.get("error_type") or "other",
            "evidence_text": evidence_text,
            "calculation_check": None,
        },
        "context_issue": None,
        "reason": decision["reason"],
        "suggestion": decision.get("suggestion") or "",
        "verdict": status,
    }
    validated, error = validate_transition_judge_output(
        canonical, ordered_solution, generated_question, stage_payload
    )
    if validated is not None and error_type_normalized:
        validated["error_type_normalized"] = True
    if validated is not None:
        validated.update(certificate_fields)
    return validated, error


def _transition_stage_index(
    result: dict[str, Any],
    transition_payload: dict[str, Any],
) -> int | None:
    transition = result.get("first_invalid_transition")
    if not isinstance(transition, dict):
        return None
    return next(
        (
            stage_index
            for stage_index, stage in enumerate(transition_payload.get("stages") or [])
            if int(stage["solution_index"]) == int(transition["solution_index"])
            and stage["from_order"] == transition["from_order"]
            and int(stage["to_order"]) == int(transition["to_order"])
        ),
        None,
    )


def apply_mandatory_arithmetic_result(
    correctness_result: dict[str, Any],
    transition_payload: dict[str, Any],
    ordered_solution: dict[str, Any],
    generated_question: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Giữ lỗi deterministic đã được grounding nếu LLM không tìm lỗi sớm hơn."""

    stages = transition_payload.get("stages") or []
    mandatory_stages = [
        (stage_index, stage, stage.get("mandatory_code_issue"))
        for stage_index, stage in enumerate(stages)
        if isinstance(stage, dict)
        and isinstance(stage.get("mandatory_code_issue"), dict)
    ]
    hard_stages = _hard_invalid_stages(transition_payload)
    if (
        not mandatory_stages
        or not hard_stages
        or mandatory_stages[0][0] != hard_stages[0][0]
        or correctness_result.get("contract_valid") is not True
        or correctness_result.get("certificate_disposition") != "accept"
    ):
        return correctness_result, False
    mandatory_index, stage, mandatory_issue = mandatory_stages[0]
    llm_index = _transition_stage_index(correctness_result, transition_payload)
    if llm_index is not None and llm_index <= mandatory_index:
        return correctness_result, False

    evidence = str(mandatory_issue["evidence_text"])
    expected_result = str(mandatory_issue["expected_result"])
    actual_result = str(mandatory_issue["actual_result"])
    calculation_check = {
        "expected_result": expected_result,
        "actual_result": actual_result,
        "matches": False,
    }
    error_type = "calculation_error"
    reason = (
        f"Phép tính {evidence} sai; kết quả đúng là {expected_result}, "
        f"không phải {actual_result}."
    )
    suggestion = f"Sửa kết quả của phép tính thành {expected_result}."
    role = correctness_result.get("semantic_role")
    is_uncertain = role == "uncertain"
    canonical = {
        "first_invalid_transition": {
            "solution_index": int(stage["solution_index"]),
            "from_order": stage["from_order"],
            "to_order": int(stage["to_order"]),
            "error_type": error_type,
            "evidence_text": evidence,
            "calculation_check": calculation_check,
        },
        "context_issue": None,
        "reason": (
            "Không xác định chắc chắn vai trò ngữ nghĩa của phép tính bị certificate đánh dấu sai."
            if is_uncertain
            else reason
        ),
        "suggestion": "Cần kiểm tra thủ công bước bị cảnh báo." if is_uncertain else suggestion,
        "verdict": "uncertain" if is_uncertain else "bad",
    }
    validated, error = validate_transition_judge_output(
        canonical,
        ordered_solution,
        generated_question,
        transition_payload,
    )
    if validated is None:
        return correctness_result, False
    return {
        "contract_valid": True,
        "http_retry_count": int(correctness_result.get("http_retry_count") or 0),
        "semantic_role": role,
        "certificate_disposition": correctness_result.get("certificate_disposition"),
        "role_evidence": correctness_result.get("role_evidence"),
        **validated,
    }, True


def apply_hard_invalid_equivalence_result(
    correctness_result: dict[str, Any],
    transition_payload: dict[str, Any],
    ordered_solution: dict[str, Any],
    generated_question: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Enforce the earliest hard math certificate using the LLM semantic role."""

    candidates = _hard_invalid_stages(transition_payload)
    if not candidates or correctness_result.get("contract_valid") is not True:
        return correctness_result, False
    role = correctness_result.get("semantic_role")
    disposition = correctness_result.get("certificate_disposition")
    if disposition == "ignore" and role in {
        "hypothetical",
        "rejected",
        "self_corrected",
    }:
        return correctness_result, False
    if disposition != "accept" or role not in {"asserted_active", "uncertain"}:
        return correctness_result, False

    candidate_index, stage = candidates[0]
    analysis = stage["code_analysis"]
    llm_index = _transition_stage_index(correctness_result, transition_payload)
    if llm_index is not None and llm_index <= candidate_index:
        return correctness_result, False

    before = str(stage["bieu_thuc_truoc"])
    after = str(stage["bieu_thuc_sau"])
    issue_type = str(analysis.get("issue_type") or "")
    error_type = {
        "calculation_error": "calculation_error",
        "sign_error": "sign_error",
        "invalid_equivalence": "non_equivalent_transformation",
        "inequality_direction_error": "incorrect_transformation",
        "division_by_zero": "calculation_error",
    }.get(issue_type, "incorrect_transformation")
    is_uncertain = role == "uncertain"
    canonical = {
        "first_invalid_transition": {
            "solution_index": int(stage["solution_index"]),
            "from_order": stage["from_order"],
            "to_order": int(stage["to_order"]),
            "error_type": error_type,
            "evidence_text": after,
            "calculation_check": None,
        },
        "context_issue": None,
        "reason": (
            "Không xác định chắc chắn vai trò ngữ nghĩa của bước bị certificate toán học đánh dấu sai."
            if is_uncertain
            else f"Phép biến đổi từ '{before}' sang '{after}' không hợp lệ về mặt toán học."
        ),
        "suggestion": (
            "Cần kiểm tra thủ công bước bị cảnh báo."
            if is_uncertain
            else "Sửa bước biến đổi để biểu thức sau hợp lệ so với biểu thức trước."
        ),
        "verdict": "uncertain" if is_uncertain else "bad",
    }
    validated, _error = validate_transition_judge_output(
        canonical,
        ordered_solution,
        generated_question,
        transition_payload,
    )
    if validated is None:
        return correctness_result, False
    return {
        "contract_valid": True,
        "http_retry_count": int(correctness_result.get("http_retry_count") or 0),
        "semantic_role": role,
        "certificate_disposition": disposition,
        "role_evidence": correctness_result.get("role_evidence"),
        **validated,
    }, True


def validate_process_presentation_judge_output(
    parsed: Any,
    ordered_solution: dict[str, Any],
    generated_question: dict[str, Any],
    transition_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    try:
        semantic = ProcessPresentationSemanticOutput.model_validate(parsed).model_dump()
    except ValidationError as exc:
        return None, validation_error_text(exc)

    redundant_candidate = next(
        (
            candidate
            for candidate in _redundant_state_candidates(
                ordered_solution,
                transition_payload,
            )
            if semantic["solution_index"] is not None
            and semantic["state_order"] is not None
            and int(candidate["solution_index"]) == int(semantic["solution_index"])
            and int(candidate["state_order"]) == int(semantic["state_order"])
        ),
        None,
    )
    if (
        redundant_candidate is not None
        and semantic["verdict"] == "bad"
        and semantic["error_type"] == "missing_major_step"
    ):
        semantic = {
            **semantic,
            "error_type": "redundant_step",
            "reason": (
                f"Bước {redundant_candidate['source_text'].strip()} đúng toán học nhưng thừa; "
                "bỏ bước này vẫn nối được hai trạng thái trước và sau bằng một transition verified_valid + hard."
            ),
            "suggestion": "Bỏ bước trung gian thừa và giữ đường biến đổi trực tiếp.",
            "verdict": "good",
        }

    redundant_advisory = (
        semantic["verdict"] == "good"
        and semantic["error_type"] == "redundant_step"
    )
    if semantic["verdict"] == "good" and not redundant_advisory:
        return {
            "issue": None,
            "reason": "",
            "suggestion": "",
            "verdict": "good",
        }, ""

    if (
        semantic["error_type"] is None
        or semantic["solution_index"] is None
        or not semantic["reason"].strip()
        or not semantic["suggestion"].strip()
    ):
        return None, "Process & Presentation bad/uncertain hoặc advisory phải có reason và suggestion."

    solution_index = int(semantic["solution_index"])
    state_order = semantic["state_order"]
    if state_order is not None:
        state = next(
            (
                state
                for state in ordered_solution.get("states") or []
                if int(state["solution_index"]) == solution_index
                and int(state["order"]) == int(state_order)
            ),
            None,
        )
        if state is None:
            return None, "Process state_order không tồn tại."
        if not str(state["source_path"]).startswith("/solutions/"):
            return None, "Process state issue phải trỏ tới solution."
        return {
            "issue": {
                "scope": "state",
                "solution_index": solution_index,
                "state_order": int(state_order),
                "source_path": str(state["source_path"]),
                "evidence_text": str(state["source_text"]),
                "error_type": semantic["error_type"],
            },
            "reason": semantic["reason"],
            "suggestion": semantic["suggestion"],
            "verdict": semantic["verdict"],
        }, ""

    valid_solution_paths = {
        str(solution["path"])
        for solution in compact_generated_question_payload(generated_question).get("solutions") or []
    }
    canonical_solution_path = f"/solutions/{solution_index}"
    if canonical_solution_path not in valid_solution_paths:
        return None, "Process solution_index không trỏ tới solution hợp lệ."
    return {
        "issue": {
            "scope": "global",
            "solution_index": solution_index,
            "state_order": None,
            "source_path": canonical_solution_path,
            "evidence_text": None,
            "error_type": semantic["error_type"],
        },
        "reason": semantic["reason"],
        "suggestion": semantic["suggestion"],
        "verdict": semantic["verdict"],
    }, ""


def validate_combined_contract_correction_output(
    parsed: Any,
    ordered_solution: dict[str, Any],
    generated_question: dict[str, Any],
    transition_payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    try:
        output = CombinedJudgeCorrectionOutput.model_validate(parsed).model_dump()
    except ValidationError as exc:
        return None, validation_error_text(exc)

    correctness_semantic = {
        "error_type": output["correctness_error_type"],
        "solution_index": output["correctness_solution_index"],
        "from_order": output["correctness_from_order"],
        "to_order": output["correctness_to_order"],
        "reason": output["correctness_reason"],
        "suggestion": output["correctness_suggestion"],
        "semantic_role": output["correctness_semantic_role"],
        "certificate_disposition": output["correctness_certificate_disposition"],
        "role_evidence": output["correctness_role_evidence"],
        "status": output["correctness_status"],
    }
    process_semantic = {
        "error_type": output["process_error_type"],
        "solution_index": output["process_solution_index"],
        "state_order": output["process_state_order"],
        "reason": output["process_reason"],
        "suggestion": output["process_suggestion"],
        "verdict": output["process_verdict"],
    }
    correctness_result, correctness_error = validate_correctness_semantic_output(
        correctness_semantic,
        ordered_solution,
        generated_question,
        transition_payload,
    )
    if correctness_result is None:
        return None, f"Correctness correction invalid: {correctness_error}"
    process_result, process_error = validate_process_presentation_judge_output(
        process_semantic,
        ordered_solution,
        generated_question,
        transition_payload,
    )
    if process_result is None:
        return None, f"Process correction invalid: {process_error}"
    return {
        "correctness_result": correctness_result,
        "process_presentation_result": process_result,
    }, ""


def _call_solution_splitter(
    generated_question: dict[str, Any],
    client: LLMClient,
    model: str,
    *,
    debug: bool,
) -> tuple[dict[str, Any] | None, str]:
    messages = build_solution_splitter_messages(generated_question)
    try:
        debug_llm_messages(step="solution_state_splitter", model=model, messages=messages, debug=debug)
        response = client.chat_completion(model=model, messages=messages, temperature=0)
    except Exception as exc:
        return None, _RuntimeFailureDetail(str(exc))
    parsed, parse_ok, parse_error = parse_json_output(str(response.get("content") or ""))
    if not parse_ok or parsed is None:
        return None, parse_error or "Splitter không trả JSON hợp lệ."
    return validate_splitter_output(parsed, generated_question)


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


def _call_transition_judge(
    *,
    generated_question: dict[str, Any],
    ordered_solution: dict[str, Any],
    schema_validation_result: dict[str, Any],
    client: LLMClient,
    model: str,
    debug: bool,
    transition_payload: dict[str, Any],
    contract_retry_error: str | None = None,
    contract_retry_candidate: str | None = None,
    retry_transient_once: bool = False,
) -> dict[str, Any]:
    return _call_structured_judge(
        generated_question=generated_question,
        schema_validation_result=schema_validation_result,
        client=client,
        model=model,
        debug=debug,
        step="solution_correctness_judge",
        messages=build_generated_question_judge_messages(
            generated_question,
            load_text(CRITERIA_PATH),
            ordered_solution,
            transition_payload,
            contract_retry_error,
            contract_retry_candidate,
        ),
        validator=lambda parsed: validate_correctness_semantic_output(
            parsed,
            ordered_solution,
            generated_question,
            transition_payload,
        ),
        invalid_json_message="Gemma Correctness không trả JSON hợp lệ.",
        response_format=correctness_response_format(),
        max_tokens=2048,
        retry_transient_once=retry_transient_once,
    )


def _call_process_presentation_judge(
    *,
    generated_question: dict[str, Any],
    ordered_solution: dict[str, Any],
    schema_validation_result: dict[str, Any],
    client: LLMClient,
    model: str,
    debug: bool,
    transition_payload: dict[str, Any] | None = None,
    contract_retry_error: str | None = None,
    contract_retry_candidate: str | None = None,
) -> dict[str, Any]:
    return _call_structured_judge(
        generated_question=generated_question,
        schema_validation_result=schema_validation_result,
        client=client,
        model=model,
        debug=debug,
        step="solution_process_presentation_judge",
        messages=build_process_presentation_judge_messages(
            generated_question,
            load_text(PROCESS_PRESENTATION_CRITERIA_PATH),
            ordered_solution,
            contract_retry_error,
            contract_retry_candidate,
            transition_payload,
        ),
        validator=lambda parsed: validate_process_presentation_judge_output(
            parsed,
            ordered_solution,
            generated_question,
            transition_payload,
        ),
        invalid_json_message="Gemma Process & Presentation không trả JSON hợp lệ.",
        response_format=process_presentation_response_format(ordered_solution),
        max_tokens=2048,
    )


def _call_combined_contract_correction(
    *,
    generated_question: dict[str, Any],
    ordered_solution: dict[str, Any],
    transition_payload: dict[str, Any],
    schema_validation_result: dict[str, Any],
    correctness_candidate: Any,
    process_candidate: Any,
    client: LLMClient,
    model: str,
    debug: bool,
) -> dict[str, Any]:
    return _call_structured_judge(
        generated_question=generated_question,
        schema_validation_result=schema_validation_result,
        client=client,
        model=model,
        debug=debug,
        step="solution_combined_contract_correction",
        messages=build_combined_contract_correction_messages(
            correctness_candidate,
            process_candidate,
            ordered_solution,
        ),
        validator=lambda parsed: validate_combined_contract_correction_output(
            parsed,
            ordered_solution,
            generated_question,
            transition_payload,
        ),
        invalid_json_message="Combined Judge correction không trả JSON hợp lệ.",
        response_format=combined_correction_response_format(ordered_solution),
        max_tokens=3072,
        retry_transient_once=True,
    )


@dataclass(frozen=True)
class AnchoredSolutionIssue:
    source: Literal["correctness", "process_presentation"]
    verdict: Literal["bad", "uncertain", "warning"]
    scope: Literal["state", "global"]
    solution_index: int | None
    source_path: str
    anchor_order: int | None
    char_start: int | None
    char_end: int | None
    error_type: str
    reason: str
    suggestion: str


def normalize_correctness_result(
    result: dict[str, Any],
    ordered_solution: dict[str, Any],
    generated_question: dict[str, Any],
) -> AnchoredSolutionIssue | None:
    if result["verdict"] == "good":
        return None

    transition = result["first_invalid_transition"]
    if transition is None:
        return AnchoredSolutionIssue(
            source="correctness",
            verdict="uncertain",
            scope="global",
            solution_index=None,
            source_path="/solutions",
            anchor_order=None,
            char_start=None,
            char_end=None,
            error_type="other",
            reason=result["reason"],
            suggestion=result.get("suggestion") or "",
        )
    transition_orders = (
        (transition["to_order"],)
        if transition["from_order"] is None
        else (transition["to_order"], transition["from_order"])
    )
    candidate_states = [
        state
        for order in transition_orders
        for state in ordered_solution.get("states") or []
        if int(state["solution_index"]) == int(transition["solution_index"])
        and int(state["order"]) == int(order)
    ]
    evidence = transition["evidence_text"]
    state = next(
        state
        for state in candidate_states
        if _normalized_evidence_in_source(evidence, state["source_text"])
    )
    char_start = _normalized_evidence_position(evidence, state["source_text"])
    return AnchoredSolutionIssue(
        source="correctness",
        verdict=result["verdict"],
        scope="state",
        solution_index=int(transition["solution_index"]),
        source_path=state["source_path"],
        anchor_order=int(state["order"]),
        char_start=char_start,
        char_end=char_start + len(normalize_math_evidence(evidence)),
        error_type=transition["error_type"],
        reason=result["reason"],
        suggestion=result["suggestion"],
    )


def normalize_process_presentation_result(
    result: dict[str, Any],
    ordered_solution: dict[str, Any],
) -> AnchoredSolutionIssue | None:
    if result["verdict"] == "good" and not (
        isinstance(result.get("issue"), dict)
        and result["issue"].get("error_type") == "redundant_step"
    ):
        return None

    issue = result["issue"]
    normalized_verdict = "warning" if result["verdict"] == "good" else result["verdict"]
    if issue["scope"] == "global":
        return AnchoredSolutionIssue(
            source="process_presentation",
            verdict=normalized_verdict,
            scope="global",
            solution_index=int(issue["solution_index"]),
            source_path=issue["source_path"],
            anchor_order=None,
            char_start=None,
            char_end=None,
            error_type=issue["error_type"],
            reason=result["reason"],
            suggestion=result["suggestion"],
        )

    state = next(
        state
        for state in ordered_solution.get("states") or []
        if int(state["solution_index"]) == int(issue["solution_index"])
        and int(state["order"]) == int(issue["state_order"])
    )
    char_start = _normalized_evidence_position(
        issue["evidence_text"], state["source_text"]
    )
    return AnchoredSolutionIssue(
        source="process_presentation",
        verdict=normalized_verdict,
        scope="state",
        solution_index=int(issue["solution_index"]),
        source_path=issue["source_path"],
        anchor_order=int(issue["state_order"]),
        char_start=char_start,
        char_end=char_start + len(normalize_math_evidence(issue["evidence_text"])),
        error_type=issue["error_type"],
        reason=result["reason"],
        suggestion=result["suggestion"],
    )


def _select_earliest_solution_issue(
    issues: list[AnchoredSolutionIssue],
) -> AnchoredSolutionIssue | None:
    source_rank = {"process_presentation": 0, "correctness": 1}
    state_issues = [issue for issue in issues if issue.scope == "state"]
    if state_issues:
        return min(
            state_issues,
            key=lambda issue: (
                issue.solution_index if issue.solution_index is not None else -1,
                issue.anchor_order if issue.anchor_order is not None else -1,
                issue.char_start if issue.char_start is not None else 0,
                source_rank[issue.source],
            ),
        )

    global_issues = [issue for issue in issues if issue.scope == "global"]
    if global_issues:
        return min(
            global_issues,
            key=lambda issue: (
                issue.solution_index if issue.solution_index is not None else -1,
                source_rank[issue.source],
            ),
        )
    return None


def aggregate_specialized_judge_results(
    correctness_result: Any,
    process_presentation_result: Any,
    ordered_solution: dict[str, Any],
    generated_question: dict[str, Any],
    *,
    strict_mode: bool,
    index: int,
) -> dict[str, Any]:
    """Chuẩn hóa hai contract và chọn issue xuất hiện sớm nhất."""

    if not isinstance(correctness_result, dict):
        return fail_closed_output(
            "Correctness Judge handoff không phải canonical result.",
            generated_question=generated_question,
            index=index,
        )
    if correctness_result.get("contract_valid") is not True:
        return fail_closed_output(
            "Correctness Judge trả output không đúng contract: "
            + str(correctness_result.get("contract_error") or ""),
            generated_question=generated_question,
            index=index,
        )
    if not isinstance(process_presentation_result, dict):
        return fail_closed_output(
            "Process & Presentation Judge handoff không phải canonical result.",
            generated_question=generated_question,
            index=index,
        )
    if process_presentation_result.get("contract_valid") is not True:
        return fail_closed_output(
            "Process & Presentation Judge trả output không đúng contract: "
            + str(process_presentation_result.get("contract_error") or ""),
            generated_question=generated_question,
            index=index,
        )
    try:
        correctness_payload = CanonicalCorrectnessResult.model_validate(
            {
                name: correctness_result[name]
                for name in CanonicalCorrectnessResult.model_fields
                if name in correctness_result
            }
        ).model_dump()
    except (ValidationError, KeyError) as exc:
        return fail_closed_output(
            "Correctness Judge handoff không còn hợp lệ: " + str(exc),
            generated_question=generated_question,
            index=index,
        )
    try:
        presentation_payload = ProcessPresentationJudgeOutput.model_validate(
            {
                name: process_presentation_result[name]
                for name in ProcessPresentationJudgeOutput.model_fields
                if name in process_presentation_result
            }
        ).model_dump()
    except (ValidationError, KeyError) as exc:
        return fail_closed_output(
            "Process & Presentation Judge handoff không còn hợp lệ: " + str(exc),
            generated_question=generated_question,
            index=index,
        )
    try:
        normalized_issues = [
            issue
            for issue in (
                normalize_correctness_result(
                    correctness_payload, ordered_solution, generated_question
                ),
                normalize_process_presentation_result(
                    presentation_payload, ordered_solution
                ),
            )
            if issue is not None
        ]
    except (KeyError, StopIteration, TypeError, ValueError) as exc:
        return fail_closed_output(
            "Judge handoff grounding không còn hợp lệ: " + str(exc),
            generated_question=generated_question,
            index=index,
        )
    blocking_issues = [issue for issue in normalized_issues if issue.verdict != "warning"]
    selected = _select_earliest_solution_issue(blocking_issues or normalized_issues)
    if selected is not None:
        advisory = selected.verdict == "warning"
        payload = {
            "is_good": False,
            "issues": [{
                "severity": "warning" if advisory else "needs_review",
                "category": "solution_quality",
                "location": selected.source_path,
                "reason": selected.reason,
                "suggestion": selected.suggestion,
                "repair_intent": "clean_solution_reasoning" if advisory else "needs_manual_review",
            }],
        }
    else:
        payload = {"is_good": True, "issues": []}
    return normalize_generated_question_result(
        payload,
        strict_mode=strict_mode,
        generated_question=generated_question,
        index=index,
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
    """Solution Judge: run specialized models without semantic model fallback."""

    fast_model = generated_question_splitter_model(config)
    correctness_model = generated_question_correctness_model(config)
    presentation_model = generated_question_presentation_model(config)
    dedicated_correctness = bool(
        getattr(config, "solution_correctness_model", "")
    )
    split, gemma_split_error = _call_solution_splitter(
        generated_question,
        client,
        fast_model,
        debug=debug,
    )
    splitter_runtime_seen = isinstance(gemma_split_error, _RuntimeFailureDetail)
    splitter_attempts = 1
    splitter_fallback_called = False
    code_splitter_used = False
    if split is None:
        split_error = gemma_split_error
        split, code_split_error = split_solution_with_code(generated_question)
        if split is not None:
            code_splitter_used = True
        else:
            split_error = f"{split_error} | Code splitter: {code_split_error}"
        if split is None:
            detail = "Không tách được ordered states nguyên văn: " + split_error
            result = (
                _judge_runtime_needs_review_output(
                    detail,
                    generated_question=generated_question,
                    strict_mode=strict_mode,
                    index=index,
                )
                if splitter_runtime_seen
                else _judge_contract_needs_review_output(
                    detail,
                    generated_question=generated_question,
                    strict_mode=strict_mode,
                    index=index,
                )
            )
            result.update(
                judge_model=fast_model,
                judge_attempt_count=splitter_attempts,
                judge_fallback_called=splitter_fallback_called,
                judge_gemma_run_count=0,
                judge_fallback_reason=(
                    "splitter_runtime_failure"
                    if splitter_runtime_seen
                    else "splitter_contract_failure"
                ),
            )
            return result

    ordered_solution = split
    context_result = build_context_issue_from_requirements(
        ordered_solution,
        generated_question,
        strict_mode=strict_mode,
        index=index,
    )
    if context_result is not None:
        context_result.update(
            judge_model=fast_model,
            judge_attempt_count=splitter_attempts,
            judge_fallback_called=splitter_fallback_called,
            judge_gemma_run_count=0,
            judge_fallback_reason=(
                "reasoning_splitter_fallback"
                if splitter_fallback_called
                else "code_splitter_fallback" if code_splitter_used
                else None
            ),
        )
        return context_result

    transition_payload = analyze_transition_stages(
        build_transition_stages(ordered_solution, generated_question)
    )

    def call_correctness(
        model: str,
        contract_retry_error: str | None = None,
        contract_retry_candidate: str | None = None,
        *,
        retry_transient_once: bool = True,
    ) -> dict[str, Any]:
        return _call_transition_judge(
            generated_question=generated_question,
            ordered_solution=ordered_solution,
            schema_validation_result=schema_validation_result,
            client=client,
            model=model,
            debug=debug,
            transition_payload=transition_payload,
            contract_retry_error=contract_retry_error,
            contract_retry_candidate=contract_retry_candidate,
            retry_transient_once=retry_transient_once,
        )

    def call_process_presentation(
        model: str,
        contract_retry_error: str | None = None,
        contract_retry_candidate: str | None = None,
    ) -> dict[str, Any]:
        return _call_process_presentation_judge(
            generated_question=generated_question,
            ordered_solution=ordered_solution,
            schema_validation_result=schema_validation_result,
            client=client,
            model=model,
            debug=debug,
            transition_payload=transition_payload,
            contract_retry_error=contract_retry_error,
            contract_retry_candidate=contract_retry_candidate,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        correctness_future = executor.submit(call_correctness, correctness_model)
        process_presentation_future = executor.submit(
            call_process_presentation, presentation_model
        )
        correctness_result = correctness_future.result()
        process_presentation_result = process_presentation_future.result()

    correctness_metrics = {
        "correctness_primary_26b_calls": 1,
        "correctness_contract_correction_26b_calls": 0,
        "correctness_rejudge_26b_calls": 0,
        "primary_12b_calls": int(not dedicated_correctness),
        "contract_correction_12b_calls": 0,
        "fallback_26b_calls": 0,
        "fallback_reason_contract": 0,
        "fallback_reason_resolver_mismatch": 0,
        "http_retry_count": 0,
        "non_canonical_count": 0,
        "runtime_count": 0,
        "correctness_error_type_normalized_count": 0,
    }

    def record_correctness_call(value: Any) -> None:
        if not isinstance(value, dict):
            correctness_metrics["non_canonical_count"] += 1
            return
        correctness_metrics["http_retry_count"] += int(
            value.get("http_retry_count") or 0
        )
        if value.get("failure_kind") == "runtime":
            correctness_metrics["runtime_count"] += 1
        elif value.get("contract_valid") is not True:
            correctness_metrics["non_canonical_count"] += 1
        if value.get("error_type_normalized") is True:
            correctness_metrics["correctness_error_type_normalized_count"] += 1

    def record_process_call(value: Any) -> None:
        if not isinstance(value, dict):
            correctness_metrics["non_canonical_count"] += 1
            return
        correctness_metrics["http_retry_count"] += int(
            value.get("http_retry_count") or 0
        )
        if value.get("failure_kind") == "runtime":
            correctness_metrics["runtime_count"] += 1
        elif value.get("contract_valid") is not True:
            correctness_metrics["non_canonical_count"] += 1

    record_correctness_call(correctness_result)
    record_process_call(process_presentation_result)
    correctness_valid = (
        isinstance(correctness_result, dict)
        and correctness_result.get("contract_valid") is True
    )
    presentation_valid = (
        isinstance(process_presentation_result, dict)
        and process_presentation_result.get("contract_valid") is True
    )
    correctness_runtime_seen = (
        isinstance(correctness_result, dict)
        and correctness_result.get("failure_kind") == "runtime"
    )
    presentation_runtime_seen = (
        isinstance(process_presentation_result, dict)
        and process_presentation_result.get("failure_kind") == "runtime"
    )
    correctness_fallback_called = False
    presentation_fallback_called = False
    contract_correction_call_count = 0
    reasoning_fallback_reason: str | None = None

    # Both parallel Judges share one contract-correction budget per object.
    # When both fail, one flat 26B call repairs both envelopes and each result is
    # still validated/grounded independently before Aggregate can see it.
    if (
        not correctness_valid
        and not correctness_runtime_seen
        and not presentation_valid
        and not presentation_runtime_seen
    ):
        contract_correction_call_count = 1
        correctness_metrics["correctness_contract_correction_26b_calls"] += 1
        combined_correction = _call_combined_contract_correction(
            generated_question=generated_question,
            ordered_solution=ordered_solution,
            transition_payload=transition_payload,
            schema_validation_result=schema_validation_result,
            correctness_candidate=correctness_result,
            process_candidate=process_presentation_result,
            client=client,
            model=correctness_model,
            debug=debug,
        )
        correctness_metrics["http_retry_count"] += int(
            combined_correction.get("http_retry_count") or 0
        )
        if combined_correction.get("contract_valid") is True:
            correctness_result = {
                "contract_valid": True,
                "http_retry_count": 0,
                **combined_correction["correctness_result"],
            }
            process_presentation_result = {
                "contract_valid": True,
                "http_retry_count": 0,
                **combined_correction["process_presentation_result"],
            }
            correctness_valid = True
            presentation_valid = True
        elif combined_correction.get("failure_kind") == "runtime":
            correctness_metrics["runtime_count"] += 1
            correctness_runtime_seen = True
            presentation_runtime_seen = True
        else:
            correctness_metrics["non_canonical_count"] += 1
    elif not correctness_valid and not correctness_runtime_seen:
        contract_correction_call_count = 1
        correctness_metrics["correctness_contract_correction_26b_calls"] += 1
        correctness_metrics["contract_correction_12b_calls"] += int(not dedicated_correctness)
        correctness_result = call_correctness(
            correctness_model,
            str(
                correctness_result.get("contract_error")
                if isinstance(correctness_result, dict)
                else "Output is not canonical."
            ),
            str(
                correctness_result.get("raw_candidate")
                if isinstance(correctness_result, dict)
                else ""
            ),
        )
        record_correctness_call(correctness_result)
        correctness_valid = (
            isinstance(correctness_result, dict)
            and correctness_result.get("contract_valid") is True
        )
        correctness_runtime_seen = (
            correctness_runtime_seen
            or (
                isinstance(correctness_result, dict)
                and correctness_result.get("failure_kind") == "runtime"
            )
        )
    elif not presentation_valid and not presentation_runtime_seen:
        contract_correction_call_count = 1
        correctness_metrics["contract_correction_12b_calls"] += 1
        process_presentation_result = call_process_presentation(
            presentation_model,
            str(
                process_presentation_result.get("contract_error")
                if isinstance(process_presentation_result, dict)
                else "Output is not canonical."
            ),
            str(
                process_presentation_result.get("raw_candidate")
                if isinstance(process_presentation_result, dict)
                else ""
            ),
        )
        record_process_call(process_presentation_result)
        presentation_valid = (
            isinstance(process_presentation_result, dict)
            and process_presentation_result.get("contract_valid") is True
        )
        presentation_runtime_seen = (
            presentation_runtime_seen
            or (
                isinstance(process_presentation_result, dict)
                and process_presentation_result.get("failure_kind") == "runtime"
            )
        )

    correctness_valid = (
        isinstance(correctness_result, dict)
        and correctness_result.get("contract_valid") is True
    )
    presentation_valid = (
        isinstance(process_presentation_result, dict)
        and process_presentation_result.get("contract_valid") is True
    )
    final_correctness_result = correctness_result
    if correctness_valid:
        final_correctness_result, arithmetic_applied = apply_mandatory_arithmetic_result(
            correctness_result,
            transition_payload,
            ordered_solution,
            generated_question,
        )
        if not arithmetic_applied:
            final_correctness_result, _ = apply_hard_invalid_equivalence_result(
                final_correctness_result,
                transition_payload,
                ordered_solution,
                generated_question,
            )

    if not (
        isinstance(final_correctness_result, dict)
        and final_correctness_result.get("contract_valid") is True
        and isinstance(process_presentation_result, dict)
        and process_presentation_result.get("contract_valid") is True
    ):
        invalid_scopes = []
        for scope, value in (
            ("correctness", final_correctness_result),
            ("process_presentation", process_presentation_result),
        ):
            if isinstance(value, dict) and value.get("contract_valid") is True:
                continue
            contract_error = (
                str(value.get("contract_error") or "output không canonical")
                if isinstance(value, dict)
                else "output không phải object"
            )
            invalid_scopes.append(f"{scope}: {contract_error}")
        detail = (
            "Không aggregate semantic vì Judge handoff invalid: "
            + " | ".join(invalid_scopes)
        )
        unresolved_runtime = (
            correctness_runtime_seen
            and not (
                isinstance(final_correctness_result, dict)
                and final_correctness_result.get("contract_valid") is True
            )
        ) or (
            presentation_runtime_seen
            and not (
                isinstance(process_presentation_result, dict)
                and process_presentation_result.get("contract_valid") is True
            )
        )
        result = (
            _judge_runtime_needs_review_output(
                detail,
                generated_question=generated_question,
                strict_mode=strict_mode,
                index=index,
            )
            if unresolved_runtime
            else _judge_contract_needs_review_output(
                detail,
                generated_question=generated_question,
                strict_mode=strict_mode,
                index=index,
            )
        )
    else:
        result = aggregate_specialized_judge_results(
            final_correctness_result,
            process_presentation_result,
            ordered_solution,
            generated_question,
            strict_mode=strict_mode,
            index=index,
        )
    result.update(
        judge_model=correctness_model,
        splitter_model=fast_model,
        correctness_primary_model=correctness_model,
        process_presentation_model=presentation_model,
        final_correctness_model=correctness_model,
        judge_attempt_count=(
            splitter_attempts
            + 2
            + contract_correction_call_count
            + int(correctness_fallback_called)
            + int(presentation_fallback_called)
        ),
        judge_fallback_called=(
            splitter_fallback_called
            or correctness_fallback_called
            or presentation_fallback_called
        ),
        judge_gemma_run_count=2,
        judge_fallback_reason=(
            reasoning_fallback_reason
            if reasoning_fallback_reason
            else "reasoning_splitter_fallback" if splitter_fallback_called
            else "code_splitter_fallback" if code_splitter_used
            else None
        )
    )
    result["correctness_fallback_called"] = correctness_fallback_called
    result.update(correctness_metrics)
    if debug:
        print(
            "[DEBUG_GENERATED_QUESTION_FALLBACK] "
            + json.dumps(
                {
                    "id": generated_question.get("id")
                    or generated_question.get("_id"),
                    "correctness_fallback": correctness_fallback_called,
                    "presentation_fallback": presentation_fallback_called,
                    "splitter_model": fast_model,
                    "correctness_model": correctness_model,
                    "presentation_model": presentation_model,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
    return result


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

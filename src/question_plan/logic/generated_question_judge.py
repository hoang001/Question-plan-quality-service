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
from .code_transition_analyzer import analyze_transition_stages


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
- Chỉ trả quyết định semantic; không trả location, source_path, evidence_text, before, after hoặc code_analysis review.
- Không dùng chuỗi "None", "null" hoặc "N/A" thay cho JSON null."""
PROCESS_PRESENTATION_OUTPUT_INVARIANTS = """### ĐIỀU KIỆN BẤT BIẾN CỦA ĐẦU RA
- Nếu verdict="good": ưu tiên error_type=null, solution_index=null, state_order=null, reason="", suggestion=""; code sẽ bỏ qua nhận xét thừa và canonical hóa kết quả good.
- Nếu verdict="bad" hoặc verdict="uncertain": error_type, solution_index, reason và suggestion bắt buộc; state_order=null chỉ khi lỗi áp dụng cho toàn solution.
- Không trả source_path, evidence_text hoặc scope; code tự dựng các metadata này từ anchor đã validate.
- Không dùng chuỗi "None", "null" hoặc "N/A" thay cho giá trị JSON; dùng JSON null hoặc "" đúng theo schema."""
FIRST_TRANSITION_RULE = """### QUY TẮC BƯỚC CHUYỂN ĐẦU TIÊN
- Bắt buộc đọc `initial_transition` trước các transition `loi_giai`.
- Nếu lỗi đầu tiên ở initial transition, trả `from_order=null` và `to_order` của first state.
- Evidence phải nằm nguyên văn trong first state.
- Không dùng state phía sau để hợp thức hóa lỗi đứng trước.
- Đọc `code_analysis` như tín hiệu advisory để ưu tiên vị trí cần kiểm tra; không sao chép annotation vào output."""
CODE_SPLITTER_MAX_TOKENS = 160
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_BOUNDARY_PATTERNS = (
    re.compile(r"\\Rightarrow|\\implies|⇒|=>"),
    re.compile(r"\r?\n+"),
    re.compile(r"[.!?;:,]+(?:[\"'»”)\]]*)\s+"),
    re.compile(r"\b(?:Suy ra|Do đó|Vậy|Khi đó|Ta có|Tiếp theo|Mặt khác)\b", re.IGNORECASE),
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

    context_texts = [
        str(block["text"])
        for block in payload.get("instruction") or []
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ]
    context_texts.extend(
        str(block["text"])
        for item in payload.get("questionItems") or []
        for block in item.get("stem") or []
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    )
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
    return validate_splitter_output(
        {"context_requirements": [], "states": states},
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
                "### CONTEXT REQUIREMENT ANALYSIS\n"
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
                "context_requirements=[]. evidence_text phải là đoạn nguyên văn trong QUESTION CONTEXT thể hiện "
                "sự phụ thuộc. Nếu không có dependency bổ sung bắt buộc, trả context_requirements=[].\n"
                "Không trả verdict, context_issue, is_good, error_type hoặc suggestion; không đánh giá solution.\n\n"
                "### VISUAL DESCRIPTION\n"
                "Với mỗi phần image_url được đính kèm, bắt buộc trả đúng một phần tử visual_descriptions. "
                "Sao chép chính xác source_path và asset_url từ manifest ảnh. description phải mô tả bằng tiếng Việt "
                "những gì thực sự nhìn thấy và đủ chi tiết để Judge phía sau kiểm chứng bài: loại hình/bảng/đồ thị, "
                "mọi nhãn, hàng/cột, mốc, giá trị, dấu, chiều biến thiên, điểm và quan hệ có liên quan. "
                "Không tự giải bài, không suy đoán phần không nhìn rõ. Nếu không đọc được ảnh, đặt description đúng bằng "
                "'Không thể đọc nội dung ảnh.' và trả context requirement insufficient tương ứng; flow sẽ dừng trước Correctness. "
                "Nếu không có ảnh đính kèm, trả visual_descriptions=[].\n\n"
                "### STATE SPLITTING\n"
                "Chỉ trả context_requirements, visual_descriptions và states. Mỗi source_text phải là một lát cắt nguyên văn của đúng source_path. "
                "Trong từng solution, order bắt đầu từ 0 và liên tục qua các text block theo thứ tự gốc. "
                "Khi ghép source_text của các state thuộc cùng source_path theo order, kết quả phải giống tuyệt đối "
                "text gốc từng ký tự. Không trả transitions và không bỏ qua solution block.\n\n"
                f"JSON SCHEMA:\n{contract_schema_text(SolutionSplitOutput)}\n\n"
                f"QUESTION CONTEXT:\n{json.dumps(question_context, ensure_ascii=False, indent=2)}\n\n"
                f"TEXTUAL VISUAL DESCRIPTIONS:\n{json.dumps(visual_descriptions, ensure_ascii=False, indent=2)}\n\n"
                f"SOLUTION PAYLOAD:\n{json.dumps(payload.get('solutions') or [], ensure_ascii=False, indent=2)}\n\n"
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
    """Expose only hard-invalid arithmetic advice to Correctness.

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
        and analysis.get("status") == "verified_invalid"
        and analysis.get("strength") == "hard"
    ):
        prompt_stage["code_analysis"] = {
            name: analysis[name]
            for name in _CORRECTNESS_HARD_ANALYSIS_FIELDS
            if analysis.get(name) is not None
        }
    return prompt_stage


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
            "Chỉ sửa lỗi contract của candidate và trả lại đúng 7 field. Không đổi quyết định semantic "
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
                "đó là tín hiệu advisory và chính bạn phải quyết định đúng/sai. "
                "Chỉ trả JSON đúng schema, không sửa dữ liệu."
            ),
        },
        {
            "role": "user",
            "content": (
                f"### TIÊU CHÍ\n{criteria_text}\n\n"
                "### HỢP ĐỒNG ĐẦU RA\n"
                "Chỉ trả 7 field semantic theo thứ tự: error_type, solution_index, from_order, to_order, reason, suggestion, status.\n\n"
                f"### NGỮ CẢNH CÂU HỎI\n{_prompt_json(payload)}\n\n"
                f"### LỜI GIẢI THEO THỨ TỰ\n{_prompt_json(stages)}\n\n"
                "### RÀNG BUỘC CUỐI\n"
                "- Chỉ trả lỗi correctness đầu tiên.\n"
                "- `verified_invalid + hard` là tín hiệu cơ học mạnh cần ưu tiên kiểm tra, nhưng không phải field cần sao chép vào output. "
                "Nếu đồng ý thì báo lỗi transition theo contract semantic; nếu không đồng ý thì tiếp tục đánh giá bình thường. Các issue_type cơ học gồm "
                "sign_error, calculation_error, invalid_equivalence hoặc inequality_direction_error.\n"
                "- Với `verified_invalid + hard`, phải đọc nguyên văn toàn bộ state để xác định biểu thức bị cảnh báo có thực sự "
                "được dùng làm tiền đề hoặc kết luận hay không. Chỉ được bỏ qua cảnh báo khi lời giải có bằng chứng rõ ràng rằng "
                "biểu thức đó bị bác bỏ/sửa lại, hoặc chỉ được trích dẫn/đặt làm giả thiết phản ví dụ và không được dùng cho bước sau. "
                "Nếu không có bằng chứng rõ ràng hoặc biểu thức được dùng tiếp, phải xử lý nó như lỗi correctness.\n"
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
                f"### RULES\n{criteria_text}\n\n"
                f"### OUTPUT SCHEMA\n{_prompt_json(ProcessPresentationSemanticOutput.model_json_schema())}\n\n"
                f"### QUESTION CONTEXT\n{_prompt_json(context)}\n\n"
                f"### ORDERED SOLUTION STATES\n{_prompt_json(solution_states)}\n\n"
                f"### SOLUTION PATHS\n{_prompt_json(solution_paths)}\n\n"
                "### FINAL CONSTRAINTS\n"
                "- Chỉ trả 6 field theo thứ tự: error_type, solution_index, state_order, reason, suggestion, verdict.\n"
                "- Lỗi tại state phải trả đúng solution_index và state_order; code tự dựng source_path và evidence_text.\n"
                "- Lỗi toàn bộ solution trả solution_index và state_order=null.\n"
                "- Dùng missing_major_step khi lời giải thiếu biến đổi chính cần thiết cho học sinh theo dõi; "
                "không dùng nó cho phép tính sai, thiếu nghiệm, thiếu trường hợp hoặc thiếu điều kiện toán học.\n"
                "- Đọc toàn bộ state: nếu trạng thái trung gian đã xuất hiện nguyên văn thì không được báo nó bị thiếu.\n"
                "- Không kiểm tra lại hay sửa kết quả toán học.\n"
                "- verdict=good: code bỏ qua các field nhận xét còn lại và canonical hóa thành không có issue.\n"
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
                "Sửa đồng thời candidate Correctness và Process & Presentation thành 13 field phẳng có prefix. "
                "Mọi field đều bắt buộc; dùng null đúng kiểu. Chỉ dùng solution_index/state order trong canonical_anchors.\n\n"
                f"### INPUT CẦN SỬA\n{_prompt_json(correction_input)}\n\n"
                "### INVARIANT CORRECTNESS\n"
                f"{CORRECTNESS_OUTPUT_INVARIANTS}\n\n"
                "### INVARIANT PROCESS & PRESENTATION\n"
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
        _problem_anchor(compact_generated_question_payload(generated_question))
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

    status = decision["status"]
    if status == "good":
        canonical = {
            "first_invalid_transition": None,
            "context_issue": None,
            "reason": "",
            "suggestion": "",
            "verdict": "good",
        }
        return validate_transition_judge_output(
            canonical, ordered_solution, generated_question, transition_payload
        )

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
        }, ""
    if decision.get("solution_index") is None or decision.get("to_order") is None:
        return None, "mechanical: Correctness anchor chỉ được bỏ hoàn toàn cho status=uncertain."

    stage_payload = transition_payload or build_transition_stages(
        ordered_solution, generated_question
    )
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
    if not mandatory_stages or correctness_result.get("contract_valid") is not True:
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
        "reason": reason,
        "suggestion": suggestion,
        "verdict": "bad",
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
        **validated,
    }, True


def validate_process_presentation_judge_output(
    parsed: Any,
    ordered_solution: dict[str, Any],
    generated_question: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    try:
        semantic = ProcessPresentationSemanticOutput.model_validate(parsed).model_dump()
    except ValidationError as exc:
        return None, validation_error_text(exc)

    if semantic["verdict"] == "good":
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
        return None, "Process & Presentation bad/uncertain phải có reason và suggestion."

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
        ),
        validator=lambda parsed: validate_process_presentation_judge_output(
            parsed,
            ordered_solution,
            generated_question,
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
    verdict: Literal["bad", "uncertain"]
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
    if result["verdict"] == "good":
        return None

    issue = result["issue"]
    if issue["scope"] == "global":
        return AnchoredSolutionIssue(
            source="process_presentation",
            verdict=result["verdict"],
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
        verdict=result["verdict"],
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
    selected = _select_earliest_solution_issue(normalized_issues)
    if selected is not None:
        payload = {
            "is_good": False,
            "issues": [{
                "severity": "needs_review",
                "category": "solution_quality",
                "location": selected.source_path,
                "reason": selected.reason,
                "suggestion": selected.suggestion,
                "repair_intent": "needs_manual_review",
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
        final_correctness_result, _ = (
            apply_mandatory_arithmetic_result(
                correctness_result,
                transition_payload,
                ordered_solution,
                generated_question,
            )
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

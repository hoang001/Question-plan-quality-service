"""LLM judge đánh giá chất lượng solution của generated question object."""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..infra.config import (
    AppConfig,
    generated_question_fast_model,
    generated_question_reasoning_model,
)
from ..infra.debug import debug_llm_messages, llm_prompt_debug_enabled
from ..infra.llm_client import LLMClient
from ..shared.utils import parse_json_output
from ..schemas.generated_question_contracts import (
    SolutionSplitOutput,
    TransitionJudgeOutput,
    contract_schema_text,
    validation_error_text,
)
from .generated_question_schema import (
    fail_closed_output,
    normalize_generated_question_result,
)


KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"
CRITERIA_PATH = KNOWLEDGE_DIR / "generated_question_quality_criteria.md"


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


def build_solution_splitter_messages(generated_question: dict[str, Any]) -> list[dict[str, str]]:
    payload = compact_generated_question_payload(generated_question)
    return [
        {
            "role": "system",
            "content": (
                "Bạn là Solution State Splitter. Chỉ chia text thành ordered states, không đánh giá đúng sai. "
                "Tuyệt đối không thêm, bớt, sửa, chuẩn hóa hoặc diễn giải bất kỳ chữ, khoảng trắng, xuống dòng, "
                "dấu câu, ký hiệu hay LaTeX nào. Chỉ trả một JSON object hợp lệ, không markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                "Chỉ trả field states. Mỗi source_text phải là một lát cắt nguyên văn của đúng source_path. "
                "Trong từng solution, order bắt đầu từ 0 và liên tục qua các text block theo thứ tự gốc. "
                "Khi ghép source_text của các state thuộc cùng source_path theo order, kết quả phải giống tuyệt đối "
                "text gốc từng ký tự. Không trả transitions và không bỏ qua solution block.\n\n"
                f"JSON SCHEMA:\n{contract_schema_text(SolutionSplitOutput)}\n\n"
                f"SOLUTION PAYLOAD:\n{json.dumps(payload.get('solutions') or [], ensure_ascii=False, indent=2)}"
            ),
        },
    ]


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


def _prepend_problem_context(
    split: dict[str, Any],
    generated_question: dict[str, Any],
) -> dict[str, Any]:
    anchor = _problem_anchor(compact_generated_question_payload(generated_question))
    if anchor is None:
        return split
    states: list[dict[str, Any]] = []
    solution_indexes = sorted({int(state["solution_index"]) for state in split["states"]})
    for solution_index in solution_indexes:
        solution_states = [state for state in split["states"] if int(state["solution_index"]) == solution_index]
        states.append({"solution_index": solution_index, "order": 0, **anchor})
        states.extend({**state, "order": int(state["order"]) + 1} for state in solution_states)
    return {"states": states}


def build_generated_question_judge_messages(
    generated_question: dict[str, Any],
    criteria_text: str,
    ordered_solution: dict[str, Any],
) -> list[dict[str, str]]:
    """Yêu cầu LLM tự đánh giá và chỉ trả kết luận good/bad/uncertain."""

    payload = compact_generated_question_payload(generated_question)
    payload.pop("solutions", None)
    stages = build_transition_stages(ordered_solution)
    return [
        {
            "role": "system",
            "content": (
                "Bạn là Solution Quality Judge chuyên nghiệp. Nhiệm vụ của bạn là đánh giá tính đúng đắn "
                "và chất lượng của lời giải toán học dựa trên `question_context` và các `stages_co_dinh`.\n\n"
                "<cac_rang_buoc_bat_buoc>\n"
                "- CHỈ đánh giá solution; dùng instruction, stem và interaction type làm ngữ cảnh nền.\n"
                "- TUYỆT ĐỐI KHÔNG đánh giá answerSpec, expected, options hoặc hints.\n"
                "- KHÔNG sửa solution đầu vào hoặc tạo final answer mới.\n"
                "- CHỈ trả duy nhất một JSON object đúng schema; không markdown, không preamble.\n"
                '- Toàn bộ "reason" và "suggestion" PHẢI viết bằng tiếng Việt.\n'
                "</cac_rang_buoc_bat_buoc>"
            ),
        },
        {
            "role": "user",
            "content": (
                f"<huong_dan_danh_gia>\n{criteria_text}\n</huong_dan_danh_gia>\n\n"
                "<vi_du_kiem_tra_loi>\n"
                "Ví dụ 1 — lỗi tính toán xuất hiện trước:\n"
                "2x^3 + 3 = 19 → 2x^3 = 18 → x^3 = 9 → x = 2\n"
                "Kết luận: bad tại transition đầu tiên, vì 19 - 3 = 16, không phải 18. "
                "Không được bỏ qua để báo lỗi phía sau.\n\n"
                "Ví dụ 2 — thiếu bước biến đổi cốt lõi:\n"
                "2x^3 + 3 = 19 → 2x^3 = 16 → x = 2\n"
                "Kết luận: bad tại transition thứ hai, vì thiếu trạng thái trung gian x^3 = 8.\n"
                "</vi_du_kiem_tra_loi>\n\n"
                f"<question_context>\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n</question_context>\n\n"
                f"<stages_co_dinh>\n{json.dumps(stages, ensure_ascii=False, indent=2)}\n</stages_co_dinh>\n\n"
                "<output_schema_json>\n"
                f"{contract_schema_text(TransitionJudgeOutput)}\n"
                "</output_schema_json>\n\n"
                "<rang_buoc_cuoi>\n"
                "- Không chia lại stage và không sửa các trường dữ liệu của stage.\n"
                "- Điền kết quả kiểm tra từ stage 1 theo đúng thứ tự.\n"
                "- Gặp trang_thai_buoc=false đầu tiên thì trả stage đó và dừng.\n"
                "- Không dùng đáp án cuối đúng để hợp thức hóa transition sai.\n"
                "- Không thêm, sửa hoặc đổi solution_index/order.\n"
                "- Chỉ trả JSON đúng schema.\n"
                "</rang_buoc_cuoi>"
            ),
        },
    ]


def build_transition_stages(ordered_solution: dict[str, Any]) -> dict[str, Any]:
    """Ghép các state liền kề thành stage cố định để Judge chỉ việc kiểm tra."""

    stages: list[dict[str, Any]] = []
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
        stages.append({
            "solution_index": solution_index,
            "stage": 0,
            "source_path": states[0]["source_path"],
            "noi_dung": states[0]["source_text"],
        })
        for previous, current in zip(states, states[1:]):
            stages.append({
                "solution_index": solution_index,
                "stage": int(current["order"]),
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
) -> tuple[dict[str, Any] | None, str]:
    """Chỉ kiểm tra contract và vị trí transition do LLM kết luận."""

    try:
        output = TransitionJudgeOutput.model_validate(parsed).model_dump()
    except ValidationError as exc:
        return None, validation_error_text(exc)

    expected: list[tuple[int, int, int]] = []
    solution_indexes = sorted({int(state["solution_index"]) for state in ordered_solution.get("states") or []})
    for solution_index in solution_indexes:
        states = [
            state
            for state in ordered_solution.get("states") or []
            if int(state["solution_index"]) == solution_index
        ]
        expected.extend(
            (solution_index, order, order + 1)
            for order in range(len(states) - 1)
        )
    verdict = output["verdict"]
    stage_results = output["stage_results"]
    if len(stage_results) > len(expected):
        return None, "Judge trả nhiều stage hơn input."
    for position, stage_result in enumerate(stage_results):
        key = (
            stage_result["solution_index"],
            stage_result["from_order"],
            stage_result["to_order"],
        )
        if key != expected[position] or stage_result["stage"] != stage_result["to_order"]:
            return None, "stage_result không đúng thứ tự hoặc không trỏ tới hai state liền kề."
        if not stage_result["kiem_tra_lap_luan"].strip():
            return None, "Mỗi stage_result phải có kiem_tra_lap_luan."

    failed = [position for position, item in enumerate(stage_results) if not item["trang_thai_buoc"]]
    if len(failed) > 1 or (failed and failed[0] != len(stage_results) - 1):
        return None, "Judge phải dừng ngay sau stage có trang_thai_buoc=false."
    if not failed and len(stage_results) != len(expected):
        return None, "Judge chưa kiểm tra đủ các stage theo thứ tự."

    if verdict == "good":
        if failed:
            return None, "Kết luận good không được có stage sai."
        output["reason"] = ""
        output["suggestion"] = ""
        output["first_invalid_transition"] = None
        return output, ""
    if not output["reason"].strip() or not output["suggestion"].strip():
        return None, "Kết luận bad/uncertain phải có reason và suggestion."
    if failed:
        failed_stage = stage_results[failed[0]]
        if not failed_stage["reason"].strip():
            return None, "Stage sai phải có reason."
        output["first_invalid_transition"] = {
            "solution_index": failed_stage["solution_index"],
            "from_order": failed_stage["from_order"],
            "to_order": failed_stage["to_order"],
        }
    else:
        output["first_invalid_transition"] = None
    return output, ""


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
        return None, str(exc)
    content = str(response.get("content") or "")
    fenced_candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.IGNORECASE | re.DOTALL)
    candidates = list(reversed(fenced_candidates)) if fenced_candidates else [content]
    last_error = "Splitter không trả JSON hợp lệ."
    for candidate in candidates:
        parsed, parse_ok, parse_error = parse_json_output(candidate)
        if not parse_ok or parsed is None:
            last_error = parse_error or last_error
            continue
        split, error = validate_splitter_output(parsed, generated_question)
        if split is not None:
            return split, ""
        last_error = error
    return None, last_error


def _call_transition_judge(
    *,
    generated_question: dict[str, Any],
    ordered_solution: dict[str, Any],
    schema_validation_result: dict[str, Any],
    client: LLMClient,
    model: str,
    debug: bool,
) -> dict[str, Any]:
    messages = build_generated_question_judge_messages(
        generated_question,
        load_text(CRITERIA_PATH),
        ordered_solution,
    )
    prompt_chars = sum(len(message.get("content") or "") for message in messages)
    start = time.perf_counter()
    try:
        debug_llm_messages(step="solution_transition_judge", model=model, messages=messages, debug=debug)
        response = client.chat_completion(model=model, messages=messages, temperature=0)
        elapsed = float(response.get("latency_seconds") or (time.perf_counter() - start))
        debug_generated_question_call(
            step="solution_transition_judge",
            model=model,
            generated_question=generated_question,
            schema_validation_result=schema_validation_result,
            prompt_chars=prompt_chars,
            elapsed_seconds=elapsed,
            debug=debug,
        )
        content = str(response.get("content") or "")
        fenced_candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.IGNORECASE | re.DOTALL)
        candidates = list(reversed(fenced_candidates)) if fenced_candidates else [content]
        last_error = "Gemma không trả JSON hợp lệ."
        for candidate in candidates:
            parsed, parse_ok, parse_error = parse_json_output(candidate)
            if not parse_ok or parsed is None:
                last_error = parse_error or last_error
                continue
            output, error = validate_transition_judge_output(parsed, ordered_solution)
            if output is not None:
                return {"contract_valid": True, **output}
            last_error = error
        return {"contract_valid": False, "contract_error": last_error}
    except Exception as exc:
        debug_generated_question_call(
            step="solution_transition_judge_error",
            model=model,
            generated_question=generated_question,
            schema_validation_result=schema_validation_result,
            prompt_chars=prompt_chars,
            elapsed_seconds=time.perf_counter() - start,
            debug=debug,
        )
        return {"contract_valid": False, "contract_error": str(exc)}


def _first_transition_problem(result: dict[str, Any]) -> dict[str, Any] | None:
    if result.get("verdict") == "good":
        return None
    transition = result.get("first_invalid_transition")
    return {**transition, "verdict": result["verdict"]} if transition else None


def _transition_issue(
    judge_result: dict[str, Any],
    ordered_solution: dict[str, Any],
) -> dict[str, Any]:
    transition = judge_result.get("first_invalid_transition")
    state = next(
        (
            state
            for state in ordered_solution["states"]
            if transition
            and state["solution_index"] == transition["solution_index"]
            and state["order"] == transition["to_order"]
        ),
        None,
    )
    return {
        "severity": "needs_review",
        "category": "solution_quality",
        "location": state["source_path"] if state else "/solutions",
        "reason": judge_result["reason"],
        "suggestion": judge_result["suggestion"],
        "repair_intent": "needs_manual_review",
    }


def _single_transition_judge_result(
    judge_result: dict[str, Any],
    ordered_solution: dict[str, Any],
    generated_question: dict[str, Any],
    *,
    strict_mode: bool,
    index: int,
) -> dict[str, Any]:
    """Chuyển một output Qwen hợp lệ về result hiện tại mà không lộ contract nội bộ."""

    problem = _first_transition_problem(judge_result)
    is_good = judge_result["verdict"] == "good"
    payload = {
        "is_good": is_good,
        "issues": [] if is_good else [_transition_issue(judge_result, ordered_solution)],
    }
    selected = (
        {"from_order": problem["from_order"], "to_order": problem["to_order"]}
        if problem
        else None
    )
    result = normalize_generated_question_result(
        payload,
        strict_mode=strict_mode,
        generated_question=generated_question,
        index=index,
    )
    result["_judge_transition_decision"] = {
        "decision_source": "qwen_fallback",
        "confirmed_by_both": False,
        "selected_transition": selected,
    }
    return result


def aggregate_transition_judge_results(
    results: list[dict[str, Any]],
    ordered_solution: dict[str, Any],
    generated_question: dict[str, Any],
    *,
    strict_mode: bool,
    index: int,
) -> dict[str, Any]:
    valid_results = [result for result in results if result.get("contract_valid")]
    decision_source = "contract_failure"
    confirmed_by_both = False
    selected: dict[str, Any] | None = None

    if not valid_results:
        result = fail_closed_output(
            "Cả hai Gemma trả output transition không đúng contract.",
            generated_question=generated_question,
            index=index,
        )
    elif len(valid_results) == 1:
        judge_result = valid_results[0]
        problem = _first_transition_problem(judge_result)
        if judge_result["verdict"] == "good":
            result = fail_closed_output(
                "Một Gemma sai contract; kết quả Gemma còn lại không đủ để xác nhận lời giải đúng.",
                generated_question=generated_question,
                index=index,
            )
        else:
            selected = problem
            result = normalize_generated_question_result(
                {"is_good": False, "issues": [_transition_issue(judge_result, ordered_solution)]},
                strict_mode=strict_mode,
                generated_question=generated_question,
                index=index,
            )
    else:
        non_good = [item for item in valid_results if item["verdict"] != "good"]
        if non_good:
            with_transition = [
                (item, problem)
                for item in non_good
                if (problem := _first_transition_problem(item)) is not None
            ]
            if with_transition:
                selected_result, selected = min(
                    with_transition,
                    key=lambda pair: (pair[1]["to_order"], pair[1]["solution_index"]),
                )
            else:
                selected_result = non_good[0]
            same_transition = len(non_good) == 2 and all(
                item.get("first_invalid_transition")
                == selected_result.get("first_invalid_transition")
                for item in non_good
            )
            confirmed_by_both = (
                same_transition
                and all(item["verdict"] == "bad" for item in non_good)
            )
            decision_source = "gemma_agreement" if confirmed_by_both else "gemma_disagreement"
            result = normalize_generated_question_result(
                {"is_good": False, "issues": [_transition_issue(selected_result, ordered_solution)]},
                strict_mode=strict_mode,
                generated_question=generated_question,
                index=index,
            )
        else:
            decision_source = "gemma_agreement"
            confirmed_by_both = True
            result = normalize_generated_question_result(
                {"is_good": True, "issues": []},
                strict_mode=strict_mode,
                generated_question=generated_question,
                index=index,
            )

    result["_judge_transition_decision"] = {
        "decision_source": decision_source,
        "confirmed_by_both": confirmed_by_both,
        "selected_transition": (
            {"from_order": selected["from_order"], "to_order": selected["to_order"]}
            if selected
            else None
        ),
    }
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
    """Solution Judge: Gemma/Qwen split, rồi hai Gemma Judge chạy độc lập."""

    fast_model = generated_question_fast_model(config)
    reasoning_model = generated_question_reasoning_model(config)
    split, gemma_split_error = _call_solution_splitter(
        generated_question,
        client,
        fast_model,
        debug=debug,
    )
    splitter_attempts = 1
    splitter_fallback_called = False
    if split is None:
        split_error = gemma_split_error
        if (
            getattr(config, "use_fallback_judge", True)
            and reasoning_model
            and reasoning_model != fast_model
        ):
            splitter_fallback_called = True
            splitter_attempts += 1
            split, split_error = _call_solution_splitter(
                generated_question,
                client,
                reasoning_model,
                debug=debug,
            )
        if split is None:
            result = fail_closed_output(
                "Không tách được ordered states nguyên văn: " + split_error,
                generated_question=generated_question,
                index=index,
            )
            result.update(
                judge_model=reasoning_model if splitter_fallback_called else fast_model,
                judge_attempt_count=splitter_attempts,
                judge_fallback_called=splitter_fallback_called,
                judge_gemma_run_count=0,
                judge_gemma_agreement=None,
                judge_fallback_reason="splitter_contract_failure",
            )
            return result

    ordered_solution = _prepend_problem_context(split, generated_question)

    def call_gemma() -> dict[str, Any]:
        return _call_transition_judge(
            generated_question=generated_question,
            ordered_solution=ordered_solution,
            schema_validation_result=schema_validation_result,
            client=client,
            model=fast_model,
            debug=debug,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        gemma_results = [future.result() for future in [executor.submit(call_gemma) for _ in range(2)]]
    gemma_contract_failure = any(not item.get("contract_valid") for item in gemma_results)
    qwen_judge_called = False
    qwen_result: dict[str, Any] | None = None
    if (
        gemma_contract_failure
        and getattr(config, "use_fallback_judge", True)
        and reasoning_model
        and reasoning_model != fast_model
    ):
        qwen_judge_called = True
        qwen_result = _call_transition_judge(
            generated_question=generated_question,
            ordered_solution=ordered_solution,
            schema_validation_result=schema_validation_result,
            client=client,
            model=reasoning_model,
            debug=debug,
        )
    if qwen_result and qwen_result.get("contract_valid"):
        result = _single_transition_judge_result(
            qwen_result,
            ordered_solution,
            generated_question,
            strict_mode=strict_mode,
            index=index,
        )
    else:
        result = aggregate_transition_judge_results(
            gemma_results,
            ordered_solution,
            generated_question,
            strict_mode=strict_mode,
            index=index,
        )
    decision = result["_judge_transition_decision"]
    result.update(
        judge_model=reasoning_model if qwen_judge_called else fast_model,
        judge_attempt_count=splitter_attempts + 2 + int(qwen_judge_called),
        judge_fallback_called=splitter_fallback_called or qwen_judge_called,
        judge_gemma_run_count=2,
        judge_gemma_agreement=decision["decision_source"] == "gemma_agreement",
        judge_fallback_reason=(
            "gemma_contract_failure"
            if qwen_judge_called
            else "gemma_splitter_contract_failure" if splitter_fallback_called else None
        ),
    )
    return result

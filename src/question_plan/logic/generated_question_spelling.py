"""Spelling/wording checker cho text nodes trong generated question object."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ..infra.config import AppConfig
from ..infra.debug import debug_llm_messages
from ..infra.llm_client import LLMClient
from ..shared.utils import parse_json_output
from .generated_question_schema import location_to_json_pointer, make_issue


KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"
SPELLING_RULES_PATH = KNOWLEDGE_DIR / "generated_question_spelling_rules.md"
SPELLING_OUTPUT_SCHEMA_PATH = KNOWLEDGE_DIR / "generated_question_spelling_output_schema.md"
SPELLING_WHITELIST_PATH = KNOWLEDGE_DIR / "spelling_whitelist_vi.txt"
MATH_PATTERN = re.compile(r"(\$\$.*?\$\$|\$.*?\$|```.*?```)", re.DOTALL)
VIETNAMESE_ACCENT_CHARS = set("àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_whitelist() -> set[str]:
    if not SPELLING_WHITELIST_PATH.exists():
        return set()
    return {
        line.strip().lower()
        for line in SPELLING_WHITELIST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def protect_math_segments(text: str) -> tuple[str, list[str]]:
    segments: list[str] = []

    def replace(match: re.Match[str]) -> str:
        placeholder = f"__MATH_SEGMENT_{len(segments)}__"
        segments.append(match.group(0))
        return placeholder

    return MATH_PATTERN.sub(replace, text), segments


def restore_math_segments(text: str, segments: list[str]) -> str:
    restored = text
    for index, segment in enumerate(segments):
        restored = restored.replace(f"__MATH_SEGMENT_{index}__", segment)
    return restored


def get_nested_text_nodes(blocks: Any, base_path: str, role: str, interaction_type: str = "") -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    if not isinstance(blocks, list):
        return nodes
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        path = f"{base_path}/{index}"
        if isinstance(block.get("text"), str):
            nodes.append(
                {
                    "path": f"{path}/text",
                    "text": block["text"],
                    "role": role,
                    "interaction_type": interaction_type,
                }
            )
        content = block.get("content")
        if isinstance(content, list):
            nodes.extend(get_nested_text_nodes(content, f"{path}/content", role, interaction_type))
    return nodes


def extract_text_nodes_for_spelling(generated_question: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    nodes.extend(get_nested_text_nodes(generated_question.get("instruction"), "/instruction", "instruction"))

    for item_index, item in enumerate(generated_question.get("questionItems") or []):
        if not isinstance(item, dict):
            continue
        item_path = f"/questionItems/{item_index}"
        nodes.extend(get_nested_text_nodes(item.get("stem"), f"{item_path}/stem", "stem"))
        for interaction_index, interaction in enumerate(item.get("interactions") or []):
            if not isinstance(interaction, dict):
                continue
            interaction_type = str(interaction.get("type") or interaction.get("interactionType") or "").strip()
            options = ((interaction.get("config") or {}).get("options") if isinstance(interaction.get("config"), dict) else None)
            if isinstance(options, list):
                for option_index, option in enumerate(options):
                    if isinstance(option, dict):
                        nodes.extend(
                            get_nested_text_nodes(
                                option.get("content"),
                                f"{item_path}/interactions/{interaction_index}/config/options/{option_index}/content",
                                "option",
                                interaction_type,
                            )
                        )
        for hint_index, hint in enumerate(item.get("hints") or []):
            if isinstance(hint, dict):
                nodes.extend(get_nested_text_nodes(hint.get("content"), f"{item_path}/hints/{hint_index}/content", "hint"))

    for solution_index, solution in enumerate(generated_question.get("solutions") or []):
        if isinstance(solution, dict):
            nodes.extend(
                get_nested_text_nodes(
                    solution.get("solutionContent"),
                    f"/solutions/{solution_index}/solutionContent",
                    "solution",
                )
            )
    return nodes


def spelling_issue(
    *,
    severity: str,
    location: str,
    reason: str,
    suggestion: str,
    error_snippet: str,
    corrected_text: str | None = None,
) -> dict[str, Any]:
    issue = make_issue(
        severity=severity,
        category="spelling_wording",
        location=location,
        reason=reason,
        suggestion=suggestion,
        error_snippet=error_snippet,
        repair_intent="fix_spelling_wording",
    )
    if corrected_text is not None:
        issue["corrected_text"] = corrected_text
    return issue


def rule_based_text_hygiene_issues(node: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(node.get("text") or "")
    protected, _segments = protect_math_segments(text)
    location = str(node.get("path") or "")
    role = str(node.get("role") or "")
    issues: list[dict[str, Any]] = []

    if role in {"instruction", "stem"} and len(protected.strip()) < 5:
        issues.append(
            spelling_issue(
                severity="needs_review",
                location=location,
                reason="Text quan trọng quá ngắn hoặc gần như rỗng.",
                suggestion="Review và bổ sung diễn đạt rõ ràng hơn.",
                error_snippet=text[:120],
            )
        )
    if re.search(r"[^\S\r\n]{2,}", protected):
        issues.append(
            spelling_issue(
                severity="warning",
                location=location,
                reason="Text có nhiều khoảng trắng liên tiếp.",
                suggestion="Rút gọn khoảng trắng thừa.",
                error_snippet=text[:160],
                corrected_text=re.sub(r"[^\S\r\n]{2,}", " ", text),
            )
        )
    if re.search(r"\s+([,.;:!?])", protected):
        issues.append(
            spelling_issue(
                severity="warning",
                location=location,
                reason="Text có khoảng trắng trước dấu câu.",
                suggestion="Xóa khoảng trắng trước dấu câu.",
                error_snippet=text[:160],
                corrected_text=re.sub(r"\s+([,.;:!?])", r"\1", text),
            )
        )
    if re.search(r"([,.;:!?])(?=[^\s\d,.;:!?])", protected):
        issues.append(
            spelling_issue(
                severity="warning",
                location=location,
                reason="Text có dấu câu không được theo sau bởi khoảng trắng.",
                suggestion="Bổ sung khoảng trắng sau dấu câu nếu đó là câu tiếng Việt.",
                error_snippet=text[:160],
            )
        )
    if re.search(r"([,.;!?])\1{1,}", protected):
        issues.append(
            spelling_issue(
                severity="warning",
                location=location,
                reason="Text có dấu câu lặp bất thường.",
                suggestion="Review và bỏ dấu câu lặp nếu không có chủ ý.",
                error_snippet=text[:160],
            )
        )
    repeated_word_match = re.search(r"\b([A-Za-zÀ-ỹ]+)\s+\1\b", protected, flags=re.IGNORECASE)
    if repeated_word_match:
        issues.append(
            spelling_issue(
                severity="warning",
                location=location,
                reason=f"Text có từ lặp liên tiếp: `{repeated_word_match.group(0)}`.",
                suggestion="Xóa từ bị lặp nếu không có chủ ý.",
                error_snippet=text[:160],
                corrected_text=re.sub(r"\b([A-Za-zÀ-ỹ]+)\s+\1\b", r"\1", text, flags=re.IGNORECASE),
            )
        )
    if any(token in protected for token in ("�", "Ã", "Â", "Æ", "á»", "Ä")):
        issues.append(
            spelling_issue(
                severity="needs_review",
                location=location,
                reason="Text có dấu hiệu lỗi encoding/mojibake.",
                suggestion="Review lại encoding tiếng Việt của text.",
                error_snippet=text[:160],
            )
        )
    if len(protected) > 650 and protected.count(".") + protected.count("?") + protected.count("!") <= 1:
        issues.append(
            spelling_issue(
                severity="needs_review",
                location=location,
                reason="Câu quá dài, khó đọc hoặc thiếu ngắt câu.",
                suggestion="Tách câu hoặc thêm dấu câu phù hợp, không đổi nội dung toán.",
                error_snippet=text[:180],
            )
        )
    if abs(protected.count("(") - protected.count(")")) >= 2:
        issues.append(
            spelling_issue(
                severity="needs_review",
                location=location,
                reason="Dấu ngoặc thường có vẻ mất cân bằng.",
                suggestion="Review lại cặp dấu ngoặc trong câu.",
                error_snippet=text[:180],
            )
        )
    return issues


def dictionary_domain_issues(node: dict[str, Any], whitelist: set[str]) -> list[dict[str, Any]]:
    text = str(node.get("text") or "")
    protected, _segments = protect_math_segments(text)
    if not protected.strip() or len(protected) < 20:
        return []
    lowered = protected.lower()
    for phrase in whitelist:
        lowered = lowered.replace(phrase, " ")
    words = re.findall(r"[A-Za-zÀ-ỹ_]{3,}", lowered)
    if not words:
        return []
    ascii_words = [word for word in words if word.isascii() and "_" not in word]
    if len(ascii_words) >= 6 and not any(char in lowered for char in VIETNAMESE_ACCENT_CHARS):
        return [
            spelling_issue(
                severity="needs_review",
                location=str(node.get("path") or ""),
                reason="Text tiếng Việt có dấu hiệu mất dấu hoặc chứa quá nhiều token Latin lạ.",
                suggestion="Review chính tả tiếng Việt, giữ nguyên biến/ký hiệu toán.",
                error_snippet=text[:180],
            )
        ]
    return []


def build_generated_question_spelling_messages(
    text_nodes: list[dict[str, Any]],
    rules_text: str,
    output_schema_text: str,
) -> list[dict[str, str]]:
    payload_nodes = []
    for node in text_nodes:
        protected_text, segments = protect_math_segments(str(node.get("text") or ""))
        payload_nodes.append(
            {
                "path": node.get("path"),
                "role": node.get("role"),
                "interaction_type": node.get("interaction_type") or "",
                "text": node.get("text"),
                "math_protected_text": protected_text,
                "math_segments": segments,
            }
        )
    language_policy = "Mọi chuỗi người đọc trong JSON output phải viết bằng tiếng Việt có dấu."
    return [
        {
            "role": "system",
            "content": (
                "Bạn là spelling/wording judge cho generated question object. "
                "Chỉ kiểm tra chính tả/diễn đạt của text nodes được cung cấp. "
                "Không sửa nội dung toán, đáp án, answerSpecs hoặc LaTeX. "
                f"{language_policy}"
            ),
        },
        {
            "role": "user",
            "content": (
                "Hãy kiểm tra spelling/wording theo rules và output schema.\n\n"
                "Bắt buộc:\n"
                "- Chỉ báo lỗi chính tả/diễn đạt.\n"
                "- Không thay đổi math segments hoặc LaTeX.\n"
                "- Không đổi nghĩa chuyên môn.\n"
                "- Không trả markdown hoặc giải thích ngoài JSON.\n"
                f"- {language_policy}\n\n"
                "SPELLING RULES:\n"
                f"{rules_text}\n\n"
                "OUTPUT SCHEMA:\n"
                f"{output_schema_text}\n\n"
                "TEXT NODES:\n"
                f"{json.dumps(payload_nodes, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def normalize_spelling_issue(issue: Any, index: int) -> dict[str, Any]:
    data = issue if isinstance(issue, dict) else {}
    normalized = spelling_issue(
        severity=str(data.get("severity") or "warning").strip(),
        location=str(data.get("location") or f"spelling.issues[{index}]").strip(),
        reason=str(data.get("reason") or "Phát hiện nghi vấn chính tả/diễn đạt.").strip(),
        suggestion=str(data.get("suggestion") or "Review lại text node.").strip(),
        error_snippet=str(data.get("error_snippet") or "").strip(),
        corrected_text=data.get("corrected_text") if isinstance(data.get("corrected_text"), str) else None,
    )
    normalized["category"] = "spelling_wording"
    normalized["repair_intent"] = "fix_spelling_wording"
    return normalized


def normalize_spelling_llm_result(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {
            "spelling_status": "checked",
            "llm_checked": True,
            "issues": [
                spelling_issue(
                    severity="needs_review",
                    location="spelling",
                    reason="LLM spelling judge không trả JSON object hợp lệ.",
                    suggestion="Review thủ công hoặc chạy lại spelling judge.",
                    error_snippet="",
                )
            ],
        }
    return {
        "spelling_status": "checked",
        "llm_checked": True,
        "issues": [normalize_spelling_issue(issue, index) for index, issue in enumerate(parsed.get("issues") or [])],
    }


def check_spelling_and_wording(
    generated_question: dict[str, Any],
    *,
    config: AppConfig | None = None,
    client: LLMClient | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    nodes = extract_text_nodes_for_spelling(generated_question)
    whitelist = load_whitelist()
    rule_issues: list[dict[str, Any]] = []
    for node in nodes:
        rule_issues.extend(rule_based_text_hygiene_issues(node))
        rule_issues.extend(dictionary_domain_issues(node, whitelist))

    if not rule_issues:
        return {
            "spelling_status": "checked",
            "llm_checked": False,
            "text_node_count": len(nodes),
            "issues": rule_issues,
        }

    if config is None or client is None:
        return {
            "spelling_status": "checked",
            "llm_checked": False,
            "text_node_count": len(nodes),
            "issues": rule_issues,
        }

    candidate_nodes = [node for node in nodes if any(issue.get("location") == node.get("path") for issue in rule_issues)]
    if not candidate_nodes:
        candidate_nodes = nodes[:20]
    rules_text = load_text(SPELLING_RULES_PATH)
    output_schema_text = load_text(SPELLING_OUTPUT_SCHEMA_PATH)
    messages = build_generated_question_spelling_messages(candidate_nodes, rules_text, output_schema_text)
    start = time.perf_counter()
    try:
        debug_llm_messages(step="generated_question_spelling", model=config.primary_judge_model, messages=messages, debug=debug)
        response = client.chat_completion(model=config.primary_judge_model, messages=messages, temperature=0)
        parsed, ok, parse_error = parse_json_output(str(response.get("content") or ""))
        if not ok:
            llm_result = normalize_spelling_llm_result(
                {
                    "issues": [
                        {
                            "severity": "needs_review",
                            "location": "spelling",
                            "reason": parse_error or "Không parse được output spelling judge.",
                            "suggestion": "Review thủ công hoặc chạy lại spelling judge.",
                        }
                    ]
                }
            )
        else:
            llm_result = normalize_spelling_llm_result(parsed)
        merged = [*rule_issues]
        seen = {(issue.get("location"), issue.get("reason")) for issue in merged}
        for issue in llm_result.get("issues") or []:
            key = (issue.get("location"), issue.get("reason"))
            if key not in seen:
                merged.append(issue)
                seen.add(key)
        return {
            "spelling_status": "checked",
            "llm_checked": True,
            "text_node_count": len(nodes),
            "spelling_latency_seconds": response.get("latency_seconds", time.perf_counter() - start),
            "issues": merged,
        }
    except Exception as exc:
        return {
            "spelling_status": "checked",
            "llm_checked": False,
            "text_node_count": len(nodes),
            "issues": [
                *rule_issues,
                spelling_issue(
                    severity="needs_review",
                    location="spelling",
                    reason=f"Spelling judge gặp lỗi runtime: {exc}",
                    suggestion="Review thủ công hoặc chạy lại spelling judge.",
                    error_snippet="",
                ),
            ],
        }

"""Tiện ích nhỏ dùng chung trong nhiều pipeline.

File này chứa helper parse JSON từ output model, đọc JSON list, tạo filename an
toàn và chuẩn hóa id/question object cho các script chạy chính.
"""

import json
import re
from pathlib import Path
from typing import Any


def parse_json_output(content: str) -> tuple[dict[str, Any] | None, bool, str | None]:
    text = content.strip()
    candidates = [text]
    if "```" in text:
        stripped = text.replace("```json", "```")
        parts = stripped.split("```")
        candidates.extend(part.strip() for part in parts if part.strip())

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed, True, None
    return None, False, "Không parse được output của model thành JSON."


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} phải chứa một JSON list.")
    return data


def safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "unknown"


def extract_question_object(case: dict[str, Any]) -> dict[str, Any]:
    question_object = case.get("question_object")
    if isinstance(question_object, dict):
        return question_object
    return case


def normalize_case_id(case: dict[str, Any]) -> str:
    question_object = extract_question_object(case)
    return str(case.get("case_id") or case.get("id") or question_object.get("id") or "unknown")


def expected_is_valid(case: dict[str, Any]) -> bool | None:
    if "expected_is_valid" in case:
        return case["expected_is_valid"]
    expected = case.get("expected")
    if isinstance(expected, dict):
        return expected.get("is_valid")
    return None

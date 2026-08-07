"""Debug an toàn cho prompt gửi tới LLM."""

from __future__ import annotations

import json
import os
import sys
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def llm_prompt_debug_enabled(explicit_debug: bool = False) -> bool:
    if explicit_debug:
        return True
    return os.getenv("DEBUG_LLM_PROMPT", "").strip().lower() in TRUE_VALUES


def debug_llm_messages(
    *,
    step: str,
    model: str,
    messages: list[dict[str, Any]],
    debug: bool = False,
) -> None:
    """In metadata prompt, không in full nội dung record/prompt."""

    if not llm_prompt_debug_enabled(debug):
        return

    summary: dict[str, Any] = {
        "step": step,
        "model": model,
        "message_count": len(messages),
        "messages": [],
    }
    for index, message in enumerate(messages, start=1):
        content = message.get("content") or ""
        text_content = "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ) if isinstance(content, list) else str(content)
        image_count = sum(
            1
            for part in content
            if isinstance(part, dict) and part.get("type") == "image_url"
        ) if isinstance(content, list) else 0
        summary["messages"].append(
            {
                "index": index,
                "role": message.get("role", ""),
                "content_chars": len(text_content),
                "content_lines": text_content.count("\n") + 1,
                "image_count": image_count,
            }
        )
    print("[DEBUG_LLM_PROMPT] " + json.dumps(summary, ensure_ascii=False), file=sys.stderr)


def debug_loop_event(*, event: str, debug: bool = False, **metadata: Any) -> None:
    """In metadata vòng loop/refinement khi debug được bật."""

    if not llm_prompt_debug_enabled(debug):
        return

    summary = {"event": event, **metadata}
    print("[DEBUG_QUESTION_PLAN_LOOP] " + json.dumps(summary, ensure_ascii=False), file=sys.stderr)

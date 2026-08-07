"""Standalone Chandra OCR 2 adapter for OpenAI-compatible endpoints."""

from __future__ import annotations

from typing import Any

from ..infra.llm_client import ApiError, LLMClient
from .visual_context import (
    LoadedVisualAsset,
    VisualExtractionResult,
    classify_visual_type,
    extract_chandra_structured_analysis,
    extract_coordinates,
    extract_text_facts,
)


CHANDRA_EXTRACTION_PROMPT = """Convert this image into faithful structured Markdown.
Preserve every visible Vietnamese/English text, mathematical expression, label,
table cell, row and column. For a graph, chart, geometry figure or diagram,
describe only visible axes, labels, marked points, values, arrows and relations.
Do not solve the exercise, infer hidden facts or judge a solution. Return only
the extracted Markdown/HTML content, with no preamble."""


class ChandraVisualExtractor:
    """Call Chandra once and normalize its OCR text into the comparison contract."""

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str = "chandra-ocr-2",
        max_tokens: int = 8192,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max(256, min(int(max_tokens), 36_000))

    def extract(self, asset: LoadedVisualAsset) -> VisualExtractionResult:
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CHANDRA_EXTRACTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": asset.data_url(), "detail": "high"},
                    },
                ],
            }
        ]
        try:
            response = self.client.chat_completion(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=self.max_tokens,
                retry_transient_once=True,
            )
        except ApiError as exc:
            return VisualExtractionResult(
                extractor="chandra",
                status="runtime_error",
                source=asset.reference.source,
                mime_type=asset.mime_type,
                sha256=asset.sha256,
                width=asset.width,
                height=asset.height,
                model=self.model,
                error=str(exc),
            )

        content = str(response.get("content") or "").strip()
        context = "\n".join(
            part
            for part in (asset.reference.alt_text.strip(), content)
            if part
        )
        structured_analysis = extract_chandra_structured_analysis(content)
        return VisualExtractionResult(
            extractor="chandra",
            status="extracted" if content else "partial",
            source=asset.reference.source,
            mime_type=asset.mime_type,
            sha256=asset.sha256,
            width=asset.width,
            height=asset.height,
            visual_type=classify_visual_type(context),
            transcription=content,
            facts=extract_text_facts(content),
            coordinates=extract_coordinates(content),
            signals={
                "input_mode": "data_url",
                "http_retry_count": int(response.get("http_retry_count") or 0),
                "endpoint": response.get("endpoint"),
                "structured_analysis": structured_analysis,
                "structured_analysis_block_count": len(structured_analysis),
            },
            model=str(response.get("model") or self.model),
            latency_seconds=float(response.get("latency_seconds") or 0),
        )

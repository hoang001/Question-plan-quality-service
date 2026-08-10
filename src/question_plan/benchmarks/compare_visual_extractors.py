"""Compare Chandra OCR 2 with the deterministic code vision baseline.

This command is standalone and has no imports from the generated-question flow.
It accepts an image URL/path or a JSON fixture containing Markdown image links.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..infra.config import load_config
from ..infra.llm_client import LLMClient
from ..vision import (
    ChandraVisualExtractor,
    CodeVisionParser,
    extract_asset_references,
    load_visual_asset,
)
from ..vision.visual_context import direct_asset_reference


SERVICE_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="So sánh Chandra OCR 2 với code vision parser độc lập."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="File JSON, file ảnh local hoặc URL ảnh.",
    )
    parser.add_argument("--output", help="Đường dẫn JSON report.")
    parser.add_argument("--model", default="chandra-ocr-2")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Không gọi Chandra; chỉ chạy baseline code.",
    )
    return parser.parse_args()


def load_references(input_value: str):
    parsed = urlparse(input_value)
    if parsed.scheme in {"http", "https"} or input_value.startswith("data:image/"):
        return [direct_asset_reference(input_value)], None

    input_path = Path(input_value).expanduser().resolve()
    if not input_path.exists():
        raise ValueError(f"Không tìm thấy input: {input_path}")
    if input_path.suffix.lower() != ".json":
        return [direct_asset_reference(str(input_path))], input_path

    with input_path.open("r", encoding="utf-8-sig") as file:
        payload: Any = json.load(file)
    references = []
    if isinstance(payload, list):
        for item in payload:
            references.extend(extract_asset_references(item))
    else:
        references.extend(extract_asset_references(payload))
    return references, input_path


def default_output_path(input_path: Path | None) -> Path:
    if input_path is not None:
        return input_path.with_name(f"{input_path.stem}_visual_comparison.json")
    return Path.cwd() / "visual_comparison.json"


def compare(
    input_value: str,
    *,
    model: str,
    limit: int,
    max_tokens: int,
    code_only: bool,
) -> dict[str, Any]:
    references, input_path = load_references(input_value)
    selected = references[: max(0, int(limit))]
    code_parser = CodeVisionParser()
    chandra = None
    if not code_only:
        config = load_config(SERVICE_ROOT)
        chandra = ChandraVisualExtractor(
            LLMClient(config), model=model, max_tokens=max_tokens
        )

    rows: list[dict[str, Any]] = []
    for index, reference in enumerate(selected, start=1):
        print(
            f"[{index}/{len(selected)}] {reference.question_id or '-'} {reference.source}",
            file=sys.stderr,
        )
        row: dict[str, Any] = {"asset": reference.model_dump()}
        try:
            asset = load_visual_asset(reference)
        except Exception as exc:  # report per asset; do not abort a comparison batch
            row["load_error"] = str(exc)
            rows.append(row)
            continue
        row["asset_metadata"] = {
            "mime_type": asset.mime_type,
            "sha256": asset.sha256,
            "width": asset.width,
            "height": asset.height,
            "image_format": asset.image_format,
            "byte_count": len(asset.content),
        }
        row["code"] = code_parser.parse(asset).model_dump()
        if chandra is not None:
            row["chandra"] = chandra.extract(asset).model_dump()
        rows.append(row)

    return {
        "input": input_value,
        "model": None if code_only else model,
        "asset_count_found": len(references),
        "asset_count_compared": len(selected),
        "results": rows,
        "notes": {
            "code_parser": "Chỉ metadata, alt text và tín hiệu raster đơn giản; không OCR/suy luận toán.",
            "chandra": "OCR/caption có cấu trúc; không được yêu cầu giải hoặc chấm bài.",
            "pipeline_integrated": False,
        },
        "suggested_output": str(default_output_path(input_path)),
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    try:
        report = compare(
            args.input,
            model=args.model,
            limit=args.limit,
            max_tokens=args.max_tokens,
            code_only=args.code_only,
        )
        output = (
            Path(args.output).expanduser().resolve()
            if args.output
            else Path(report["suggested_output"]).resolve()
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(output)
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

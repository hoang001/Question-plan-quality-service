"""Deterministic asset loading and visual baseline extraction.

The code parser deliberately stays conservative: it extracts metadata and
simple raster/layout signals, but never claims to understand a mathematical
diagram. Chandra or another VLM remains responsible for OCR/captioning.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import mimetypes
import re
import socket
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote_to_bytes, urljoin, urlparse

import requests
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field


MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_REDIRECTS = 3
MARKDOWN_IMAGE_PATTERN = re.compile(
    r"!\[(?P<alt>.*?)\]\((?P<url>https?://[^)\s]+)\)", re.DOTALL
)
DIRECT_IMAGE_URL_PATTERN = re.compile(
    r"https?://[^\s<>\])]+\.(?:png|jpe?g|webp|gif|bmp)(?:\?[^\s<>\])]*)?",
    re.IGNORECASE,
)
COORDINATE_PATTERN = re.compile(
    r"\(\s*(?P<x>[+-]?(?:\d+(?:[.,]\d+)?|\d*\.\d+))\s*[,;]\s*"
    r"(?P<y>[+-]?(?:\d+(?:[.,]\d+)?|\d*\.\d+))\s*\)"
)
JSON_COORDINATE_PATTERN = re.compile(
    r'[{,]\s*["\']x["\']\s*:\s*(?P<x>[+-]?\d+(?:\.\d+)?)\s*,\s*'
    r'["\']y["\']\s*:\s*(?P<y>[+-]?\d+(?:\.\d+)?)'
)
EQUATION_PATTERN = re.compile(
    r"(?<!\w)(?:[A-Za-z][A-Za-z0-9_]*\s*=\s*[^,;.\n]{1,80}|"
    r"\d+(?:[.,]\d+)?\s*[+−\-*/] ?\s*\d+(?:[.,]\d+)?\s*=\s*\d+(?:[.,]\d+)?)"
)


class AssetReference(BaseModel):
    """A visual asset found in a question object or supplied directly."""

    question_id: str | None = None
    source_path: str
    source: str
    alt_text: str = ""
    context_text: str = ""


class VisualExtractionResult(BaseModel):
    """Common serializable output used to compare both extractors."""

    extractor: Literal["code", "chandra"]
    status: Literal["extracted", "partial", "unavailable", "runtime_error"]
    source: str
    mime_type: str | None = None
    sha256: str | None = None
    width: int | None = None
    height: int | None = None
    visual_type: Literal[
        "table", "graph", "chart", "geometry", "diagram", "image", "unknown"
    ] = "unknown"
    transcription: str = ""
    facts: list[str] = Field(default_factory=list)
    coordinates: list[dict[str, float]] = Field(default_factory=list)
    signals: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None
    latency_seconds: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class LoadedVisualAsset:
    reference: AssetReference
    content: bytes
    mime_type: str
    sha256: str
    width: int | None
    height: int | None
    image_format: str | None

    def data_url(self) -> str:
        encoded = base64.b64encode(self.content).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _walk_strings(value: Any, path: str = ""):
    if isinstance(value, str):
        yield path or "/", value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{path}/{index}")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(item, f"{path}/{_json_pointer_token(str(key))}")


def extract_asset_references(value: Any) -> list[AssetReference]:
    """Find Markdown images/direct image URLs without mutating the input."""

    question_id = None
    if isinstance(value, dict):
        question_id = str(value.get("id") or value.get("_id") or "") or None
    references: list[AssetReference] = []
    seen: set[tuple[str, str]] = set()
    for path, text in _walk_strings(value):
        markdown_spans: list[tuple[int, int]] = []
        for match in MARKDOWN_IMAGE_PATTERN.finditer(text):
            markdown_spans.append(match.span())
            key = (path, match.group("url"))
            if key in seen:
                continue
            seen.add(key)
            references.append(
                AssetReference(
                    question_id=question_id,
                    source_path=path,
                    source=match.group("url"),
                    alt_text=match.group("alt").strip(),
                    context_text=text,
                )
            )
        for match in DIRECT_IMAGE_URL_PATTERN.finditer(text):
            if any(start <= match.start() < end for start, end in markdown_spans):
                continue
            key = (path, match.group(0))
            if key in seen:
                continue
            seen.add(key)
            references.append(
                AssetReference(
                    question_id=question_id,
                    source_path=path,
                    source=match.group(0),
                    context_text=text,
                )
            )
    return references


def attach_remote_visual_assets(
    messages: list[dict[str, Any]],
    value: Any,
) -> list[dict[str, Any]]:
    """Attach question images using the multimodal shape accepted by the LLM backend.

    The input records and original messages are not mutated. The same remote image
    is attached only once even when its Markdown reference occurs in several fields.
    Local/data assets are deliberately excluded here: production question objects
    currently expose public HTTP(S) assets, matching ``cli.py --ping``.
    """

    references: list[AssetReference] = []
    seen_sources: set[str] = set()
    for reference in extract_asset_references(value):
        source = reference.source.strip()
        if not source.startswith(("http://", "https://")) or source in seen_sources:
            continue
        seen_sources.add(source)
        references.append(reference)
    if not references:
        return messages

    copied = [dict(message) for message in messages]
    user_index = next(
        (
            index
            for index in range(len(copied) - 1, -1, -1)
            if copied[index].get("role") == "user"
        ),
        None,
    )
    if user_index is None or not isinstance(copied[user_index].get("content"), str):
        return messages

    manifest = "\n".join(
        f"- Ảnh {index}: source_path={reference.source_path}; asset_url={reference.source}"
        + (f" — mô tả text: {reference.alt_text}" if reference.alt_text else "")
        for index, reference in enumerate(references, start=1)
    )
    text_content = (
        str(copied[user_index]["content"])
        + "\n\n### ẢNH ĐƯỢC ĐÍNH KÈM TRỰC TIẾP\n"
        + manifest
        + "\nCác phần image_url ngay sau phần text này là nội dung ảnh thật theo đúng thứ tự trên. "
        "Hãy đọc ảnh khi quyết định; không coi ảnh là thiếu chỉ vì nguồn ban đầu là URL."
    )
    copied[user_index]["content"] = [
        {"type": "text", "text": text_content},
        *[
            {"type": "image_url", "image_url": {"url": reference.source}}
            for reference in references
        ],
    ]
    return copied


def direct_asset_reference(source: str) -> AssetReference:
    return AssetReference(source_path="/", source=source)


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Chỉ hỗ trợ URL ảnh http/https hợp lệ.")
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443)
        }
    except socket.gaierror as exc:
        raise ValueError(f"Không phân giải được host ảnh: {parsed.hostname}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("URL ảnh trỏ tới địa chỉ nội bộ/không public.")


def _read_response_bytes(response: requests.Response, max_bytes: int) -> bytes:
    declared = response.headers.get("Content-Length")
    if declared and int(declared) > max_bytes:
        raise ValueError(f"Ảnh vượt giới hạn {max_bytes} bytes.")
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        size += len(chunk)
        if size > max_bytes:
            raise ValueError(f"Ảnh vượt giới hạn {max_bytes} bytes.")
        chunks.append(chunk)
    return b"".join(chunks)


def _download_public_asset(
    url: str,
    *,
    timeout_seconds: int,
    max_bytes: int,
    session: requests.Session | None,
) -> tuple[bytes, str | None]:
    requester = session or requests.Session()
    current = url
    for redirect_index in range(MAX_REDIRECTS + 1):
        _validate_public_url(current)
        response = requester.get(
            current,
            timeout=timeout_seconds,
            stream=True,
            allow_redirects=False,
            headers={"Accept": "image/*"},
        )
        if response.status_code in {301, 302, 303, 307, 308}:
            if redirect_index == MAX_REDIRECTS:
                raise ValueError("URL ảnh redirect quá số lần cho phép.")
            location = response.headers.get("Location")
            if not location:
                raise ValueError("Redirect ảnh thiếu Location header.")
            current = urljoin(current, location)
            continue
        response.raise_for_status()
        return _read_response_bytes(response, max_bytes), response.headers.get(
            "Content-Type"
        )
    raise ValueError("Không tải được ảnh.")


def _decode_data_url(source: str, max_bytes: int) -> tuple[bytes, str]:
    header, separator, payload = source.partition(",")
    if not separator or not header.startswith("data:image/"):
        raise ValueError("Data URL phải chứa image MIME type.")
    mime_type = header[5:].split(";", 1)[0].lower()
    content = (
        base64.b64decode(payload, validate=True)
        if ";base64" in header
        else unquote_to_bytes(payload)
    )
    if len(content) > max_bytes:
        raise ValueError(f"Ảnh vượt giới hạn {max_bytes} bytes.")
    return content, mime_type


def _inspect_image(content: bytes) -> tuple[str, int, int, str]:
    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
        with Image.open(BytesIO(content)) as image:
            image_format = str(image.format or "").upper() or "UNKNOWN"
            mime_type = Image.MIME.get(image_format, "application/octet-stream")
            return mime_type, int(image.width), int(image.height), image_format
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Nội dung tải về không phải ảnh raster hợp lệ.") from exc


def load_visual_asset(
    reference: AssetReference,
    *,
    timeout_seconds: int = 20,
    max_bytes: int = MAX_IMAGE_BYTES,
    session: requests.Session | None = None,
) -> LoadedVisualAsset:
    """Load URL/local/data image once so both extractors compare identical bytes."""

    source = reference.source.strip()
    header_type: str | None = None
    if source.startswith("data:"):
        content, header_type = _decode_data_url(source, max_bytes)
    elif urlparse(source).scheme in {"http", "https"}:
        content, header_type = _download_public_asset(
            source,
            timeout_seconds=timeout_seconds,
            max_bytes=max_bytes,
            session=session,
        )
    else:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Không tìm thấy file ảnh: {path}")
        if path.stat().st_size > max_bytes:
            raise ValueError(f"Ảnh vượt giới hạn {max_bytes} bytes.")
        content = path.read_bytes()
        header_type = mimetypes.guess_type(path.name)[0]

    detected_type, width, height, image_format = _inspect_image(content)
    mime_type = detected_type if detected_type.startswith("image/") else header_type
    return LoadedVisualAsset(
        reference=reference,
        content=content,
        mime_type=mime_type or "application/octet-stream",
        sha256=hashlib.sha256(content).hexdigest(),
        width=width,
        height=height,
        image_format=image_format,
    )


def extract_coordinates(text: str) -> list[dict[str, float]]:
    coordinates: list[dict[str, float]] = []
    seen: set[tuple[float, float]] = set()
    for match in COORDINATE_PATTERN.finditer(text or ""):
        x = float(match.group("x").replace(",", "."))
        y = float(match.group("y").replace(",", "."))
        if (x, y) not in seen:
            seen.add((x, y))
            coordinates.append({"x": x, "y": y})
    for match in JSON_COORDINATE_PATTERN.finditer(text or ""):
        x = float(match.group("x"))
        y = float(match.group("y"))
        if (x, y) not in seen:
            seen.add((x, y))
            coordinates.append({"x": x, "y": y})
    return coordinates


def extract_chandra_structured_analysis(text: str) -> list[Any]:
    """Parse JSON blocks emitted inside Chandra's native analyze tags."""

    blocks: list[Any] = []
    for raw in re.findall(r"<analyze>(.*?)</analyze>", text or "", re.DOTALL):
        try:
            blocks.append(json.loads(raw.strip()))
        except ValueError:
            continue
    return blocks


def extract_text_facts(text: str) -> list[str]:
    facts: list[str] = []
    for match in COORDINATE_PATTERN.finditer(text or ""):
        fact = match.group(0).strip()
        if fact not in facts:
            facts.append(fact)
    for match in EQUATION_PATTERN.finditer(text or ""):
        fact = match.group(0).strip()
        if fact and fact not in facts:
            facts.append(fact)
    return facts[:100]


def classify_visual_type(text: str, signals: dict[str, Any] | None = None) -> str:
    normalized = (text or "").lower()
    keyword_groups = (
        ("table", ("table", "bảng", "hàng", "cột", "tần số")),
        ("graph", ("đồ thị", "graph", "x-axis", "y-axis", "trục tọa độ")),
        ("chart", ("biểu đồ", "chart", "bar chart", "pie chart")),
        ("geometry", ("tam giác", "đường tròn", "hình học", "triangle", "circle")),
        ("diagram", ("sơ đồ", "diagram", "flowchart", "mermaid")),
    )
    for visual_type, keywords in keyword_groups:
        if any(keyword in normalized for keyword in keywords):
            return visual_type
    values = signals or {}
    horizontal = int(values.get("long_horizontal_lines") or 0)
    vertical = int(values.get("long_vertical_lines") or 0)
    if horizontal >= 3 and vertical >= 2:
        return "table"
    if horizontal >= 1 and vertical >= 1:
        return "graph"
    return "image" if values else "unknown"


class CodeVisionParser:
    """Cheap deterministic baseline for raster layout and supplied alt text."""

    def __init__(self, *, thumbnail_size: int = 512):
        self.thumbnail_size = max(64, min(int(thumbnail_size), 1024))

    def parse(self, asset: LoadedVisualAsset) -> VisualExtractionResult:
        with Image.open(BytesIO(asset.content)) as image:
            rgb = image.convert("RGB")
            rgb.thumbnail((self.thumbnail_size, self.thumbnail_size))
            width, height = rgb.size
            pixels = list(rgb.getdata())

        darkness_threshold = 80
        dark_mask = [max(pixel) <= darkness_threshold for pixel in pixels]
        saturated_mask = [max(pixel) - min(pixel) >= 70 for pixel in pixels]
        dark_ratio = sum(dark_mask) / max(1, len(dark_mask))
        saturated_ratio = sum(saturated_mask) / max(1, len(saturated_mask))

        long_rows = 0
        for y in range(height):
            start = y * width
            if sum(dark_mask[start : start + width]) >= width * 0.55:
                long_rows += 1
        long_columns = 0
        for x in range(width):
            if sum(dark_mask[y * width + x] for y in range(height)) >= height * 0.55:
                long_columns += 1

        signals = {
            "parser_scope": "metadata_and_simple_raster_signals_only",
            "thumbnail_width": width,
            "thumbnail_height": height,
            "dark_pixel_ratio": round(dark_ratio, 6),
            "saturated_pixel_ratio": round(saturated_ratio, 6),
            "long_horizontal_lines": long_rows,
            "long_vertical_lines": long_columns,
            "alt_text_available": bool(asset.reference.alt_text.strip()),
        }
        context = "\n".join(
            part
            for part in (
                asset.reference.alt_text.strip(),
                asset.reference.context_text.strip(),
            )
            if part
        )
        facts = extract_text_facts(context)
        status = "extracted" if context or long_rows or long_columns else "partial"
        return VisualExtractionResult(
            extractor="code",
            status=status,
            source=asset.reference.source,
            mime_type=asset.mime_type,
            sha256=asset.sha256,
            width=asset.width,
            height=asset.height,
            visual_type=classify_visual_type(context, signals),
            transcription=asset.reference.alt_text.strip(),
            facts=facts,
            coordinates=extract_coordinates(context),
            signals=signals,
        )

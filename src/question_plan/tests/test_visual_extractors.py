import base64
from io import BytesIO

from PIL import Image, ImageDraw

from src.question_plan.vision.chandra_extractor import ChandraVisualExtractor
from src.question_plan.vision.visual_context import (
    AssetReference,
    CodeVisionParser,
    extract_asset_references,
    load_visual_asset,
)


def png_bytes_with_grid() -> bytes:
    image = Image.new("RGB", (240, 160), "white")
    draw = ImageDraw.Draw(image)
    for x in (20, 100, 180):
        draw.line((x, 10, x, 150), fill="black", width=3)
    for y in (20, 70, 120):
        draw.line((10, y, 230, y), fill="black", width=3)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def data_url(content: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(content).decode("ascii")


def test_extract_asset_reference_preserves_alt_text_and_json_pointer():
    payload = {
        "_id": "visual-1",
        "instruction": [{
            "type": "text",
            "text": "Dựa vào ![Đồ thị qua A(-1, -2) và O(0, 0)](https://example.com/a.webp)",
        }],
    }

    references = extract_asset_references(payload)

    assert len(references) == 1
    assert references[0].question_id == "visual-1"
    assert references[0].source_path == "/instruction/0/text"
    assert references[0].alt_text.startswith("Đồ thị qua A")


def test_code_parser_extracts_metadata_coordinates_and_grid_signals():
    reference = AssetReference(
        source_path="/instruction/0/text",
        source=data_url(png_bytes_with_grid()),
        alt_text="Bảng có điểm A(-1, -2) và O(0, 0).",
    )
    asset = load_visual_asset(reference)

    result = CodeVisionParser().parse(asset)

    assert result.status == "extracted"
    assert result.width == 240
    assert result.height == 160
    assert result.visual_type == "table"
    assert result.coordinates == [{"x": -1.0, "y": -2.0}, {"x": 0.0, "y": 0.0}]
    assert result.signals["long_horizontal_lines"] >= 3
    assert result.signals["long_vertical_lines"] >= 3


def test_chandra_uses_multimodal_data_url_and_common_contract():
    class FakeClient:
        def __init__(self):
            self.request = None

        def chat_completion(self, **kwargs):
            self.request = kwargs
            return {
                "model": "chandra-ocr-2",
                "endpoint": "/v1/chat/completions",
                "latency_seconds": 0.2,
                "http_retry_count": 0,
                "content": (
                    '<analyze>[{"series":"Đồ thị hàm số"}]</analyze>\n'
                    'Điểm {"x": -1, "y": -2} và O(0, 0).'
                ),
            }

    client = FakeClient()
    reference = AssetReference(
        source_path="/",
        source=data_url(png_bytes_with_grid()),
    )
    asset = load_visual_asset(reference)

    result = ChandraVisualExtractor(client).extract(asset)

    image_part = client.request["messages"][0]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert client.request["temperature"] == 0
    assert result.status == "extracted"
    assert result.visual_type == "graph"
    assert {"x": -1.0, "y": -2.0} in result.coordinates
    assert result.signals["structured_analysis_block_count"] == 1

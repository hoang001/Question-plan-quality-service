"""Standalone visual extraction helpers.

These modules are intentionally not imported by the generated-question flow.
They exist so visual extraction can be evaluated before pipeline integration.
"""

from .chandra_extractor import ChandraVisualExtractor
from .visual_context import (
    AssetReference,
    CodeVisionParser,
    LoadedVisualAsset,
    VisualExtractionResult,
    extract_asset_references,
    load_visual_asset,
)

__all__ = [
    "AssetReference",
    "ChandraVisualExtractor",
    "CodeVisionParser",
    "LoadedVisualAsset",
    "VisualExtractionResult",
    "extract_asset_references",
    "load_visual_asset",
]

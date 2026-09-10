"""Traceable dataset canonicalization, deduplication, splits and manifests."""

from adaptive_math.data.canonicalize import QuarantineReason, QuarantineRecord, canonicalize_source
from adaptive_math.data.deduplicate import DedupResult, deduplicate
from adaptive_math.data.manifest import (
    DataManifest,
    DedupManifest,
    SourceManifest,
    SplitManifest,
    load_manifest,
    write_manifest,
)
from adaptive_math.data.sources import SourceRegistry, SourceSpec, load_source_records
from adaptive_math.data.split import SplitConfig, assign_splits

__all__ = [
    "DataManifest",
    "DedupManifest",
    "DedupResult",
    "QuarantineReason",
    "QuarantineRecord",
    "SourceManifest",
    "SourceRegistry",
    "SourceSpec",
    "SplitConfig",
    "SplitManifest",
    "assign_splits",
    "canonicalize_source",
    "deduplicate",
    "load_manifest",
    "load_source_records",
    "write_manifest",
]

"""Deterministic answer extraction, normalization and verification."""

from adaptive_math.verifier.extractor import (
    ExtractResult,
    ExtractStatus,
    extract,
    extract_final_answer,
    extract_solution_answer,
    extract_terminal_solution_answer,
)
from adaptive_math.verifier.normalizer import normalize_surface
from adaptive_math.verifier.numeric import VerifierConfig
from adaptive_math.verifier.service import VerifierResult, VerifierStatus, verify_answer

__all__ = [
    "ExtractResult",
    "ExtractStatus",
    "VerifierConfig",
    "VerifierResult",
    "VerifierStatus",
    "extract",
    "extract_final_answer",
    "extract_solution_answer",
    "extract_terminal_solution_answer",
    "normalize_surface",
    "verify_answer",
]

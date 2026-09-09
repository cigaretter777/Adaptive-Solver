"""Deterministic hashing and surface canonicalization for content-addressed tasks."""

import hashlib
import unicodedata


def canonical_text(text: str) -> str:
    """Normalize to a canonical surface form: NFKC, unified line endings,
    whitespace collapsed to single spaces, trimmed. No algebra is applied."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(normalized.split())


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_source_hash(source_record: bytes) -> str:
    """SHA-256 over the exact, unmodified source record bytes."""
    return sha256_hex(source_record)


def make_task_id(dataset: str, problem: str) -> str:
    """Content-addressed task id: dataset plus the first 20 hex characters of
    SHA-256(canonical_text(problem))."""
    digest = sha256_hex(canonical_text(problem).encode("utf-8"))
    return f"{dataset}:{digest[:20]}"

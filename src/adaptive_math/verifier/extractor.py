"""Bounded final-answer extraction from raw model output.

Extraction is purely textual: it never executes input and never raises for
arbitrary text. Every outcome is an ExtractResult with status OK, MISSING or
AMBIGUOUS. Priority: <final> tags, then \\boxed{...}, then $...$ / $$...$$.
"""

from enum import StrEnum

import orjson
from pydantic import BaseModel, ConfigDict, Field

from adaptive_math.core.types import JSONValue

MAX_INPUT_CHARS = 32768
MAX_NESTING = 64

_FINAL_OPEN = "<final>"
_FINAL_CLOSE = "</final>"


class ExtractStatus(StrEnum):
    OK = "ok"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"


class ExtractResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ExtractStatus
    value: str | None = None
    details: dict[str, JSONValue] = Field(default_factory=dict)


def extract(raw: str) -> ExtractResult:
    if len(raw) > MAX_INPUT_CHARS:
        return _missing(f"input exceeds {MAX_INPUT_CHARS} chars")
    if _FINAL_OPEN in raw:
        return _extract_final_tags(raw)
    boxed = _scan_boxed(raw)
    if boxed:
        if len(boxed) == 1:
            return _ok(boxed[0])
        return _ambiguous(f"{len(boxed)} boxed answers found")
    displays = _scan_dollar_groups(raw, "$$")
    if displays:
        if len(displays) == 1:
            return _ok(displays[0])
        return _ambiguous(f"{len(displays)} dollar groups found")
    singles = _scan_dollar_groups(raw, "$")
    if singles:
        if len(singles) == 1:
            return _ok(singles[0])
        return _ambiguous(f"{len(singles)} dollar groups found")
    return _missing("no final answer marker found")


def extract_final_answer(raw: str) -> str | None:
    result = extract(raw)
    return result.value if result.status is ExtractStatus.OK else None


def _extract_final_tags(text: str) -> ExtractResult:
    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        start = text.find(_FINAL_OPEN, pos)
        if start == -1:
            break
        end = text.find(_FINAL_CLOSE, start + len(_FINAL_OPEN))
        if end == -1:
            break
        spans.append((start + len(_FINAL_OPEN), end))
        pos = end + len(_FINAL_CLOSE)
    if not spans:
        return _missing("final tag present but no complete pair")
    if len(spans) > 1:
        return _ambiguous(f"{len(spans)} final tags found")
    inner = text[spans[0][0] : spans[0][1]].strip()
    if not inner:
        return _missing("empty final tag")
    try:
        payload = orjson.loads(inner)
    except orjson.JSONDecodeError:
        return _missing("final payload is not valid JSON")
    if not isinstance(payload, dict):
        return _missing("final payload is not a JSON object")
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer:
        return _missing("final payload has no non-empty string answer field")
    return _ok(answer)


def _scan_boxed(text: str) -> list[str]:
    """Find complete \\boxed{...} groups with a single-pass balanced-brace
    scanner. One or two preceding backslashes are accepted."""
    values: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        idx = text.find("boxed", i)
        if idx == -1:
            break
        backslashes = 0
        j = idx - 1
        while j >= 0 and text[j] == "\\":
            backslashes += 1
            j -= 1
        if backslashes not in (1, 2):
            i = idx + len("boxed")
            continue
        k = idx + len("boxed")
        while k < n and text[k] in " \t\r\n":
            k += 1
        if k >= n or text[k] != "{":
            i = idx + len("boxed")
            continue
        depth = 1
        start = k + 1
        p = start
        while p < n:
            ch = text[p]
            if ch == "\\":
                p += 2  # skip escaped char, e.g. \{ or \}
                continue
            if ch == "{":
                depth += 1
                if depth > MAX_NESTING:
                    p = n  # treat as unbalanced; stops this candidate
                    break
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    values.append(text[start:p].strip())
                    break
            p += 1
        i = max(p + 1, idx + len("boxed"))
    return values


def _scan_dollar_groups(text: str, marker: str) -> list[str]:
    """Find non-empty text enclosed in a pair of the given dollar marker.
    Escaped markers are skipped; unpaired markers contribute nothing."""
    groups: list[str] = []
    pos = 0
    while True:
        start = text.find(marker, pos)
        if start == -1:
            break
        if start > 0 and text[start - 1] == "\\":
            pos = start + len(marker)
            continue
        end = text.find(marker, start + len(marker))
        if end == -1:
            break
        inner = text[start + len(marker) : end].strip()
        if inner:
            groups.append(inner)
        pos = end + len(marker)
    return groups


def _ok(value: str) -> ExtractResult:
    return ExtractResult(status=ExtractStatus.OK, value=value)


def _missing(reason: str) -> ExtractResult:
    return ExtractResult(status=ExtractStatus.MISSING, details={"reason": reason})


def _ambiguous(reason: str) -> ExtractResult:
    return ExtractResult(status=ExtractStatus.AMBIGUOUS, details={"reason": reason})

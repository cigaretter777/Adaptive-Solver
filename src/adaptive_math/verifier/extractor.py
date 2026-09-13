"""Bounded final-answer extraction.

Two modes with deliberately different semantics:

- ``extract`` is the strict trajectory-output extractor for model responses:
  priority <final> tags, then \\boxed{...}, then $...$ / $$...$$; multiple
  candidates are AMBIGUOUS because a protocol-valid turn has one answer.
- ``extract_solution_answer`` is for dataset worked solutions, a different
  genre: the final answer is conventionally the LAST \\boxed{...}, or
  follows a prose anchor ("final answer is ..."). Solutions may far exceed
  MAX_INPUT_CHARS, so only a bounded tail window is scanned. Without a
  boxed group or anchor the result is MISSING — never a guess among inline
  math spans.

Both modes are purely textual: they never execute input and never raise.
"""

import re
from enum import StrEnum

import orjson
from pydantic import BaseModel, ConfigDict, Field

from adaptive_math.core.types import JSONValue

MAX_INPUT_CHARS = 32768
MAX_NESTING = 64
SOLUTION_TAIL_WINDOW = 32768
TERMINAL_TAIL_WINDOW = 4096
TERMINAL_MAX_NONEMPTY_LINES = 12

_FINAL_OPEN = "<final>"
_FINAL_CLOSE = "</final>"

_PROSE_ANCHOR = re.compile(r"(?:final\s+answer|answer)\s*(?:is\s*:?|:)\s*", re.IGNORECASE)
# A math run after an anchor: LaTeX commands, escaped chars, digits and
# operators. Bare words (e.g. "dollars per item") terminate the run.
_MATH_RUN = re.compile(r"(?:\\[a-zA-Z]+|\\.|[\d.,+\-*/^=(){}\[\]\s])+")
_TERMINAL_ASSIGNMENT = re.compile(
    r"^\s*(?:(?:therefore|hence|so)\s*,?\s*)?"
    r"(?:\\[a-zA-Z]+(?:\s+[A-Za-z])?|[A-Za-z][A-Za-z0-9_]*)\s*=\s*(?P<rhs>.+?)\s*$",
    re.IGNORECASE,
)
_TERMINAL_CONCLUSION_CUE = re.compile(
    r"\b(?:therefore|hence|thus|consequently|it follows|which means|we obtain|we get|"
    r"which simplifies to)\b",
    re.IGNORECASE,
)


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


def extract_solution_answer(solution: str) -> ExtractResult:
    """Extract the final answer from a full worked solution.

    Priority: last complete \\boxed{...} in the bounded tail window, then
    prose anchors ("final answer is"/"answer is:"/"answer:") scanned last to
    first for a math-run value, then a whole-text single $...$/$$...$$ span
    (some solutions are just "$D$"). Anything else is MISSING; this never
    reports AMBIGUOUS and never guesses among spans embedded in prose.
    """
    if not solution.strip():
        return _missing("empty solution")
    tail = solution[-SOLUTION_TAIL_WINDOW:]
    boxed = _scan_boxed(tail)
    if boxed:
        return _ok(boxed[-1], details={"method": "last_boxed"})
    # last anchor first: a trailing "the answer is discussed above" must not
    # hide an earlier usable "the answer is 42"
    for anchor in reversed(list(_PROSE_ANCHOR.finditer(tail))):
        value = _anchor_value(tail[anchor.end() :])
        if value is not None:
            return _ok(value, details={"method": "prose_anchor"})
    whole = _whole_text_math_span(tail)
    if whole is not None:
        return _ok(whole, details={"method": "whole_text_span"})
    return _missing("no boxed group and no usable prose anchor found")


def extract_terminal_solution_answer(solution: str) -> ExtractResult:
    """Extract only the final explicit assignment from a worked-solution tail.

    This fallback deliberately accepts less syntax than ``extract_solution_answer``.
    It never scans the full solution or picks an earlier verifier-matching value.
    """
    tail = solution[-TERMINAL_TAIL_WINDOW:]
    lines = [line.strip() for line in tail.splitlines() if line.strip()][-TERMINAL_MAX_NONEMPTY_LINES:]
    for line in reversed(lines):
        match = _TERMINAL_ASSIGNMENT.match(line)
        if match is None:
            continue
        value = _anchor_value(match.group("rhs"))
        if value is not None:
            return _ok(value, details={"method": "terminal_assignment"})
        return _missing("terminal assignment has no usable value")
    for line in reversed(lines):
        if "=" not in line or _TERMINAL_CONCLUSION_CUE.search(line) is None:
            continue
        value = _anchor_value(line.rsplit("=", maxsplit=1)[1])
        if value is not None:
            return _ok(value, details={"method": "terminal_conclusion_equation"})
    return _missing("no explicit terminal assignment in bounded tail")


def _whole_text_math_span(text: str) -> str | None:
    """Return the content when the ENTIRE text is one $...$ or $$...$$ span.

    Some dataset solutions are nothing but the final answer ("$D$"). A math
    span embedded in prose is deliberately not accepted here.
    """
    stripped = text.strip()
    for marker in ("$$", "$"):
        if (
            stripped.startswith(marker)
            and stripped.endswith(marker)
            and len(stripped) > 2 * len(marker)
        ):
            inner = stripped[len(marker) : -len(marker)]
            if marker not in inner.replace("\\" + marker, ""):
                inner = inner.strip()
                if inner:
                    return inner
    return None


def _anchor_value(rest: str) -> str | None:
    text = rest.strip()
    if not text:
        return None
    parenthesized = _balanced_parenthesized_prefix(text)
    if parenthesized is not None:
        return parenthesized
    if text.startswith("$"):
        end = text.find("$", 1)
        if end > 1:
            inner = text[1:end].strip()
            return inner or None
    if text.startswith("\\boxed"):
        boxed = _scan_boxed(text)
        if boxed:
            return boxed[0]
    match = _MATH_RUN.match(text)
    if match is None:
        return None
    value = match.group(0).strip().rstrip(".")
    if value.endswith("(") and len(value) > 1 and value[-2].isspace():
        value = value[:-1].rstrip()
    return value or None


def _balanced_parenthesized_prefix(text: str) -> str | None:
    """Return a complete leading parenthesized answer, never an open prose suffix."""
    if not text.startswith("("):
        return None
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[: index + 1]
    return None


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


def _ok(value: str, details: dict[str, JSONValue] | None = None) -> ExtractResult:
    return ExtractResult(status=ExtractStatus.OK, value=value, details=details or {})


def _missing(reason: str) -> ExtractResult:
    return ExtractResult(status=ExtractStatus.MISSING, details={"reason": reason})


def _ambiguous(reason: str) -> ExtractResult:
    return ExtractResult(status=ExtractStatus.AMBIGUOUS, details={"reason": reason})

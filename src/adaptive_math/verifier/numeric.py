"""Bounded, exact parsing for numeric and collection answers.

All value arithmetic is exact (Decimal/Fraction); floats are never used for
comparison. Every parser enforces digit, exponent, item and depth limits
before any potentially expensive operation.
"""

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from pydantic import BaseModel, ConfigDict, Field

_INTEGER_RE = re.compile(r"[+-]?[0-9]+")
_DECIMAL_RE = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_GROUPED_INT_RE = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+")
_GROUPED_DEC_RE = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+\.\d+")


class VerifierConfig(BaseModel):
    """Limits and tolerances for deterministic answer verification."""

    model_config = ConfigDict(extra="forbid")

    real_atol: float = Field(default=1e-9, ge=0)
    real_rtol: float = Field(default=1e-9, ge=0)
    max_digits: int = Field(default=256, gt=0)
    max_abs_exponent: int = Field(default=1000, ge=0)
    max_items: int = Field(default=128, gt=0)
    max_depth: int = Field(default=8, ge=0)


@dataclass(frozen=True)
class ParseResult[T]:
    ok: bool
    value: T | None = None
    error_code: str | None = None


type Collection = Fraction | tuple[Collection, ...] | frozenset[Collection]


@dataclass(frozen=True)
class IntervalValue:
    left_open: bool
    right_open: bool
    left: Fraction
    right: Fraction


def parse_integer(text: str, config: VerifierConfig) -> ParseResult[Fraction]:
    """Strict integer syntax, accepting safe thousands grouping ("1,234").
    error_code "not_integer" means the text is a well-formed number of
    another form; any other error is unparseable."""
    stripped = _strip_thousands(text)
    if stripped is not None:
        text = stripped
    if _INTEGER_RE.fullmatch(text):
        return _bounded_integer(text, config)
    if _DECIMAL_RE.fullmatch(text) or _fraction_parts_ok(text):
        return ParseResult(ok=False, error_code="not_integer")
    return ParseResult(ok=False, error_code="bad_syntax")


def parse_number(text: str, config: VerifierConfig) -> ParseResult[Fraction]:
    """Exact numeric parsing: integers, decimals, scientific notation,
    safe thousands grouping and simple fractions. Never converts to float."""
    stripped = _strip_thousands(text)
    if stripped is not None:
        text = stripped
    if "/" in text:
        if text.count("/") != 1:
            return ParseResult(ok=False, error_code="bad_syntax")
        numerator, denominator = text.split("/", 1)
        if not _INTEGER_RE.fullmatch(numerator) or not _INTEGER_RE.fullmatch(denominator):
            return ParseResult(ok=False, error_code="bad_syntax")
        r_num = _bounded_integer(numerator, config)
        r_den = _bounded_integer(denominator, config)
        if not r_num.ok or r_num.value is None:
            return ParseResult(ok=False, error_code=r_num.error_code)
        if not r_den.ok or r_den.value is None:
            return ParseResult(ok=False, error_code=r_den.error_code)
        if r_den.value == 0:
            return ParseResult(ok=False, error_code="division_by_zero")
        return ParseResult(ok=True, value=Fraction(r_num.value, r_den.value))
    if not _DECIMAL_RE.fullmatch(text):
        return ParseResult(ok=False, error_code="bad_syntax")
    return _bounded_decimal(text, config)


def parse_value(text: str, config: VerifierConfig, depth: int) -> ParseResult[Collection]:
    """Recursive parser for scalars, sets and tuples."""
    if depth > config.max_depth:
        return ParseResult(ok=False, error_code="too_deep")
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        inner = stripped[1:-1].strip()
        if not inner:
            return ParseResult(ok=True, value=frozenset())
        r_elements = _parse_elements(inner, config, depth)
        if not r_elements.ok or r_elements.value is None:
            return ParseResult(ok=False, error_code=r_elements.error_code)
        return ParseResult(ok=True, value=frozenset(r_elements.value))
    if stripped.startswith("(") and stripped.endswith(")"):
        inner = stripped[1:-1].strip()
        if not inner:
            return ParseResult(ok=False, error_code="bad_syntax")
        r_elements = _parse_elements(inner, config, depth)
        if not r_elements.ok or r_elements.value is None:
            return ParseResult(ok=False, error_code=r_elements.error_code)
        return ParseResult(ok=True, value=tuple(r_elements.value))
    result = parse_number(stripped, config)
    if not result.ok:
        return ParseResult(ok=False, error_code=result.error_code)
    return ParseResult(ok=True, value=result.value)


def _parse_elements(
    inner: str, config: VerifierConfig, depth: int
) -> ParseResult[list[Collection]]:
    parts = _split_top_level(inner)
    if len(parts) > config.max_items:
        return ParseResult(ok=False, error_code="too_many_items")
    elements: list[Collection] = []
    for part in parts:
        if not part.strip():
            return ParseResult(ok=False, error_code="bad_syntax")
        result = parse_value(part, config, depth + 1)
        if not result.ok or result.value is None:
            return ParseResult(ok=False, error_code=result.error_code)
        elements.append(result.value)
    return ParseResult(ok=True, value=elements)


def parse_interval(text: str, config: VerifierConfig) -> ParseResult[IntervalValue]:
    stripped = text.strip()
    if len(stripped) < 3:
        return ParseResult(ok=False, error_code="bad_syntax")
    if stripped[0] not in "([" or stripped[-1] not in ")]":
        return ParseResult(ok=False, error_code="bad_syntax")
    parts = _split_top_level(stripped[1:-1])
    if len(parts) != 2:
        return ParseResult(ok=False, error_code="bad_syntax")
    r_left = parse_number(parts[0].strip(), config)
    r_right = parse_number(parts[1].strip(), config)
    if not r_left.ok or r_left.value is None:
        return ParseResult(ok=False, error_code=r_left.error_code)
    if not r_right.ok or r_right.value is None:
        return ParseResult(ok=False, error_code=r_right.error_code)
    return ParseResult(
        ok=True,
        value=IntervalValue(
            left_open=stripped[0] == "(",
            right_open=stripped[-1] == ")",
            left=r_left.value,
            right=r_right.value,
        ),
    )


def canonical_form(value: Collection) -> str:
    """Deterministic canonical string form for a parsed value."""
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, tuple):
        return "(" + ", ".join(canonical_form(v) for v in value) + ")"
    return "{" + ", ".join(sorted(canonical_form(v) for v in value)) + "}"


def interval_form(value: IntervalValue) -> str:
    left = "(" if value.left_open else "["
    right = ")" if value.right_open else "]"
    return f"{left}{value.left}, {value.right}{right}"


def tolerance_fraction(value: float) -> Fraction:
    """Exact rational for a float tolerance given as a decimal literal."""
    return Fraction(str(value))


def _bounded_integer(text: str, config: VerifierConfig) -> ParseResult[Fraction]:
    digits = text.lstrip("+-")
    if len(digits) > config.max_digits:
        return ParseResult(ok=False, error_code="too_many_digits")
    try:
        return ParseResult(ok=True, value=Fraction(int(text)))
    except ValueError:
        return ParseResult(ok=False, error_code="bad_syntax")


def _bounded_decimal(text: str, config: VerifierConfig) -> ParseResult[Fraction]:
    mantissa = text
    exponent = 0
    if "e" in mantissa or "E" in mantissa:
        mantissa, exp_part = re.split("[eE]", mantissa, maxsplit=1)
        try:
            exponent = int(exp_part)
        except ValueError:
            return ParseResult(ok=False, error_code="bad_syntax")
    if abs(exponent) > config.max_abs_exponent:
        return ParseResult(ok=False, error_code="exponent_out_of_range")
    digit_count = sum(1 for ch in mantissa if ch.isdigit())
    if digit_count > config.max_digits:
        return ParseResult(ok=False, error_code="too_many_digits")
    try:
        value = Fraction(Decimal(mantissa)) * Fraction(10) ** exponent
    except (InvalidOperation, ValueError):
        return ParseResult(ok=False, error_code="bad_syntax")
    return ParseResult(ok=True, value=value)


def _strip_thousands(text: str) -> str | None:
    """Return the text with safe thousands separators removed, or None when
    the comma grouping is not a complete grouped number. Applied per numeric
    token so collection elements are never merged across commas."""
    if _GROUPED_INT_RE.fullmatch(text) or _GROUPED_DEC_RE.fullmatch(text):
        return text.replace(",", "")
    return None


def _fraction_parts_ok(text: str) -> bool:
    if text.count("/") != 1:
        return False
    numerator, denominator = text.split("/", 1)
    return bool(_INTEGER_RE.fullmatch(numerator) and _INTEGER_RE.fullmatch(denominator))


def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch in "{(":
            depth += 1
            current.append(ch)
        elif ch in "})":
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current or not parts:
        parts.append("".join(current))
    return parts

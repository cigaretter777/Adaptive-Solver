"""Surface normalization for extracted answers.

Representation-preserving cleanup only: NFKC, trim, unicode minus to ASCII,
one outer math delimiter, frac spelling and safe thousands separators.
It must never simplify algebra.
"""

import re

from adaptive_math.core.hashing import canonical_text

_FRAC_SPELLING = re.compile(r"\\[dt]frac\b")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_DELIMITER_PAIRS = (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$"))


def normalize_surface(value: str) -> str:
    text = canonical_text(value)
    text = text.replace("\u2212", "-")
    text = _FRAC_SPELLING.sub(r"\\frac", text)
    text = _THOUSANDS.sub("", text)
    return _strip_outer_delimiters(text)


def _strip_outer_delimiters(text: str) -> str:
    # fixed-point stripping keeps normalization idempotent for inputs like
    # "$ $x$ $" where one pass exposes another outer delimiter pair.
    while True:
        stripped = _strip_one(text)
        if stripped == text:
            return text
        text = stripped


def _strip_one(text: str) -> str:
    for opener, closer in _DELIMITER_PAIRS:
        if text.startswith(opener) and text.endswith(closer):
            inner = text[len(opener) : len(text) - len(closer)].strip()
            if inner:
                return inner
    return text

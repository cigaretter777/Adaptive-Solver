import pytest
from hypothesis import given
from hypothesis import strategies as st

from adaptive_math.verifier import normalize_surface


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("−3", "-3"),  # unicode minus
        ("−３", "-3"),  # unicode minus + fullwidth digit
        (r"\dfrac{1}{2}", r"\frac{1}{2}"),
        (r"\tfrac{x}{2}", r"\frac{x}{2}"),
        (r"\frac{1}{2}", r"\frac{1}{2}"),  # already normalized
        ("$x$", "x"),
        ("$$x^2$$", "x^2"),
        (r"\(x\)", "x"),
        (r"\[x\]", "x"),
        ("1,234", "1234"),
        ("1,234,567", "1234567"),
        ("1,234.5", "1234.5"),
        ("12,34", "12,34"),  # not a safe thousands grouping
        ("(1,2,3)", "(1,2,3)"),  # tuple separators untouched
        ("  $ x $  ", "x"),
        ("", ""),
    ],
)
def test_surface_normalization_cases(raw: str, expected: str) -> None:
    assert normalize_surface(raw) == expected


def test_normalization_does_not_simplify_algebra() -> None:
    assert normalize_surface("x+x") == "x+x"
    assert normalize_surface("2 + 2") == "2 + 2"
    assert normalize_surface(r"\frac{2}{4}") == r"\frac{2}{4}"


def test_unpaired_delimiters_are_left_alone() -> None:
    assert normalize_surface("$x") == "$x"
    assert normalize_surface("x$") == "x$"


@given(st.text(max_size=2048))
def test_surface_normalization_is_idempotent(value: str) -> None:
    once = normalize_surface(value)
    assert normalize_surface(once) == once

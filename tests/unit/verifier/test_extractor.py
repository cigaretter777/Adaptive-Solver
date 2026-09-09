import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from adaptive_math.verifier import ExtractResult, ExtractStatus, extract, extract_final_answer

FIXTURE = Path(__file__).parents[2] / "fixtures" / "verifier_cases.jsonl"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (r"Therefore \\boxed{42}.", "42"),
        (r"\\boxed{\\frac{1}{2}}", r"\\frac{1}{2}"),
        ("<final>{\"answer\":\"-3\"}</final>", "-3"),
        ("No final answer", None),
    ],
)
def test_extract_final_answer(raw: str, expected: str | None) -> None:
    assert extract_final_answer(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (r"Therefore \boxed{42}.", "42"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"The answer is $\frac{1}{2}$ dollars.", r"\frac{1}{2}"),
    ],
)
def test_extract_single_backslash_variants(raw: str, expected: str) -> None:
    assert extract_final_answer(raw) == expected


def test_fixture_cases_are_all_consumed() -> None:
    rows = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    assert len(rows) > 0
    for row in rows:
        assert extract_final_answer(row["raw"]) == row["expected"]
    # every fixture line must be exercised exactly once
    assert len(rows) == sum(1 for line in FIXTURE.read_text().splitlines() if line.strip())


@pytest.mark.parametrize(
    ("raw", "status"),
    [
        (r"The answer is \boxed{42", ExtractStatus.MISSING),  # unbalanced braces
        (r"\boxed{1} and \boxed{2}", ExtractStatus.AMBIGUOUS),  # multiple boxed
        ("<final></final>", ExtractStatus.MISSING),  # empty final
        ("<final>not json</final>", ExtractStatus.MISSING),  # unparseable payload
        ("<final>{\"answer\":42}</final>", ExtractStatus.MISSING),  # non-string answer
        ("<final>{\"confidence\":0.9}</final>", ExtractStatus.MISSING),  # missing answer key
        ('<final>{"answer":"1"}</final><final>{"answer":"2"}</final>', ExtractStatus.AMBIGUOUS),
        ("", ExtractStatus.MISSING),
        ("   ", ExtractStatus.MISSING),
        ("Cost is $5", ExtractStatus.MISSING),  # single dollar, no pair
        ("$a=1$ and $b=2$", ExtractStatus.AMBIGUOUS),  # multiple dollar groups
        ("no markers at all", ExtractStatus.MISSING),
    ],
)
def test_adversarial_extraction_never_throws(raw: str, status: ExtractStatus) -> None:
    result = extract(raw)
    assert result.status is status
    assert result.value is None or result.status is ExtractStatus.OK


def test_deeply_nested_braces_are_bounded() -> None:
    raw = r"\boxed{" + "{" * 20000 + "x" + "}" * 20000 + "}"
    result = extract(raw)
    assert result.status is ExtractStatus.MISSING


def test_overlong_input_is_rejected_bounded() -> None:
    result = extract("x" * 32769)
    assert result.status is ExtractStatus.MISSING
    assert "32768" in str(result.details)


def test_final_tag_takes_priority_over_boxed() -> None:
    raw = '<final>{"answer":"7"}</final> or \\boxed{3}'
    result = extract(raw)
    assert result.status is ExtractStatus.OK
    assert result.value == "7"


def test_final_payload_with_extra_keys_is_ok() -> None:
    raw = '<final>{"answer":"-3","confidence":0.9}</final>'
    result = extract(raw)
    assert result.status is ExtractStatus.OK
    assert result.value == "-3"


def test_extract_result_serializes_stably() -> None:
    result = extract(r"\boxed{42}")
    assert ExtractResult.model_validate_json(result.model_dump_json()) == result


@given(st.text(max_size=60000))
def test_extract_never_raises(value: str) -> None:
    # extraction must return a structured result, never throw, for any input
    result = extract(value)
    assert isinstance(result, ExtractResult)
    assert result.status in ExtractStatus

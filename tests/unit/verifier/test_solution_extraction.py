"""Solution-mode extraction: full worked solutions, not short answer strings.

The strict ``extract`` remains the trajectory-output extractor (multiple
boxed/dollar groups are AMBIGUOUS there). Dataset solutions follow different
conventions: the final answer is the LAST \\boxed, or follows a prose anchor
like "final answer is". Extraction works on a bounded tail window because
solutions may far exceed MAX_INPUT_CHARS while the answer sits at the end.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from adaptive_math.verifier import (
    ExtractResult,
    ExtractStatus,
    extract_solution_answer,
    extract_terminal_solution_answer,
)


def test_last_boxed_wins_over_intermediate_steps() -> None:
    solution = (
        r"First we estimate \boxed{2}. Refining with the second method "
        r"gives \boxed{42}. Therefore the result is confirmed."
    )
    result = extract_solution_answer(solution)
    assert result.status is ExtractStatus.OK
    assert result.value == "42"


def test_single_boxed_still_works() -> None:
    result = extract_solution_answer(r"Some reasoning here, so \boxed{\frac{1}{2}}.")
    assert result.status is ExtractStatus.OK
    assert result.value == r"\frac{1}{2}"


def test_prose_anchor_final_answer_is() -> None:
    solution = (
        "Let $x = a^2 + b^2$. Then $x \\equiv 1 \\pmod 4$, so the sum "
        "cannot be divisible by 4. The final answer is 42."
    )
    result = extract_solution_answer(solution)
    assert result.status is ExtractStatus.OK
    assert result.value == "42"


def test_prose_anchor_with_latex_value() -> None:
    solution = "After simplification, the final answer is $\\frac{3}{4}$."
    result = extract_solution_answer(solution)
    assert result.status is ExtractStatus.OK
    assert result.value == r"\frac{3}{4}"


def test_prose_anchor_case_insensitive_and_colon_variant() -> None:
    for solution in (
        "Thus the Final Answer: 7 dollars per item.",
        "we conclude that the answer is 7.",
    ):
        result = extract_solution_answer(solution)
        assert result.status is ExtractStatus.OK, solution
        assert result.value == "7"


def test_boxed_takes_priority_over_prose_anchor() -> None:
    solution = "The final answer is 2, but the corrected value is \\boxed{42}."
    result = extract_solution_answer(solution)
    assert result.value == "42"


def test_no_marker_at_all_is_missing() -> None:
    result = extract_solution_answer("Just some words without any answer.")
    assert result.status is ExtractStatus.MISSING


def test_anchor_with_is_and_colon_combined() -> None:
    # observed in OpenR1: "Therefore, the answer is: 180."
    result = extract_solution_answer(
        "Some reasoning with $x=1$ inline. Therefore, the answer is: 180."
    )
    assert result.status is ExtractStatus.OK
    assert result.value == "180"


def test_whole_solution_single_math_span_is_the_answer() -> None:
    # observed in OpenR1: the entire solution is just "$D$"
    assert extract_solution_answer("$D$").value == "D"
    assert extract_solution_answer("  $$42$$  ").value == "42"


def test_math_span_embedded_in_prose_is_not_the_answer() -> None:
    # a single $ group inside prose must not be mistaken for the answer
    result = extract_solution_answer("Let $x = 1$ and then compute the result.")
    assert result.status is ExtractStatus.MISSING
    result = extract_solution_answer("$x=1$ then $y=2$")
    assert result.status is ExtractStatus.MISSING


def test_earlier_anchor_used_when_last_one_has_no_value() -> None:
    solution = "We get the answer is 42. The answer is discussed in the note above."
    result = extract_solution_answer(solution)
    assert result.status is ExtractStatus.OK
    assert result.value == "42"


def test_multiple_dollar_groups_without_anchor_is_missing() -> None:
    # unlike short answer strings, we do not guess among inline math spans
    solution = "Let $x = 1$ and $y = 2$, then $z = 3$ follows."
    result = extract_solution_answer(solution)
    assert result.status is ExtractStatus.MISSING


def test_unbalanced_boxed_falls_back_to_anchor() -> None:
    solution = "We get \\boxed{42 and then the final answer is 7."
    result = extract_solution_answer(solution)
    assert result.status is ExtractStatus.OK
    assert result.value == "7"


def test_very_long_solution_scans_bounded_tail() -> None:
    filler = "Intermediate reasoning step. " * 20000  # ~540k chars
    solution = filler + "The final answer is 42."
    result = extract_solution_answer(solution)
    assert result.status is ExtractStatus.OK
    assert result.value == "42"


def test_anchor_value_stops_at_sentence_end() -> None:
    solution = "All checks pass. The final answer is 120. This completes the proof."
    result = extract_solution_answer(solution)
    assert result.value == "120"


def test_anchor_strips_only_an_explanatory_open_paren_suffix() -> None:
    result = extract_solution_answer("Answer: 10 (all triples of solutions are counted).")
    assert result.status is ExtractStatus.OK
    assert result.value == "10"


@pytest.mark.parametrize("solution, expected", [("Answer: (1,2)", "(1,2)"), ("Answer: (B)", "(B)")])
def test_anchor_preserves_complete_parenthesized_answers(solution: str, expected: str) -> None:
    assert extract_solution_answer(solution).value == expected


def test_terminal_assignment_fallback_uses_the_last_explicit_assignment() -> None:
    solution = "152 x=38\nx=\\frac{38}{152}\nx=\\frac{1}{4}\nTOTAL 6 POINTS"
    result = extract_terminal_solution_answer(solution)
    assert result.status is ExtractStatus.OK
    assert result.value == r"\frac{1}{4}"


def test_terminal_assignment_fallback_does_not_guess_from_inline_math() -> None:
    result = extract_terminal_solution_answer("We noted x=1 in an earlier calculation, then finish.")
    assert result.status is ExtractStatus.MISSING


def test_terminal_conclusion_equation_uses_the_last_equation_rhs() -> None:
    result = extract_terminal_solution_answer(
        "The preceding cases are impossible. Therefore, we have "
        r"f(\alpha)=f(-1-\beta), which means \alpha + \beta = -1"
    )

    assert result.status is ExtractStatus.OK
    assert result.value == "-1"
    assert result.details == {"method": "terminal_conclusion_equation"}


@pytest.mark.parametrize("empty", ["", "   "])
def test_empty_solutions_are_missing(empty: str) -> None:
    assert extract_solution_answer(empty).status is ExtractStatus.MISSING


@given(st.text(max_size=60000))
def test_solution_extraction_never_raises(value: str) -> None:
    result = extract_solution_answer(value)
    assert isinstance(result, ExtractResult)
    assert result.status in ExtractStatus
    # solution mode resolves ambiguity by convention, so it never reports it
    assert result.status is not ExtractStatus.AMBIGUOUS

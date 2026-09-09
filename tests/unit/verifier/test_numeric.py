from adaptive_math.core.types import AnswerType, ReferenceAnswer
from adaptive_math.verifier import VerifierConfig, VerifierResult, VerifierStatus, verify_answer


def verify(
    prediction: str, value: str, answer_type: AnswerType, **config_kwargs: object
) -> VerifierResult:
    config = VerifierConfig(**config_kwargs)
    return verify_answer(prediction, ReferenceAnswer(value=value, answer_type=answer_type), config)


def test_verifier_status_members() -> None:
    assert {s.value for s in VerifierStatus} == {
        "correct",
        "incorrect",
        "invalid_prediction",
        "invalid_reference",
        "timeout",
        "internal_error",
    }


def test_reward_is_one_only_for_correct() -> None:
    assert verify("2", "2", AnswerType.INTEGER).reward == 1.0
    assert verify("3", "2", AnswerType.INTEGER).reward == 0.0
    assert verify("abc", "2", AnswerType.INTEGER).reward == 0.0


def test_integer_exact_equality() -> None:
    assert verify("2", "2", AnswerType.INTEGER).status is VerifierStatus.CORRECT
    assert verify("0042", "42", AnswerType.INTEGER).status is VerifierStatus.CORRECT
    assert verify("−42", "-42", AnswerType.INTEGER).status is VerifierStatus.CORRECT
    assert verify("1,234", "1234", AnswerType.INTEGER).status is VerifierStatus.CORRECT
    assert verify("2", "3", AnswerType.INTEGER).status is VerifierStatus.INCORRECT


def test_numeric_coercion_is_gated_by_answer_type() -> None:
    # 2 equals 2.0 only when the answer type permits numeric coercion
    assert verify("2.0", "2", AnswerType.INTEGER).status is VerifierStatus.INCORRECT
    assert verify("2.0", "2", AnswerType.RATIONAL).status is VerifierStatus.CORRECT


def test_rational_fraction_decimal_equivalence() -> None:
    assert verify("0.5", "1/2", AnswerType.RATIONAL).status is VerifierStatus.CORRECT
    assert verify("2/4", "1/2", AnswerType.RATIONAL).status is VerifierStatus.CORRECT
    # 1/3 is not 0.333 without a configured tolerance
    assert verify("0.333", "1/3", AnswerType.RATIONAL).status is VerifierStatus.INCORRECT
    assert verify("0.333", "1/3", AnswerType.REAL).status is VerifierStatus.INCORRECT


def test_denominator_zero_is_invalid() -> None:
    assert verify("1/0", "1/2", AnswerType.RATIONAL).status is VerifierStatus.INVALID_PREDICTION
    assert verify("1/2", "1/0", AnswerType.RATIONAL).status is VerifierStatus.INVALID_REFERENCE


def test_real_tolerance_uses_atol_plus_rtol() -> None:
    result = verify("3.14159265358979", "3.141592653589793", AnswerType.REAL)
    assert result.status is VerifierStatus.CORRECT
    assert result.details["atol"] == 1e-9
    assert result.details["rtol"] == 1e-9
    assert float(result.details["abs_diff"]) < 1e-12
    assert verify("0.2", "0.1", AnswerType.REAL).status is VerifierStatus.INCORRECT


def test_real_zero_reference_relies_on_atol() -> None:
    assert verify("0.0000000001", "0", AnswerType.REAL).status is VerifierStatus.CORRECT
    assert (
        verify("0.0000000001", "0", AnswerType.REAL, real_atol=1e-12).status
        is VerifierStatus.INCORRECT
    )


def test_scientific_notation() -> None:
    assert verify("1e3", "1000", AnswerType.REAL).status is VerifierStatus.CORRECT
    assert verify("1e3", "1000", AnswerType.RATIONAL).status is VerifierStatus.CORRECT


def test_acceptable_forms_are_tried() -> None:
    reference = ReferenceAnswer(
        value="1/2", answer_type=AnswerType.RATIONAL, acceptable_forms=("0.5",)
    )
    config = VerifierConfig()
    assert verify_answer("0.5", reference, config).status is VerifierStatus.CORRECT
    assert verify_answer("1/4", reference, config).status is VerifierStatus.INCORRECT


def test_thousands_grouping_is_resolved_type_aware() -> None:
    assert verify("1,234", "1234", AnswerType.INTEGER).status is VerifierStatus.CORRECT
    assert verify("1,234,567", "1234567", AnswerType.INTEGER).status is VerifierStatus.CORRECT
    assert verify("1,234.5", "1234.5", AnswerType.RATIONAL).status is VerifierStatus.CORRECT
    # a comma between single digits is not a grouping and must not be accepted
    assert verify("12,34", "1234", AnswerType.INTEGER).status is VerifierStatus.INVALID_PREDICTION


def test_bounded_digit_and_exponent_limits() -> None:
    huge = "1" * 300
    assert verify(huge, "1", AnswerType.INTEGER).status is VerifierStatus.INVALID_PREDICTION
    assert verify("1e9999", "1", AnswerType.REAL).status is VerifierStatus.INVALID_PREDICTION
    assert verify("1e9999", "1e9999", AnswerType.REAL).status is VerifierStatus.INVALID_REFERENCE


def test_invalid_reference_and_prediction_are_distinct() -> None:
    assert verify("abc", "2", AnswerType.INTEGER).status is VerifierStatus.INVALID_PREDICTION
    assert verify("2", "abc", AnswerType.INTEGER).status is VerifierStatus.INVALID_REFERENCE
    result = verify("2", "2", AnswerType.INTEGER)
    assert result.normalized_prediction == "2"
    assert result.normalized_reference == "2"


def test_result_round_trips() -> None:
    result = verify("2", "2", AnswerType.INTEGER)
    assert VerifierResult.model_validate_json(result.model_dump_json()) == result

from adaptive_math.core.types import AnswerType, ReferenceAnswer
from adaptive_math.verifier import VerifierConfig, VerifierResult, VerifierStatus, verify_answer


def verify(prediction: str, value: str, answer_type: AnswerType) -> VerifierResult:
    return verify_answer(prediction, ReferenceAnswer(value=value, answer_type=answer_type), VerifierConfig())


def test_sets_are_order_independent_and_duplicate_insensitive() -> None:
    assert verify("{1,2,3}", "{3,1,2}", AnswerType.SET).status is VerifierStatus.CORRECT
    assert verify("{1,1,2}", "{1,2}", AnswerType.SET).status is VerifierStatus.CORRECT
    assert verify("{1,2}", "{1,3}", AnswerType.SET).status is VerifierStatus.INCORRECT


def test_set_elements_use_exact_numeric_equality() -> None:
    assert verify("{1/2}", "{0.5}", AnswerType.SET).status is VerifierStatus.CORRECT


def test_scalar_never_equals_singleton_set() -> None:
    assert verify("2", "{2}", AnswerType.SET).status is VerifierStatus.INCORRECT


def test_tuples_are_ordered() -> None:
    assert verify("(1,2,3)", "(1,2,3)", AnswerType.TUPLE).status is VerifierStatus.CORRECT
    assert verify("(1,2,3)", "(3,2,1)", AnswerType.TUPLE).status is VerifierStatus.INCORRECT
    assert verify("(1,2)", "(1,2,3)", AnswerType.TUPLE).status is VerifierStatus.INCORRECT
    assert verify("(1/2, 2)", "(0.5, 2)", AnswerType.TUPLE).status is VerifierStatus.CORRECT


def test_intervals_preserve_open_and_closed_endpoints() -> None:
    assert verify("(1,5)", "(1, 5)", AnswerType.INTERVAL).status is VerifierStatus.CORRECT
    assert verify("(1,5)", "[1,5]", AnswerType.INTERVAL).status is VerifierStatus.INCORRECT
    assert verify("[1,5)", "[1,5]", AnswerType.INTERVAL).status is VerifierStatus.INCORRECT
    assert verify("(1.5, 2.5)", "(3/2, 5/2)", AnswerType.INTERVAL).status is VerifierStatus.CORRECT


def test_malformed_collections_are_invalid() -> None:
    assert verify("{1,2", "{1,2}", AnswerType.SET).status is VerifierStatus.INVALID_PREDICTION
    assert verify("((1,2)", "(1,2)", AnswerType.TUPLE).status is VerifierStatus.INVALID_PREDICTION
    assert verify("(1,5)", "(1,x)", AnswerType.INTERVAL).status is VerifierStatus.INVALID_REFERENCE


def test_nested_collections() -> None:
    assert verify("{1, (2,3)}", "{1, (2,3)}", AnswerType.SET).status is VerifierStatus.CORRECT
    assert verify("{1, {2, 3}}", "{1, {3, 2}}", AnswerType.SET).status is VerifierStatus.CORRECT
    assert verify("{1, (2,3)}", "{(2,3), 1}", AnswerType.SET).status is VerifierStatus.CORRECT
    assert verify("{1, {2, 3}}", "{1, {2, 4}}", AnswerType.SET).status is VerifierStatus.INCORRECT


def test_empty_set() -> None:
    assert verify("{}", "{}", AnswerType.SET).status is VerifierStatus.CORRECT
    assert verify("{}", "{1}", AnswerType.SET).status is VerifierStatus.INCORRECT


def test_collection_size_and_depth_limits() -> None:
    too_many = "{" + ",".join(str(i) for i in range(129)) + "}"
    assert verify(too_many, too_many, AnswerType.SET).status is VerifierStatus.INVALID_REFERENCE
    assert verify(too_many, "{1}", AnswerType.SET).status is VerifierStatus.INVALID_PREDICTION
    deep = "{" * 10 + "1" + "}" * 10
    assert verify(deep, deep, AnswerType.SET).status is VerifierStatus.INVALID_REFERENCE
    assert verify(deep, "{1}", AnswerType.SET).status is VerifierStatus.INVALID_PREDICTION


def test_wrong_kind_predictions_are_incorrect() -> None:
    assert verify("(1,2)", "{1,2}", AnswerType.SET).status is VerifierStatus.INCORRECT
    assert verify("2.0", "2", AnswerType.INTEGER).status is VerifierStatus.INCORRECT

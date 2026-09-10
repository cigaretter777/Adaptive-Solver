"""Offline-only answer evaluation boundary for reinforcement learning."""

from adaptive_math.core.types import ReferenceAnswer
from adaptive_math.verifier.service import VerifierResult, verify_answer


class HiddenVerifier:
    """Retains a reference privately and exposes only a terminal verdict."""

    def __init__(self, reference: ReferenceAnswer, task_id: str) -> None:
        self.__reference = reference
        self.__task_id = task_id

    def evaluate(self, answer: str) -> VerifierResult:
        return verify_answer(answer, self.__reference, task_id=self.__task_id)

from adaptive_math.core.hashing import make_task_id
from adaptive_math.core.types import AnswerType, LabeledMathTask, MathTask, ReferenceAnswer
from adaptive_math.data.deduplicate import deduplicate


def make_task(problem: str, dataset: str = "d", split: str = "train") -> LabeledMathTask:
    return LabeledMathTask(
        task=MathTask(
            task_id=make_task_id(dataset, problem),
            problem=problem,
            answer_type=AnswerType.INTEGER,
            dataset=dataset,
            split=split,
            source_hash="a" * 64,
            pipeline_version="canonicalize-v1",
        ),
        reference=ReferenceAnswer(value="1", answer_type=AnswerType.INTEGER),
    )


def long_problem(suffix: str) -> str:
    return (
        "Consider the sequence of positive integers defined recursively by "
        "a_{n+1} = a_n^2 - a_n + 1 with initial term a_1 = 2. Determine the "
        "remainder when a_2024 is divided by 997. Justify every step of your "
        "computation and express the final result as an integer " + suffix
    )


def test_exact_duplicates_are_removed() -> None:
    tasks = [
        make_task("What is 1+1?"),
        make_task("What is 1+1?"),
        make_task("What is 2+2?"),
    ]
    result = deduplicate(tasks, seed=1, eval_datasets=set())
    assert len(result.kept) == 2
    assert result.exact_removed == 1
    assert result.near_removed == 0


def test_near_duplicates_above_threshold_are_removed() -> None:
    a = make_task(long_problem("ending one."))
    b = make_task(long_problem("ending one?"))
    result = deduplicate([a, b], seed=1, eval_datasets=set(), threshold=0.90)
    assert len(result.kept) == 1
    assert result.near_removed == 1


def test_distinct_problems_are_kept() -> None:
    tasks = [
        make_task("Prove that every prime greater than 2 is odd."),
        make_task("Compute the area of a circle with radius 3."),
        make_task("Solve the system of equations x+y=3 and x-y=1."),
    ]
    result = deduplicate(tasks, seed=1, eval_datasets=set())
    assert len(result.kept) == 3


def test_eval_members_win_in_cross_split_clusters() -> None:
    train = make_task(long_problem("shared tail."), split="train")
    frozen = make_task(long_problem("shared tail?"), dataset="eval_src", split="frozen_eval")
    result = deduplicate([train, frozen], seed=1, eval_datasets={"eval_src"})
    assert len(result.kept) == 1
    assert result.kept[0].task.split == "frozen_eval"


def test_exact_duplicate_across_splits_keeps_eval() -> None:
    train = make_task("What is 7*8?", split="train")
    frozen = make_task("What is 7*8?", dataset="eval_src", split="frozen_eval")
    result = deduplicate([train, frozen], seed=1, eval_datasets={"eval_src"})
    assert len(result.kept) == 1
    assert result.kept[0].task.dataset == "eval_src"


def test_deduplication_is_deterministic() -> None:
    tasks = [
        make_task("What is 1+1?"),
        make_task(long_problem("a.")),
        make_task(long_problem("a?")),
        make_task("What is 2+2?"),
    ]
    first = deduplicate(tasks, seed=7, eval_datasets=set())
    second = deduplicate(list(reversed(tasks)), seed=7, eval_datasets=set())
    assert [t.task.task_id for t in first.kept] == [t.task.task_id for t in second.kept]
    assert first.exact_removed == second.exact_removed
    assert first.near_removed == second.near_removed

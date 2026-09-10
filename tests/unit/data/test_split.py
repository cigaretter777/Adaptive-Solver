from adaptive_math.core.hashing import make_task_id
from adaptive_math.core.types import AnswerType, LabeledMathTask, MathTask, ReferenceAnswer
from adaptive_math.data.sources import SourceRegistry, SourceSpec
from adaptive_math.data.split import SplitConfig, assign_splits


def make_task(i: int, dataset: str) -> LabeledMathTask:
    problem = f"What is the value of {i}+{i}?"
    return LabeledMathTask(
        task=MathTask(
            task_id=make_task_id(dataset, problem),
            problem=problem,
            answer_type=AnswerType.INTEGER,
            dataset=dataset,
            split="unassigned",
            source_hash="a" * 64,
            pipeline_version="canonicalize-v1",
        ),
        reference=ReferenceAnswer(value=str(2 * i), answer_type=AnswerType.INTEGER),
    )


def make_registry() -> SourceRegistry:
    return SourceRegistry(
        version="1.0",
        sources={
            "train_src": SourceSpec.model_validate(
                {
                    "name": "train_src",
                    "uri": "file://synthetic",
                    "revision": "a" * 40,
                    "license": "apache-2.0",
                    "intended_use": "train",
                    "citation": "synthetic",
                    "loader": "synthetic",
                    "loader_params": {},
                    "answer_type": AnswerType.INTEGER,
                }
            ),
            "eval_src": SourceSpec.model_validate(
                {
                    "name": "eval_src",
                    "uri": "file://synthetic",
                    "revision": "b" * 40,
                    "license": "apache-2.0",
                    "intended_use": "eval",
                    "citation": "synthetic",
                    "loader": "synthetic",
                    "loader_params": {},
                    "answer_type": AnswerType.INTEGER,
                }
            ),
        },
    )


def test_frozen_eval_never_overlaps_training_splits() -> None:
    registry = make_registry()
    tasks = [make_task(i, "train_src") for i in range(200)] + [
        make_task(i + 1000, "eval_src") for i in range(20)
    ]
    for seed in (1, 2, 3, 20260910):
        splits = assign_splits(tasks, registry, seed=seed, config=SplitConfig())
        frozen_ids = {t.task.task_id for t in splits["frozen_eval"]}
        for name in ("train", "sft_dev", "rl_dev"):
            assert frozen_ids.isdisjoint({t.task.task_id for t in splits[name]})
        assert len(splits["frozen_eval"]) == 20  # all eval-source tasks


def test_train_split_ratios_are_respected() -> None:
    registry = make_registry()
    tasks = [make_task(i, "train_src") for i in range(1000)]
    splits = assign_splits(tasks, registry, seed=42, config=SplitConfig())
    assert len(splits["train"]) == 900
    assert len(splits["sft_dev"]) == 50
    assert len(splits["rl_dev"]) == 50
    assert splits["frozen_eval"] == []


def test_split_assignment_is_deterministic() -> None:
    registry = make_registry()
    tasks = [make_task(i, "train_src") for i in range(100)] + [
        make_task(i + 500, "eval_src") for i in range(10)
    ]
    first = assign_splits(tasks, registry, seed=11, config=SplitConfig())
    second = assign_splits(list(reversed(tasks)), registry, seed=11, config=SplitConfig())
    for name in ("train", "sft_dev", "rl_dev", "frozen_eval"):
        assert [t.task.task_id for t in first[name]] == [
            t.task.task_id for t in second[name]
        ]


def test_relabeled_tasks_carry_the_assigned_split() -> None:
    registry = make_registry()
    tasks = [make_task(i, "train_src") for i in range(30)]
    splits = assign_splits(tasks, registry, seed=5, config=SplitConfig())
    for name, members in splits.items():
        assert all(t.task.split == name for t in members)

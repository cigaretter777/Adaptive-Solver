import sys
from types import SimpleNamespace

from adaptive_math.data.sources import SourceSpec, _adapt_records, load_source_records


def test_dapo_records_are_adapted_to_canonical_problem_and_answer_fields() -> None:
    spec = SourceSpec.model_validate(
        {
            "name": "dapo_math_17k",
            "uri": "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k",
            "revision": "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
            "license": "apache-2.0",
            "intended_use": "train",
            "citation": "DAPO",
            "loader": "dapo",
            "answer_type": "integer",
        }
    )
    records = [
        {
            "prompt": [{"role": "user", "content": "Solve 2 + 2."}],
            "reward_model": {"ground_truth": "4"},
            "extra_info": {"index": "example"},
        }
    ]

    assert _adapt_records(spec, records) == [
        {
            "problem": "Solve 2 + 2.",
            "answer": "4",
            "source_index": "example",
        }
    ]


def test_huggingface_loader_passes_the_pinned_revision(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def load_dataset(**kwargs):
        calls.append(kwargs)
        return [{"problem": "What is 1+1?", "solution": "\\boxed{2}"}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    spec = SourceSpec.model_validate(
        {
            "name": "numinamath_tir",
            "uri": "https://huggingface.co/datasets/AI-MO/NuminaMath-TIR",
            "revision": "77a91d7b7a1a98ac4b1beb7d86c09d156b935dcd",
            "license": "apache-2.0",
            "intended_use": "train",
            "citation": "NuminaMath",
            "loader": "numinamath_tir",
            "loader_params": {"dataset": "AI-MO/NuminaMath-TIR"},
            "answer_type": "expression",
        }
    )

    records, resolved = load_source_records(spec, sample=1)
    assert records == [{"problem": "What is 1+1?", "solution": "\\boxed{2}"}]
    assert resolved is None
    assert calls == [
        {
            "path": "AI-MO/NuminaMath-TIR",
            "revision": "77a91d7b7a1a98ac4b1beb7d86c09d156b935dcd",
            "split": "train[:1]",
        }
    ]

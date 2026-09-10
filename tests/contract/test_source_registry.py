from pathlib import Path

import yaml

from adaptive_math.data.sources import SourceRegistry

REGISTRY_PATH = Path(__file__).parents[2] / "configs" / "data" / "sources.yaml"


def test_primary_registry_contains_only_approved_train_and_eval_sources() -> None:
    registry = SourceRegistry.model_validate(yaml.safe_load(REGISTRY_PATH.read_text()))

    assert set(registry.sources) == {
        "openr1_math_220k",
        "dapo_math_17k",
        "numinamath_tir",
        "omni_math",
    }
    assert {name for name, spec in registry.sources.items() if spec.intended_use == "train"} == {
        "openr1_math_220k",
        "dapo_math_17k",
        "numinamath_tir",
    }
    assert registry.eval_source_names() == {"omni_math"}
    assert all(len(spec.revision) == 40 for spec in registry.sources.values())
    assert {spec.license for spec in registry.sources.values()} <= {"apache-2.0"}

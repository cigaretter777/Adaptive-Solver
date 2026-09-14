"""Config-workflow tests for run_sft.py — no torch required.

The training itself is exercised by the GPU-gated
tests/integration/test_sft_overfit.py; here we pin config loading,
documented overrides, placeholder rejection, manifest-hash gating and
atomic artifact writing.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from adaptive_math.core.hashing import sha256_hex
from adaptive_math.training.config import SFTConfig

REPO_ROOT = Path(__file__).parents[3]
CONFIGS = REPO_ROOT / "configs" / "sft"

_spec = importlib.util.spec_from_file_location("run_sft", REPO_ROOT / "scripts" / "train" / "run_sft.py")
run_sft = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(run_sft)

QWEN3_1_7B_SHA = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
QWEN3_4B_SHA = "1cfa9a7208912126459214e8b04321603b3df60c"


@pytest.mark.parametrize(
    ("filename", "model_id", "revision"),
    [
        ("qwen3_1_7b_lora.yaml", "Qwen/Qwen3-1.7B", QWEN3_1_7B_SHA),
        ("qwen3_1_7b_smoke.yaml", "Qwen/Qwen3-1.7B", QWEN3_1_7B_SHA),
        ("qwen3_4b_lora.yaml", "Qwen/Qwen3-4B", QWEN3_4B_SHA),
    ],
)
def test_shipped_configs_load_with_immutable_revisions(
    filename: str, model_id: str, revision: str
) -> None:
    config = run_sft.load_config(CONFIGS / filename)
    assert config.model_id == model_id
    assert config.model_revision == revision
    assert config.tokenizer_revision == revision
    assert config.precision == "bf16"


def test_shipped_configs_use_expected_data_hash_policy() -> None:
    # Template configs stay guarded by a placeholder hash, while the frozen
    # production dp_v1 config is intentionally bound to a real manifest hash.
    formal_config = "qwen3_1_7b_dp_v1.yaml"

    for path in sorted(CONFIGS.glob("*.yaml")):
        config = run_sft.load_config(path)

        if path.name == formal_config:
            assert config.data_manifest_sha256 != run_sft.PLACEHOLDER_HASH
        else:
            assert config.data_manifest_sha256 == run_sft.PLACEHOLDER_HASH
            with pytest.raises(SystemExit):
                run_sft.main(["--config", str(path), "--dry-run"])


def _usable_args(config_path: Path, manifest: Path, *extra: str) -> list[str]:
    digest = sha256_hex(manifest.read_bytes())
    return [
        "--config", str(config_path),
        "--set", f"data_manifest_sha256={digest}",
        *extra,
    ]


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    path = tmp_path / "sft_manifest.json"
    path.write_text(json.dumps({"records": 200}))
    return path


def test_dry_run_prints_resolved_config_and_writes_nothing(
    manifest: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _usable_args(CONFIGS / "qwen3_1_7b_lora.yaml", manifest, "--dry-run")
    assert run_sft.main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["model_id"] == "Qwen/Qwen3-1.7B"
    assert payload["overrides"]["data_manifest_sha256"] == sha256_hex(manifest.read_bytes())
    assert not (tmp_path / "artifacts").exists()


def test_documented_lr_pilot_override(manifest: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _usable_args(
        CONFIGS / "qwen3_1_7b_lora.yaml", manifest, "--set", "learning_rate=2e-5", "--dry-run"
    )
    assert run_sft.main(args) == 0
    assert json.loads(capsys.readouterr().out)["config"]["learning_rate"] == pytest.approx(2e-5)


def test_override_rejects_unknown_key_and_bad_value(manifest: Path) -> None:
    with pytest.raises(SystemExit):
        run_sft.main(_usable_args(CONFIGS / "qwen3_1_7b_smoke.yaml", manifest, "--set", "bogus=1", "--dry-run"))
    with pytest.raises(SystemExit):
        run_sft.main(
            _usable_args(
                CONFIGS / "qwen3_1_7b_smoke.yaml", manifest,
                "--set", "learning_rate=not_a_number", "--dry-run",
            )
        )


def test_manifest_hash_mismatch_blocks_training(manifest: Path) -> None:
    wrong = "f" * 64
    with pytest.raises(SystemExit):
        run_sft.main(
            [
                "--config", str(CONFIGS / "qwen3_1_7b_smoke.yaml"),
                "--set", f"data_manifest_sha256={wrong}",
                "--data", "unused.parquet",
                "--data-manifest", str(manifest),
            ]
        )


def test_matching_hash_writes_run_headers_then_training(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[SFTConfig, Path]] = []

    def fake_training(config: SFTConfig, data: Path) -> None:
        calls.append((config, data))

    monkeypatch.setattr(run_sft, "run_training", fake_training)
    output = tmp_path / "run"
    data = tmp_path / "records.parquet"
    data.write_bytes(b"parquet")
    args = _usable_args(
        CONFIGS / "qwen3_1_7b_smoke.yaml",
        manifest,
        "--set", f"output_dir={output}",
        "--data", str(data),
        "--data-manifest", str(manifest),
    )
    assert run_sft.main(args) == 0
    assert len(calls) == 1

    resolved = yaml.safe_load((output / "resolved_config.yaml").read_text())
    assert resolved["overrides"]["output_dir"] == str(output)
    assert len(resolved["config_hash"]) == 64
    environment = json.loads((output / "environment.json").read_text())
    assert environment["packages"]["torch"] in ("missing",) or environment["packages"]["torch"]
    assert "git_sha" in environment


def test_atomic_write_text_overwrites_without_temp_residue(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "file.txt"
    run_sft.atomic_write_text(target, "first")
    run_sft.atomic_write_text(target, "second")
    assert target.read_text() == "second"
    assert list(tmp_path.rglob("*.tmp")) == []


def test_atomic_replace_dir_swaps_previous_content(tmp_path: Path) -> None:
    final = tmp_path / "adapter"
    final.mkdir()
    (final / "old.bin").write_text("old")
    staged = tmp_path / "adapter.tmp"
    staged.mkdir()
    (staged / "new.bin").write_text("new")
    run_sft.atomic_replace_dir(staged, final)
    assert (final / "new.bin").read_text() == "new"
    assert not (final / "old.bin").exists()
    assert not staged.exists()
    assert list(tmp_path.glob("*.old")) == []


def test_final_metrics_keep_aggregate_and_last_logged_loss_distinct() -> None:
    row = run_sft.final_metrics_row(
        {"train_loss": 0.733803628146319},
        [{"loss": 0.9}, {"loss": 0.8432}, {"train_loss": 0.733803628146319}],
        records=1447,
        rejected=0,
    )
    assert row == {
        "event": "final",
        "train_loss": 0.733803628146319,
        "last_logged_loss": 0.8432,
        "records": 1447,
        "rejected": 0,
    }

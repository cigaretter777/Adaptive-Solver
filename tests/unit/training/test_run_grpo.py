import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
_spec = importlib.util.spec_from_file_location("run_grpo", REPO_ROOT / "scripts" / "train" / "run_grpo.py")
run_grpo = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(run_grpo)


def test_grpo_config_requires_immutable_input_hashes(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("task_pool_sha256: '0'\n")

    with pytest.raises(run_grpo.ConfigError, match="required keys"):
        run_grpo.load_config(config)


def test_grpo_dry_run_writes_nothing_and_reports_pinned_backend(capsys) -> None:
    assert run_grpo.main(["--config", "configs/grpo/qwen3_1_7b_smoke_r0.yaml", "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "20bd331bdbc9026a5668e11362178e10ab7400c8" in output
    assert "adaptive_math" in output


def test_grpo_output_directory_can_be_overridden_for_unique_cloud_run(tmp_path: Path) -> None:
    config = run_grpo.load_config(REPO_ROOT / "configs/grpo/qwen3_1_7b_smoke_r0.yaml")

    overridden = run_grpo.with_output_dir(config, tmp_path / "unique-run")

    assert overridden.output_dir == tmp_path / "unique-run"

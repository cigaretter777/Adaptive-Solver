from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts" / "eval" / "run_rollout_health.py"


def load_runner():
    spec = spec_from_file_location("run_rollout_health", RUNNER)
    assert spec is not None
    assert spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_match_rollout_health_smoke_gate() -> None:
    runner = load_runner()
    args = runner.build_parser().parse_args([])

    assert Path(args.data) == Path("data/processed/v1/rl_dev.parquet")
    assert Path(args.model) == Path(
        "artifacts/models/qwen3_1_7b_sft_dp_v1_merged"
    )
    assert Path(args.agent_config) == Path("configs/agent/default.yaml")
    assert Path(args.reward_config) == Path("configs/reward/r0.yaml")

    assert args.task_count == 10
    assert args.group_size == 4
    assert args.selection_seed == 42
    assert args.max_new_tokens == 1024
    assert args.temperature == 0.7


def test_default_output_dir_matches_gate_shape() -> None:
    runner = load_runner()
    args = runner.build_parser().parse_args([])

    assert runner.default_output_dir(args) == Path(
        "artifacts/rollout_health/smoke_10x4_seed42"
    )


def test_prepare_output_dir_rejects_non_empty_directory(tmp_path: Path) -> None:
    runner = load_runner()

    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "partial.jsonl").write_text("partial")

    try:
        runner.prepare_output_dir(output_dir)
    except FileExistsError:
        pass
    else:
        raise AssertionError("non-empty output directory must be rejected")

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
_spec = importlib.util.spec_from_file_location(
    "check_backend_contract", REPO_ROOT / "scripts" / "train" / "check_backend_contract.py"
)
backend_contract = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(backend_contract)


def test_backend_contract_dry_run_validates_pinned_manifest(capsys) -> None:
    assert backend_contract.main(["--manifest", "third_party/manifest.json", "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "20bd331bdbc9026a5668e11362178e10ab7400c8" in output

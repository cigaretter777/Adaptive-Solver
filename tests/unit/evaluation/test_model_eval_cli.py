"""Evaluation preflight rejects changed data before loading a model."""

import importlib.util
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

import adaptive_math
from adaptive_math.agent.model_client import ChatMessage
from adaptive_math.core.hashing import sha256_hex
from adaptive_math.training.config import SFTConfig
from adaptive_math.training.sft_io import write_records
from adaptive_math.training.sft_records import SFTBehavior, SFTTrajectoryRecord
from adaptive_math.verifier.service import VerifierResult, VerifierStatus

ROOT = Path(__file__).parents[3]
SPEC = importlib.util.spec_from_file_location("run_model_eval", ROOT / "scripts/eval/run_model_eval.py")
run_model_eval = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_model_eval)


def fixture_inputs(tmp_path: Path) -> dict[str, Path]:
    eval_path = tmp_path / "frozen_eval.parquet"
    rows = [{
        "task_id": f"omni_math:{i}", "problem": f"What is {i}+1?", "answer_type": "integer",
        "dataset": "omni_math", "split": "frozen_eval", "source_hash": "a" * 64,
        "pipeline_version": "test", "metadata": "{}", "reference_value": str(i + 1),
        "reference_acceptable_forms": [],
    } for i in (2, 1)]
    pq.write_table(pa.Table.from_pylist(rows), eval_path)
    data_manifest = tmp_path / "data_manifest.json"
    data_manifest.write_text(json.dumps({"splits": {"frozen_eval": {
        "file_hash": sha256_hex(eval_path.read_bytes()), "count": 2,
    }}}))
    sft_path = tmp_path / "sft.parquet"
    write_records([SFTTrajectoryRecord(
        task_id="openr1:1", behavior=SFTBehavior.DIRECT,
        messages=(ChatMessage(role="system", content="solve"),
                  ChatMessage(role="user", content="1+0"),
                  ChatMessage(role="assistant", content='<final>{"answer":"1"}</final>')),
        final_answer="1",
        verifier_result=VerifierResult(status=VerifierStatus.CORRECT, reward=1,
                                       normalized_prediction="1", normalized_reference="1"),
        source_trace_ids=("trace",), tool_execution_ids=(),
        tokenizer_revision="test", transform_version="test",
    )], sft_path)
    sft_manifest = tmp_path / "sft_manifest.json"
    sft_manifest.write_text(json.dumps({
        "parquet_sha256": sha256_hex(sft_path.read_bytes()), "total_records": 1,
    }))
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "Qwen/Qwen3-1.7B",
    }))
    training_config = SFTConfig.model_validate({
        **yaml.safe_load((ROOT / "configs/sft/qwen3_1_7b_lora.yaml").read_text()),
        "model_revision": "a" * 40, "tokenizer_revision": "a" * 40,
        "data_manifest_sha256": sha256_hex(sft_manifest.read_bytes()),
    })
    config_hash = sha256_hex(training_config.model_dump_json().encode())
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump({
        "config": training_config.model_dump(mode="json"), "config_hash": config_hash,
    }))
    (adapter / "COMPLETE").write_text("2026-09-13T00:00:00+00:00\n" + config_hash)
    return {"eval": eval_path, "data_manifest": data_manifest, "sft": sft_path,
            "sft_manifest": sft_manifest, "adapter": adapter}


def test_preflight_selects_stable_prefix_and_records_hashes(tmp_path: Path) -> None:
    paths = fixture_inputs(tmp_path)
    tasks, sft_ids, manifest = run_model_eval.preflight(
        eval_parquet=paths["eval"], data_manifest=paths["data_manifest"],
        sft_parquet=paths["sft"], sft_manifest=paths["sft_manifest"],
        adapter=paths["adapter"], limit=1,
        model_id="Qwen/Qwen3-1.7B", model_revision="a" * 40,
        tokenizer_revision="a" * 40, max_new_tokens=1024, batch_size=2,
    )
    assert [task.task.task_id for task in tasks] == ["omni_math:1"]
    assert sft_ids == {"openr1:1"}
    assert manifest["adapter_sha256"] == sha256_hex(b"adapter")
    assert manifest["generation_config"]["do_sample"] is False
    assert manifest["generation_config"]["batch_size"] == 2


def test_preflight_rejects_incomplete_adapter(tmp_path: Path) -> None:
    paths = fixture_inputs(tmp_path)
    (paths["adapter"] / "COMPLETE").unlink()
    with pytest.raises(ValueError, match="COMPLETE"):
        run_model_eval.preflight(
            eval_parquet=paths["eval"], data_manifest=paths["data_manifest"],
            sft_parquet=paths["sft"], sft_manifest=paths["sft_manifest"],
            adapter=paths["adapter"], limit=1,
            model_id="Qwen/Qwen3-1.7B", model_revision="a" * 40,
            tokenizer_revision="a" * 40, max_new_tokens=1024,
        )


def test_preflight_rejects_unexpected_adapter_hash(tmp_path: Path) -> None:
    paths = fixture_inputs(tmp_path)
    with pytest.raises(ValueError, match="adapter SHA-256"):
        run_model_eval.preflight(
            eval_parquet=paths["eval"], data_manifest=paths["data_manifest"],
            sft_parquet=paths["sft"], sft_manifest=paths["sft_manifest"],
            adapter=paths["adapter"], limit=1,
            model_id="Qwen/Qwen3-1.7B", model_revision="a" * 40,
            tokenizer_revision="a" * 40, max_new_tokens=1024,
            expected_adapter_sha256="0" * 64,
        )


def test_preflight_rejects_adapter_trained_on_another_sft_manifest(tmp_path: Path) -> None:
    paths = fixture_inputs(tmp_path)
    resolved_path = tmp_path / "resolved_config.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved["config"]["data_manifest_sha256"] = "b" * 64
    resolved_path.write_text(yaml.safe_dump(resolved))
    with pytest.raises(ValueError, match="training config"):
        run_model_eval.preflight(
            eval_parquet=paths["eval"], data_manifest=paths["data_manifest"],
            sft_parquet=paths["sft"], sft_manifest=paths["sft_manifest"],
            adapter=paths["adapter"], limit=1,
            model_id="Qwen/Qwen3-1.7B", model_revision="a" * 40,
            tokenizer_revision="a" * 40, max_new_tokens=1024,
        )


def test_preflight_rejects_eval_hash_mismatch(tmp_path: Path) -> None:
    paths = fixture_inputs(tmp_path)
    paths["eval"].write_bytes(paths["eval"].read_bytes() + b"changed")
    with pytest.raises(ValueError, match="eval parquet hash"):
        run_model_eval.preflight(
            eval_parquet=paths["eval"], data_manifest=paths["data_manifest"],
            sft_parquet=paths["sft"], sft_manifest=paths["sft_manifest"],
            adapter=paths["adapter"], limit=1,
            model_id="Qwen/Qwen3-1.7B", model_revision="a" * 40,
            tokenizer_revision="a" * 40, max_new_tokens=1024,
        )


def test_generator_batches_prompts_with_attention_mask_and_disables_adapter_for_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[tuple[str, bool]] = []

    class Tensor:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

        def to(self, _device):
            return self

    class Output:
        def __getitem__(self, _index):
            return Tensor((3,))

    class BatchEncoding(dict):
        def to(self, _device):
            return self

    class Tokenizer:
        chat_template = "template"
        eos_token_id = 0

        def apply_chat_template(self, payload, **_kwargs):
            assert payload[0]["role"] == "user"
            return payload[0]["content"]

        def __call__(self, prompts, **kwargs):
            assert prompts in (["1+0", "2+0"], ["1+0"])
            assert kwargs == {"return_tensors": "pt", "padding": True,
                              "return_attention_mask": True}
            return BatchEncoding(input_ids=Tensor((2, 2)), attention_mask=Tensor((2, 2)))

        def decode(self, _tokens, **_kwargs):
            return '<final>{"answer":"1"}</final>'

    class Model:
        disabled = False

        def parameters(self):
            return iter([types.SimpleNamespace(device="cpu")])

        def eval(self):
            return self

        @contextmanager
        def disable_adapter(self):
            self.disabled = True
            try:
                yield
            finally:
                self.disabled = False

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            assert "attention_mask" in kwargs
            calls.append(("generate", self.disabled))
            return Output()

    @contextmanager
    def inference_mode():
        yield

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        bfloat16="bf16", inference_mode=inference_mode,
    ))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *_a, **_kw: Tokenizer()),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *_a, **_kw: Model()),
    ))
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(
        PeftModel=types.SimpleNamespace(from_pretrained=lambda base, _path: base),
    ))
    generate = run_model_eval.make_generator("Qwen/Qwen3-1.7B", "a" * 40, "a" * 40,
                                              tmp_path, 16)
    message = (types.SimpleNamespace(model_dump=lambda: {"role": "user", "content": "1+0"}),)
    second_message = (types.SimpleNamespace(model_dump=lambda: {"role": "user", "content": "2+0"}),)
    turns = generate.generate_batch("base", [(None, message), (None, second_message)])
    assert [turn.text for turn in turns] == ['<final>{"answer":"1"}</final>'] * 2
    assert generate("sft", None, message).text.startswith("<final>")
    assert calls == [("generate", True), ("generate", False)]


def test_preflight_rejects_editable_install_from_another_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    paths = fixture_inputs(tmp_path)
    monkeypatch.setattr(adaptive_math, "__file__", str(tmp_path / "other_checkout" / "__init__.py"))
    with pytest.raises(ValueError, match="import path"):
        run_model_eval.preflight(
            eval_parquet=paths["eval"], data_manifest=paths["data_manifest"],
            sft_parquet=paths["sft"], sft_manifest=paths["sft_manifest"],
            adapter=paths["adapter"], limit=1,
            model_id="Qwen/Qwen3-1.7B", model_revision="a" * 40,
            tokenizer_revision="a" * 40, max_new_tokens=1024,
        )

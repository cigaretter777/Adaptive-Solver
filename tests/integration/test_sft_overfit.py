"""GPU-gated 200-record SFT overfit smoke (Training Plan Task 4).

Not default CI: set ADAPTIVE_MATH_RUN_GPU_TESTS=1 on a CUDA machine with
the `runtime` extra installed and HF access to Qwen/Qwen3-1.7B.

Gates: loss decreases, the saved adapter reloads, and >=98% of training
prompts emit a parseable terminal FINAL action under greedy generation.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from adaptive_math.agent.actions import FinalAction
from adaptive_math.agent.model_client import ChatMessage
from adaptive_math.agent.parser import parse_action
from adaptive_math.core.hashing import sha256_hex
from adaptive_math.training.sft_io import write_records
from adaptive_math.training.sft_records import SFTBehavior, SFTTrajectoryRecord
from adaptive_math.verifier.service import VerifierResult, VerifierStatus

pytestmark = pytest.mark.skipif(
    os.environ.get("ADAPTIVE_MATH_RUN_GPU_TESTS") != "1",
    reason="GPU smoke: set ADAPTIVE_MATH_RUN_GPU_TESTS=1 (downloads Qwen3-1.7B)",
)

REPO_ROOT = Path(__file__).parents[2]
_spec = importlib.util.spec_from_file_location(
    "run_sft", REPO_ROOT / "scripts" / "train" / "run_sft.py"
)
run_sft = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(run_sft)

SYSTEM_PROMPT = "Solve the arithmetic problem. Respond with exactly one final action."


def _record(index: int) -> SFTTrajectoryRecord:
    a, b = index % 20 + 1, index // 20 + 1
    problem = f"What is {a}+{b}?"
    answer = str(a + b)
    return SFTTrajectoryRecord(
        task_id=f"overfit:{index:04d}",
        behavior=SFTBehavior.DIRECT,
        messages=(
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=problem),
            ChatMessage(role="assistant", content=f'<final>{{"answer":"{answer}"}}</final>'),
        ),
        final_answer=answer,
        verifier_result=VerifierResult(
            status=VerifierStatus.CORRECT,
            reward=1.0,
            normalized_prediction=answer,
            normalized_reference=answer,
        ),
        source_trace_ids=("overfit",),
        tool_execution_ids=(),
        tokenizer_revision="unresolved",
        transform_version="sft-v1",
    )


def test_sft_overfit_gates(tmp_path: Path) -> None:
    records = [_record(i) for i in range(200)]
    data = tmp_path / "sft_overfit.parquet"
    write_records(records, data)
    manifest = tmp_path / "sft_overfit_manifest.json"
    manifest.write_text(json.dumps({"records": len(records), "purpose": "overfit-smoke"}))

    output = tmp_path / "run"
    rc = run_sft.main(
        [
            "--config", str(REPO_ROOT / "configs" / "sft" / "qwen3_1_7b_smoke.yaml"),
            "--set", f"data_manifest_sha256={sha256_hex(manifest.read_bytes())}",
            "--set", f"output_dir={output}",
            "--data", str(data),
            "--data-manifest", str(manifest),
        ]
    )
    assert rc == 0

    # gate 1: loss decreases over the overfit run
    metrics = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    losses = [m["loss"] for m in metrics if "loss" in m]
    assert len(losses) >= 4
    assert losses[-1] < losses[0]

    # gate 2: adapter checkpoint is complete and reloads onto the base model
    adapter = output / "adapter"
    assert (adapter / "COMPLETE").exists()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = run_sft.load_config(REPO_ROOT / "configs" / "sft" / "qwen3_1_7b_smoke.yaml")
    base = AutoModelForCausalLM.from_pretrained(
        config.model_id, revision=config.model_revision, torch_dtype=torch.bfloat16
    )
    reloaded = PeftModel.from_pretrained(base, str(adapter))
    assert reloaded.peft_config["default"].r == config.lora_rank

    # gate 3: >=98% of training prompts emit a parseable terminal action
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.tokenizer_revision)
    from adaptive_math.agent.model_client import GenerationConfig
    from adaptive_math.agent.transformers_client import TransformersModelClient

    client = TransformersModelClient(
        tokenizer, reloaded.to("cuda"), model_id=config.model_id
    )
    import asyncio

    valid = 0
    prompts = records[:50]
    for record in prompts:
        turn = asyncio.run(
            client.generate(record.messages[:2], GenerationConfig(max_new_tokens=64, temperature=0.0))
        )
        parsed = parse_action(turn.text)
        if isinstance(parsed.action, FinalAction):
            valid += 1
    assert valid / len(prompts) >= 0.98

"""Exact and near deduplication on canonical problem text.

Exact deduplication hashes canonical_text(problem). Near deduplication uses
character 5-gram MinHash signatures with a fixed similarity threshold and a
seeded hash function, so rebuilds are deterministic. In cross-split
clusters the evaluation member always wins.
"""

import hashlib
from collections.abc import Callable

from datasketch import MinHash, MinHashLSH
from pydantic import BaseModel, ConfigDict

from adaptive_math.core.hashing import canonical_text, sha256_hex
from adaptive_math.core.types import LabeledMathTask


class DedupResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kept: list[LabeledMathTask]
    exact_removed: int
    near_removed: int


def deduplicate(
    tasks: list[LabeledMathTask],
    *,
    seed: int,
    eval_datasets: set[str],
    threshold: float = 0.90,
    num_perm: int = 128,
) -> DedupResult:
    ordered = sorted(tasks, key=lambda t: t.task.task_id)

    groups: dict[str, list[LabeledMathTask]] = {}
    for task in ordered:
        key = sha256_hex(canonical_text(task.task.problem).encode())
        groups.setdefault(key, []).append(task)
    exact_kept = [_winner(members, eval_datasets) for _, members in sorted(groups.items())]
    exact_removed = len(ordered) - len(exact_kept)

    hashfunc = _seeded_hash(seed)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    signatures: dict[str, MinHash] = {}
    for task in exact_kept:
        signature = MinHash(num_perm=num_perm, hashfunc=hashfunc)
        for shingle in _shingles(task.task.problem):
            signature.update(shingle)
        signatures[task.task.task_id] = signature
        lsh.insert(task.task.task_id, signature)

    removed_ids: set[str] = set()
    near_kept: list[LabeledMathTask] = []
    tasks_by_id = {t.task.task_id: t for t in exact_kept}
    for task in exact_kept:
        if task.task.task_id in removed_ids:
            continue
        candidate_ids = {
            cid
            for cid in lsh.query(signatures[task.task.task_id])
            if cid != task.task.task_id and cid not in removed_ids
        }
        cluster_members = [task] + [tasks_by_id[cid] for cid in sorted(candidate_ids)]
        winner = _winner(cluster_members, eval_datasets)
        for member in cluster_members:
            if member.task.task_id != winner.task.task_id:
                removed_ids.add(member.task.task_id)
        if winner.task.task_id == task.task.task_id:
            near_kept.append(task)
    near_removed = len(exact_kept) - len(near_kept)
    return DedupResult(kept=near_kept, exact_removed=exact_removed, near_removed=near_removed)


def _winner(members: list[LabeledMathTask], eval_datasets: set[str]) -> LabeledMathTask:
    def sort_key(task: LabeledMathTask) -> tuple[int, str]:
        is_eval = 0 if task.task.dataset in eval_datasets else 1
        return (is_eval, task.task.task_id)

    return min(members, key=sort_key)


def _shingles(text: str) -> set[bytes]:
    normalized = canonical_text(text)
    if len(normalized) < 5:
        return {normalized.encode()}
    return {normalized[i : i + 5].encode() for i in range(len(normalized) - 4)}


def _seeded_hash(seed: int) -> Callable[[bytes], int]:
    seed_bytes = seed.to_bytes(8, "big", signed=True)

    def hashfn(value: bytes) -> int:
        return int.from_bytes(
            hashlib.sha256(seed_bytes + value).digest()[:8], "big", signed=False
        )

    return hashfn

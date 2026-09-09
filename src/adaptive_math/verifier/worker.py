"""Isolated worker process for symbolic verification.

Untrusted expressions are parsed and compared in a spawned worker process.
Two independent limits protect the parent:

- a per-request CPU timer (SIGPROF) inside the worker interrupts a runaway
  comparison and returns a timeout verdict while the worker survives;
- a parent-side wall-clock deadline kills and replaces the whole worker if
  it stops responding (SIGPROF-blocked C calls, hangs) or crashes.

Resource limits are applied best-effort; RLIMIT_AS is not enforced on macOS,
where the wall-clock and CPU limits still guarantee bounded work.
"""

import multiprocessing as mp
import queue
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from adaptive_math.verifier.symbolic import compare_expressions

Comparator = Callable[[str, str, str], dict[str, object]]


@dataclass(frozen=True)
class WorkerConfig:
    timeout_seconds: float = 2.0
    memory_mb: int = 512
    cpu_seconds: int = 2


class _CpuLimitExceeded(Exception):
    pass


def _with_cpu_limit(
    cpu_seconds: int, comparator: Comparator, prediction: str, reference: str, task_id: str
) -> dict[str, object]:
    def handler(signum: int, frame: object) -> None:
        raise _CpuLimitExceeded

    previous = signal.signal(signal.SIGPROF, handler)
    signal.setitimer(signal.ITIMER_PROF, cpu_seconds, 0)
    try:
        return comparator(prediction, reference, task_id)
    except _CpuLimitExceeded:
        return _error_verdict(
            "timeout", f"cpu limit {cpu_seconds}s exceeded inside the worker"
        )
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGPROF, previous)


def _worker_main(
    task_queue: Any,
    result_queue: Any,
    comparator: Comparator,
    config: WorkerConfig,
) -> None:
    _apply_limits(config)
    while True:
        item = task_queue.get()
        if item is None:  # shutdown sentinel
            break
        request_id, prediction, reference, task_id = item
        try:
            verdict = _with_cpu_limit(
                config.cpu_seconds, comparator, prediction, reference, task_id
            )
        except BaseException as exc:  # the worker must never die on bad input
            verdict = _error_verdict(
                "internal_error", f"comparator raised {type(exc).__name__}"
            )
        try:
            result_queue.put((request_id, verdict))
        except Exception:
            break  # parent is gone


def _apply_limits(config: WorkerConfig) -> None:
    try:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS,
            (config.memory_mb * 1024 * 1024, config.memory_mb * 1024 * 1024),
        )
    except (ImportError, ValueError, OSError):
        pass  # macOS does not enforce RLIMIT_AS; timeouts still bound the work


def _error_verdict(status: str, reason: str) -> dict[str, object]:
    return {
        "status": status,
        "reward": 0.0,
        "normalized_prediction": None,
        "normalized_reference": None,
        "details": {"reason": reason},
    }


class SymbolicWorker:
    """Owns one spawned worker process; replaces it after timeout or crash."""

    def __init__(self, config: WorkerConfig | None = None, comparator: Comparator | None = None) -> None:
        self._config = config or WorkerConfig()
        self._comparator = comparator or compare_expressions
        self._ctx = mp.get_context("spawn")
        self._task_queue: Any = None
        self._result_queue: Any = None
        self._process: Any = None
        self._next_id = 0

    def compare(self, prediction: str, reference: str, task_id: str) -> dict[str, object]:
        self._ensure_worker()
        request_id = self._next_id
        self._next_id += 1
        self._task_queue.put((request_id, prediction, reference, task_id))
        deadline = time.monotonic() + self._config.timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                verdict = _error_verdict("timeout", "worker wall-clock deadline exceeded")
                self._restart()
                return verdict
            try:
                got_id, payload = self._result_queue.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if not self._process.is_alive():
                    verdict = _error_verdict("internal_error", "worker process died")
                    self._reset()
                    return verdict
                continue
            if got_id != request_id:
                continue  # stale result from a previous generation
            return cast("dict[str, object]", payload)

    def close(self) -> None:
        self._shutdown()

    def _ensure_worker(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        if self._process is not None:
            self._join_quietly(self._process)
        self._task_queue = self._ctx.Queue()
        self._result_queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(self._task_queue, self._result_queue, self._comparator, self._config),
            daemon=True,
        )
        self._process.start()

    def _restart(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._join_quietly(self._process)
            if self._process.is_alive():
                self._process.kill()
                self._join_quietly(self._process)
        self._process = None

    def _reset(self) -> None:
        if self._process is not None:
            self._join_quietly(self._process)
        self._process = None

    def _shutdown(self) -> None:
        if self._process is None:
            return
        try:
            self._task_queue.put(None)
        except Exception:
            pass
        self._process.terminate()
        self._join_quietly(self._process)
        self._process = None

    @staticmethod
    def _join_quietly(process: Any) -> None:
        try:
            process.join(timeout=5)
        except Exception:
            pass

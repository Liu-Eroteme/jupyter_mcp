"""Background execution primitives: bounded output buffer, tasks, executor.

Execution is asynchronous under the hood: every run is submitted to a
per-notebook executor thread that owns the kernel connection. Tool calls
wait a bounded time for results; long-running cells keep executing in the
background while their output accumulates in a live buffer the agent can
read mid-run (like a human watching a cell in the Jupyter frontend).

Correctness leans on the existing freshness design: each task captures its
cell's source and revision at submission. A cell edited after submission is
superseded (queued) or keeps its outputs off the file (running) — and the
rev+epoch stamps mean anything questionable simply reads as stale.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

#: adjacent stream writes coalesce into one output; a single coalesced block
#: keeps head+tail once it exceeds these caps (training loops must neither
#: bloat the .ipynb nor flood the agent)
STREAM_HEAD_CHARS = 16_000
STREAM_TAIL_CHARS = 16_000
#: rich outputs (display_data etc.) keep head+tail by count
HEAD_OUTPUTS = 40
TAIL_OUTPUTS = 40


def _cap_text(text: str) -> str:
    if len(text) <= STREAM_HEAD_CHARS + STREAM_TAIL_CHARS + 200:
        return text
    dropped = len(text) - STREAM_HEAD_CHARS - STREAM_TAIL_CHARS
    return (
        text[:STREAM_HEAD_CHARS]
        + f"\n[... {dropped} chars dropped ...]\n"
        + text[-STREAM_TAIL_CHARS:]
    )


class OutputBuffer:
    """Thread-safe, bounded accumulator of nbformat output dicts.

    The executor thread appends as IOPub messages arrive; tool threads may
    snapshot at any time for a live mid-run view.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._head: list[dict] = []
        self._tail: deque[dict] = deque(maxlen=TAIL_OUTPUTS)
        self._dropped = 0
        self._pending_clear = False

    def add(self, out: dict) -> None:
        with self._lock:
            if self._pending_clear:
                self._clear_locked()
            last = self._tail[-1] if self._tail else (self._head[-1] if self._head else None)
            if (
                out.get("output_type") == "stream"
                and last is not None
                and last.get("output_type") == "stream"
                and last.get("name") == out.get("name")
            ):
                last["text"] = _cap_text(last.get("text", "") + out.get("text", ""))
                return
            if len(self._head) < HEAD_OUTPUTS and not self._tail and not self._dropped:
                self._head.append(out)
                return
            if len(self._tail) == self._tail.maxlen:
                self._dropped += 1
            self._tail.append(out)

    def clear(self, wait: bool = False) -> None:
        """clear_output message: wait=True defers until the next output."""
        with self._lock:
            if wait:
                self._pending_clear = True
            else:
                self._clear_locked()

    def _clear_locked(self) -> None:
        self._head.clear()
        self._tail.clear()
        self._dropped = 0
        self._pending_clear = False

    def snapshot(self) -> list[dict]:
        with self._lock:
            outs = list(self._head)
            if self._dropped:
                outs.append(
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": f"[... {self._dropped} outputs dropped ...]\n",
                    }
                )
            outs.extend(self._tail)
            return outs

    def last_line(self) -> str:
        """Most recent non-empty output line — the mid-run progress teaser."""
        with self._lock:
            for out in (*reversed(self._tail), *reversed(self._head)):
                if out.get("output_type") == "stream":
                    for line in reversed(out.get("text", "").splitlines()):
                        if line.strip():
                            return line.strip()
                elif out.get("output_type") == "error":
                    return f"{out.get('ename', 'Error')}: {out.get('evalue', '')}"
                else:
                    return f"<{out.get('output_type')}>"
            return ""


class Batch:
    """One submission (one run call); fails fast within itself."""

    def __init__(self) -> None:
        self.tasks: list[ExecTask] = []
        self.failed = False

    def add(self, task: ExecTask) -> None:
        self.tasks.append(task)

    def wait(self, timeout: float | None) -> bool:
        """True if every task finished within `timeout` seconds."""
        deadline = None if timeout is None else time.monotonic() + timeout
        for task in self.tasks:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not task.done.wait(remaining):
                return False
        return True


@dataclass
class ExecTask:
    """One cell execution. The cell name is the task's identity."""

    name: str
    code: str
    rev: str
    timeout: float
    batch: Batch
    adhoc: bool = False  # e.g. inspect: no cell lookup, no write-back
    status: str = "queued"  # queued|running|ok|error|timeout|skipped|cancelled|superseded
    note: str = ""
    execution_count: int | None = None
    buffer: OutputBuffer = field(default_factory=OutputBuffer)
    started_at: float | None = None
    done: threading.Event = field(default_factory=threading.Event)

    def elapsed(self) -> float:
        return 0.0 if self.started_at is None else time.monotonic() - self.started_at


class Executor:
    """One worker thread per notebook: owns all kernel execution.

    Serializing every execution (including ad-hoc inspection) through this
    thread is what makes concurrent tool calls safe — the kernel client is
    never used from two threads at once.
    """

    def __init__(self, run_task, name: str) -> None:
        self._run_task = run_task
        self._cond = threading.Condition()
        self._queue: deque[ExecTask] = deque()
        self._current: ExecTask | None = None
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"jmcp-exec-{name}")
        self._thread.start()

    def submit(self, tasks: list[ExecTask]) -> None:
        with self._cond:
            if self._stopped:
                raise RuntimeError("executor is stopped")
            self._queue.extend(tasks)
            self._cond.notify()

    def busy(self) -> bool:
        with self._cond:
            return self._current is not None or bool(self._queue)

    def activity(self) -> tuple[ExecTask | None, list[str]]:
        with self._cond:
            return self._current, [t.name for t in self._queue]

    def running_task(self, name: str) -> ExecTask | None:
        with self._cond:
            if self._current is not None and self._current.name == name:
                return self._current
            return None

    def cancel_queued(self) -> list[str]:
        with self._cond:
            cancelled = []
            for task in self._queue:
                task.status = "cancelled"
                task.done.set()
                cancelled.append(task.name)
            self._queue.clear()
            return cancelled

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            for task in self._queue:
                task.status = "cancelled"
                task.done.set()
            self._queue.clear()
            self._cond.notify()

    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                task = self._queue.popleft()
                self._current = task
            try:
                self._run_task(task)
            except Exception as e:  # a task must never kill the worker
                task.status = "error"
                task.note = f"{type(e).__name__}: {e}"
                task.batch.failed = True
            finally:
                with self._cond:
                    self._current = None
                task.done.set()

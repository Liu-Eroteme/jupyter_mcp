"""Per-notebook sessions: model + kernel + DAG + summaries, plus a registry."""

from __future__ import annotations

import atexit
import threading
import time
from pathlib import Path
from typing import Callable, TypeVar, cast

from nbformat import NotebookNode

from .condense import Condensed, condense_outputs
from .dag import NotebookGraph, build_graph
from .errors import CellNotFound, ExternalModification, JupyterMcpError
from .kernel import DEFAULT_EXEC_TIMEOUT, KERNEL_IDLE_TTL, KernelSession
from .model import META_NS, CellRef, NotebookFile, cell_meta
from .summaries import Summarizer
from .tasks import Batch, ExecTask, Executor

T = TypeVar("T")


class NotebookSession:
    def __init__(self, path: Path, summarizer: Summarizer):
        self.path = path.resolve()
        self.summarizer = summarizer
        self.nbfile = NotebookFile(self.path)
        self.nbfile.load()
        self._kernel: KernelSession | None = None
        self._executor: Executor | None = None
        #: guards nbfile state; the executor thread and tool threads both
        #: read and write it (RLock: mutate() nests guard_mutation())
        self._lock = threading.RLock()

    # ------------------------------------------------------------ freshness

    def refresh_reads(self) -> bool:
        """Reload silently if the file changed on disk (read paths)."""
        with self._lock:
            if self.nbfile.disk_changed():
                self.nbfile.load()
                return True
            return False

    def guard_mutation(self) -> None:
        """Mutation paths: an external change means the caller's revs are
        void — reload and abort so it re-reads."""
        with self._lock:
            if self.nbfile.disk_changed():
                self.nbfile.load()
                raise ExternalModification(str(self.path))

    def mutate(self, op: str, fn: Callable[[], T]) -> T:
        with self._lock:
            self.guard_mutation()
            self.nbfile.snapshot(op)
            try:
                result = fn()
            except Exception:
                # disk still holds the last saved state (save only runs on
                # success) — reload so a partially-applied mutation can't
                # linger in memory and get persisted by the next operation
                self.nbfile.load()
                raise
            self.nbfile.save()
            return result

    # ------------------------------------------------------------------ dag

    def graph(self) -> NotebookGraph:
        with self._lock:
            return build_graph(
                [(r.name, r.cell.cell_type, r.cell.source) for r in self.nbfile.refs()]
            )

    def stale_names(self, graph: NotebookGraph | None = None) -> list[str]:
        """Code cells whose *current kernel state* doesn't reflect their
        source, plus every downstream dependent — in document order.

        A cell counts as fresh only if it executed with its current source
        (`last_exec_rev`) on the kernel that is alive right now
        (`last_exec_epoch`). A new or restarted kernel has a different epoch,
        so persisted metadata can never claim freshness against empty state.
        """
        with self._lock:
            graph = graph or self.graph()
            epoch = (
                self._kernel.epoch if self._kernel is not None and self._kernel.alive else None
            )
            changed = set()
            for ref in self.nbfile.refs():
                if ref.cell.cell_type != "code" or not ref.cell.source.strip():
                    continue
                meta = cell_meta(ref.cell)
                if meta.get("last_exec_rev") != ref.rev or meta.get("last_exec_epoch") != epoch or epoch is None:
                    changed.add(ref.name)
            stale = graph.stale_closure(changed)
            code_names = {
                r.name for r in self.nbfile.refs() if r.cell.cell_type == "code" and r.cell.source.strip()
            }
            return [n for n in stale if n in code_names]

    # ---------------------------------------------------------------- kernel

    def kernel(self) -> KernelSession:
        if self._kernel is None:
            kernel_name = self.nbfile.nb.metadata.get("kernelspec", {}).get("name")
            self._kernel = KernelSession(self.path, kernel_name)
        return self._kernel

    def shutdown_kernel(self) -> None:
        if self._kernel is not None:
            self._kernel.shutdown()
            self._kernel = None

    def kernel_status(self) -> str:
        k = self._kernel
        if k is None or k.epoch is None:
            return "not started (launches on first execution)"
        if not k.alive:
            return "dead (restarts on next execution)"
        desc = k.note or f"kernelspec {k.kernel_name!r}"
        return f"alive — {desc} (epoch {k.epoch})"

    def expire_idle_kernel(self, ttl: float) -> bool:
        """Shut the kernel down if it has been idle longer than `ttl`."""
        if self.busy():
            return False  # a long-running cell is not idle
        k = self._kernel
        if k is not None and k.alive and time.monotonic() - k.last_used > ttl:
            self.shutdown_kernel()
            return True
        return False

    # ------------------------------------------------------------- execution
    # All execution — batches and ad-hoc inspection — goes through one
    # executor thread per notebook, so the kernel client is never used from
    # two threads and tool calls can return while cells keep running.

    def _ensure_executor(self) -> Executor:
        if self._executor is None:
            self._executor = Executor(self._run_task, self.path.stem)
        return self._executor

    def busy(self) -> bool:
        return self._executor is not None and self._executor.busy()

    def activity(self) -> tuple[ExecTask | None, list[str]]:
        """(currently running task | None, queued cell names)."""
        if self._executor is None:
            return None, []
        return self._executor.activity()

    def running_task(self, name: str) -> ExecTask | None:
        if self._executor is None:
            return None
        return self._executor.running_task(name)

    def interrupt(self, clear_queue: bool = True) -> str:
        ex = self._executor
        current, queued = (ex.activity() if ex is not None else (None, []))
        if current is None and not queued:
            return "Nothing is running or queued."
        cancelled = ex.cancel_queued() if ex is not None and clear_queue else []
        parts = []
        if current is not None:
            if self._kernel is not None:
                self._kernel.interrupt()
            parts.append(f"Interrupted {current.name!r} after {current.elapsed():.0f}s")
        if cancelled:
            parts.append(f"cancelled queued: {', '.join(cancelled)}")
        return "; ".join(parts) + "."

    def submit_cells(self, names: list[str], timeout: float = DEFAULT_EXEC_TIMEOUT) -> Batch:
        """Queue the named cells (document order) for background execution."""
        with self._lock:
            refs = {r.name: r for r in self.nbfile.refs()}
            for name in names:
                if name not in refs:
                    raise CellNotFound(name, list(refs))
                if refs[name].cell.cell_type != "code":
                    raise JupyterMcpError(f"Cell {name!r} is a {refs[name].cell.cell_type} cell.")
            ordered = [r.name for r in self.nbfile.refs() if r.name in set(names)]
            self.guard_mutation()
            self.nbfile.snapshot(f"execute-{ordered[0]}" if ordered else "execute")
            batch = Batch()
            for name in ordered:
                ref = refs[name]
                batch.add(
                    ExecTask(name=name, code=ref.cell.source, rev=ref.rev, timeout=timeout, batch=batch)
                )
        self._ensure_executor().submit(batch.tasks)
        return batch

    def execute_cells(
        self, names: list[str], timeout: float = DEFAULT_EXEC_TIMEOUT
    ) -> list[tuple[str, str, Condensed]]:
        """Submit and wait to completion (synchronous façade over the executor)."""
        batch = self.submit_cells(names, timeout)
        batch.wait(None)
        return self.batch_results(batch)

    def batch_results(self, batch: Batch) -> list[tuple[str, str, Condensed]]:
        results: list[tuple[str, str, Condensed]] = []
        for task in batch.tasks:
            if task.status in ("skipped", "cancelled", "superseded"):
                results.append((task.name, task.status, Condensed(text=f"({task.note or task.status})")))
                continue
            condensed = condense_outputs(task.buffer.snapshot())
            if task.note:
                condensed.text = f"[{task.note}]\n{condensed.text}"
            results.append((task.name, task.status, condensed))
        return results

    def run_adhoc(self, code: str, timeout: float) -> ExecTask:
        """Run non-cell code (inspection); routed through the executor so the
        kernel client is never shared across threads."""
        if self.busy():
            raise JupyterMcpError(
                "Kernel is busy (a cell is running or queued) — this would queue "
                "behind it. Wait for it, or interrupt() first."
            )
        batch = Batch()
        task = ExecTask(name="(adhoc)", code=code, rev="", timeout=timeout, batch=batch, adhoc=True)
        batch.add(task)
        self._ensure_executor().submit([task])
        task.done.wait()  # bounded by the kernel-side timeout
        return task

    def _run_task(self, task: ExecTask) -> None:
        """Executor-thread callback: run one task against the kernel."""
        if task.adhoc:
            kernel = self.kernel()
            kernel.ensure_started()
            task.status, task.started_at = "running", time.monotonic()
            res = kernel.execute(task.code, timeout=task.timeout, buffer=task.buffer)
            task.status, task.note = res.status, res.note
            task.execution_count = res.execution_count
            return
        if task.batch.failed:
            task.status, task.note = "skipped", "skipped: earlier cell failed"
            return
        with self._lock:
            ref = self._find(task.name)
            if ref is None or ref.rev != task.rev:
                task.status = "superseded"
                task.note = "superseded: cell changed after submission; run again for the current version"
                return
            meta = cell_meta(ref.cell)
            # any execution attempt voids the old stamp: a failed or
            # interrupted run can still mutate kernel state, so freshness
            # must be re-earned by a successful run
            meta.pop("last_exec_rev", None)
            meta.pop("last_exec_epoch", None)
            meta.pop("output_summary", None)  # outputs are being replaced
        kernel = self.kernel()
        kernel.ensure_started()  # epoch must exist before stamping freshness
        task.status, task.started_at = "running", time.monotonic()
        res = kernel.execute(task.code, timeout=task.timeout, buffer=task.buffer)
        task.status, task.note = res.status, res.note
        task.execution_count = res.execution_count
        with self._lock:
            ref = self._find(task.name)
            if ref is not None and ref.rev == task.rev:
                ref.cell.outputs = [_as_output_node(o) for o in task.buffer.snapshot()]
                ref.cell.execution_count = res.execution_count
                if res.status == "ok":
                    meta = cell_meta(ref.cell)
                    meta["last_exec_rev"] = task.rev
                    meta["last_exec_epoch"] = kernel.epoch
            elif ref is not None:
                extra = "cell changed during the run; outputs not persisted"
                task.note = f"{task.note}; {extra}" if task.note else extra
            self.nbfile.save()
        if task.status != "ok":
            task.batch.failed = True

    def _find(self, name: str) -> CellRef | None:
        for r in self.nbfile.refs():
            if r.name == name:
                return r
        return None

    def stop(self) -> None:
        if self._executor is not None:
            self._executor.stop()
            self._executor = None
        self.shutdown_kernel()


def _as_output_node(out: dict) -> NotebookNode:
    from nbformat import from_dict

    # from_dict is typed for arbitrary nesting; a dict input yields a node
    return cast(NotebookNode, from_dict(out))


class Registry:
    def __init__(self) -> None:
        self._sessions: dict[Path, NotebookSession] = {}
        self.summarizer = Summarizer()
        atexit.register(self.shutdown_all)

    def get(self, path: str) -> NotebookSession:
        p = Path(path).expanduser().resolve()
        if p.suffix != ".ipynb":
            raise JupyterMcpError(f"{p} is not a .ipynb file.")
        for other in list(self._sessions.values()):
            other.expire_idle_kernel(KERNEL_IDLE_TTL)
        session = self._sessions.get(p)
        if session is not None:
            if not p.exists():
                self.evict(p)
                raise JupyterMcpError(
                    f"{p} was deleted or moved on disk; its session and kernel were "
                    "shut down. Check the path, or recreate it with create_notebook."
                )
            return session
        if not p.exists():
            raise JupyterMcpError(
                f"{p} does not exist. Use create_notebook to start a new one."
            )
        session = NotebookSession(p, self.summarizer)
        self._sessions[p] = session
        return session

    def evict(self, p: Path) -> None:
        session = self._sessions.pop(p, None)
        if session is not None:
            try:
                session.stop()
            except Exception:
                pass

    def register_new(self, path: str, kernel_name: str = "python3") -> NotebookSession:
        p = Path(path).expanduser().resolve()
        NotebookFile.create(p, kernel_name=kernel_name)
        session = NotebookSession(p, self.summarizer)
        self._sessions[p] = session
        return session

    def shutdown_all(self) -> None:
        for session in self._sessions.values():
            try:
                session.stop()
            except Exception:
                pass

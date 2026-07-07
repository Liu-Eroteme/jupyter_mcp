"""Per-notebook sessions: model + kernel + DAG + summaries, plus a registry."""

from __future__ import annotations

import atexit
from pathlib import Path
from typing import Callable, TypeVar, cast

from nbformat import NotebookNode

from .condense import Condensed, condense_outputs
from .dag import NotebookGraph, build_graph
from .errors import ExternalModification, JupyterMcpError
from .kernel import DEFAULT_EXEC_TIMEOUT, KernelSession
from .model import META_NS, NotebookFile, cell_meta
from .summaries import Summarizer

T = TypeVar("T")


class NotebookSession:
    def __init__(self, path: Path, summarizer: Summarizer):
        self.path = path.resolve()
        self.summarizer = summarizer
        self.nbfile = NotebookFile(self.path)
        self.nbfile.load()
        self._kernel: KernelSession | None = None

    # ------------------------------------------------------------ freshness

    def refresh_reads(self) -> bool:
        """Reload silently if the file changed on disk (read paths)."""
        if self.nbfile.disk_changed():
            self.nbfile.load()
            return True
        return False

    def guard_mutation(self) -> None:
        """Mutation paths: an external change means the caller's revs are
        void — reload and abort so it re-reads."""
        if self.nbfile.disk_changed():
            self.nbfile.load()
            raise ExternalModification(str(self.path))

    def mutate(self, op: str, fn: Callable[[], T]) -> T:
        self.guard_mutation()
        self.nbfile.snapshot(op)
        try:
            result = fn()
        except Exception:
            # disk still holds the last saved state (save only runs on
            # success) — reload so a partially-applied mutation can't linger
            # in memory and get persisted by the next operation
            self.nbfile.load()
            raise
        self.nbfile.save()
        return result

    # ------------------------------------------------------------------ dag

    def graph(self) -> NotebookGraph:
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

    def execute_cells(
        self, names: list[str], timeout: float = DEFAULT_EXEC_TIMEOUT
    ) -> list[tuple[str, str, Condensed]]:
        """Execute the named cells in document order; persist outputs.

        Returns [(name, status, condensed)] and stops after the first cell
        that errors or times out (remaining cells are reported as skipped).
        """
        refs = {r.name: r for r in self.nbfile.refs()}
        for name in names:
            if name not in refs:
                from .errors import CellNotFound

                raise CellNotFound(name, list(refs))
            if refs[name].cell.cell_type != "code":
                raise JupyterMcpError(f"Cell {name!r} is a {refs[name].cell.cell_type} cell.")
        ordered = [r.name for r in self.nbfile.refs() if r.name in set(names)]

        self.guard_mutation()
        self.nbfile.snapshot(f"execute-{ordered[0]}" if ordered else "execute")
        kernel = self.kernel()
        kernel.ensure_started()  # epoch must exist before stamping freshness

        results: list[tuple[str, str, Condensed]] = []
        failed = False
        for name in ordered:
            if failed:
                results.append((name, "skipped", Condensed(text="(skipped: earlier cell failed)")))
                continue
            ref = self.nbfile.get(name)
            meta = cell_meta(ref.cell)
            # any execution attempt voids the old stamp: a failed or
            # interrupted run can still mutate kernel state, so freshness
            # must be re-earned by a successful run
            meta.pop("last_exec_rev", None)
            meta.pop("last_exec_epoch", None)
            meta.pop("output_summary", None)  # outputs are being replaced
            res = kernel.execute(ref.cell.source, timeout=timeout)
            ref.cell.outputs = [_as_output_node(o) for o in res.outputs]
            ref.cell.execution_count = res.execution_count
            condensed = condense_outputs(ref.cell.outputs)
            if res.note:
                condensed.text = f"[{res.note}]\n{condensed.text}"
            if res.status == "ok":
                meta["last_exec_rev"] = ref.rev
                meta["last_exec_epoch"] = kernel.epoch
            else:
                failed = True
            results.append((name, res.status, condensed))
        self.nbfile.save()
        return results


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
        if p in self._sessions:
            return self._sessions[p]
        if not p.exists():
            raise JupyterMcpError(
                f"{p} does not exist. Use create_notebook to start a new one."
            )
        session = NotebookSession(p, self.summarizer)
        self._sessions[p] = session
        return session

    def register_new(self, path: str, kernel_name: str = "python3") -> NotebookSession:
        p = Path(path).expanduser().resolve()
        NotebookFile.create(p, kernel_name=kernel_name)
        session = NotebookSession(p, self.summarizer)
        self._sessions[p] = session
        return session

    def shutdown_all(self) -> None:
        for session in self._sessions.values():
            try:
                session.shutdown_kernel()
            except Exception:
                pass

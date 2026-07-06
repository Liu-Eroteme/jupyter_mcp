"""Persistent Jupyter kernel per notebook (via jupyter_client).

The kernel starts lazily on first execution, runs with the notebook's
directory as cwd (so relative paths behave like in the editor), and keeps
state between tool calls — enabling incremental execution instead of
whole-notebook re-runs.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass, field
from pathlib import Path

from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel
from jupyter_client.manager import KernelManager
from nbformat.v4 import output_from_msg

from .errors import KernelError

STARTUP_TIMEOUT = 60.0
DEFAULT_EXEC_TIMEOUT = 120.0


@dataclass
class ExecResult:
    status: str  # "ok" | "error" | "timeout"
    outputs: list[dict] = field(default_factory=list)
    execution_count: int | None = None
    note: str = ""


def resolve_kernel_name(requested: str | None) -> tuple[str, str]:
    """Return (kernel_name, note). Falls back to python3 if unavailable."""
    specs = KernelSpecManager().find_kernel_specs()
    if requested and requested in specs:
        return requested, ""
    fallback = "python3" if "python3" in specs else next(iter(specs), "python3")
    note = (
        f"kernelspec {requested!r} not installed; using {fallback!r}"
        if requested and requested != fallback
        else ""
    )
    return fallback, note


class KernelSession:
    def __init__(self, notebook_path: Path, kernel_name: str | None):
        self.notebook_path = notebook_path
        self.requested_kernel = kernel_name
        self.kernel_name: str | None = None
        self.note = ""
        self._km: KernelManager | None = None
        self._kc = None

    @property
    def alive(self) -> bool:
        return self._km is not None and self._km.is_alive()

    def start(self) -> None:
        self.kernel_name, self.note = resolve_kernel_name(self.requested_kernel)
        try:
            km = KernelManager(kernel_name=self.kernel_name)
            km.start_kernel(cwd=str(self.notebook_path.parent))
        except NoSuchKernel as e:
            raise KernelError(f"Cannot start kernel {self.kernel_name!r}: {e}") from e
        kc = km.client()
        kc.start_channels()
        try:
            kc.wait_for_ready(timeout=STARTUP_TIMEOUT)
        except RuntimeError as e:
            km.shutdown_kernel(now=True)
            raise KernelError(f"Kernel {self.kernel_name!r} did not become ready: {e}") from e
        self._km, self._kc = km, kc

    def ensure_started(self) -> None:
        if not self.alive:
            self.shutdown()
            self.start()

    def restart(self) -> None:
        if self._km is not None and self._km.is_alive():
            self._km.restart_kernel(now=True)
            self._kc.wait_for_ready(timeout=STARTUP_TIMEOUT)
        else:
            self.ensure_started()

    def shutdown(self) -> None:
        if self._kc is not None:
            try:
                self._kc.stop_channels()
            except Exception:
                pass
            self._kc = None
        if self._km is not None:
            try:
                self._km.shutdown_kernel(now=True)
            except Exception:
                pass
            self._km = None

    def interrupt(self) -> None:
        if self._km is not None and self._km.is_alive():
            self._km.interrupt_kernel()

    # ------------------------------------------------------------ execution

    def execute(self, code: str, timeout: float = DEFAULT_EXEC_TIMEOUT) -> ExecResult:
        """Run code, collecting outputs as nbformat output dicts."""
        self.ensure_started()
        kc = self._kc
        msg_id = kc.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)

        outputs: list[dict] = []
        execution_count: int | None = None
        status = "ok"
        note = ""
        idle = False
        pending_clear = False

        while not idle:
            try:
                msg = kc.get_iopub_msg(timeout=timeout)
            except queue.Empty:
                self.interrupt()
                status, note = "timeout", f"no kernel output within {timeout:.0f}s; kernel interrupted"
                # drain the interrupt-induced messages briefly
                try:
                    while True:
                        msg = kc.get_iopub_msg(timeout=2)
                        if (
                            msg["parent_header"].get("msg_id") == msg_id
                            and msg["msg_type"] == "status"
                            and msg["content"]["execution_state"] == "idle"
                        ):
                            break
                except queue.Empty:
                    pass
                break
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            mtype = msg["msg_type"]
            content = msg["content"]
            if mtype == "status":
                idle = content["execution_state"] == "idle"
            elif mtype == "execute_input":
                execution_count = content.get("execution_count", execution_count)
            elif mtype == "clear_output":
                if content.get("wait"):
                    pending_clear = True
                else:
                    outputs.clear()
            elif mtype in ("stream", "display_data", "execute_result", "error"):
                if pending_clear:
                    outputs.clear()
                    pending_clear = False
                try:
                    out = output_from_msg(msg)
                except ValueError:
                    continue
                if mtype == "error":
                    status = "error"
                if mtype == "execute_result":
                    execution_count = content.get("execution_count", execution_count)
                outputs.append(dict(out))

        # consume the shell reply so the channel stays clean
        try:
            reply = kc.get_shell_msg(timeout=5)
            if status == "ok" and reply["content"].get("status") == "error":
                status = "error"
        except queue.Empty:
            pass

        return ExecResult(status=status, outputs=outputs, execution_count=execution_count, note=note)


INSPECT_HELPER = r'''
def __jmcp_inspect__(name):
    import reprlib
    g = globals()
    if name not in g:
        print(f"NameError: {name!r} is not defined in the kernel")
        return
    obj = g[name]
    t = type(obj)
    print(f"type: {t.__module__}.{t.__qualname__}")
    for attr in ("shape",):
        if hasattr(obj, attr):
            try:
                print(f"{attr}: {getattr(obj, attr)}")
            except Exception:
                pass
    if hasattr(obj, "schema") and not callable(getattr(obj, "schema", None)):
        try:
            print(f"schema: {obj.schema}")
        except Exception:
            pass
    elif hasattr(obj, "dtypes") and hasattr(obj, "columns"):
        try:
            cols = list(obj.columns)
            print(f"columns ({len(cols)}): {cols}")
            print(f"dtypes: {list(map(str, obj.dtypes))}")
        except Exception:
            pass
    try:
        n = len(obj)
        print(f"len: {n}")
    except Exception:
        pass
    r = reprlib.Repr()
    r.maxstring = 800
    r.maxother = 2000
    try:
        head = obj.head(5) if hasattr(obj, "head") and callable(obj.head) else obj
        print(repr(head)[:2500])
    except Exception:
        print(r.repr(obj))
'''


def inspect_code(name: str) -> str:
    if not name.isidentifier():
        raise KernelError(f"{name!r} is not a valid identifier.")
    return f"{INSPECT_HELPER}\n__jmcp_inspect__({name!r})\ndel __jmcp_inspect__\n"

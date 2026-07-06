"""Typed errors surfaced to MCP tool callers with actionable context."""

from __future__ import annotations


class JupyterMcpError(Exception):
    """Base error; the message is shown verbatim to the calling agent."""


class CellNotFound(JupyterMcpError):
    def __init__(self, name: str, available: list[str]):
        self.name = name
        self.available = available
        super().__init__(
            f"No cell named {name!r}. Available cells: {', '.join(available) or '(none)'}"
        )


class RevisionMismatch(JupyterMcpError):
    """The caller's expected_rev does not match the cell's current content.

    Carries the current cell state so the caller can re-orient without an
    extra read round-trip.
    """

    def __init__(self, name: str, expected: str, actual: str, current_source: str):
        self.name = name
        self.expected = expected
        self.actual = actual
        self.current_source = current_source
        super().__init__(
            f"Revision mismatch for cell {name!r}: expected rev {expected}, "
            f"current rev is {actual}. The cell changed since you last read it. "
            f"Current source:\n{current_source}"
        )


class ExternalModification(JupyterMcpError):
    def __init__(self, path: str):
        super().__init__(
            f"Notebook file {path} was modified on disk by another program since it "
            "was loaded. It has been reloaded — re-read the affected cells (revisions "
            "may have changed) and retry."
        )


class DuplicateCellName(JupyterMcpError):
    def __init__(self, name: str):
        super().__init__(f"A cell named {name!r} already exists — cell names must be unique.")


class KernelError(JupyterMcpError):
    pass


class NothingToUndo(JupyterMcpError):
    def __init__(self) -> None:
        super().__init__("No snapshots available to undo.")

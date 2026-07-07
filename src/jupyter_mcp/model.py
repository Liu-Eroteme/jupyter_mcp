"""Notebook file model: nbformat IO, cell names, revisions, snapshots, undo.

Design notes:

- One cell = code + outputs, addressed by a unique, human-meaningful *name*
  stored under ``cell.metadata["jupyter_mcp"]["name"]``. The nbformat cell
  ``id`` remains the stable machine key; names are the agent-facing key.
- Every cell has a *revision* — a short content hash of its source. Mutating
  operations require the caller to pass the revision it last read
  (optimistic locking), which makes wrong-target and stale edits structurally
  impossible instead of merely unlikely.
- Before every mutation the current file bytes are snapshotted, giving a
  cheap undo stack (kept outside the project tree, under ``~/.cache``).
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path

import nbformat
from nbformat import NotebookNode

from .errors import (
    CellNotFound,
    DuplicateCellName,
    JupyterMcpError,
    NothingToUndo,
    RevisionMismatch,
)

META_NS = "jupyter_mcp"
SNAPSHOT_ROOT = Path.home() / ".cache" / "jupyter_mcp" / "snapshots"
MAX_SNAPSHOTS = 20

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 40) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len]
        if "-" in slug:  # don't truncate mid-word
            slug = slug.rsplit("-", 1)[0]
    return slug.strip("-")


def source_rev(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:10]


def cell_meta(cell: NotebookNode) -> dict:
    return cell.metadata.setdefault(META_NS, {})


def cell_name(cell: NotebookNode) -> str | None:
    return cell.metadata.get(META_NS, {}).get("name")


@dataclass
class CellRef:
    index: int
    cell: NotebookNode

    @property
    def name(self) -> str:
        return cell_name(self.cell) or ""

    @property
    def rev(self) -> str:
        return source_rev(self.cell.source)


def _auto_name(cell: NotebookNode, index: int) -> str:
    """Derive a name from content: first md heading / first comment, else id."""
    source = cell.get("source", "")
    if cell.cell_type == "markdown":
        for line in source.splitlines():
            line = line.strip()
            if line.startswith("#"):
                slug = slugify(line.lstrip("#"))
                if slug:
                    return f"md-{slug}"
        slug = slugify(source[:60])
        return f"md-{slug}" if slug else f"md-cell-{index}"
    for line in source.splitlines():
        line = line.strip()
        if line.startswith("#"):
            slug = slugify(line.lstrip("#"))
            if slug:
                return slug
        elif line:
            break
    slug = slugify(source.strip().splitlines()[0] if source.strip() else "")
    return slug or f"cell-{str(cell.get('id', index))[:6]}"


class NotebookFile:
    """A notebook on disk plus the bookkeeping the MCP layer needs."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.nb: NotebookNode = None  # type: ignore[assignment]
        self._disk_sha: str | None = None
        # seed from disk so snapshots from a previous server process remain
        # undoable (filenames sort chronologically)
        self._undo_stack: list[Path] = sorted(self._snapshot_dir().glob("*.ipynb"))
        self._names_dirty = False

    # ------------------------------------------------------------- loading

    @classmethod
    def create(cls, path: Path, kernel_name: str = "python3") -> "NotebookFile":
        path = path.resolve()
        if path.exists():
            raise JupyterMcpError(f"{path} already exists.")
        nb = nbformat.v4.new_notebook()
        nb.metadata["kernelspec"] = {
            "name": kernel_name,
            "display_name": kernel_name,
            "language": "python",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, str(path))
        nbf = cls(path)
        nbf.load()
        return nbf

    def load(self) -> None:
        raw = self.path.read_bytes()
        self._disk_sha = hashlib.sha256(raw).hexdigest()
        self.nb = nbformat.reads(raw.decode("utf-8"), as_version=4)
        self._names_dirty = self._ensure_names()

    def disk_changed(self) -> bool:
        """True if the file on disk differs from what we last loaded/saved."""
        if self._disk_sha is None:
            return True
        if not self.path.exists():
            return True
        return hashlib.sha256(self.path.read_bytes()).hexdigest() != self._disk_sha

    def save(self) -> None:
        nbformat.validate(self.nb)
        nbformat.write(self.nb, str(self.path))
        self._disk_sha = hashlib.sha256(self.path.read_bytes()).hexdigest()

    # --------------------------------------------------------------- names

    def _ensure_names(self) -> bool:
        """Assign auto-names to unnamed cells. Returns True if any were added.

        Does not write to disk — a foreign notebook is only rewritten once the
        first real mutation happens.
        """
        changed = False
        seen: set[str] = set()
        for i, cell in enumerate(self.nb.cells):
            name = cell_name(cell)
            if not name:
                name = _auto_name(cell, i)
                changed = True
            base, n = name, 2
            while name in seen:
                name = f"{base}-{n}"
                n += 1
                changed = changed or name != cell_name(cell)
            if name != cell_name(cell):
                cell_meta(cell)["name"] = name
                changed = True
            seen.add(name)
        return changed

    def _check_new_name(self, name: str) -> str:
        name = name.strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
            raise JupyterMcpError(
                f"Invalid cell name {name!r}: use kebab-case (lowercase letters, "
                "digits, hyphens; must start with a letter or digit)."
            )
        if name in self.names():
            raise DuplicateCellName(name)
        return name

    # ------------------------------------------------------------ accessors

    @property
    def cells(self) -> list[NotebookNode]:
        return self.nb.cells

    def refs(self) -> list[CellRef]:
        return [CellRef(i, c) for i, c in enumerate(self.nb.cells)]

    def names(self) -> list[str]:
        return [cell_name(c) or "" for c in self.nb.cells]

    def get(self, name: str) -> CellRef:
        for i, cell in enumerate(self.nb.cells):
            if cell_name(cell) == name:
                return CellRef(i, cell)
        raise CellNotFound(name, self.names())

    def check_rev(self, name: str, expected_rev: str) -> CellRef:
        ref = self.get(name)
        actual = ref.rev
        if actual != expected_rev:
            raise RevisionMismatch(name, expected_rev, actual, ref.cell.source)
        return ref

    # ------------------------------------------------------------ mutations
    # Callers (the session layer) are responsible for snapshotting first and
    # saving afterwards; these methods only touch the in-memory model.

    def _resolve_insert_index(self, after: str | None, index: int | None) -> int:
        if after is not None and index is not None:
            raise JupyterMcpError("Pass either 'after' or 'index', not both.")
        if after is not None:
            if after == "":  # prepend sentinel
                return 0
            return self.get(after).index + 1
        if index is not None:
            return max(0, min(index, len(self.nb.cells)))
        return len(self.nb.cells)  # append

    def add_cell(
        self,
        name: str,
        source: str,
        cell_type: str = "code",
        after: str | None = None,
        index: int | None = None,
    ) -> CellRef:
        name = self._check_new_name(name)
        if cell_type == "code":
            cell = nbformat.v4.new_code_cell(source)
        elif cell_type == "markdown":
            cell = nbformat.v4.new_markdown_cell(source)
        else:
            raise JupyterMcpError(f"Unsupported cell_type {cell_type!r} (code|markdown).")
        cell_meta(cell)["name"] = name
        pos = self._resolve_insert_index(after, index)
        self.nb.cells.insert(pos, cell)
        return CellRef(pos, cell)

    def update_cell(
        self,
        name: str,
        expected_rev: str,
        source: str | None = None,
        new_name: str | None = None,
    ) -> CellRef:
        ref = self.check_rev(name, expected_rev)
        if source is None and new_name is None:
            raise JupyterMcpError("update_cell: pass source and/or new_name.")
        if new_name is not None and new_name != name:
            new_name = self._check_new_name(new_name)
            cell_meta(ref.cell)["name"] = new_name
        if source is not None:
            ref.cell.source = source
            if ref.cell.cell_type == "code":
                ref.cell.outputs = []
                ref.cell.execution_count = None
                # the cached output summary describes outputs that no longer exist
                cell_meta(ref.cell).pop("output_summary", None)
        return CellRef(ref.index, ref.cell)

    def remove_cell(self, name: str, expected_rev: str) -> None:
        ref = self.check_rev(name, expected_rev)
        del self.nb.cells[ref.index]

    def move_cell(
        self,
        name: str,
        expected_rev: str,
        after: str | None = None,
        index: int | None = None,
    ) -> CellRef:
        ref = self.check_rev(name, expected_rev)
        # validate the full request BEFORE popping — a rejected move must not
        # leave the in-memory model missing the cell
        if after == name:
            raise JupyterMcpError("Cannot move a cell after itself.")
        if after is not None and index is not None:
            raise JupyterMcpError("Pass either 'after' or 'index', not both.")
        if after:  # "" is the prepend sentinel
            self.get(after)
        cell = self.nb.cells.pop(ref.index)
        pos = self._resolve_insert_index(after, index)
        self.nb.cells.insert(pos, cell)
        return CellRef(pos, cell)

    # ------------------------------------------------------------ snapshots

    def _snapshot_dir(self) -> Path:
        digest = hashlib.sha256(str(self.path).encode()).hexdigest()[:12]
        d = SNAPSHOT_ROOT / digest
        d.mkdir(parents=True, exist_ok=True)
        return d

    def snapshot(self, op: str) -> Path:
        """Copy the current on-disk bytes aside before a mutation."""
        d = self._snapshot_dir()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        op_slug = slugify(op, 24) or "op"
        target = d / f"{stamp}-{int(time.time() * 1000) % 1000:03d}-{op_slug}.ipynb"
        target.write_bytes(self.path.read_bytes())
        self._undo_stack.append(target)
        existing = sorted(d.glob("*.ipynb"))
        for old in existing[:-MAX_SNAPSHOTS]:
            old.unlink(missing_ok=True)
            if old in self._undo_stack:
                self._undo_stack.remove(old)
        return target

    def undo_last(self) -> str:
        """Restore the most recent snapshot. Returns the op it undid."""
        while self._undo_stack:
            snap = self._undo_stack.pop()
            if snap.exists():
                self.path.write_bytes(snap.read_bytes())
                snap.unlink(missing_ok=True)
                self.load()
                return snap.stem.split("-", 3)[-1]
        raise NothingToUndo()

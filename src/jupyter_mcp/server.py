"""MCP server exposing the notebook toolkit.

Conventions:
- Cells are addressed by unique name (see notebook_overview for the list).
- Every read shows each cell's `rev`; every mutation REQUIRES the rev you
  last read (`expected_rev`) — a mismatch means the cell changed and you must
  re-read it. This replaces confirmation round-trips.
- Mutations snapshot first; `undo_last` restores the previous state.
"""

from __future__ import annotations

import functools
import re
import sys
import traceback
from typing import Any

from mcp.server.fastmcp import FastMCP, Image

from .condense import Condensed, condense_outputs
from .dag import NotebookGraph
from .errors import JupyterMcpError
from .kernel import inspect_code
from .model import CellRef, cell_meta
from .session import NotebookSession, Registry
from .summaries import get_summary, get_tldr, output_hash

mcp = FastMCP("jupyter-eda")
registry = Registry()


def _tool_errors(fn):
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any):
        try:
            return fn(*args, **kwargs)
        except JupyterMcpError as e:
            return f"ERROR: {e}"
        except Exception as e:  # last resort — never leak a raw traceback to the wire
            traceback.print_exc(file=sys.stderr)
            return f"ERROR (internal {type(e).__name__}): {e} — traceback logged to server stderr."

    return wrapper


# ------------------------------------------------------------------ helpers


def _deps_line(graph: NotebookGraph, name: str) -> str:
    parents = graph.parents.get(name, {})
    children = graph.children.get(name, {})
    parts = []
    if parents:
        ps = ", ".join(f"{p}({','.join(sorted(v))})" for p, v in sorted(parents.items()))
        parts.append(f"← {ps}")
    if children:
        parts.append("→ " + ", ".join(sorted(children)))
    return " | ".join(parts) if parts else "(no dependencies)"


def _cell_header(session: NotebookSession, ref: CellRef, graph: NotebookGraph, stale: set[str]) -> str:
    flags = [ref.cell.cell_type]
    if ref.name in stale:
        flags.append("STALE")
    ec = ref.cell.get("execution_count")
    if ec:
        flags.append(f"exec#{ec}")
    return f"[{ref.index}] {ref.name}  (rev {ref.rev}; {', '.join(flags)})"


def _render_cell(
    session: NotebookSession,
    ref: CellRef,
    graph: NotebookGraph,
    stale: set[str],
    view: str,
) -> tuple[str, list[Image]]:
    lines = [_cell_header(session, ref, graph, stale)]
    if ref.cell.cell_type == "code":
        lines.append(f"  deps: {_deps_line(graph, ref.name)}")
        parents = graph.parents.get(ref.name, {})
        for parent in sorted(parents):
            try:
                lines.append(f"    {parent}: {get_tldr(session.nbfile.get(parent).cell)}")
            except JupyterMcpError:
                pass
    images: list[Image] = []
    if view in ("full", "code"):
        lines.append("source:")
        lines.append(ref.cell.source if ref.cell.source.strip() else "(empty)")
    if view in ("full", "outputs") and ref.cell.cell_type == "code":
        running = session.running_task(ref.name)
        if running is not None:
            condensed = condense_outputs(running.buffer.snapshot())
            lines.append(f"output (running for {running.elapsed():.0f}s — so far):")
        else:
            condensed = condense_outputs(ref.cell.get("outputs", []))
            lines.append("output:")
        lines.append(condensed.text)
        images = [Image(data=png, format="png") for png in condensed.images]
    return "\n".join(lines), images


def _select_refs(
    session: NotebookSession, names: list[str] | None, indices: str | None
) -> list[CellRef]:
    refs = session.nbfile.refs()
    if names:
        by_name = {r.name: r for r in refs}
        missing = [n for n in names if n not in by_name]
        if missing:
            from .errors import CellNotFound

            raise CellNotFound(missing[0], list(by_name))
        return [by_name[n] for n in names]
    if indices:
        m = re.fullmatch(r"(-?\d+)?:(-?\d+)?", indices.strip())
        if not m:
            raise JupyterMcpError(f"indices must look like '2:5', got {indices!r}")
        start = int(m.group(1)) if m.group(1) else None
        end = int(m.group(2)) if m.group(2) else None
        return refs[start:end]
    return refs


def _mutation_footer(session: NotebookSession, focus: str | None = None) -> str:
    graph = session.graph()
    stale = session.stale_names(graph)
    lines = []
    if focus and focus in graph.order:
        lines.append(f"deps: {_deps_line(graph, focus)}")
    if stale:
        lines.append(f"stale cells (run_stale to refresh): {', '.join(stale)}")
    if graph.undefined:
        undef = "; ".join(f"{n}: {sorted(v)}" for n, v in graph.undefined.items())
        lines.append(f"lint — names used but never defined in this notebook: {undef}")
    return "\n".join(lines)


# -------------------------------------------------------------------- tools


@mcp.tool()
@_tool_errors
def create_notebook(path: str, kernel_name: str = "python3") -> str:
    """Create a new empty notebook at `path` (must not exist yet)."""
    session = registry.register_new(path, kernel_name)
    return f"Created {session.path} (kernel: {kernel_name})."


@mcp.tool()
@_tool_errors
def notebook_overview(path: str, refresh_summaries: bool = True) -> str:
    """Index of the notebook: one line per cell (index, name, revision,
    staleness, one-line summary) plus dependency edges and lint findings.
    Start here when opening a notebook. Summaries marked with * are
    deterministic fallbacks, not LLM-generated."""
    session = registry.get(path)
    session.refresh_reads()
    graph = session.graph()
    notice = ""
    if refresh_summaries:
        result = session.summarizer.refresh(session.nbfile, graph)
        if result.refreshed:
            session.nbfile.save()
        notice = result.notice
    stale = set(session.stale_names(graph))
    current, queued_names = session.activity()
    queued = set(queued_names)
    kernel_line = session.kernel_status()
    if current is not None:
        kernel_line += f" — busy: {current.name!r} running for {current.elapsed():.0f}s"
        if queued_names:
            kernel_line += f", {len(queued_names)} queued"
    lines = [
        f"# {session.path} — {len(session.nbfile.cells)} cells",
        f"kernel: {kernel_line}",
    ]
    for ref in session.nbfile.refs():
        if current is not None and ref.name == current.name:
            marker = " RUNNING"
        elif ref.name in queued:
            marker = " QUEUED"
        elif ref.name in stale:
            marker = " STALE"
        else:
            marker = ""
        lines.append(
            f"[{ref.index:>2}] {ref.name}  (rev {ref.rev}; {ref.cell.cell_type}{marker})"
        )
        lines.append(f"     {get_tldr(ref.cell)}")
        if ref.cell.cell_type == "code" and graph.parents.get(ref.name):
            lines.append(f"     {_deps_line(graph, ref.name)}")
    footer = []
    if stale:
        footer.append(f"stale: {', '.join(n for n in graph.order if n in stale)}")
    for name, missing in graph.undefined.items():
        footer.append(f"lint {name}: uses undefined {sorted(missing)}")
    for name, deps in graph.deps.items():
        if deps.parse_error:
            footer.append(f"lint {name}: syntax error ({deps.parse_error})")
    if notice:
        footer.append(f"note: {notice}")
    if footer:
        lines.append("")
        lines.extend(footer)
    return "\n".join(lines)


@mcp.tool()
@_tool_errors
def read_cells(
    path: str,
    names: list[str] | None = None,
    indices: str | None = None,
    view: str = "full",
) -> list:
    """Read cells with code and condensed outputs (charts attached as images).
    Select by `names` (list of cell names), `indices` (python slice string
    like '0:5'), or neither for the whole notebook. `view`: full | code |
    outputs."""
    if view not in ("full", "code", "outputs"):
        raise JupyterMcpError("view must be one of: full, code, outputs")
    session = registry.get(path)
    session.refresh_reads()
    graph = session.graph()
    stale = set(session.stale_names(graph))
    blocks: list = []
    text_parts: list[str] = []
    for ref in _select_refs(session, names, indices):
        text, images = _render_cell(session, ref, graph, stale, view)
        text_parts.append(text)
        if images:
            blocks.append("\n\n---\n\n".join(text_parts))
            text_parts = []
            blocks.extend(images)
    if text_parts:
        blocks.append("\n\n---\n\n".join(text_parts))
    return blocks


def _after_mutation(session: NotebookSession, header: str, focus: str, run: str):
    """Shared tail for add/update: either report, or fold in a stale run."""
    if run not in ("none", "stale"):
        raise JupyterMcpError('run must be "none" or "stale"')
    if run == "stale":
        stale = session.stale_names()
        if not stale:
            return f"{header}\nNothing is stale — no cells executed."
        blocks = _run_and_render(session, stale, timeout_seconds=120, quiet=False, wait_seconds=60)
        if blocks and isinstance(blocks[0], str):
            blocks[0] = f"{header}\n\n{blocks[0]}"
        else:
            blocks.insert(0, header)
        return blocks
    return f"{header}\n" + _mutation_footer(session, focus)


@mcp.tool()
@_tool_errors
def add_cell(
    path: str,
    name: str,
    source: str,
    cell_type: str = "code",
    after: str | None = None,
    index: int | None = None,
    run: str = "none",
):
    """Add a cell. `name` must be unique kebab-case. Placement: `after` (an
    existing cell name; '' prepends), `index`, or omit both to append.
    `run="stale"` immediately executes every stale cell (the add→run loop in
    one call) and returns the execution results."""
    session = registry.get(path)
    ref: CellRef = session.mutate(
        f"add-{name}",
        lambda: session.nbfile.add_cell(name, source, cell_type, after, index),
    )
    header = f"Added {ref.cell.cell_type} cell {ref.name!r} at index {ref.index} (rev {ref.rev})."
    return _after_mutation(session, header, ref.name, run)


@mcp.tool()
@_tool_errors
def update_cell(
    path: str,
    name: str,
    expected_rev: str,
    source: str | None = None,
    new_name: str | None = None,
    run: str = "none",
):
    """Replace a cell's source and/or rename it. `expected_rev` must be the
    rev from your latest read of this cell (optimistic locking). Updating
    source clears the cell's outputs and marks it (and dependents) stale.
    `run="stale"` immediately executes every stale cell (the edit→run loop in
    one call) and returns the execution results."""
    session = registry.get(path)
    ref: CellRef = session.mutate(
        f"update-{name}",
        lambda: session.nbfile.update_cell(name, expected_rev, source, new_name),
    )
    header = f"Updated cell {ref.name!r} (new rev {ref.rev})."
    return _after_mutation(session, header, ref.name, run)


@mcp.tool()
@_tool_errors
def remove_cell(path: str, name: str, expected_rev: str) -> str:
    """Delete a cell (requires its current rev; undo with undo_last)."""
    session = registry.get(path)
    session.mutate(f"remove-{name}", lambda: session.nbfile.remove_cell(name, expected_rev))
    return f"Removed cell {name!r}.\n" + _mutation_footer(session)


@mcp.tool()
@_tool_errors
def move_cell(
    path: str,
    name: str,
    expected_rev: str,
    after: str | None = None,
    index: int | None = None,
) -> str:
    """Move a cell: `after` an existing cell ('' moves to top) or to `index`."""
    session = registry.get(path)
    ref: CellRef = session.mutate(
        f"move-{name}", lambda: session.nbfile.move_cell(name, expected_rev, after, index)
    )
    return f"Moved cell {name!r} to index {ref.index}.\n" + _mutation_footer(session, name)


@mcp.tool()
@_tool_errors
def run(
    path: str,
    cells: list[str] | None = None,
    fresh_deps: bool = True,
    timeout_seconds: float = 120,
    quiet: bool = False,
    wait_seconds: float = 60,
) -> list:
    """Execute cells on the notebook's persistent kernel.

    Default (no `cells`): every stale cell in document order — minimal
    recompute after edits. With `cells`: exactly those cells, even if fresh
    (API calls, randomness), preceded by their stale ancestors so inputs are
    trustworthy (`fresh_deps=false` skips the ancestor pass). Outputs are
    persisted and returned condensed; charts come back as images; `quiet`
    collapses ok cells to status lines. If everything finishes within
    `wait_seconds` you get full results; otherwise execution continues in the
    background — watch it via notebook_overview / read_cells (live output),
    stop it via interrupt."""
    session = registry.get(path)
    session.refresh_reads()
    graph = session.graph()
    stale = session.stale_names(graph)
    if cells is None:
        if not stale:
            return ["Nothing is stale — all code cells are up to date."]
        targets = stale
    else:
        known = {r.name for r in session.nbfile.refs()}
        for c in cells:
            if c not in known:
                from .errors import CellNotFound

                raise CellNotFound(c, sorted(known))
        wanted = set(cells)
        if fresh_deps:
            upstream: set[str] = set()
            for c in cells:
                upstream |= graph.ancestors(c)
            wanted |= upstream & set(stale)
        targets = [r.name for r in session.nbfile.refs() if r.name in wanted]
    return _run_and_render(session, targets, timeout_seconds, quiet, wait_seconds)


@mcp.tool()
@_tool_errors
def interrupt(path: str, clear_queue: bool = True) -> str:
    """Interrupt the currently running cell (KeyboardInterrupt in the kernel)
    and cancel queued cells. Kernel state (variables) survives; interrupted
    cells read as stale until a successful re-run."""
    session = registry.get(path)
    return session.interrupt(clear_queue)


def _run_and_render(
    session: NotebookSession,
    names: list[str],
    timeout_seconds: float,
    quiet: bool,
    wait_seconds: float,
) -> list:
    batch = session.submit_cells(names, timeout=timeout_seconds)
    if batch.wait(wait_seconds):
        return _render_exec_results(session, session.batch_results(batch), quiet=quiet)
    return [_progress_snapshot(batch, wait_seconds)]


def _progress_snapshot(batch, waited: float) -> str:
    lines = [f"Still running after the {waited:.0f}s check-in — execution continues in the background."]
    for t in batch.tasks:
        if t.status == "running":
            last = t.buffer.last_line()
            tail = f" — last output: {last}" if last else ""
            lines.append(f"running: {t.name} ({t.elapsed():.0f}s){tail}")
    queued = [t.name for t in batch.tasks if t.status == "queued"]
    if queued:
        lines.append(f"queued: {', '.join(queued)}")
    done = [(t.name, t.status) for t in batch.tasks if t.done.is_set()]
    if done:
        lines.append("finished: " + ", ".join(f"{n} ({s})" for n, s in done))
    lines.append(
        "Watch progress with notebook_overview / read_cells (live output so far); stop with interrupt."
    )
    return "\n".join(lines)


def _render_exec_results(
    session: NotebookSession,
    results: list[tuple[str, str, Condensed]],
    quiet: bool = False,
) -> list:
    blocks: list = []
    text_parts: list[str] = []
    for name, status, condensed in results:
        rev = ""
        try:
            rev = f" rev {session.nbfile.get(name).rev};"
        except JupyterMcpError:
            pass
        if quiet and status == "ok":
            text_parts.append(f"## {name} — ok{rev}")
        else:
            text_parts.append(f"## {name} — {status}{rev}\n{condensed.text}")
        for png in condensed.images:
            blocks.append("\n\n".join(text_parts))
            text_parts = []
            blocks.append(Image(data=png, format="png"))
    remaining_stale = session.stale_names()
    text_parts.append(
        f"still stale: {', '.join(remaining_stale)}" if remaining_stale else "all cells up to date"
    )
    blocks.append("\n\n".join(text_parts))
    return blocks


@mcp.tool()
@_tool_errors
def restart_kernel(path: str) -> str:
    """Restart the notebook's kernel (all in-memory state is lost; every code
    cell becomes stale)."""
    session = registry.get(path)
    if session.busy():
        raise JupyterMcpError("Kernel is busy (cell running or queued). interrupt first, then restart.")
    session.kernel().restart()
    # freshness is scoped to the kernel epoch, which just changed — no
    # metadata rewrite needed for cells to read as stale
    return "Kernel restarted. All code cells are now stale."


@mcp.tool()
@_tool_errors
def inspect_variable(path: str, variable: str, timeout_seconds: float = 30) -> list:
    """Inspect a live kernel variable without adding a cell: type, shape,
    schema/columns, length, plus its richest Jupyter repr — dataframes
    condense to a CSV table, figures come back as images, everything else
    falls back to a (pretty) repr."""
    session = registry.get(path)
    task = session.run_adhoc(inspect_code(variable), timeout=timeout_seconds)
    condensed = condense_outputs(task.buffer.snapshot())
    text = f"[{task.note}]\n{condensed.text}" if task.note else condensed.text
    blocks: list = [text]
    blocks.extend(Image(data=png, format="png") for png in condensed.images)
    return blocks


@mcp.tool()
@_tool_errors
def undo_last(path: str) -> str:
    """Restore the notebook to its state before the most recent mutation."""
    session = registry.get(path)
    if session.busy():
        raise JupyterMcpError("Cannot undo while cells are running or queued. interrupt first.")
    # an undo overwrites the file — external edits void it like any mutation
    session.guard_mutation()
    op = session.nbfile.undo_last()
    return f"Undid {op!r}. Re-read cells before further edits (revisions changed)."


@mcp.tool()
@_tool_errors
def summarize_cells(path: str, names: list[str] | None = None, include_outputs: bool = True) -> str:
    """Detailed summaries (LLM): per-cell description plus, optionally, a
    summary of each cell's current output. Cheaper than reading full cells
    when orienting in a large notebook."""
    session = registry.get(path)
    session.refresh_reads()
    graph = session.graph()
    result = session.summarizer.refresh(session.nbfile, graph, names)
    notices = [result.notice] if result.notice else []
    if include_outputs:
        items = []
        for ref in _select_refs(session, names, None):
            if ref.cell.cell_type == "code" and ref.cell.get("outputs"):
                items.append((ref.name, condense_outputs(ref.cell.outputs).text))
        out_result = session.summarizer.summarize_outputs(session.nbfile, items)
        if out_result.notice:
            notices.append(out_result.notice)
    session.nbfile.save()

    lines = []
    for ref in _select_refs(session, names, None):
        summ = get_summary(ref.cell)
        lines.append(f"[{ref.index}] {ref.name} (rev {ref.rev})")
        if summ:
            lines.append(f"  {summ['tldr']}")
            if summ.get("description"):
                lines.append(f"  {summ['description']}")
        else:
            lines.append(f"  {get_tldr(ref.cell)}")
        out_summ = cell_meta(ref.cell).get("output_summary")
        if include_outputs and out_summ and ref.cell.get("outputs"):
            # only show a summary that describes the CURRENT outputs
            current = condense_outputs(ref.cell.outputs).text
            if out_summ.get("output_hash") == output_hash(current):
                lines.append(f"  output: {out_summ['text']}")
    if notices:
        lines.append("")
        lines.extend(f"note: {n}" for n in dict.fromkeys(notices))
    return "\n".join(lines)


@mcp.tool()
@_tool_errors
def search_cells(path: str, query: str, regex: bool = False) -> str:
    """Search cell sources, names, summaries, and condensed outputs. Returns
    matching cells with the matching lines."""
    session = registry.get(path)
    session.refresh_reads()
    try:
        pattern = re.compile(query if regex else re.escape(query), re.IGNORECASE)
    except re.error as e:
        raise JupyterMcpError(f"Invalid regex: {e}") from e
    hits: list[str] = []
    for ref in session.nbfile.refs():
        matches: list[str] = []
        if pattern.search(ref.name):
            matches.append("(name match)")
        summ = get_summary(ref.cell)
        if summ and (pattern.search(summ.get("tldr", "")) or pattern.search(summ.get("description", ""))):
            matches.append("(summary match)")
        for i, line in enumerate(ref.cell.source.splitlines(), 1):
            if pattern.search(line):
                matches.append(f"  L{i}: {line.strip()[:120]}")
        if ref.cell.cell_type == "code" and ref.cell.get("outputs"):
            out_matches = [
                f"  out: {line.strip()[:120]}"
                for line in condense_outputs(ref.cell.outputs).text.splitlines()
                if pattern.search(line)
            ]
            matches.extend(out_matches[:5])  # cap noise from repetitive outputs
        if matches:
            hits.append(f"[{ref.index}] {ref.name} (rev {ref.rev})\n" + "\n".join(matches))
    return "\n\n".join(hits) if hits else f"No matches for {query!r}."


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()

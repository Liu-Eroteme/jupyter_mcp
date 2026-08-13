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
import os
import re
import sys
import traceback
from typing import Any

from mcp.server.fastmcp import FastMCP, Image

from . import kb as kb_mod
from .condense import condense_outputs
from .dag import NotebookGraph
from .errors import JupyterMcpError
from .kernel import DEFAULT_EXEC_TIMEOUT, find_project_python, inspect_code, resolve_kernel_name
from .model import META_NS, CellRef, cell_meta
from .session import CellResult, NotebookSession, Registry
from .summaries import get_summary, get_tldr, output_hash, summary_marker

mcp = FastMCP("jupyter-eda")
registry = Registry()

DEFAULT_WAIT_SECONDS = 60.0
MAX_DEFAULT_SECONDS = 86_400.0  # sanity cap for configure_notebook values


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


def _fmt_secs(seconds: float) -> str:
    if seconds >= 90:
        return f"{int(seconds // 60)}m{seconds % 60:02.0f}s"
    if seconds >= 10:
        return f"{seconds:.0f}s"
    return f"{seconds:.1f}s"


def _last_run_secs(ref: CellRef) -> float | None:
    secs = cell_meta(ref.cell).get("last_exec_seconds")
    return secs if isinstance(secs, (int, float)) else None


def _cell_header(session: NotebookSession, ref: CellRef, graph: NotebookGraph, stale: set[str]) -> str:
    flags = [ref.cell.cell_type]
    if ref.name in stale:
        flags.append("STALE")
    ec = ref.cell.get("execution_count")
    if ec:
        flags.append(f"exec#{ec}")
    secs = _last_run_secs(ref)
    if secs is not None:
        flags.append(f"last run {_fmt_secs(secs)}")
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
            outputs = ref.cell.get("outputs", [])
            condensed = condense_outputs(outputs)
            if outputs and ref.name in stale:
                lines.append("output (from a previous run — cell is stale):")
            else:
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


def _kb_crossref(session: NotebookSession) -> list[str]:
    """Knowledge-bundle hits for this notebook's code — the hook that makes
    the KB pay off (see ROADMAP). Env-configured bundles only; any failure
    must never break the overview."""
    if not os.environ.get(kb_mod.KB_ENV):
        return []
    try:
        entries = kb_mod.load_bundle(kb_mod.resolve_bundle(None))
        cells = [
            (r.name, r.cell.source)
            for r in session.nbfile.refs()
            if r.cell.cell_type == "code"
        ]
        return kb_mod.crossref(entries, cells)
    except Exception:
        return []


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
    # surface what the kernelspec will actually resolve to — invisible until
    # first execution otherwise, and reassuring to see up front
    if kernel_name in ("", "python3"):
        python = find_project_python(session.path)
        resolution = (
            f"will run on the project venv: {python}"
            if python is not None
            else "no project venv with ipykernel found nearby; will use the 'python3' kernelspec"
        )
    else:
        resolved, note = resolve_kernel_name(kernel_name)
        resolution = note or f"will use kernelspec {resolved!r}"
    return f"Created {session.path} (kernel: {kernel_name} — {resolution})."


@mcp.tool()
@_tool_errors
def configure_notebook(
    path: str,
    default_timeout_seconds: float | None = None,
    default_wait_seconds: float | None = None,
) -> str:
    """Set notebook-level defaults for run's timeout_seconds / wait_seconds,
    used whenever a run call doesn't pass them explicitly. Pass 0 to clear a
    default back to the global (timeout 120s, wait 60s); pass nothing to just
    report the current defaults. Stored in notebook metadata (undoable)."""
    session = registry.get(path)

    def describe() -> str:
        d = session.nbfile.defaults()
        if not d:
            return "no notebook defaults set (globals: timeout 120s, wait 60s)"
        parts = [f"{k.removesuffix('_seconds')} {v:g}s" for k, v in sorted(d.items())]
        return f"defaults: {', '.join(parts)}"

    updates = {
        "timeout_seconds": default_timeout_seconds,
        "wait_seconds": default_wait_seconds,
    }
    if all(v is None for v in updates.values()):
        session.refresh_reads()
        return describe()
    for key, value in updates.items():
        if value is not None and not (0 <= value <= MAX_DEFAULT_SECONDS):
            raise JupyterMcpError(f"default_{key} must be between 0 and {MAX_DEFAULT_SECONDS:g}.")

    def apply() -> None:
        meta = session.nbfile.nb.metadata.setdefault(META_NS, {})
        defaults = meta.setdefault("defaults", {})
        for key, value in updates.items():
            if value is None:
                continue
            if value == 0:
                defaults.pop(key, None)
            else:
                defaults[key] = value
        if not defaults:
            meta.pop("defaults", None)

    session.mutate("configure-defaults", apply)
    return f"Updated. Now: {describe()}"


@mcp.tool()
@_tool_errors
def notebook_overview(path: str, refresh_summaries: bool = True) -> str:
    """Index of the notebook: one line per cell (index, name, revision,
    staleness, one-line summary) plus dependency edges and lint findings.
    Start here when opening a notebook. Summary markers: ~ = LLM-generated
    (approximate — verify exact numbers against the cell itself), * =
    deterministic fallback (first comment/heading line)."""
    session = registry.get(path)
    session.refresh_reads()
    graph = session.graph()
    notice = ""
    if refresh_summaries:
        result = session.summarizer.refresh(session.nbfile, graph)
        notice = result.notice
        if result.refreshed and not session.save_if_unchanged():
            extra = "file changed on disk during summarization — reloaded; summaries regenerate next call"
            notice = f"{notice}; {extra}" if notice else extra
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
    nb_defaults = session.nbfile.defaults()
    if nb_defaults:
        parts = [f"{k.removesuffix('_seconds')} {v:g}s" for k, v in sorted(nb_defaults.items())]
        lines.append(f"run defaults (configure_notebook): {', '.join(parts)}")
    for ref in session.nbfile.refs():
        if current is not None and ref.name == current.name:
            marker = " RUNNING"
        elif ref.name in queued:
            marker = " QUEUED"
        elif ref.name in stale:
            marker = " STALE"
        else:
            marker = ""
        secs = _last_run_secs(ref)
        timing = f"; last run {_fmt_secs(secs)}" if secs is not None else ""
        lines.append(
            f"[{ref.index:>2}] {ref.name}  (rev {ref.rev}; {ref.cell.cell_type}{timing}{marker})"
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
    footer.extend(_kb_crossref(session))
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


def _exec_params(
    session: NotebookSession, timeout_seconds: float | None, wait_seconds: float | None
) -> tuple[float, float]:
    """Resolve execution timing: explicit arg > notebook default > global."""
    defaults = session.nbfile.defaults()
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else defaults.get("timeout_seconds", DEFAULT_EXEC_TIMEOUT)
    )
    wait = (
        wait_seconds
        if wait_seconds is not None
        else defaults.get("wait_seconds", DEFAULT_WAIT_SECONDS)
    )
    return float(timeout), float(wait)


def _after_mutation(session: NotebookSession, header: str, focus: str, run: str):
    """Shared tail for add/update: either report, or fold in a stale run."""
    if run not in ("none", "stale"):
        raise JupyterMcpError('run must be "none" or "stale"')
    if run == "stale":
        stale = session.stale_names()
        if not stale:
            return f"{header}\nNothing is stale — no cells executed."
        timeout, wait = _exec_params(session, None, None)
        blocks = _run_and_render(session, stale, timeout_seconds=timeout, quiet=False, wait_seconds=wait)
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
    timeout_seconds: float | None = None,
    quiet: bool = False,
    wait_seconds: float | None = None,
) -> list:
    """Execute cells on the notebook's persistent kernel.

    Default (no `cells`): every stale cell in document order — minimal
    recompute after edits. With `cells`: exactly those cells, even if fresh
    (API calls, randomness), preceded by their stale ancestors so inputs are
    trustworthy (`fresh_deps=false` skips the ancestor pass). Outputs are
    persisted and returned condensed; charts come back as images; `quiet`
    collapses ok cells to status lines, except cells you named in `cells` —
    their output always comes through. `timeout_seconds`/`wait_seconds`
    fall back to notebook defaults (configure_notebook), then 120s/60s.
    If everything finishes within `wait_seconds` you get full results;
    otherwise execution continues in the background — watch it via
    notebook_overview / read_cells (live output), stop it via interrupt."""
    session = registry.get(path)
    session.refresh_reads()
    timeout, wait = _exec_params(session, timeout_seconds, wait_seconds)
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
    return _run_and_render(session, targets, timeout, quiet, wait, always_full=set(cells or ()))


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
    always_full: set[str] = frozenset(),
) -> list:
    batch = session.submit_cells(names, timeout=timeout_seconds)
    if batch.wait(wait_seconds):
        return _render_exec_results(
            session, session.batch_results(batch), quiet=quiet, always_full=always_full
        )
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
    results: list[CellResult],
    quiet: bool = False,
    always_full: set[str] = frozenset(),
) -> list:
    blocks: list = []
    text_parts: list[str] = []
    for result in results:
        name, status, condensed = result.name, result.status, result.condensed
        rev = ""
        try:
            rev = f" rev {session.nbfile.get(name).rev};"
        except JupyterMcpError:
            pass
        took = f" ({_fmt_secs(result.duration)})" if result.duration is not None else ""
        if quiet and status == "ok" and name not in always_full:
            text_parts.append(f"## {name} — ok{took}{rev}")
        else:
            text_parts.append(f"## {name} — {status}{took}{rev}\n{condensed.text}")
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
    """Detailed summaries: per-cell description plus, optionally, a summary
    of each cell's current output. Cheaper than reading full cells when
    orienting in a large notebook. Summaries marked ~ are LLM-generated —
    approximate by nature, so verify exact numbers against the cell itself;
    * marks deterministic fallbacks (first comment/heading line)."""
    session = registry.get(path)
    session.refresh_reads()
    graph = session.graph()
    stale = set(session.stale_names(graph))
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
    if not session.save_if_unchanged():
        notices.append(
            "file changed on disk during summarization — reloaded; summaries regenerate next call"
        )

    lines = []
    for ref in _select_refs(session, names, None):
        summ = get_summary(ref.cell)
        stale_flag = "; STALE" if ref.name in stale else ""
        lines.append(f"[{ref.index}] {ref.name} (rev {ref.rev}{stale_flag})")
        if summ:
            lines.append(f"  {summ['tldr']}{summary_marker(summ)}")
            if summ.get("description"):
                lines.append(f"  {summ['description']}")
        else:
            lines.append(f"  {get_tldr(ref.cell)}")
        out_summ = cell_meta(ref.cell).get("output_summary")
        if include_outputs and out_summ and ref.cell.get("outputs"):
            # only show a summary that describes the CURRENT outputs
            current = condense_outputs(ref.cell.outputs).text
            if out_summ.get("output_hash") == output_hash(current):
                label = (
                    "output (from a previous run — cell is stale)"
                    if ref.name in stale
                    else "output"
                )
                lines.append(f"  {label}: {out_summ['text']} ~")
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


# ------------------------------------------------------------ knowledge base


@mcp.tool()
@_tool_errors
def kb_search(
    query: str,
    type: str | None = None,
    tag: str | None = None,
    bundle: str | None = None,
) -> str:
    """Search the knowledge bundle — durable notes about data sources,
    internal tooling, and recipes, accreted across notebooks and sessions.
    Check here before EDA on an unfamiliar source: units, caveats, join keys
    live here. Plain word matching over frontmatter + body; `type` filters
    (Data Source | Tool | Recipe), as does `tag`. The bundle directory comes
    from `bundle` or the JUPYTER_MCP_KB_PATH env var."""
    root = kb_mod.resolve_bundle(bundle)
    hits = kb_mod.search(kb_mod.load_bundle(root), query, type=type, tag=tag)
    if not hits:
        filters = "".join(f", {k}={v}" for k, v in (("type", type), ("tag", tag)) if v)
        return f"No KB entries match {query!r}{filters} in {root}."
    lines = []
    for hit in hits[:10]:
        e = hit.entry
        header = f"{e.rel_path} ({e.type}) — {e.title}"
        if e.resource:
            header += f" [resource: {e.resource}]"
        lines.append(header)
        if e.description:
            lines.append(f"  {e.description}")
        lines.extend(f"  > {ml}" for ml in hit.matching_lines)
    if len(hits) > 10:
        lines.append(f"… and {len(hits) - 10} more — narrow the query or filter.")
    lines.append("(kb_get a path for the full entry)")
    return "\n".join(lines)


@mcp.tool()
@_tool_errors
def kb_get(path: str, bundle: str | None = None) -> str:
    """Read one knowledge-bundle entry in full (frontmatter + body)."""
    root = kb_mod.resolve_bundle(bundle)
    e = kb_mod.get_entry(root, path)
    meta = "\n".join(f"{k}: {v}" for k, v in e.meta.items())
    return f"# {e.rel_path}\n{meta}\n\n{e.body.strip()}"


@mcp.tool()
@_tool_errors
def kb_upsert(
    path: str,
    type: str,
    title: str | None = None,
    description: str | None = None,
    resource: str | None = None,
    tags: list[str] | None = None,
    body: str | None = None,
    bundle: str | None = None,
) -> str:
    """Create or update a knowledge-bundle entry. Record durable facts worth
    re-finding: data-source caveats (units, quirks, join keys), internal
    tooling install/usage, known-good recipes. Update-don't-duplicate: `path`
    (bundle-relative, e.g. 'data-sources/omniplus') is the key; omitted
    fields keep their existing values. `resource` is the canonical URI (file
    glob, table name, package) used to crossref entries into
    notebook_overview automatically. Writes are logged to the bundle's
    log.md."""
    root = kb_mod.resolve_bundle(bundle, create=True)
    entry, created = kb_mod.upsert(
        root,
        path,
        type=type,
        title=title,
        description=description,
        resource=resource,
        tags=tags,
        body=body,
    )
    return f"{'Created' if created else 'Updated'} {entry.rel_path} in bundle {root}."


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()

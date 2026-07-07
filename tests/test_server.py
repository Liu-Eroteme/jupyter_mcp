"""End-to-end tests through the MCP tool functions, including a real kernel."""

import pytest

from jupyter_mcp import server
from jupyter_mcp.model import NotebookFile


def _rev(path, name):
    nbf = NotebookFile(path)
    nbf.load()
    return nbf.get(name).rev


@pytest.fixture
def nb(tmp_path):
    path = tmp_path / "eda.ipynb"
    out = server.create_notebook(str(path))
    assert "Created" in out
    return path


def test_edit_workflow(nb):
    assert "Added" in server.add_cell(str(nb), "load", "data = [1, 2, 3]")
    assert "Added" in server.add_cell(str(nb), "total", "total = sum(data)", after="load")
    assert "Added" in server.add_cell(str(nb), "report", "print(total)")

    overview = server.notebook_overview(str(nb))
    assert "load" in overview and "total" in overview and "report" in overview
    assert "STALE" in overview  # nothing executed yet
    assert "← load(data)" in overview

    # wrong rev is rejected with re-orientation info
    res = server.update_cell(str(nb), "total", "badrev0000", source="total = sum(data) * 2")
    assert res.startswith("ERROR") and "Revision mismatch" in res

    res = server.update_cell(str(nb), "total", _rev(nb, "total"), source="total = sum(data) * 2")
    assert "Updated" in res and "stale" in res

    # duplicate name rejected
    res = server.add_cell(str(nb), "load", "x = 1")
    assert res.startswith("ERROR")


def test_read_and_search(nb):
    server.add_cell(str(nb), "load", "# load the data\ndata = [1, 2, 3]")
    server.add_cell(str(nb), "total", "total = sum(data)")
    blocks = server.read_cells(str(nb), names=["total"])
    text = blocks[0]
    assert "total" in text and "sum(data)" in text and "← load(data)" in text

    hits = server.search_cells(str(nb), "sum")
    assert "total" in hits
    assert "No matches" in server.search_cells(str(nb), "zzz-not-there")


def test_undo(nb):
    server.add_cell(str(nb), "keep", "x = 1")
    server.add_cell(str(nb), "oops", "y = 2")
    out = server.undo_last(str(nb))
    assert "Undid" in out
    overview = server.notebook_overview(str(nb))
    assert "oops" not in overview and "keep" in overview


@pytest.mark.kernel
def test_execute_and_staleness(nb):
    server.add_cell(str(nb), "load", "data = [1, 2, 3]")
    server.add_cell(str(nb), "total", "total = sum(data)\nprint('total is', total)")
    server.add_cell(str(nb), "broken", "print(undefined_variable)")

    blocks = server.run(str(nb))
    text = "\n".join(b for b in blocks if isinstance(b, str))
    assert "total is 6" in text
    assert "NameError" in text
    assert "still stale: broken" in text  # failed cell remains stale

    # fix the broken cell and re-run: only it should execute
    server.update_cell(str(nb), "broken", _rev(nb, "broken"), source="print('fixed', total)")
    blocks = server.run(str(nb))
    text = "\n".join(b for b in blocks if isinstance(b, str))
    assert "fixed 6" in text
    assert "## load" not in text  # untouched upstream not re-run
    assert "all cells up to date" in text

    # editing upstream makes dependents stale again
    server.update_cell(str(nb), "load", _rev(nb, "load"), source="data = [10, 20]")
    overview = server.notebook_overview(str(nb))
    assert overview.count("STALE") == 3


@pytest.mark.kernel
def test_inspect_variable(nb):
    server.add_cell(str(nb), "setup", "nums = list(range(100))")
    server.run(str(nb), cells=["setup"])
    out = server.inspect_variable(str(nb), "nums")
    assert "type: builtins.list" in out[0]
    assert "len: 100" in out[0]
    out = server.inspect_variable(str(nb), "not_defined")
    assert "NameError" in out[0]


@pytest.mark.kernel
def test_execute_returns_image(nb):
    server.add_cell(
        str(nb),
        "tiny-png",
        "import base64\n"
        "from IPython.display import display, Image as IPyImage\n"
        "png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==')\n"
        "display(IPyImage(data=png))",
    )
    blocks = server.run(str(nb), cells=["tiny-png"])
    from mcp.server.fastmcp import Image

    assert any(isinstance(b, Image) for b in blocks)


@pytest.mark.kernel
def test_staleness_scoped_to_kernel_epoch(nb, tmp_path):
    """Regression: persisted freshness must not survive into a NEW kernel.

    Found while dogfooding: a fresh server process trusted last_exec_rev from
    a previous process's kernel, so run_stale executed a 'minimal' set
    against an empty kernel and hit NameError on upstream variables.
    """
    from jupyter_mcp.session import NotebookSession
    from jupyter_mcp.summaries import Summarizer

    server.add_cell(str(nb), "load", "data = [1, 2]")
    server.add_cell(str(nb), "use", "print(sum(data))")
    server.run(str(nb))
    # same session + live kernel: nothing stale
    assert server.registry.get(str(nb)).stale_names() == []

    # simulate a new server process: fresh session, no kernel started
    fresh = NotebookSession(nb, Summarizer())
    assert fresh.stale_names() == ["load", "use"]

    # and executing through the fresh session must include upstream cells
    results = fresh.execute_cells(fresh.stale_names())
    try:
        assert [(name, status) for name, status, _ in results] == [
            ("load", "ok"),
            ("use", "ok"),
        ]
    finally:
        fresh.stop()


@pytest.mark.kernel
def test_restart_kernel_marks_all_stale(nb):
    server.add_cell(str(nb), "setup", "x = 1")
    server.run(str(nb), cells=["setup"])
    assert "STALE" not in server.notebook_overview(str(nb))
    server.restart_kernel(str(nb))
    assert "STALE" in server.notebook_overview(str(nb))


# ------------------------------------------------- review-round regressions


def _disk_names(path):
    nbf = NotebookFile(path)
    nbf.load()
    return nbf.names()


def test_failed_move_preserves_notebook(nb):
    """Regression: move_cell popped the cell before validating, corrupting
    the in-memory model; the next successful mutation then persisted the
    corruption, deleting the cell from disk."""
    server.add_cell(str(nb), "a", "x = 1")
    server.add_cell(str(nb), "b", "y = 2")
    session = server.registry.get(str(nb))
    rev_a = _rev(nb, "a")

    res = server.move_cell(str(nb), "a", rev_a, after="a")
    assert res.startswith("ERROR")
    assert session.nbfile.names() == ["a", "b"]

    res = server.move_cell(str(nb), "a", rev_a, after="missing")
    assert res.startswith("ERROR")
    assert "a, b" in res  # error reports the real cells, not corrupted state
    assert session.nbfile.names() == ["a", "b"]

    res = server.move_cell(str(nb), "a", rev_a, after="b", index=0)
    assert res.startswith("ERROR")
    assert session.nbfile.names() == ["a", "b"]

    server.add_cell(str(nb), "c", "z = 3")
    assert _disk_names(nb) == ["a", "b", "c"]


def test_update_source_clears_output_summary(nb):
    from jupyter_mcp.model import cell_meta

    server.add_cell(str(nb), "calc", "print(1)")
    session = server.registry.get(str(nb))
    ref = session.nbfile.get("calc")
    cell_meta(ref.cell)["output_summary"] = {
        "output_hash": "aaaaaaaaaa",
        "text": "OLD SUMMARY",
        "source": "llm",
    }
    session.nbfile.save()

    server.update_cell(str(nb), "calc", ref.rev, source="print(2)")
    assert "output_summary" not in cell_meta(session.nbfile.get("calc").cell)
    assert "OLD SUMMARY" not in server.summarize_cells(str(nb), names=["calc"])


def test_output_summary_shown_only_when_hash_matches(nb):
    import nbformat

    from jupyter_mcp.condense import condense_outputs
    from jupyter_mcp.model import cell_meta
    from jupyter_mcp.summaries import output_hash

    server.add_cell(str(nb), "calc", "print(1)")
    session = server.registry.get(str(nb))
    ref = session.nbfile.get("calc")
    ref.cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="1\n")]
    cell_meta(ref.cell)["output_summary"] = {
        "output_hash": "stale00000",
        "text": "WRONG",
        "source": "llm",
    }
    session.nbfile.save()
    assert "WRONG" not in server.summarize_cells(str(nb), names=["calc"])

    current = condense_outputs(ref.cell.outputs).text
    cell_meta(ref.cell)["output_summary"] = {
        "output_hash": output_hash(current),
        "text": "RIGHT",
        "source": "llm",
    }
    session.nbfile.save()
    assert "output: RIGHT" in server.summarize_cells(str(nb), names=["calc"])


def test_kernel_startup_failure_surfaces_in_results(nb, monkeypatch):
    """Startup failures (e.g. PermissionError from socket creation) must
    surface as an actionable message, never a raw traceback. With the
    executor they arrive as an error result on the affected cell."""
    from jupyter_mcp import kernel as kernel_mod

    def denied(self, **kwargs):
        raise PermissionError("socket bind not permitted")

    monkeypatch.setattr(kernel_mod.KernelManager, "start_kernel", denied)
    server.add_cell(str(nb), "a", "x = 1")
    res = server.run(str(nb), cells=["a"])
    text = "\n".join(b for b in res if isinstance(b, str))
    assert "## a — error" in text
    assert "socket bind not permitted" in text


def test_undo_rejects_external_modification(nb):
    server.add_cell(str(nb), "keep", "x = 1")
    external = nb.read_text().replace("x = 1", "x = 42")
    nb.write_text(external)
    res = server.undo_last(str(nb))
    assert res.startswith("ERROR") and "modified on disk" in res
    assert "x = 42" in nb.read_text()  # the external edit was not clobbered


@pytest.mark.kernel
def test_failed_run_voids_freshness_stamp(nb):
    """Regression: a failed run kept the old stamp, so reverting the source
    made the cell read fresh while the kernel held the failed run's state."""
    from jupyter_mcp.model import cell_meta

    server.add_cell(str(nb), "a", "x = 1")
    server.run(str(nb), cells=["a"])
    session = server.registry.get(str(nb))
    assert session.stale_names() == []

    server.update_cell(str(nb), "a", _rev(nb, "a"), source="x = 2\nraise ValueError('boom')")
    server.run(str(nb))
    assert "last_exec_rev" not in cell_meta(session.nbfile.get("a").cell)

    # reverting must NOT resurrect freshness: the kernel already ran x = 2
    server.update_cell(str(nb), "a", _rev(nb, "a"), source="x = 1")
    assert session.stale_names() == ["a"]


@pytest.mark.kernel
def test_edit_revert_without_run_stays_fresh(nb):
    """Edit + revert with no execution attempt in between is genuinely fresh."""
    server.add_cell(str(nb), "a", "x = 1")
    server.run(str(nb), cells=["a"])
    session = server.registry.get(str(nb))
    server.update_cell(str(nb), "a", _rev(nb, "a"), source="x = 999")
    assert session.stale_names() == ["a"]
    server.update_cell(str(nb), "a", _rev(nb, "a"), source="x = 1")
    assert session.stale_names() == []


# ------------------------------------------------------------ round 3: P1-P3


def test_deleted_notebook_clean_error(nb):
    server.add_cell(str(nb), "a", "x = 1")
    nb.unlink()
    res = server.notebook_overview(str(nb))
    assert isinstance(res, str) and res.startswith("ERROR") and "deleted or moved" in res
    # the session was evicted: a second call reports a plain missing file
    res = server.notebook_overview(str(nb))
    assert res.startswith("ERROR") and "does not exist" in res


def test_internal_errors_never_leak_raw(nb, monkeypatch):
    monkeypatch.setattr(
        server.registry, "get", lambda path: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    res = server.notebook_overview(str(nb))
    assert isinstance(res, str)
    assert res.startswith("ERROR (internal RuntimeError)") and "boom" in res


def test_overview_shows_kernel_status(nb):
    server.add_cell(str(nb), "a", "x = 1")
    overview = server.notebook_overview(str(nb))
    assert "kernel: not started" in overview


def test_run_param_validated(nb):
    res = server.add_cell(str(nb), "a", "x = 1", run="bogus")
    assert res.startswith("ERROR") and "run must be" in res


def test_undo_survives_new_session(nb):
    """The undo stack is seeded from on-disk snapshots, so undo works after
    a server restart."""
    server.add_cell(str(nb), "keep", "x = 1")
    server.add_cell(str(nb), "oops", "y = 2")

    fresh = NotebookFile(nb)  # simulates a new server process
    fresh.load()
    op = fresh.undo_last()
    assert op == "add-oops"
    assert _disk_names(nb) == ["keep"]


def test_search_finds_output_lines(nb):
    import nbformat

    server.add_cell(str(nb), "calc", "print(compute())")
    session = server.registry.get(str(nb))
    ref = session.nbfile.get("calc")
    ref.cell.outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="ratio: 3.532\n")
    ]
    session.nbfile.save()
    hits = server.search_cells(str(nb), "3.532")
    assert "calc" in hits and "out: ratio: 3.532" in hits


@pytest.mark.kernel
def test_update_cell_run_stale(nb):
    server.add_cell(str(nb), "load", "data = [1, 2, 3]")
    server.add_cell(str(nb), "use", "print('sum', sum(data))")
    server.run(str(nb))

    res = server.update_cell(
        str(nb), "load", _rev(nb, "load"), source="data = [10, 20]", run="stale"
    )
    assert isinstance(res, list)
    text = "\n".join(b for b in res if isinstance(b, str))
    assert "Updated cell 'load'" in text
    assert "## load — ok" in text and "sum 30" in text
    assert "all cells up to date" in text

    # nothing stale afterwards: run="stale" reports instead of executing
    res = server.update_cell(str(nb), "use", _rev(nb, "use"), new_name="report", run="stale")
    assert isinstance(res, str) and "Nothing is stale" in res


@pytest.mark.kernel
def test_quiet_execution(nb):
    server.add_cell(str(nb), "load", "data = [1, 2, 3]")
    server.add_cell(str(nb), "use", "print('sum', sum(data))")
    server.add_cell(str(nb), "broken", "boom()")
    blocks = server.run(str(nb), quiet=True)
    text = "\n".join(b for b in blocks if isinstance(b, str))
    assert "## load — ok" in text and "## use — ok" in text
    assert "sum 6" not in text  # ok output collapsed
    assert "NameError" in text  # errors still come through in full


@pytest.mark.kernel
def test_idle_kernel_ttl(nb):
    server.add_cell(str(nb), "a", "x = 1")
    server.run(str(nb), cells=["a"])
    session = server.registry.get(str(nb))
    assert session.stale_names() == []

    kernel = session._kernel
    assert kernel is not None
    kernel.last_used -= 10 * 24 * 3600  # simulate long idleness
    overview = server.notebook_overview(str(nb))  # registry access sweeps
    assert session._kernel is None
    assert "kernel: not started" in overview
    assert "STALE" in overview  # a future kernel has a new epoch


# ------------------------------------------- round 4: unified run + async


@pytest.mark.kernel
def test_run_with_cells_freshens_stale_ancestors(nb):
    server.add_cell(str(nb), "load", "data = [1, 2, 3]")
    server.add_cell(str(nb), "use", "print('sum', sum(data))")
    server.run(str(nb))

    # edit upstream: both stale; ask only for the downstream cell
    server.update_cell(str(nb), "load", _rev(nb, "load"), source="data = [10, 20]")
    blocks = server.run(str(nb), cells=["use"])
    text = "\n".join(b for b in blocks if isinstance(b, str))
    assert "## load — ok" in text  # stale ancestor ran first
    assert "sum 30" in text
    assert "all cells up to date" in text

    # fresh_deps=False runs exactly the requested cell against old state
    server.update_cell(str(nb), "load", _rev(nb, "load"), source="data = [100]")
    blocks = server.run(str(nb), cells=["use"], fresh_deps=False)
    text = "\n".join(b for b in blocks if isinstance(b, str))
    assert "## load" not in text
    assert "sum 30" in text  # kernel still holds the previous data


@pytest.mark.kernel
def test_run_reruns_fresh_cells(nb):
    server.add_cell(str(nb), "tick", "import random\nprint(random.random())")
    server.run(str(nb))
    assert server.registry.get(str(nb)).stale_names() == []
    blocks = server.run(str(nb), cells=["tick"])  # fresh, but requested
    text = "\n".join(b for b in blocks if isinstance(b, str))
    assert "## tick — ok" in text


@pytest.mark.kernel
def test_background_run_live_view_interrupt(nb):
    import time as _time

    server.add_cell(
        str(nb),
        "slow",
        "import time\nfor i in range(100):\n    print('tick', i)\n    time.sleep(0.2)",
    )
    server.add_cell(str(nb), "after", "print('after')")

    blocks = server.run(str(nb), wait_seconds=1)
    text = "\n".join(b for b in blocks if isinstance(b, str))
    assert "background" in text and "running: slow" in text and "queued: after" in text

    session = server.registry.get(str(nb))
    assert session.busy()

    # overview shows activity; read_cells shows accumulated output
    overview = server.notebook_overview(str(nb), refresh_summaries=False)
    assert "RUNNING" in overview and "QUEUED" in overview and "busy" in overview
    cell_view = server.read_cells(str(nb), names=["slow"], view="outputs")
    view_text = "\n".join(b for b in cell_view if isinstance(b, str))
    assert "running for" in view_text and "tick" in view_text

    # busy guards
    assert server.restart_kernel(str(nb)).startswith("ERROR")
    assert server.undo_last(str(nb)).startswith("ERROR")
    assert server.inspect_variable(str(nb), "i").startswith("ERROR")

    out = server.interrupt(str(nb))
    assert "Interrupted 'slow'" in out and "after" in out  # queued cancelled

    deadline = _time.monotonic() + 15
    while session.busy() and _time.monotonic() < deadline:
        _time.sleep(0.2)
    assert not session.busy()

    # the interrupted cell reads stale; kernel survived
    assert "slow" in session.stale_names()
    out = server.inspect_variable(str(nb), "i")
    assert "type: builtins.int" in out[0]


@pytest.mark.kernel
def test_inspect_rich_reprs(nb):
    server.add_cell(
        str(nb),
        "objs",
        "import base64\n"
        "class HtmlTable:\n"
        "    def _repr_html_(self):\n"
        "        return '<table><tr><th>a</th><th>b</th></tr>"
        "<tr><td>1</td><td>2</td></tr></table>'\n"
        "    def __repr__(self):\n"
        "        return 'HtmlTable()'\n"
        "class Png:\n"
        "    def _repr_png_(self):\n"
        "        return base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==')\n"
        "t = HtmlTable()\n"
        "p = Png()",
    )
    server.run(str(nb))

    out = server.inspect_variable(str(nb), "t")
    assert "[table as CSV" in out[0] and "a,b" in out[0]

    out = server.inspect_variable(str(nb), "p")
    from mcp.server.fastmcp import Image

    assert any(isinstance(b, Image) for b in out)

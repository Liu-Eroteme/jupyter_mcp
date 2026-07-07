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

    blocks = server.run_stale(str(nb))
    text = "\n".join(b for b in blocks if isinstance(b, str))
    assert "total is 6" in text
    assert "NameError" in text
    assert "still stale: broken" in text  # failed cell remains stale

    # fix the broken cell and re-run: only it should execute
    server.update_cell(str(nb), "broken", _rev(nb, "broken"), source="print('fixed', total)")
    blocks = server.run_stale(str(nb))
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
    server.execute_cells(str(nb), names=["setup"])
    out = server.inspect_variable(str(nb), "nums")
    assert "type: builtins.list" in out
    assert "len: 100" in out
    out = server.inspect_variable(str(nb), "not_defined")
    assert "NameError" in out


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
    blocks = server.execute_cells(str(nb), names=["tiny-png"])
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
    server.run_stale(str(nb))
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
        fresh.shutdown_kernel()


@pytest.mark.kernel
def test_restart_kernel_marks_all_stale(nb):
    server.add_cell(str(nb), "setup", "x = 1")
    server.execute_cells(str(nb), names=["setup"])
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


def test_kernel_startup_failure_returns_tool_error(nb, monkeypatch):
    """Regression: non-JupyterMcpError startup failures (e.g. PermissionError
    from socket creation) escaped the tool wrapper as raw tracebacks."""
    from jupyter_mcp import kernel as kernel_mod

    def denied(self, **kwargs):
        raise PermissionError("socket bind not permitted")

    monkeypatch.setattr(kernel_mod.KernelManager, "start_kernel", denied)
    server.add_cell(str(nb), "a", "x = 1")
    res = server.execute_cells(str(nb), names=["a"])
    assert isinstance(res, str)
    assert res.startswith("ERROR") and "socket bind not permitted" in res


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
    server.execute_cells(str(nb), names=["a"])
    session = server.registry.get(str(nb))
    assert session.stale_names() == []

    server.update_cell(str(nb), "a", _rev(nb, "a"), source="x = 2\nraise ValueError('boom')")
    server.run_stale(str(nb))
    assert "last_exec_rev" not in cell_meta(session.nbfile.get("a").cell)

    # reverting must NOT resurrect freshness: the kernel already ran x = 2
    server.update_cell(str(nb), "a", _rev(nb, "a"), source="x = 1")
    assert session.stale_names() == ["a"]


@pytest.mark.kernel
def test_edit_revert_without_run_stays_fresh(nb):
    """Edit + revert with no execution attempt in between is genuinely fresh."""
    server.add_cell(str(nb), "a", "x = 1")
    server.execute_cells(str(nb), names=["a"])
    session = server.registry.get(str(nb))
    server.update_cell(str(nb), "a", _rev(nb, "a"), source="x = 999")
    assert session.stale_names() == ["a"]
    server.update_cell(str(nb), "a", _rev(nb, "a"), source="x = 1")
    assert session.stale_names() == []

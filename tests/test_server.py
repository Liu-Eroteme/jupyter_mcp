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
def test_restart_kernel_marks_all_stale(nb):
    server.add_cell(str(nb), "setup", "x = 1")
    server.execute_cells(str(nb), names=["setup"])
    assert "STALE" not in server.notebook_overview(str(nb))
    server.restart_kernel(str(nb))
    assert "STALE" in server.notebook_overview(str(nb))

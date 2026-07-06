import os

# must be set before jupyter_mcp.summaries is imported anywhere:
# tests never call the Anthropic API.
os.environ["JUPYTER_MCP_DISABLE_SUMMARIES"] = "1"

from pathlib import Path

import nbformat
import pytest

import jupyter_mcp.model as model_mod


@pytest.fixture(autouse=True)
def _isolated_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(model_mod, "SNAPSHOT_ROOT", tmp_path / "snaps")


@pytest.fixture
def make_notebook(tmp_path):
    """Write a raw .ipynb (as an editor would) and return its path."""

    def _make(cells: list[tuple[str, str]], name: str = "nb.ipynb") -> Path:
        nb = nbformat.v4.new_notebook()
        nb.metadata["kernelspec"] = {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        }
        for cell_type, source in cells:
            if cell_type == "code":
                nb.cells.append(nbformat.v4.new_code_cell(source))
            else:
                nb.cells.append(nbformat.v4.new_markdown_cell(source))
        path = tmp_path / name
        nbformat.write(nb, str(path))
        return path

    return _make

import sys
from pathlib import Path

import pytest

from jupyter_mcp.kernel import KernelSession, find_project_python


def _fake_venv(root: Path) -> Path:
    bindir = root / ".venv" / "bin"
    bindir.mkdir(parents=True)
    python = bindir / "python"
    python.symlink_to(sys.executable)  # test interpreter has ipykernel
    return python


def test_find_project_python_walks_up(tmp_path):
    python = _fake_venv(tmp_path)
    nb_dir = tmp_path / "notebooks"
    nb_dir.mkdir()
    found = find_project_python(nb_dir / "analysis.ipynb")
    assert found == python


def test_find_project_python_stops_at_git_root(tmp_path):
    _fake_venv(tmp_path)  # venv ABOVE the git root must not be found
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    assert find_project_python(repo / "nb.ipynb") is None


def test_find_project_python_none(tmp_path):
    assert find_project_python(tmp_path / "nb.ipynb") is None


@pytest.mark.kernel
def test_kernel_uses_project_venv(tmp_path):
    python = _fake_venv(tmp_path)
    nb_path = tmp_path / "nb.ipynb"
    session = KernelSession(nb_path, "python3")
    try:
        session.start()
        assert str(python) in session.note
        res = session.execute("import sys; print(sys.executable)")
        assert res.status == "ok"
        text = "".join(o.get("text", "") for o in res.outputs)
        # the kernel must run the venv interpreter, not the server's
        assert text.strip() == str(python)
    finally:
        session.shutdown()

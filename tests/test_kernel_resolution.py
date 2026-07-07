import os
import sysconfig
import venv
from pathlib import Path

import pytest

from jupyter_mcp import kernel as kernel_mod
from jupyter_mcp.kernel import KernelSession, find_project_python


def _fake_venv(root: Path) -> Path:
    """Layout-only fake: just the interpreter path the resolver probes.

    A plain file (no symlink, no real env) keeps the walk-logic unit tests
    platform-independent; pair it with a monkeypatched `_has_ipykernel`.
    """
    python = root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    return python


@pytest.fixture
def accept_any_python(monkeypatch):
    monkeypatch.setattr(kernel_mod, "_has_ipykernel", lambda python: True)


def test_find_project_python_walks_up(tmp_path, accept_any_python):
    python = _fake_venv(tmp_path)
    nb_dir = tmp_path / "notebooks"
    nb_dir.mkdir()
    assert find_project_python(nb_dir / "analysis.ipynb") == python


def test_find_project_python_stops_at_git_root(tmp_path, accept_any_python):
    _fake_venv(tmp_path)  # venv ABOVE the git root must not be found
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    assert find_project_python(repo / "nb.ipynb") is None


def test_find_project_python_requires_ipykernel(tmp_path, monkeypatch):
    _fake_venv(tmp_path)
    monkeypatch.setattr(kernel_mod, "_has_ipykernel", lambda python: False)
    assert find_project_python(tmp_path / "nb.ipynb") is None


def test_find_project_python_none(tmp_path, accept_any_python):
    assert find_project_python(tmp_path / "nb.ipynb") is None


def _real_venv(root: Path) -> Path:
    """A genuine venv (pyvenv.cfg and all) that can import ipykernel.

    A venv created *from* a venv chains to the base interpreter, not to the
    creating env's site-packages — so the test env's packages are grafted in
    via a .pth file instead of `system_site_packages`.
    """
    venv_dir = root / ".venv"
    venv.EnvBuilder(with_pip=False, symlinks=(os.name != "nt")).create(venv_dir)
    site_dir = next(
        iter(venv_dir.glob("lib/python*/site-packages")),
        venv_dir / "Lib" / "site-packages",
    )
    parent_paths = {sysconfig.get_paths()[k] for k in ("purelib", "platlib")}
    (site_dir / "_test_parent_env.pth").write_text("\n".join(sorted(parent_paths)) + "\n")
    python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    assert python.exists()
    return python


@pytest.mark.kernel
def test_kernel_uses_project_venv(tmp_path):
    python = _real_venv(tmp_path)
    nb_path = tmp_path / "nb.ipynb"
    session = KernelSession(nb_path, "python3")
    try:
        session.start()
        assert str(python) in session.note
        res = session.execute("import sys; print(sys.executable)")
        assert res.status == "ok"
        text = "".join(o.get("text", "") for o in res.outputs)
        # the kernel must run the venv interpreter, not the server's
        assert Path(text.strip()) == python
    finally:
        session.shutdown()

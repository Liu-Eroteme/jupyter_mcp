import pytest

from jupyter_mcp.errors import (
    DuplicateCellName,
    ExternalModification,
    JupyterMcpError,
    NothingToUndo,
    RevisionMismatch,
)
from jupyter_mcp.model import NotebookFile, cell_name
from jupyter_mcp.session import NotebookSession
from jupyter_mcp.summaries import Summarizer


def load(path):
    nbf = NotebookFile(path)
    nbf.load()
    return nbf


def test_auto_names_from_comments_and_headings(make_notebook):
    path = make_notebook(
        [
            ("code", "# load raw data\nimport polars as pl"),
            ("markdown", "## Step 2 — Clean"),
            ("code", "x = 1"),
            ("code", "# load raw data\nother = 2"),  # duplicate comment
        ]
    )
    nbf = load(path)
    names = nbf.names()
    assert names[0] == "load-raw-data"
    assert names[1] == "md-step-2-clean"
    assert names[3] == "load-raw-data-2"  # deduped
    assert len(set(names)) == len(names)


def test_meaningful_cell_ids_adopted_as_names(make_notebook):
    """Feedback: notebooks authored elsewhere carry deliberate ids
    (fetch-gtfs) — addressing must honor them, not synthesize from content."""
    import json

    path = make_notebook(
        [
            ("code", "import polars as pl"),
            ("code", "gtfs = fetch()"),
            ("code", "x = 1"),
            ("code", "y = 2"),
        ]
    )
    raw = json.loads(path.read_text())
    raw["cells"][0]["id"] = "imports"  # meaningful single word
    raw["cells"][1]["id"] = "fetch-gtfs"  # meaningful kebab-case
    raw["cells"][2]["id"] = "6b71d8c6"  # nbformat auto id (hex) — skip
    raw["cells"][3]["id"] = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"  # uuid — skip
    path.write_text(json.dumps(raw))

    names = load(path).names()
    assert names[0] == "imports"
    assert names[1] == "fetch-gtfs"
    assert names[2] == "x-1"  # content-derived, not the hex id
    assert names[3] == "y-2"


def test_adopted_id_colliding_with_existing_name_deduped(make_notebook):
    # (nbformat corrects duplicate *ids* to random hex on load itself, so the
    # collision that can actually reach us is id vs. stored metadata name)
    import json

    path = make_notebook([("code", "a = 1"), ("code", "b = 2")])
    raw = json.loads(path.read_text())
    raw["cells"][0]["metadata"]["jupyter_mcp"] = {"name": "setup"}
    raw["cells"][1]["id"] = "setup"
    path.write_text(json.dumps(raw))
    assert load(path).names() == ["setup", "setup-2"]


def test_rev_changes_with_source(make_notebook):
    nbf = load(make_notebook([("code", "x = 1")]))
    ref = nbf.refs()[0]
    rev = ref.rev
    nbf.update_cell(ref.name, rev, source="x = 2")
    assert nbf.refs()[0].rev != rev


def test_revision_mismatch(make_notebook):
    nbf = load(make_notebook([("code", "x = 1")]))
    name = nbf.names()[0]
    with pytest.raises(RevisionMismatch) as exc:
        nbf.update_cell(name, "wrong00000", source="x = 2")
    assert "x = 1" in str(exc.value)  # carries current source for re-orientation


def test_add_remove_move_and_placement(make_notebook):
    nbf = load(make_notebook([("code", "a = 1"), ("code", "b = 2")]))
    first, second = nbf.names()
    nbf.add_cell("new-middle", "m = 1", after=first)
    assert nbf.names()[1] == "new-middle"
    nbf.add_cell("top", "t = 1", after="")
    assert nbf.names()[0] == "top"
    ref = nbf.get("new-middle")
    nbf.move_cell("new-middle", ref.rev, index=0)
    assert nbf.names()[0] == "new-middle"
    ref = nbf.get("new-middle")
    nbf.remove_cell("new-middle", ref.rev)
    assert "new-middle" not in nbf.names()


def test_duplicate_and_invalid_names(make_notebook):
    nbf = load(make_notebook([("code", "a = 1")]))
    existing = nbf.names()[0]
    with pytest.raises(DuplicateCellName):
        nbf.add_cell(existing, "x = 1")
    with pytest.raises(JupyterMcpError):
        nbf.add_cell("Bad Name!", "x = 1")


def test_update_clears_outputs(make_notebook):
    path = make_notebook([("code", "print('hi')")])
    nbf = load(path)
    name = nbf.names()[0]
    nbf.get(name).cell.outputs = [
        {"output_type": "stream", "name": "stdout", "text": "hi\n"}
    ]
    nbf.update_cell(name, nbf.get(name).rev, source="print('bye')")
    assert nbf.get(name).cell.outputs == []


def test_snapshot_undo_roundtrip(make_notebook):
    path = make_notebook([("code", "x = 1")])
    nbf = load(path)
    name = nbf.names()[0]
    nbf.snapshot("update-x")
    nbf.update_cell(name, nbf.get(name).rev, source="x = 999")
    nbf.save()
    assert "999" in path.read_text()
    undone = nbf.undo_last()
    assert "update-x" in undone
    assert "999" not in path.read_text()
    with pytest.raises(NothingToUndo):
        nbf.undo_last()


def test_external_modification_guard(make_notebook):
    path = make_notebook([("code", "x = 1")])
    session = NotebookSession(path, Summarizer())
    # simulate the user's editor touching the file
    text = path.read_text().replace("x = 1", "x = 42")
    path.write_text(text)
    with pytest.raises(ExternalModification):
        session.guard_mutation()
    # after the guard fired, the session has reloaded the new content
    assert "x = 42" in session.nbfile.cells[0].source


def test_create_notebook(tmp_path):
    nbf = NotebookFile.create(tmp_path / "new.ipynb")
    assert nbf.path.exists()
    assert nbf.nb.metadata["kernelspec"]["name"] == "python3"
    with pytest.raises(JupyterMcpError):
        NotebookFile.create(tmp_path / "new.ipynb")


def test_names_persist_after_save(make_notebook):
    path = make_notebook([("code", "# load\nx = 1")])
    nbf = load(path)
    nbf.snapshot("noop")
    nbf.save()
    reloaded = load(path)
    assert cell_name(reloaded.cells[0]) == "load"

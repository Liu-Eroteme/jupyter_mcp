"""OKF knowledge-bundle module: parsing, search, upsert, crossref."""

import pytest

from jupyter_mcp.errors import JupyterMcpError
from jupyter_mcp.kb import (
    KbEntry,
    crossref,
    get_entry,
    load_bundle,
    parse_entry_text,
    resolve_bundle,
    search,
    upsert,
)


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "kb"
    root.mkdir()
    (root / "data-sources").mkdir()
    (root / "data-sources" / "omniplus.md").write_text(
        "---\n"
        "type: Data Source\n"
        "title: Omniplus vehicle telemetry\n"
        "description: speed is km/h; chunks overlap\n"
        "resource: omniplus_*.parquet\n"
        "tags: [telemetry, parquet]\n"
        "custom_field: keep-me\n"
        "---\n\n"
        "- speed is km/h — API docs are outdated\n"
        "- dedupe on signal_timestamp\n"
    )
    (root / "recipes").mkdir()
    (root / "recipes" / "h3-joins.md").write_text(
        "---\ntype: Recipe\ntitle: H3 spatial joins\ntags: [h3, geo]\n---\n\nUse h3.latlng_to_cell.\n"
    )
    (root / "index.md").write_text("# Index\n")  # reserved, not an entry
    (root / "log.md").write_text("# Change log\n")
    return root


def test_parse_tolerates_missing_and_broken_frontmatter():
    assert parse_entry_text("no frontmatter") == ({}, "no frontmatter")
    meta, body = parse_entry_text("---\n[:broken yaml\n---\nbody")
    assert meta == {} and body == "body"
    meta, body = parse_entry_text("---\ntype: Tool\n---\nbody")
    assert meta["type"] == "Tool" and body == "body"


def test_load_bundle_skips_reserved(bundle):
    entries = load_bundle(bundle)
    paths = {e.rel_path for e in entries}
    assert paths == {"data-sources/omniplus.md", "recipes/h3-joins.md"}


def test_search_ranks_frontmatter_above_body(bundle):
    entries = load_bundle(bundle)
    hits = search(entries, "telemetry")
    assert hits and hits[0].entry.rel_path == "data-sources/omniplus.md"
    # body-line context comes back for orientation
    hits = search(entries, "km/h")
    assert hits and any("km/h" in line for line in hits[0].matching_lines)


def test_search_filters(bundle):
    entries = load_bundle(bundle)
    assert len(search(entries, "", type="Recipe")) == 1
    assert len(search(entries, "", tag="parquet")) == 1
    assert search(entries, "", type="Data Source", tag="h3") == []
    # every query word must appear somewhere
    assert search(entries, "telemetry nonexistent-word") == []
    # bare search without any intent returns nothing (no bundle dumps)
    assert search(entries, "") == []


def test_upsert_creates_updates_and_logs(bundle):
    entry, created = upsert(
        bundle,
        "tools/internal-lib",
        type="Tool",
        title="internal-lib",
        description="install from artifactory",
        body="pip index-url ...",
    )
    assert created and entry.rel_path == "tools/internal-lib.md"
    assert (bundle / "tools" / "internal-lib.md").exists()

    # update: unknown fields preserved, passed fields overwrite, key stays
    entry, created = upsert(
        bundle,
        "data-sources/omniplus.md",
        type="Data Source",
        description="UPDATED",
    )
    assert not created
    assert entry.meta["custom_field"] == "keep-me"
    assert entry.description == "UPDATED"
    assert entry.title == "Omniplus vehicle telemetry"  # omitted → unchanged
    assert "dedupe on signal_timestamp" in entry.body  # body omitted → kept

    log = (bundle / "log.md").read_text()
    assert "created tools/internal-lib.md (agent:claude)" in log
    assert "updated data-sources/omniplus.md (agent:claude)" in log


def test_upsert_rejects_escapes_and_reserved(bundle):
    with pytest.raises(JupyterMcpError):
        upsert(bundle, "../evil", type="Tool")
    with pytest.raises(JupyterMcpError):
        upsert(bundle, "log.md", type="Tool")


def test_get_entry_lists_available_on_miss(bundle):
    with pytest.raises(JupyterMcpError) as exc:
        get_entry(bundle, "nope.md")
    assert "data-sources/omniplus.md" in str(exc.value)


def test_resolve_bundle(tmp_path, monkeypatch):
    monkeypatch.delenv("JUPYTER_MCP_KB_PATH", raising=False)
    with pytest.raises(JupyterMcpError):
        resolve_bundle(None)
    with pytest.raises(JupyterMcpError):
        resolve_bundle(str(tmp_path / "missing"))
    made = resolve_bundle(str(tmp_path / "missing"), create=True)
    assert made.is_dir()
    monkeypatch.setenv("JUPYTER_MCP_KB_PATH", str(made))
    assert resolve_bundle(None) == made


def test_crossref_matches_resources_and_tags(bundle):
    entries = load_bundle(bundle)
    cells = [
        ("load-edges", "df = pl.read_parquet('data/omniplus_2024.parquet')"),
        ("spatial", "import h3\ncells = h3.latlng_to_cell(lat, lng, 9)"),
        ("unrelated", "x = 1 + 1"),
    ]
    lines = crossref(entries, cells)
    assert any("load-edges" in ln and "omniplus.md" in ln for ln in lines)
    assert any("spatial" in ln and "h3-joins.md" in ln for ln in lines)
    assert not any("unrelated" in ln for ln in lines)


def test_crossref_conservative_on_short_literals(bundle):
    # a resource must not match every 3-char string in sight
    entries = [KbEntry("x.md", {"type": "Data Source", "resource": "abc"}, "")]
    assert crossref(entries, [("c", "print('abcdef is not a match target')")]) == []


# --------------------------------------------------------- MCP tool surface


def test_kb_tools_end_to_end(bundle, monkeypatch):
    from jupyter_mcp import server

    monkeypatch.setenv("JUPYTER_MCP_KB_PATH", str(bundle))

    out = server.kb_upsert(
        "data-sources/gtfs",
        type="Data Source",
        title="GTFS static feed",
        resource="gtfs/*.txt",
        tags=["transit"],
        body="- stop_times.txt is huge; stream it\n",
    )
    assert "Created data-sources/gtfs.md" in out

    out = server.kb_search("stop_times")
    assert "data-sources/gtfs.md" in out and "GTFS static feed" in out

    out = server.kb_get("data-sources/gtfs")
    assert "stream it" in out and "type: Data Source" in out

    # explicit bundle param beats env
    out = server.kb_search("stop_times", bundle=str(bundle))
    assert "gtfs" in out

    # errors surface as tool-level ERROR strings, not tracebacks
    monkeypatch.delenv("JUPYTER_MCP_KB_PATH")
    out = server.kb_search("anything")
    assert out.startswith("ERROR") and "JUPYTER_MCP_KB_PATH" in out


def test_overview_crossrefs_kb(bundle, tmp_path, monkeypatch):
    from jupyter_mcp import server

    nb_path = tmp_path / "xref.ipynb"
    server.create_notebook(str(nb_path))
    server.add_cell(str(nb_path), "load", "df = pl.read_parquet('omniplus_07.parquet')")

    # without a configured bundle: no kb lines, no error
    monkeypatch.delenv("JUPYTER_MCP_KB_PATH", raising=False)
    assert "kb:" not in server.notebook_overview(str(nb_path), refresh_summaries=False)

    monkeypatch.setenv("JUPYTER_MCP_KB_PATH", str(bundle))
    overview = server.notebook_overview(str(nb_path), refresh_summaries=False)
    assert "kb: load matches data-sources/omniplus.md" in overview

    # a broken bundle path must never break the overview
    monkeypatch.setenv("JUPYTER_MCP_KB_PATH", str(tmp_path / "gone"))
    overview = server.notebook_overview(str(nb_path), refresh_summaries=False)
    assert overview.startswith("#") and "kb:" not in overview

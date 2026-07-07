"""Fallback-summary retry semantics (no real API calls)."""

from jupyter_mcp.dag import build_graph
from jupyter_mcp.model import NotebookFile
from jupyter_mcp.summaries import BatchSummaries, CellSummary, Summarizer, get_summary


def _notebook(tmp_path):
    nbf = NotebookFile.create(tmp_path / "t.ipynb")
    nbf.add_cell("calc", "x = 1")
    nbf.save()
    graph = build_graph([("calc", "code", "x = 1")])
    return nbf, graph


def test_fallback_summaries_are_retried_once_llm_returns(tmp_path, monkeypatch):
    """Regression: fallback summaries stamped code_rev, so after a credit
    outage they were cached forever — never upgraded to LLM summaries until
    the cell's source happened to change."""
    nbf, graph = _notebook(tmp_path)

    outage = Summarizer()
    outage._disabled_reason = "simulated outage"
    outage.refresh(nbf, graph)
    summ = get_summary(nbf.get("calc").cell)
    assert summ is not None and summ["source"] == "fallback"

    # while disabled, the fallback is NOT dirty (no rewrite loop)
    assert outage.dirty_cells(nbf) == []

    # a working summarizer must retry it and upgrade to an LLM summary
    monkeypatch.delenv("JUPYTER_MCP_DISABLE_SUMMARIES", raising=False)
    working = Summarizer()
    assert working._disabled_reason is None
    monkeypatch.setattr(
        Summarizer,
        "_parse",
        lambda self, prompt, fmt: BatchSummaries(
            cells=[CellSummary(name="calc", tldr="sets x to 1", description="d")]
        ),
    )
    assert working.dirty_cells(nbf) == ["calc"]
    working.refresh(nbf, graph)
    summ = get_summary(nbf.get("calc").cell)
    assert summ is not None and summ["source"] == "llm"
    assert summ["tldr"] == "sets x to 1"

    # up-to-date LLM summaries are never re-requested
    assert working.dirty_cells(nbf) == []

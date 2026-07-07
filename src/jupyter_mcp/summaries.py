"""Lazy, batched LLM cell summaries (claude-haiku-4-5) with hash-keyed caching.

- Summaries live in cell metadata keyed by the cell's content revision, so an
  unchanged cell is never re-summarized (and produces no file diff noise).
- Refresh is *lazy*: edits only invalidate; the next overview/summarize call
  batches all dirty cells into ONE structured-output request.
- Degrades gracefully: with no API credentials (or the env kill-switch set),
  deterministic fallback summaries are used and the notice explains why.

NOTE: uses the plain `anthropic` SDK — credentials resolve from
ANTHROPIC_API_KEY or an `ant auth login` profile automatically.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from .dag import NotebookGraph
from .model import META_NS, NotebookFile, cell_meta, source_rev

SUMMARY_MODEL = "claude-haiku-4-5"
DISABLE_ENV = "JUPYTER_MCP_DISABLE_SUMMARIES"
MAX_CODE_CHARS_PER_CELL = 3000
MAX_OUTPUT_CHARS_PER_CELL = 2000

_TModel = TypeVar("_TModel", bound=BaseModel)


def output_hash(condensed_text: str) -> str:
    """Cache key tying an output_summary to the condensed output it describes."""
    return hashlib.sha256(condensed_text.encode()).hexdigest()[:10]


class CellSummary(BaseModel):
    name: str
    tldr: str
    description: str


class BatchSummaries(BaseModel):
    cells: list[CellSummary]


class OutputSummary(BaseModel):
    name: str
    summary: str


class BatchOutputSummaries(BaseModel):
    cells: list[OutputSummary]


@dataclass
class RefreshResult:
    refreshed: list[str]
    notice: str = ""


def fallback_tldr(cell_type: str, source: str) -> str:
    for line in source.splitlines():
        line = line.strip()
        if cell_type == "markdown" and line.startswith("#"):
            return line.lstrip("# ").strip()[:100]
        if cell_type == "code" and line.startswith("#"):
            return line.lstrip("# ").strip()[:100]
    for line in source.splitlines():
        if line.strip():
            return line.strip()[:100]
    return "(empty cell)"


def get_summary(cell) -> dict | None:
    """Current summary if it matches the cell's revision, else None."""
    meta = cell.metadata.get(META_NS, {})
    summ = meta.get("summary")
    if summ and summ.get("code_rev") == source_rev(cell.source):
        return summ
    return None


def get_tldr(cell) -> str:
    summ = get_summary(cell)
    if summ:
        return summ["tldr"]
    return fallback_tldr(cell.cell_type, cell.source) + " *"


class Summarizer:
    """Shared across notebook sessions; holds one lazy anthropic client."""

    def __init__(self) -> None:
        self._client = None
        self._disabled_reason: str | None = None
        if os.environ.get(DISABLE_ENV):
            self._disabled_reason = f"summaries disabled via {DISABLE_ENV}"

    # -- plumbing ------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def _parse(self, prompt: str, output_format: type[_TModel]) -> _TModel:
        client = self._get_client()
        response = client.messages.parse(
            model=SUMMARY_MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
            output_format=output_format,
        )
        if response.parsed_output is None:
            raise ValueError("model returned no parsed output")
        return response.parsed_output

    # -- code summaries --------------------------------------------------------

    def dirty_cells(self, nbfile: NotebookFile, names: list[str] | None = None) -> list[str]:
        out = []
        for ref in nbfile.refs():
            if names is not None and ref.name not in names:
                continue
            if not ref.cell.source.strip():
                continue
            summ = get_summary(ref.cell)
            if summ is None:
                out.append(ref.name)
            elif summ.get("source") == "fallback" and self._disabled_reason is None:
                # a degraded summary (credit outage, network blip) is retried
                # once the LLM is available again — never permanent
                out.append(ref.name)
        return out

    def refresh(
        self,
        nbfile: NotebookFile,
        graph: NotebookGraph,
        names: list[str] | None = None,
    ) -> RefreshResult:
        """Ensure summaries exist for dirty cells (one batched LLM call)."""
        dirty = self.dirty_cells(nbfile, names)
        if not dirty:
            return RefreshResult(refreshed=[])
        if self._disabled_reason:
            self._write_fallbacks(nbfile, dirty)
            return RefreshResult(refreshed=dirty, notice=self._disabled_reason)

        prompt = self._build_code_prompt(nbfile, graph, dirty)
        try:
            parsed: BatchSummaries = self._parse(prompt, BatchSummaries)
        except Exception as e:  # auth/network/parse — degrade, don't fail the tool
            self._disabled_reason = f"LLM summaries unavailable ({type(e).__name__}: {e}); using fallbacks"
            self._write_fallbacks(nbfile, dirty)
            return RefreshResult(refreshed=dirty, notice=self._disabled_reason)

        by_name = {c.name: c for c in parsed.cells}
        for name in dirty:
            ref = nbfile.get(name)
            got = by_name.get(name)
            if got:
                cell_meta(ref.cell)["summary"] = {
                    "code_rev": ref.rev,
                    "tldr": got.tldr.strip()[:160],
                    "description": got.description.strip(),
                    "source": "llm",
                }
            else:
                self._write_fallbacks(nbfile, [name])
        return RefreshResult(refreshed=dirty)

    def _write_fallbacks(self, nbfile: NotebookFile, names: list[str]) -> None:
        for name in names:
            ref = nbfile.get(name)
            cell_meta(ref.cell)["summary"] = {
                "code_rev": ref.rev,
                "tldr": fallback_tldr(ref.cell.cell_type, ref.cell.source),
                "description": "",
                "source": "fallback",
            }

    def _build_code_prompt(
        self, nbfile: NotebookFile, graph: NotebookGraph, dirty: list[str]
    ) -> str:
        blocks = []
        for name in dirty:
            ref = nbfile.get(name)
            parents = graph.parents.get(name, {})
            ctx_lines = []
            for parent, variables in sorted(parents.items()):
                try:
                    ptldr = get_tldr(nbfile.get(parent).cell)
                except Exception:
                    ptldr = ""
                ctx_lines.append(f"  - uses {sorted(variables)} from cell '{parent}' ({ptldr})")
            ctx = "\n".join(ctx_lines) or "  (no upstream dependencies)"
            code = ref.cell.source[:MAX_CODE_CHARS_PER_CELL]
            blocks.append(
                f"### cell name: {name} ({ref.cell.cell_type})\ncontext:\n{ctx}\ncode:\n```\n{code}\n```"
            )
        cells_text = "\n\n".join(blocks)
        return (
            f"You are summarizing cells of the Jupyter notebook '{nbfile.path.name}' "
            "for a navigation index used by data scientists and coding agents.\n"
            "For EACH cell below, produce:\n"
            "- tldr: one clause, <= 12 words, concrete (mention key variables/operations)\n"
            "- description: 1-3 sentences covering what it computes, inputs it depends on, "
            "and what downstream cells get from it\n"
            "Return exactly one entry per cell, using the exact cell names given.\n\n"
            f"{cells_text}"
        )

    # -- output summaries -------------------------------------------------------

    def summarize_outputs(
        self, nbfile: NotebookFile, items: list[tuple[str, str]]
    ) -> RefreshResult:
        """items: (cell_name, condensed_output_text). Skips up-to-date entries."""
        todo: list[tuple[str, str, str]] = []
        for name, text in items:
            ref = nbfile.get(name)
            ohash = output_hash(text)
            existing = cell_meta(ref.cell).get("output_summary")
            if existing and existing.get("output_hash") == ohash:
                continue
            todo.append((name, text, ohash))
        if not todo:
            return RefreshResult(refreshed=[])
        if self._disabled_reason:
            return RefreshResult(refreshed=[], notice=self._disabled_reason)

        blocks = [
            f"### cell name: {name}\noutput:\n```\n{text[:MAX_OUTPUT_CHARS_PER_CELL]}\n```"
            for name, text, _ in todo
        ]
        prompt = (
            f"Below are condensed outputs of cells from the notebook '{nbfile.path.name}'. "
            "For EACH cell, summarize the OUTPUT in 1-2 sentences: key numbers, shapes, "
            "findings, or errors. Use the exact cell names given.\n\n" + "\n\n".join(blocks)
        )
        try:
            parsed: BatchOutputSummaries = self._parse(prompt, BatchOutputSummaries)
        except Exception as e:
            self._disabled_reason = f"LLM summaries unavailable ({type(e).__name__}: {e}); using fallbacks"
            return RefreshResult(refreshed=[], notice=self._disabled_reason)
        by_name = {c.name: c for c in parsed.cells}
        done = []
        for name, _text, ohash in todo:
            got = by_name.get(name)
            if got:
                ref = nbfile.get(name)
                cell_meta(ref.cell)["output_summary"] = {
                    "output_hash": ohash,
                    "text": got.summary.strip(),
                    "source": "llm",
                }
                done.append(name)
        return RefreshResult(refreshed=done)

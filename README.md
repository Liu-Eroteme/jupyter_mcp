# jupyter-mcp

Kernel-attached MCP server for Jupyter EDA workflows. Built for coding
agents that iterate on notebooks: named cells, a dependency DAG, minimal
re-execution, condensed outputs, and cheap LLM summaries for navigation.

## Why

Editing notebooks through generic file tools is painful for agents:

- cells have no stable, human-meaningful addresses
- every change means re-executing the whole notebook from scratch
- raw outputs (box-drawn tables, base64 charts, ANSI noise) flood context
- nothing tells you which cells a change invalidates

This server fixes each of those with an opinionated data model.

## Concepts

| Concept | What it means |
|---|---|
| **Cell names** | Every cell has a unique kebab-case name stored in `cell.metadata.jupyter_mcp.name`. Unnamed cells get auto-names from their first comment/heading. |
| **Revisions** | Each cell has a short content hash (`rev`). Every mutation requires the `expected_rev` from your latest read — optimistic locking that makes wrong-target and stale edits structurally impossible. No confirmation round-trips. |
| **Dependency DAG** | A static AST pass (not runtime tracing) extracts per-cell defines/uses/mutations and builds last-writer-wins edges. Works on unexecuted cells, which is the whole point: edit first, then `run_stale`. |
| **Staleness** | A cell is stale when its source changed since its last successful execution — or when any upstream cell did. `run_stale` executes exactly that set, in document order, on a persistent kernel. |
| **Condensed outputs** | Streams merged, ANSI stripped, long text truncated head+tail with explicit markers. Duplicate table reprs collapse to one: uniform → CSV, ragged → JSON. Charts return as real MCP images (downscaled), so the agent *sees* them. |
| **Summaries** | Lazy, batched `claude-haiku-4-5` summaries (tldr + description + output summary) cached in cell metadata keyed by content hash. Unchanged cells are never re-summarized. Degrades to deterministic fallbacks (marked `*`) without API credentials. |
| **Snapshots / undo** | Every mutation snapshots the file first (under `~/.cache/jupyter_mcp/`); `undo_last` restores it. |

## Tools

| Tool | Purpose |
|---|---|
| `create_notebook` | New empty notebook |
| `notebook_overview` | Index: names, revs, staleness, tldrs, edges, lint |
| `read_cells` | Full cells (code + condensed outputs + images), by names/slice |
| `add_cell` / `update_cell` / `remove_cell` / `move_cell` | Mutations; all take `expected_rev` |
| `execute_cells` | Run named cells on the persistent kernel |
| `run_stale` | Minimal re-execution after edits |
| `restart_kernel` | Fresh kernel; marks everything stale |
| `inspect_variable` | Type/shape/schema/head of a live variable, no cell needed |
| `undo_last` | Restore pre-mutation snapshot |
| `summarize_cells` | Detailed LLM summaries incl. outputs |
| `search_cells` | Search source + names + summaries |

## Setup

```sh
uv sync
```

Register with Claude Code (`.mcp.json` in any project, or globally):

```json
{
  "mcpServers": {
    "jupyter": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/jupyter_mcp", "jupyter-mcp"]
    }
  }
}
```

The server is multi-notebook: every tool takes a notebook `path`, one kernel
per notebook, started lazily in the notebook's directory (so relative data
paths behave like in your editor). Kernelspec comes from the notebook's
metadata, falling back to `python3`.

### Summaries & credentials

Summaries use the plain `anthropic` SDK: credentials resolve from
`ANTHROPIC_API_KEY` or an `ant auth login` profile automatically. Cost is
negligible (Haiku, batched, hash-cached). To disable entirely set
`JUPYTER_MCP_DISABLE_SUMMARIES=1` — everything else works unchanged.

## Development

```sh
uv run pytest              # full suite
uv run pytest -m "not kernel"   # skip real-kernel integration tests
```

Layout: `src/jupyter_mcp/` is a plain library (model, dag, condense, kernel,
summaries, session) with the MCP surface isolated in `server.py`; everything
below the server is unit-testable without MCP.

## Known limitations (v1)

- The DAG is static: dynamic patterns (`globals()[name] = ...`, `exec`,
  attribute mutation through aliases) are invisible. Method calls only count
  as mutations for a known allowlist (`append`, `fit`, ...) — pure-functional
  chains (polars) intentionally create no false forward edges.
- The undo stack is per-server-process (snapshots persist on disk, but a
  restarted server won't offer them for undo).
- `%%bash` / `%%sql` style cells are treated as opaque (no dependencies).
- Concurrent edits from a live Jupyter editor are detected (the server
  reloads and rejects the mutation) but not merged.

See [docs/ROADMAP.md](docs/ROADMAP.md) for what's deliberately deferred —
including the phase-2 OKF knowledge base.

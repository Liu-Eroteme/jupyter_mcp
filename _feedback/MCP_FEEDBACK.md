# Jupyter MCP — usage feedback

Gitignored. Notes from using the prototype `jupyter` MCP server during
this project, collected as they come up.

## Session 2026-08-06 (building `01_road_graph.ipynb`, 13 cells)

### Worked well

- **Kernel/venv resolution**: the kernel bound to _this_ project's
  `.venv/bin/python` even though the server itself runs via
  `uv run --project ~/projects/jupyter_mcp`. Exactly right, zero setup.
  The overview's "using project venv interpreter" line confirms it.
- **`run="stale"` on `add_cell`/`update_cell`** is a great loop — one
  call per edit-and-run iteration. After fixing `load-edges` (the
  GIP_OBJECTID issue), its dependents (`parse-geometry`, `adjacency`)
  reran automatically, in order, unprompted. This saved real round trips.
- **Dependency tracking is accurate**, including non-obvious edges
  (it caught `arc-length-api` depending on `rng` defined in
  `spatial-index`). The `← inputs | → dependents` lines in
  `notebook_overview` orient quickly.
- **Condensed outputs**: DataFrames as CSV blocks, matplotlib figures
  returned inline as images — could verify plots at a glance.
- `expected_rev` optimistic locking is a clear, honest contract.
- Unique kebab-case cell names beat indices for addressing; renames via
  `new_name` keep that stable.

### Issues / suggestions

- **`[table as CSV, N rows]` miscounts**: N includes the header and
  dtype lines. A 10-data-row Polars table announces "12 rows". Should
  count data rows only.
- **Truncation is implicit**: a 29,658-row DataFrame renders 10 rows
  with no "(showing 10 of 29,658)" marker. Easy to misread a preview as
  the whole table when the shape line is out of scroll-back.
- **LLM cell summaries can be subtly wrong**: my `findings` markdown
  cell got summarized as "3.2 km network" (actual: 3,237 km) and "~5%
  corrupted" (the cell argues the nuance is the opposite — 0.02%
  absurd). Fine for orientation; risky if an agent trusts them as data.
  Deterministic fallbacks are marked with `*` — consider marking LLM
  summaries as approximate instead/too.
- **No per-cell timing** in run results. When `run="stale"` executes a
  chain, knowing which cell ate the wall clock would guide optimization.
- **Kernel binding invisible at `create_notebook` time** — it reports
  `kernel: python3` but which interpreter that resolves to only becomes
  visible on first run / overview. Surfacing the resolved path at
  creation would be reassuring.
- Nice-to-have: notebook-level default for `timeout_seconds` /
  `wait_seconds` instead of per-call only.

Overall: pleasant to drive an entire real session with (13 cells,
several fix-rerun cycles, two figures); the stale-propagation model is
the standout feature.

## Session 2026-08-06, part 2 (user-edited notebooks, notebook 00)

### Worked well

- **External-edit detection**: the user sed-renamed 8080→8070 across the
  repo outside the MCP. On next contact every touched cell had a new rev
  and was marked STALE, and the kernel was reset. Correct, safe, and it
  made "just rerun stale" the obvious recovery.
- **Targeted run with ancestor closure**: `run(cells=["export-route-data"])`
  on the user's notebook 00 executed exactly the stale ancestor chain
  (DB query → GTFS resolve → materialize) and skipped the display-only
  cell and the folium map cell. Precisely right, no wasted work.
- The `add_cell` error for an unknown `after=` target lists all available
  cell names — made recovery a one-step fix.

### Issues / suggestions

- **Cell-name registry ignores existing notebook cell ids.** Notebook 00
  (authored outside the MCP) has meaningful ids like `fetch-gtfs`,
  `materialize-route-geometries`, `map-route-geometries` — the MCP
  instead synthesized names from cell content
  (`from-pathlib-import-path`, `shape-points`, `import-folium`), so
  addressing by the ids visible in the file fails. Suggest: adopt
  existing ipynb cell ids as names when they are valid kebab-case and
  unique; synthesize only otherwise.
- **Cached summaries/descriptions survive external edits**: after the
  sed, a stored cell description still said "bus 8080" until summaries
  refreshed. If revs are bumped on external edit, the stale summary
  should be flagged (or dropped) too.
- **`quiet=true` swallows the target cell's prints**: for
  `run(cells=[...], quiet=true)` the explicitly-requested cell's stdout
  (a one-line "wrote …" confirmation) was collapsed along with the
  ancestors'. Keeping the requested cells' output while quieting
  ancestors would match intent better. (Figures do come through under
  quiet — good.)

## Session 2026-08-06, part 3 (nightline rework)

### Worked well

- **Edit-time lint caught a cross-cell breakage before anything ran**:
  after rewriting `route-context` (dropping `d_route`/`shape_lines`),
  the update response flagged `viz-route-context` as using names "never
  defined in this notebook". Saved a guaranteed NameError and a rerun
  cycle — probably the single most valuable assist so far.
- Stale propagation survived a session/kernel restart cleanly: one
  `run="stale"` re-ran the full 15-cell chain in correct order, mixing
  unchanged and freshly-edited cells.
- Working _inside the user's own notebook_ (adding `export-route-data`,
  `cap-at-successor-start` between their cells) felt safe throughout —
  dependency edges updated correctly around foreign code.

### Issues / suggestions

- Table row-count label still counts header + dtype lines ("11 rows"
  for a 9-row frame) — same as part 1.

# Roadmap

Design decisions already made (with rationale), deliberately deferred work,
and the phase-2 spec for the OKF knowledge base.

## Settled design decisions (v1)

| Decision | Choice | Why |
|---|---|---|
| Execution model | File-backed model + persistent kernel per notebook | Read/edit works without a kernel; execution is incremental (`run_stale`), not whole-notebook re-runs |
| Edit safety | Per-cell revision hashes as mutation preconditions + snapshot/undo | Same protection as two-phase confirmation with zero happy-path round-trips |
| Dependency DAG | Static AST def/use analysis behind a small interface | Must work on *unexecuted* cells; ipyflow's runtime-traced DAG cannot (and can't be extracted from its kernel) |
| Summaries | Lazy + batched, `claude-haiku-4-5`, structured outputs, hash-keyed in cell metadata | Edits stay instant; unchanged cells never re-summarized; no git noise |
| Scope | Multi-notebook, path param per tool | One server registration serves every project |
| Table condensing | HTML table parse → uniform: CSV, ragged: JSON | Best tokens-per-fact; duplicate MIME reprs collapse to one |
| Charts | MCP image content, downscaled to ≤1200px | The agent should *see* plots, not parse base64 |

## Phase 2 — OKF knowledge base

Goal: agents (and data scientists) accrete durable notes about **data
sources**, **internal tooling**, and **recipes** across notebooks, so the
next session doesn't rediscover what the last one learned (units, caveats,
join keys, where to install internal libs from, known-good snippets).

Format: [OKF v0.1](https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf)
— a directory ("bundle") of markdown files with YAML frontmatter. Only
`type` is required; consumers must tolerate unknown fields, partial bundles,
and broken links. Reserved files: `index.md` (progressive-disclosure
listing), `log.md` (dated change history). Shareable as an ordinary git
repo; reviewable via PRs; greppable with no tooling.

Sketch:

- **Bundle location**: configurable (env or per-tool param); may be a shared
  org repo. Entry types: `Data Source`, `Tool`, `Recipe`.
- **Frontmatter**: `type` (required), `title`, `description`, `resource`
  (canonical URI: table name, file glob, package URL — load-bearing for
  matching), `tags`, `timestamp`, `author` (`agent:claude` / `human:<name>`
  for provenance).
- **Tools**: `kb_search` (frontmatter filter + full text — plain matching,
  no embeddings in v1), `kb_get`, `kb_upsert` (update-don't-duplicate;
  maintain `timestamp`; append to `log.md`). Optionally expose entries as
  MCP resources (the mapping is 1:1).
- **The integration that makes it pay off**: on `notebook_overview` /
  session open, match identifiers in cell code (file globs, table names,
  imports) against KB `resource` URIs and tags, and surface hits inline:
  *"cells 2–3 read `omniplus_*.parquet` — known source, see
  `/data-sources/omniplus.md` (3 caveats)"*. Auto-crossref is why the KB
  lives in this server rather than a standalone one.
- **Keep decoupled**: `kb/` as its own module; only the crossref hook
  touches notebook code.

Motivating example (from the session that spawned this project): an entry
for a vehicle-telemetry parquet source would have recorded *"speed is km/h —
API docs are outdated"*, *"position timestamps are second-precision"*,
*"accuracy_factor behaves like HDOP"*, *"chunks overlap — dedupe on
signal_timestamp"*. Each was rediscovered from scratch during EDA.

## Deferred / ideas

- **Retrieval & reranking** over the KB driven by notebook session context
  (only if plain matching proves insufficient).
- **Runtime-verified DAG edges** as a refinement layer on top of the static
  pass (e.g. namespace diffing around execution — cheap, no ipyflow).
- ~~**Cross-process undo**~~ — done: the undo stack is seeded from on-disk
  snapshots.
- ~~**Cell-level diff tool**~~ — rejected: revision locking already prevents
  blind overwrites, and the agent holds both source versions anyway.
- **Notebook lint tool** as a first-class surface (currently footer notes):
  use-before-def, unused definitions, opaque cells, missing names.
- **Export tools**: notebook → script / HTML report.
- **Watch mode**: notify-on-external-change instead of reject-on-write.

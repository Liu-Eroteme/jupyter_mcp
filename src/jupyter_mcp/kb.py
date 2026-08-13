"""OKF v0.1 knowledge bundle: durable notes about data sources, tools, recipes.

A bundle is a directory of markdown files with YAML frontmatter, shareable as
an ordinary git repo (see docs/ROADMAP.md for the spec link and rationale).
Agents and data scientists accrete notes here — units, caveats, join keys,
known-good snippets — so the next session doesn't rediscover what the last
one learned.

Format contract (OKF v0.1): only ``type`` is required in frontmatter;
consumers tolerate unknown fields, partial bundles, and broken links.
Reserved files: ``index.md`` (progressive-disclosure listing), ``log.md``
(dated change history). This module treats both as non-entries.

Kept deliberately decoupled from the notebook model: only the crossref hook
in the server touches notebook code.
"""

from __future__ import annotations

import fnmatch
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .errors import JupyterMcpError

KB_ENV = "JUPYTER_MCP_KB_PATH"
#: conventional entry types (not enforced — unknown types are tolerated)
ENTRY_TYPES = ("Data Source", "Tool", "Recipe")
RESERVED = {"index.md", "log.md", "README.md"}
MAX_ENTRY_BYTES = 512_000  # skip pathological files, keep scans cheap

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass
class KbEntry:
    rel_path: str  # bundle-relative posix path — the entry's stable key
    meta: dict
    body: str

    @property
    def title(self) -> str:
        return str(self.meta.get("title") or Path(self.rel_path).stem)

    @property
    def type(self) -> str:
        return str(self.meta.get("type") or "untyped")

    @property
    def description(self) -> str:
        return str(self.meta.get("description") or "")

    @property
    def resource(self) -> str:
        return str(self.meta.get("resource") or "")

    @property
    def tags(self) -> list[str]:
        tags = self.meta.get("tags")
        if isinstance(tags, str):
            return [t.strip() for t in tags.split(",") if t.strip()]
        if isinstance(tags, list):
            return [str(t) for t in tags]
        return []


def resolve_bundle(bundle: str | None, *, create: bool = False) -> Path:
    """Bundle root: explicit param > KB_ENV. Errors carry setup guidance."""
    raw = bundle or os.environ.get(KB_ENV)
    if not raw:
        raise JupyterMcpError(
            "No knowledge bundle configured. Pass `bundle` (a directory path) or "
            f"set {KB_ENV}. A bundle is just a directory of markdown files; "
            "kb_upsert will create it."
        )
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        if not create:
            raise JupyterMcpError(
                f"Knowledge bundle {root} does not exist. kb_upsert creates it on "
                "first write, or check the path."
            )
        root.mkdir(parents=True, exist_ok=True)
    return root


def _parse_raw(text: str) -> tuple[dict | None, str]:
    """(frontmatter, body). Distinguishes the three cases: no frontmatter
    ({}), well-formed ({...}), and a block that exists but is not a YAML
    mapping (None) — writers must not treat the last one as empty."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None, text[m.end() :]
    return (meta if isinstance(meta, dict) else None), text[m.end() :]


def parse_entry_text(text: str) -> tuple[dict, str]:
    """Split frontmatter and body; tolerant of malformed/missing YAML
    (read paths: a broken block degrades to an untyped entry)."""
    meta, body = _parse_raw(text)
    return ({} if meta is None else meta), body


def load_bundle(root: Path) -> list[KbEntry]:
    """All entries in the bundle, tolerating unreadable/oversized files."""
    entries: list[KbEntry] = []
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root).as_posix()
        if path.name in RESERVED or any(part.startswith(".") for part in rel.split("/")):
            continue
        try:
            if path.stat().st_size > MAX_ENTRY_BYTES:
                continue
            meta, body = parse_entry_text(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
        entries.append(KbEntry(rel_path=rel, meta=meta, body=body))
    return entries


# -------------------------------------------------------------------- search


@dataclass
class KbHit:
    entry: KbEntry
    score: float
    matching_lines: list[str] = field(default_factory=list)


def search(
    entries: list[KbEntry],
    query: str,
    type: str | None = None,
    tag: str | None = None,
) -> list[KbHit]:
    """Rank entries by plain word matching (no embeddings, v1 by design).

    Frontmatter fields (title, tags, resource, description) weigh more than
    body text; every query word must appear somewhere for an entry to hit.
    Empty query with a type/tag filter lists that whole slice.
    """
    words = [w for w in re.split(r"\W+", query.lower()) if w]
    hits: list[KbHit] = []
    for entry in entries:
        if type and entry.type.lower() != type.lower():
            continue
        if tag and tag.lower() not in (t.lower() for t in entry.tags):
            continue
        head = " ".join(
            [entry.title, entry.description, entry.resource, " ".join(entry.tags)]
        ).lower()
        body = entry.body.lower()
        score = 0.0
        for w in words:
            in_head, in_body = w in head, w in body
            if not in_head and not in_body:
                score = -1.0
                break
            score += (3.0 if in_head else 0.0) + (1.0 if in_body else 0.0)
        if score < 0:
            continue
        if not words and not (type or tag):
            continue  # a bare kb_search() would dump the bundle — require intent
        lines = []
        if words:
            for line in entry.body.splitlines():
                lowered = line.lower()
                if any(w in lowered for w in words) and line.strip():
                    lines.append(line.strip()[:160])
                if len(lines) >= 3:
                    break
        hits.append(KbHit(entry=entry, score=score, matching_lines=lines))
    hits.sort(key=lambda h: (-h.score, h.entry.rel_path))
    return hits


# -------------------------------------------------------------------- upsert


def _slug_path(rel_path: str) -> str:
    rel = rel_path.strip().strip("/")
    if not rel.endswith(".md"):
        rel += ".md"
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise JupyterMcpError(f"Entry path must be bundle-relative, got {rel_path!r}.")
    if p.name in RESERVED:
        raise JupyterMcpError(f"{p.name} is reserved (bundle index/log), pick another name.")
    return p.as_posix()


def upsert(
    root: Path,
    rel_path: str,
    *,
    type: str,
    title: str | None = None,
    description: str | None = None,
    resource: str | None = None,
    tags: list[str] | None = None,
    body: str | None = None,
    author: str = "agent:claude",
) -> tuple[KbEntry, bool]:
    """Create or update one entry; returns (entry, created).

    Update-don't-duplicate: the bundle-relative path is the key. Unknown
    frontmatter fields on an existing entry are preserved; passed fields
    overwrite; omitted ones stay. Appends a dated line to log.md.
    """
    rel = _slug_path(rel_path)
    target = root / rel
    created = not target.exists()
    meta: dict = {}
    old_body = ""
    if not created:
        parsed, old_body = _parse_raw(target.read_text(encoding="utf-8"))
        if parsed is None:
            # bundles are hand-editable git repos — malformed YAML is the
            # expected case, and rewriting through {} would destroy every
            # existing field. Refuse instead of losing data.
            raise JupyterMcpError(
                f"Entry {rel!r} has frontmatter that does not parse as a YAML "
                "mapping — updating it would lose the existing fields. Fix the "
                "file by hand first (kb_get shows its raw content)."
            )
        meta = parsed
    meta["type"] = type
    for key, value in (
        ("title", title),
        ("description", description),
        ("resource", resource),
        ("tags", tags),
    ):
        if value is not None:
            meta[key] = value
    meta["timestamp"] = time.strftime("%Y-%m-%d")
    meta["author"] = author
    new_body = body if body is not None else old_body
    front = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).rstrip()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"---\n{front}\n---\n\n{new_body.strip()}\n", encoding="utf-8")

    log_line = f"- {meta['timestamp']}: {'created' if created else 'updated'} {rel} ({author})\n"
    log = root / "log.md"
    if not log.exists():
        log.write_text(f"# Change log\n\n{log_line}", encoding="utf-8")
    else:
        with log.open("a", encoding="utf-8") as fh:
            fh.write(log_line)
    entry_meta, entry_body = parse_entry_text(target.read_text(encoding="utf-8"))
    return KbEntry(rel_path=rel, meta=entry_meta, body=entry_body), created


def get_entry(root: Path, rel_path: str) -> KbEntry:
    rel = _slug_path(rel_path)
    target = root / rel
    if not target.exists():
        available = [e.rel_path for e in load_bundle(root)]
        raise JupyterMcpError(
            f"No entry {rel!r} in bundle {root}. Available: "
            f"{', '.join(available) or '(empty bundle)'}"
        )
    meta, body = parse_entry_text(target.read_text(encoding="utf-8"))
    return KbEntry(rel_path=rel, meta=meta, body=body)


# ------------------------------------------------------------------ crossref

_STRING_LITERAL_RE = re.compile(r"['\"]([^'\"\n]{3,200})['\"]")
_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([\w.]+)", re.MULTILINE)


def _cell_identifiers(source: str) -> tuple[set[str], set[str]]:
    """(string literals, imported top-level modules) mentioned by a cell."""
    literals = set(_STRING_LITERAL_RE.findall(source))
    imports = {m.split(".")[0] for m in _IMPORT_RE.findall(source)}
    return literals, imports


def _resource_matches(resource: str, literals: set[str]) -> bool:
    """A KB resource URI matches a cell if some literal fnmatch-es it, or the
    full resource appears inside a literal (table names embedded in SQL,
    globs inside longer paths). Deliberately NOT the reverse — a short
    literal that is merely a substring of the resource ('.txt' in
    'gtfs/*.txt') says nothing about the cell touching that source."""
    pattern = resource.lower()
    for lit in literals:
        lowered = lit.lower()
        if fnmatch.fnmatch(lowered, pattern) or fnmatch.fnmatch(Path(lowered).name, pattern):
            return True
        if len(pattern) >= 5 and pattern in lowered:
            return True
    return False


def crossref(
    entries: list[KbEntry], cells: list[tuple[str, str]], max_lines: int = 4
) -> list[str]:
    """Match notebook cells against KB entries; returns overview footer lines.

    cells: (cell_name, source) for code cells. Matching is deliberately
    conservative: resource URIs against string literals in the code, tags
    against imported module names — orientation hints, not search results.
    """
    per_cell = [(name, *_cell_identifiers(src)) for name, src in cells]
    lines: list[str] = []
    for entry in entries:
        matched: list[str] = []
        for name, literals, imports in per_cell:
            hit = bool(entry.resource) and _resource_matches(entry.resource, literals)
            hit = hit or any(t.lower() in {i.lower() for i in imports} for t in entry.tags)
            if hit:
                matched.append(name)
        if matched:
            shown = ", ".join(matched[:4]) + (", …" if len(matched) > 4 else "")
            desc = f" — {entry.description}" if entry.description else ""
            lines.append(
                f"kb: {shown} matches {entry.rel_path} ({entry.type}"
                f"{': ' + entry.resource if entry.resource else ''}){desc}"
            )
        if len(lines) >= max_lines:
            break
    return lines

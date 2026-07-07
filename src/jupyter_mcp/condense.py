"""Condense raw nbformat outputs into compact text + attachable images.

Rules (see README for rationale):

- Consecutive stream outputs are merged; ANSI escapes stripped.
- Rich MIME bundles pick ONE representation: image > html-table > markdown >
  plain text. Duplicated table reprs (text/plain + text/html) collapse to a
  single reformatted table: uniform & flat -> CSV (best tokens-per-fact),
  ragged/nested -> JSON rows, unparseable -> truncated text/plain.
- Long text is truncated head+tail with an explicit omission marker — never
  silently.
- PNG outputs are returned as image attachments (downscaled if large) so the
  calling agent can *see* charts instead of parsing base64.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

MAX_OUTPUT_CHARS = 4000
MAX_HTML_PARSE_CHARS = 1_000_000
MAX_TRACEBACK_LINES = 40
MAX_TABLE_ROWS = 50
MAX_IMAGE_DIM = 1200


@dataclass
class Condensed:
    text: str
    images: list[bytes] = field(default_factory=list)  # PNG bytes
    has_error: bool = False


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def truncate(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n… [{omitted} chars omitted] …\n{text[-tail:]}"


# ------------------------------------------------------------- html tables


class _TableParser(HTMLParser):
    """Extract the first <table> as rows of cell strings; bail on complexity."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._table_depth = 0
        self.bailed = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if self.bailed:
            return
        if tag == "table":
            self._table_depth += 1
            if self._table_depth > 1:
                self.bailed = True  # nested table
        if self._table_depth != 1:
            return
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            if any(k in ("colspan", "rowspan") for k, _ in attrs):
                self.bailed = True
                return
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if self.bailed:
            return
        if tag == "table":
            self._table_depth -= 1
        if self._table_depth < 1 and tag == "table":
            return
        if tag in ("td", "th") and self._cell is not None:
            if self._row is not None:
                self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def html_table_to_text(html: str) -> str | None:
    """Reformat an HTML table: uniform -> CSV, ragged -> JSON, else None."""
    parser = _TableParser()
    try:
        parser.feed(html)
    except Exception:
        return None
    rows = parser.rows
    if parser.bailed or not rows:
        return None
    shown = rows[:MAX_TABLE_ROWS]
    trailer = f"\n… [{len(rows) - MAX_TABLE_ROWS} more rows omitted] …" if len(rows) > MAX_TABLE_ROWS else ""
    widths = {len(r) for r in shown}
    if len(widths) == 1:
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerows(shown)
        return f"[table as CSV, {len(rows)} rows]\n{buf.getvalue().rstrip()}{trailer}"
    payload = json.dumps(shown, ensure_ascii=False)
    return f"[ragged table as JSON rows, {len(rows)} rows]\n{payload}{trailer}"


# ------------------------------------------------------------------ images


def _prepare_image(b64_png: str) -> bytes | None:
    try:
        raw = base64.b64decode(b64_png)
    except Exception:
        return None
    try:
        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(raw))
        w, h = img.size
        if max(w, h) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)))
            out = io.BytesIO()
            img.save(out, format="PNG")
            return out.getvalue()
        return raw
    except Exception:
        return raw  # pillow unavailable/failed: pass through


# ------------------------------------------------------------------- main


def _mime_bundle_text(data: dict, images: list[bytes]) -> str | None:
    """Pick ONE representation from a rich MIME bundle."""
    if "image/png" in data:
        png = data["image/png"]
        if isinstance(png, list):
            png = "".join(png)
        prepared = _prepare_image(png)
        if prepared is not None:
            images.append(prepared)
            return "[image attached]"
    if "text/html" in data:
        html = data["text/html"]
        if isinstance(html, list):
            html = "".join(html)
        # size guard: a styled mega-frame can emit megabytes of HTML
        if "<table" in html and len(html) <= MAX_HTML_PARSE_CHARS:
            table = html_table_to_text(html)
            if table is not None:
                return table
        # html without a usable table: fall through to md/plain
    if "text/markdown" in data:
        md = data["text/markdown"]
        return md if isinstance(md, str) else "".join(md)
    if "text/plain" in data:
        plain = data["text/plain"]
        return plain if isinstance(plain, str) else "".join(plain)
    if "text/latex" in data:
        latex = data["text/latex"]
        return latex if isinstance(latex, str) else "".join(latex)
    keys = ", ".join(sorted(data))
    return f"[unrendered output: {keys}]" if keys else None


def condense_outputs(outputs: Sequence[dict], max_chars: int = MAX_OUTPUT_CHARS) -> Condensed:
    """Convert a cell's nbformat outputs into one condensed text block + images."""
    parts: list[str] = []
    images: list[bytes] = []
    has_error = False
    stream_buf: list[str] = []
    stream_name = ""

    def flush_stream() -> None:
        nonlocal stream_buf, stream_name
        if stream_buf:
            text = strip_ansi("".join(stream_buf))
            label = "[stderr]\n" if stream_name == "stderr" else ""
            parts.append(label + truncate(text.rstrip("\n"), max_chars))
        stream_buf, stream_name = [], ""

    for out in outputs:
        otype = out.get("output_type")
        if otype == "stream":
            name = out.get("name", "stdout")
            if stream_buf and name != stream_name:
                flush_stream()
            stream_name = name
            text = out.get("text", "")
            stream_buf.append(text if isinstance(text, str) else "".join(text))
        elif otype in ("execute_result", "display_data"):
            flush_stream()
            text = _mime_bundle_text(out.get("data", {}), images)
            if text:
                parts.append(truncate(strip_ansi(text.rstrip("\n")), max_chars))
        elif otype == "error":
            flush_stream()
            has_error = True
            tb_lines = [strip_ansi(line) for line in out.get("traceback", [])]
            tb = "\n".join(tb_lines)
            if len(tb_lines) > MAX_TRACEBACK_LINES:
                keep_head = MAX_TRACEBACK_LINES // 2
                keep_tail = MAX_TRACEBACK_LINES - keep_head
                tb = "\n".join(
                    tb_lines[:keep_head]
                    + [f"… [{len(tb_lines) - MAX_TRACEBACK_LINES} traceback lines omitted] …"]
                    + tb_lines[-keep_tail:]
                )
            ename = out.get("ename", "Error")
            evalue = strip_ansi(out.get("evalue", ""))
            parts.append(f"ERROR {ename}: {evalue}\n{truncate(tb, max_chars)}")
    flush_stream()

    text = "\n\n".join(p for p in parts if p.strip()) or "(no output)"
    return Condensed(text=text, images=images, has_error=has_error)

import base64
import io
import json

from jupyter_mcp.condense import (
    condense_outputs,
    html_table_to_text,
    strip_ansi,
    truncate,
)


def _png_bytes(size=(4, 4)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_stream_merge_and_stderr_label():
    out = condense_outputs(
        [
            {"output_type": "stream", "name": "stdout", "text": "a\n"},
            {"output_type": "stream", "name": "stdout", "text": "b\n"},
            {"output_type": "stream", "name": "stderr", "text": "warn!\n"},
        ]
    )
    assert "a\nb" in out.text
    assert "[stderr]\nwarn!" in out.text


def test_ansi_stripped():
    assert strip_ansi("\x1b[31mred\x1b[0m") == "red"
    out = condense_outputs(
        [{"output_type": "stream", "name": "stdout", "text": "\x1b[1;32mok\x1b[0m\n"}]
    )
    assert "\x1b" not in out.text and "ok" in out.text


def test_truncation_marker():
    text = "x" * 10_000
    t = truncate(text, max_chars=1000)
    assert len(t) < 1100
    assert "chars omitted" in t


def test_uniform_table_to_csv():
    html = (
        "<table><thead><tr><th>a</th><th>b</th></tr></thead>"
        "<tbody><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr></tbody></table>"
    )
    text = html_table_to_text(html)
    assert text is not None and "CSV" in text
    assert "a,b" in text and "1,2" in text


def test_ragged_table_to_json():
    html = "<table><tr><td>a</td><td>b</td></tr><tr><td>only-one</td></tr></table>"
    text = html_table_to_text(html)
    assert text is not None and "JSON" in text
    payload = text.splitlines()[1]
    assert json.loads(payload) == [["a", "b"], ["only-one"]]


def test_complex_table_bails():
    html = "<table><tr><td colspan='2'>merged</td></tr></table>"
    assert html_table_to_text(html) is None


def test_row_count_excludes_headers():
    """Regression (feedback): a 10-data-row polars table announced '12 rows'
    because the column-name and dtype rows in <thead> were counted."""
    body = "".join(f"<tr><td>{i}</td><td>x</td></tr>" for i in range(10))
    html = (
        "<small>shape: (10, 2)</small>"
        "<table><thead><tr><th>a</th><th>b</th></tr>"
        "<tr><td>i64</td><td>str</td></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )
    text = html_table_to_text(html)
    assert text is not None
    assert "[table as CSV, 10 rows]" in text
    assert "i64,str" in text  # dtype row still shown, just not counted


def test_truncated_pandas_preview_reports_full_size():
    """Regression (feedback): a 29,658-row frame rendered 10 rows with no
    marker — the preview read as the whole table."""
    body = "".join(f"<tr><th>{i}</th><td>{i}</td><td>v</td></tr>" for i in range(5))
    html = (
        "<div><table><thead><tr><th></th><th>a</th><th>b</th></tr></thead>"
        f"<tbody>{body}<tr><th>...</th><td>...</td><td>...</td></tr>"
        "<tr><th>29657</th><td>29657</td><td>v</td></tr></tbody></table>"
        "<p>29658 rows × 3 columns</p></div>"
    )
    text = html_table_to_text(html)
    assert text is not None
    assert "showing 6 of 29,658 rows" in text


def test_truncated_polars_preview_reports_full_size():
    body = "".join(f"<tr><td>{i}</td></tr>" for i in range(4))
    html = (
        "<div><small>shape: (29_658, 1)</small>"
        "<table><thead><tr><th>a</th></tr><tr><td>i64</td></tr></thead>"
        f"<tbody>{body}<tr><td>&hellip;</td></tr><tr><td>29657</td></tr></tbody>"
        "</table></div>"
    )
    text = html_table_to_text(html)
    assert text is not None
    assert "showing 5 of 29,658 rows" in text


def test_ellipsis_without_declared_total():
    html = (
        "<table><thead><tr><th>a</th></tr></thead>"
        "<tbody><tr><td>1</td></tr><tr><td>…</td></tr><tr><td>9</td></tr></tbody></table>"
    )
    text = html_table_to_text(html)
    assert text is not None
    assert "showing 2 rows of a longer table (total unknown)" in text


def test_header_only_table_falls_back_to_raw_count():
    html = "<table><tr><th>a</th></tr><tr><th>b</th></tr></table>"
    text = html_table_to_text(html)
    assert text is not None and "2 rows" in text


def test_empty_dataframe_reports_zero_rows():
    """Review finding: an empty pandas frame reported its header row as
    '1 rows' although the declared total (0) was right there in the HTML."""
    html = (
        "<div><table><thead><tr><th></th><th>a</th><th>b</th></tr></thead>"
        "<tbody></tbody></table><p>0 rows × 2 columns</p></div>"
    )
    text = html_table_to_text(html)
    assert text is not None and "[table as CSV, 0 rows]" in text

    # polars empty frame: two thead rows (names + dtypes), declared shape 0
    html = (
        "<div><small>shape: (0, 2)</small><table><thead><tr><th>a</th><th>b</th></tr>"
        "<tr><td>i64</td><td>str</td></tr></thead><tbody></tbody></table></div>"
    )
    text = html_table_to_text(html)
    assert text is not None and "[table as CSV, 0 rows]" in text


def test_unclosed_thead_does_not_swallow_body_rows():
    """Review finding: omitted </thead> (legal HTML — tbody implicitly closes
    it) left _in_thead sticky, so every body row was counted as a header."""
    body = "".join(f"<tr><td>{i}</td></tr>" for i in range(5))
    html = f"<table><thead><tr><th>a</th></tr><tbody>{body}</tbody></table>"
    text = html_table_to_text(html)
    assert text is not None and "[table as CSV, 5 rows]" in text


def test_label_counts_emitted_rows_when_cap_fires():
    """Review finding: the label counted parsed rows while the body was
    capped at MAX_TABLE_ROWS — 'showing 200 of 29,658' above 49 rows."""
    body = "".join(f"<tr><td>{i}</td></tr>" for i in range(200))
    html = (
        f"<div><table><thead><tr><th>a</th></tr></thead><tbody>{body}</tbody>"
        "</table><p>29658 rows × 1 columns</p></div>"
    )
    text = html_table_to_text(html)
    assert text is not None
    assert "showing 49 of 29,658 rows" in text  # 50 kept rows − 1 header
    assert "more rows omitted" in text

    # without a declared total, the parsed count is exact — use it
    html = f"<table><thead><tr><th>a</th></tr></thead><tbody>{body}</tbody></table>"
    text = html_table_to_text(html)
    assert text is not None and "showing 49 of 200 rows" in text


def test_mime_bundle_prefers_table_over_plain():
    html = "<table><tr><th>x</th></tr><tr><td>1</td></tr></table>"
    out = condense_outputs(
        [
            {
                "output_type": "execute_result",
                "execution_count": 1,
                "data": {"text/plain": "shape: (1, 1)\n...box drawing...", "text/html": html},
                "metadata": {},
            }
        ]
    )
    assert "CSV" in out.text
    assert "box drawing" not in out.text


def test_image_extracted_and_placeholder():
    png = _png_bytes()
    out = condense_outputs(
        [
            {
                "output_type": "display_data",
                "data": {"image/png": base64.b64encode(png).decode()},
                "metadata": {},
            }
        ]
    )
    assert out.images and out.images[0][:8] == b"\x89PNG\r\n\x1a\n"
    assert "[image attached]" in out.text


def test_large_image_downscaled():
    png = _png_bytes(size=(2400, 100))
    out = condense_outputs(
        [
            {
                "output_type": "display_data",
                "data": {"image/png": base64.b64encode(png).decode()},
                "metadata": {},
            }
        ]
    )
    from PIL import Image

    img = Image.open(io.BytesIO(out.images[0]))
    assert max(img.size) <= 1200


def test_error_output():
    out = condense_outputs(
        [
            {
                "output_type": "error",
                "ename": "ValueError",
                "evalue": "boom",
                "traceback": ["\x1b[31mTraceback...\x1b[0m", "line 1", "ValueError: boom"],
            }
        ]
    )
    assert out.has_error
    assert "ERROR ValueError: boom" in out.text
    assert "\x1b" not in out.text


def test_empty_outputs():
    assert condense_outputs([]).text == "(no output)"

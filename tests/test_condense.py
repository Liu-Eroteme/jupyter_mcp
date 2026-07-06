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

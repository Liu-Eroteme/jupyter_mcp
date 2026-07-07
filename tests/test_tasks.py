"""OutputBuffer semantics: coalescing, capping, clear_output, last_line."""

from jupyter_mcp.tasks import (
    HEAD_OUTPUTS,
    STREAM_HEAD_CHARS,
    STREAM_TAIL_CHARS,
    TAIL_OUTPUTS,
    OutputBuffer,
)


def _stream(text, name="stdout"):
    return {"output_type": "stream", "name": name, "text": text}


def test_adjacent_streams_coalesce():
    buf = OutputBuffer()
    for i in range(100):
        buf.add(_stream(f"line {i}\n"))
    outs = buf.snapshot()
    assert len(outs) == 1
    assert "line 0" in outs[0]["text"] and "line 99" in outs[0]["text"]


def test_different_streams_do_not_coalesce():
    buf = OutputBuffer()
    buf.add(_stream("out\n"))
    buf.add(_stream("err\n", name="stderr"))
    buf.add(_stream("out2\n"))
    assert len(buf.snapshot()) == 3


def test_oversized_stream_keeps_head_and_tail():
    buf = OutputBuffer()
    for i in range(20_000):
        buf.add(_stream(f"tick {i}\n"))
    outs = buf.snapshot()
    assert len(outs) == 1
    text = outs[0]["text"]
    assert len(text) < STREAM_HEAD_CHARS + STREAM_TAIL_CHARS + 500
    assert "tick 0\n" in text
    assert "tick 19999\n" in text
    assert "chars dropped" in text


def test_output_flood_keeps_head_and_tail():
    buf = OutputBuffer()
    n = HEAD_OUTPUTS + TAIL_OUTPUTS + 50
    for i in range(n):
        buf.add({"output_type": "display_data", "data": {"text/plain": f"obj {i}"}})
    outs = buf.snapshot()
    assert len(outs) == HEAD_OUTPUTS + TAIL_OUTPUTS + 1  # + dropped marker
    assert outs[0]["data"]["text/plain"] == "obj 0"
    assert outs[-1]["data"]["text/plain"] == f"obj {n - 1}"
    assert "50 outputs dropped" in outs[HEAD_OUTPUTS]["text"]


def test_clear_output_immediate_and_deferred():
    buf = OutputBuffer()
    buf.add(_stream("before\n"))
    buf.clear()
    assert buf.snapshot() == []
    buf.add(_stream("first\n"))
    buf.clear(wait=True)  # deferred until the next output arrives
    assert buf.snapshot()[0]["text"] == "first\n"
    buf.add(_stream("after\n"))
    outs = buf.snapshot()
    assert len(outs) == 1 and outs[0]["text"] == "after\n"


def test_last_line():
    buf = OutputBuffer()
    assert buf.last_line() == ""
    buf.add(_stream("epoch 1 loss=0.9\nepoch 2 loss=0.5\n"))
    assert buf.last_line() == "epoch 2 loss=0.5"
    buf.add({"output_type": "error", "ename": "ValueError", "evalue": "boom", "traceback": []})
    assert buf.last_line() == "ValueError: boom"

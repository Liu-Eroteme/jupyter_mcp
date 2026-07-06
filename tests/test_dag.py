from jupyter_mcp.dag import analyze_source, build_graph, strip_magics


def test_simple_def_use():
    d = analyze_source("x = 1\ny = x + 1")
    assert "x" in d.defines and "y" in d.defines
    assert d.uses == set()  # x was defined before use in the same cell


def test_external_use():
    d = analyze_source("y = x + 1")
    assert d.uses == {"x"}


def test_augassign_reads_previous_value():
    d = analyze_source("x += 1")
    assert "x" in d.uses and "x" in d.defines


def test_self_reassignment_chain():
    # df = df.with_columns(...) must depend on the upstream df
    d = analyze_source("df = df.filter(x > 1)")
    assert "df" in d.uses and "df" in d.defines


def test_method_call_is_use_not_mutation():
    d = analyze_source("df.head()")
    assert "df" in d.uses
    assert "df" not in d.mutates


def test_known_mutator_method():
    d = analyze_source("items.append(3)")
    assert "items" in d.mutates and "items" in d.uses


def test_subscript_assignment_mutates():
    d = analyze_source("df['col'] = df['a'] * 2")
    assert "df" in d.mutates


def test_imports_define():
    d = analyze_source("import polars as pl\nfrom pathlib import Path")
    assert {"pl", "Path"} <= d.defines


def test_wildcard_import():
    d = analyze_source("from numpy import *")
    assert d.wildcard_import


def test_function_body_free_vars_are_uses():
    src = "def f(a):\n    return a + CONST + helper(a)"
    d = analyze_source(src)
    assert {"CONST", "helper"} <= d.uses
    assert "f" in d.defines
    assert "a" not in d.uses  # parameter is local


def test_function_body_name_defined_in_same_cell_not_a_use():
    src = "helper = lambda v: v * 2\ndef f(a):\n    return helper(a)"
    d = analyze_source(src)
    assert "helper" not in d.uses


def test_comprehension_and_lambda():
    d = analyze_source("out = [f(v) for v in values]")
    assert "values" in d.uses and "f" in d.uses


def test_comprehension_vars_not_external_uses():
    # regression: elt was visited before the generator bound its target
    d = analyze_source("cols = [c for c in df.columns if c != 'x']")
    assert "c" not in d.uses and "df" in d.uses
    d = analyze_source("nulls = {c: n for c, n in zip(a, b) if n > 0}")
    assert "c" not in d.uses and "n" not in d.uses
    assert {"a", "b"} <= d.uses


def test_comprehension_vars_do_not_leak():
    d = analyze_source("cols = [c for c in items]\nprint(c)")
    assert "c" in d.uses  # the print(c) read is genuinely external


def test_nested_multi_generator_comprehension():
    src = "out = [f(x, y) for x in xs for y in ys(x) if y]"
    d = analyze_source(src)
    assert {"f", "xs", "ys"} <= d.uses
    assert "x" not in d.uses and "y" not in d.uses


def test_builtins_ignored():
    d = analyze_source("print(len(x))")
    assert d.uses == {"x"}


def test_magics_stripped():
    d = analyze_source("%matplotlib inline\n!ls\nx = 1")
    assert d.defines == {"x"} and d.parse_error is None


def test_opaque_cell_magic():
    assert strip_magics("%%bash\necho hi") is None
    d = analyze_source("%%bash\necho hi")
    assert d.defines == set() and d.uses == set()


def test_transparent_cell_magic():
    d = analyze_source("%%time\nx = y + 1")
    assert "x" in d.defines and "y" in d.uses


def test_syntax_error_flagged():
    d = analyze_source("def broken(:")
    assert d.parse_error is not None


def test_graph_edges_and_closures():
    cells = [
        ("load", "code", "import polars as pl\ndf = pl.read_csv('x.csv')"),
        ("note", "markdown", "# notes"),
        ("clean", "code", "df = df.drop_nulls()"),
        ("stats", "code", "summary = df.describe()"),
        ("plot", "code", "plot(summary)"),
    ]
    g = build_graph(cells)
    assert g.parents["clean"] == {"load": {"df"}}
    assert g.parents["stats"] == {"clean": {"df"}}  # nearest writer wins
    assert "summary" in g.parents["plot"].get("stats", set())
    assert g.descendants("load") == {"clean", "stats", "plot"}
    assert g.ancestors("plot") == {"stats", "clean", "load"}
    assert g.stale_closure({"clean"}) == ["clean", "stats", "plot"]
    assert "plot" in g.undefined  # `plot` function never defined


def test_wildcard_fallback_edge():
    cells = [
        ("wild", "code", "from numpy import *"),
        ("use", "code", "y = array([1, 2])"),
    ]
    g = build_graph(cells)
    assert g.parents["use"] == {"wild": {"array"}}
    assert "use" not in g.undefined

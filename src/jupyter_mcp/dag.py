"""Static inter-cell dependency graph via AST def/use analysis.

Deliberately *not* ipyflow: a runtime-traced DAG only knows about executed
code, while this server's editing flow needs dependencies for cells at rest
(edited but not yet run). A static pass over the current file state covers
typical EDA notebooks; the known blind spots are handled conservatively:

- ``obj.method()`` counts as a *use* of ``obj``. It additionally counts as a
  *write* (mutation) only for a small set of known-mutating method names
  (``append``, ``fit``, ...) plus subscript/attribute assignment targets —
  so pure-functional chains (polars) don't create false forward edges.
- IPython magics / shell escapes are stripped before parsing; a cell magic
  other than the transparent ones (%%time, %%capture, ...) makes the cell
  opaque (no dependencies).
- ``from x import *`` marks the cell as a wildcard definer: later cells using
  otherwise-unresolvable names get an edge to the nearest wildcard cell.
- Function/class bodies run at call time, so their free variables count as
  uses regardless of statement order, unless bound elsewhere in the same cell.
"""

from __future__ import annotations

import ast
import builtins
import re
from dataclasses import dataclass, field

_BUILTINS = frozenset(dir(builtins)) | {"display", "get_ipython", "__name__", "__file__"}

#: method names treated as mutating their receiver (conservative allowlist)
KNOWN_MUTATORS = frozenset(
    {
        "append", "extend", "insert", "remove", "pop", "clear", "sort", "reverse",
        "update", "add", "discard", "setdefault", "popitem",
        "fit", "fit_transform", "partial_fit", "train", "load_state_dict",
        "add_trace", "update_layout", "update_traces", "add_subplot",
        "enable_string_cache", "seed",
    }
)

#: cell magics whose body is still python and should be analyzed
_TRANSPARENT_CELL_MAGICS = {"time", "timeit", "capture", "prun"}


@dataclass
class CellDeps:
    defines: set[str] = field(default_factory=set)
    uses: set[str] = field(default_factory=set)
    mutates: set[str] = field(default_factory=set)
    wildcard_import: bool = False
    parse_error: str | None = None

    @property
    def writes(self) -> set[str]:
        return self.defines | self.mutates


def strip_magics(source: str) -> str | None:
    """Remove IPython magics/shell escapes; None if the cell isn't python."""
    lines = source.splitlines()
    # cell magic?
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("%%"):
            magic = stripped[2:].split(None, 1)[0].split(".")[0]
            if magic in _TRANSPARENT_CELL_MAGICS:
                rest = lines[lines.index(line) + 1 :]
                return "\n".join(_strip_line_magics(rest))
            return None  # opaque cell magic (%%bash, %%sql, ...)
        break
    return "\n".join(_strip_line_magics(lines))


_INTROSPECT_RE = re.compile(r"[\w.]+\?{1,2}$")


def _strip_line_magics(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")) or _INTROSPECT_RE.fullmatch(stripped):
            indent = line[: len(line) - len(stripped)]
            out.append(indent + "pass")
        else:
            out.append(line)
    return out


class _ScopeVisitor(ast.NodeVisitor):
    """Collect module-level defs/uses/mutations with order-aware resolution.

    Module level: a Load counts as an external use only if the name hasn't
    been bound earlier in the same cell. Function/class bodies: free
    variables are collected and later checked against *all* module bindings
    of the cell (deferred execution ⇒ order-independent).
    """

    def __init__(self) -> None:
        self.defined: set[str] = set()          # module-level bindings (in order)
        self.uses: set[str] = set()             # external reads (module level)
        self.mutates: set[str] = set()
        self.deferred_uses: set[str] = set()    # free vars of function/class bodies
        self.wildcard = False

    # -- helpers -----------------------------------------------------------

    def _use(self, name: str) -> None:
        if name not in self.defined and name not in _BUILTINS:
            self.uses.add(name)

    def _bind_target(self, node: ast.expr) -> None:
        """Handle assignment targets: Name binds; Subscript/Attribute mutate."""
        if isinstance(node, ast.Name):
            self.defined.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self._bind_target(elt)
        elif isinstance(node, ast.Starred):
            self._bind_target(node.value)
        elif isinstance(node, (ast.Subscript, ast.Attribute)):
            base = _root_name(node)
            if base is not None:
                self._use(base)
                self.mutates.add(base)
            self.visit(node.value)
            if isinstance(node, ast.Subscript):
                self.visit(node.slice)

    # -- statements --------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for t in node.targets:
            self._bind_target(t)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value:
            self.visit(node.value)
        self._bind_target(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._use(node.target.id)  # reads previous value
            self.defined.add(node.target.id)
        else:
            self._bind_target(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # walrus
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self.defined.add(node.target.id)

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        self._bind_target(node.target)
        for stmt in node.body + node.orelse:
            self.visit(stmt)

    visit_AsyncFor = visit_For  # type: ignore[assignment]

    def visit_withitem(self, node: ast.withitem) -> None:
        self.visit(node.context_expr)
        if node.optional_vars is not None:
            self._bind_target(node.optional_vars)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.defined.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                self.wildcard = True
            else:
                self.defined.add(alias.asname or alias.name)

    def visit_Delete(self, node: ast.Delete) -> None:
        for t in node.targets:
            if isinstance(t, ast.Name):
                self._use(t.id)
                self.defined.discard(t.id)
            else:
                self._bind_target(t)

    def visit_Global(self, node: ast.Global) -> None:
        pass

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.defined.add(node.name)
        for stmt in node.body:
            self.visit(stmt)

    # -- deferred scopes ----------------------------------------------------

    def _handle_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        # decorators / defaults / annotations evaluate at definition time
        if not isinstance(node, ast.Lambda):
            for dec in node.decorator_list:
                self.visit(dec)
            if node.returns:
                self.visit(node.returns)
        for default in list(node.args.defaults) + [d for d in node.args.kw_defaults if d]:
            self.visit(default)
        for arg in _all_args(node.args):
            if arg.annotation:
                self.visit(arg.annotation)
        free = _free_names(node)
        self.deferred_uses |= free
        if not isinstance(node, ast.Lambda):
            self.defined.add(node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._handle_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for dec in node.decorator_list:
            self.visit(dec)
        for base in node.bases + [kw.value for kw in node.keywords]:
            self.visit(base)
        free = _free_names(node)
        self.deferred_uses |= free
        self.defined.add(node.name)

    # -- comprehensions -------------------------------------------------------
    # Generators must be visited before the element expression (evaluation
    # order), and comprehension variables are scoped to the comprehension —
    # they neither read from nor leak into the module scope.

    def _visit_comprehension(self, node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp) -> None:
        temp_bound: set[str] = set()
        for gen in node.generators:
            self.visit(gen.iter)
            before = set(self.defined)
            self._bind_target(gen.target)
            temp_bound |= self.defined - before
            for cond in gen.ifs:
                self.visit(cond)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.defined -= temp_bound

    visit_ListComp = _visit_comprehension  # type: ignore[assignment]
    visit_SetComp = _visit_comprehension  # type: ignore[assignment]
    visit_GeneratorExp = _visit_comprehension  # type: ignore[assignment]
    visit_DictComp = _visit_comprehension  # type: ignore[assignment]

    # -- expressions ---------------------------------------------------------

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._use(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.defined.add(node.id)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            base = _root_name(func)
            if base is not None and func.attr in KNOWN_MUTATORS:
                self._use(base)
                self.mutates.add(base)
        self.generic_visit(node)


def _all_args(args: ast.arguments) -> list[ast.arg]:
    out = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    if args.vararg:
        out.append(args.vararg)
    if args.kwarg:
        out.append(args.kwarg)
    return out


def _root_name(node: ast.expr) -> str | None:
    """`df["a"].b.c` -> 'df'; None if the chain doesn't root in a Name."""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _free_names(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef) -> set[str]:
    """Names read inside a deferred scope that aren't bound within it."""
    bound: set[str] = set()
    loaded: set[str] = set()

    class Inner(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
            else:
                bound.add(node.id)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            bound.add(node.name)
            self._nested(node)

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

        def visit_Lambda(self, node: ast.Lambda) -> None:
            self._nested(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            bound.add(node.name)
            self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                if alias.name != "*":
                    bound.add(alias.asname or alias.name)

        def _nested(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
            for arg in _all_args(node.args):
                bound.add(arg.arg)
            loaded.update(_free_names(node) - bound)

    body = node.body if not isinstance(node, ast.Lambda) else [node.body]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        for arg in _all_args(node.args):
            bound.add(arg.arg)
    inner = Inner()
    for stmt in body:
        if isinstance(stmt, ast.stmt):
            inner.visit(stmt)
        else:  # Lambda body is an expression
            inner.visit(stmt)
    return {n for n in loaded - bound if n not in _BUILTINS}


def analyze_source(source: str) -> CellDeps:
    stripped = strip_magics(source)
    if stripped is None:
        return CellDeps()
    try:
        tree = ast.parse(stripped)
    except SyntaxError as e:
        return CellDeps(parse_error=f"line {e.lineno}: {e.msg}")
    v = _ScopeVisitor()
    for stmt in tree.body:
        v.visit(stmt)
    deferred = {n for n in v.deferred_uses if n not in v.defined}
    return CellDeps(
        defines=v.defined,
        uses=v.uses | deferred,
        mutates=v.mutates,
        wildcard_import=v.wildcard,
    )


# --------------------------------------------------------------------- graph


@dataclass
class NotebookGraph:
    order: list[str]
    deps: dict[str, CellDeps]
    #: parent edges: cell -> {parent_cell: {variables provided}}
    parents: dict[str, dict[str, set[str]]]
    children: dict[str, dict[str, set[str]]]
    #: used names with no earlier definer (and no wildcard fallback)
    undefined: dict[str, set[str]]

    def ancestors(self, name: str) -> set[str]:
        return self._closure(name, self.parents)

    def descendants(self, name: str) -> set[str]:
        return self._closure(name, self.children)

    def _closure(self, name: str, edges: dict[str, dict[str, set[str]]]) -> set[str]:
        seen: set[str] = set()
        stack = list(edges.get(name, {}))
        while stack:
            n = stack.pop()
            if n not in seen:
                seen.add(n)
                stack.extend(edges.get(n, {}))
        return seen

    def stale_closure(self, changed: set[str]) -> list[str]:
        """Changed cells plus all descendants, in document order."""
        out = set(changed)
        for name in changed:
            out |= self.descendants(name)
        return [n for n in self.order if n in out]


def build_graph(cells: list[tuple[str, str, str]]) -> NotebookGraph:
    """cells: list of (name, cell_type, source) in document order."""
    order: list[str] = []
    deps: dict[str, CellDeps] = {}
    parents: dict[str, dict[str, set[str]]] = {}
    children: dict[str, dict[str, set[str]]] = {}
    undefined: dict[str, set[str]] = {}

    last_writer: dict[str, str] = {}
    last_wildcard: str | None = None

    for name, cell_type, source in cells:
        order.append(name)
        if cell_type != "code":
            deps[name] = CellDeps()
            continue
        d = analyze_source(source)
        deps[name] = d
        cell_parents: dict[str, set[str]] = {}
        missing: set[str] = set()
        for var in sorted(d.uses):
            writer = last_writer.get(var)
            if writer is not None and writer != name:
                cell_parents.setdefault(writer, set()).add(var)
            elif writer is None:
                if last_wildcard is not None:
                    cell_parents.setdefault(last_wildcard, set()).add(var)
                else:
                    missing.add(var)
        parents[name] = cell_parents
        if missing:
            undefined[name] = missing
        for parent, vars_ in cell_parents.items():
            children.setdefault(parent, {}).setdefault(name, set()).update(vars_)
        for var in d.writes:
            last_writer[var] = name
        if d.wildcard_import:
            last_wildcard = name

    return NotebookGraph(order=order, deps=deps, parents=parents, children=children, undefined=undefined)

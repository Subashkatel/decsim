"""The documentation answers for itself, against the tree it describes.

A page that is right on the day it is written and wrong a month later is
worse than no page, so the six rules below are the ones a machine can
hold. A future writer who adds a page keeps them, and nothing else about
a page is enforced here.

1. The four generated pages of docs/reference/ (map, ports, tables, cli)
   equal what tools/docs_map.py generates from the source now.
2. Every module of decsim/ carries a docstring whose first sentence ends
   within 160 characters, because the map is made of those sentences.
3. Every backticked path in docs/ and README.md exists in the tree, and
   every `path::test_name` in them names a test pytest collects.
4. Every edge of every mermaid diagram in docs/ is a real port call: the
   label reads Port.method, decsim/ports.py declares that method on that
   port, and the package at the tail of the arrow calls it. Each diagram
   declares which package each of its nodes is, in `%% Node = package`
   comment lines, so the check knows where to look.
5. Every uppercase name in backticks in docs/ and README.md is a name
   the tree defines: a module-level name of decsim/ or tests/, or a
   shell variable one of the scripts reads, apart from the foreign names
   listed in FOREIGN_NAMES; and there are seventeen plug-in tables.
6. No em dash in docs/, README.md, STYLE.md, or any docstring of the
   package, the tests or the tools.
7. Every relative link in docs/ and README.md opens a file in the tree,
   its anchor names a heading of that file, and docs/README.md links to
   every page, so a reader can reach any page from the front page.
"""

import ast
import functools
import importlib.util
import pathlib
import re
import subprocess
import sys

TESTS_FILE = pathlib.Path(__file__)
TESTS_PATH = TESTS_FILE.resolve()
CHECKOUT = TESTS_PATH.parent.parent
PACKAGE = CHECKOUT / "decsim"
DOCS = CHECKOUT / "docs"
TESTS = CHECKOUT / "tests"
TOOLS = CHECKOUT / "tools"
SLURM = CHECKOUT / "slurm"
EM_DASH = "—"
TABLE_COUNT = 17

PATH_PREFIXES = ("decsim/", "tests/", "tools/", "docs/", "configs/", "slurm/")
ROOT_FILES = ("README.md", "STYLE.md", "pyproject.toml")
BACKTICKED = re.compile(r"`([^`\n]+)`")
UPPER_NAME = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")
SHELL_NAME = re.compile(r"\$\{?([A-Z][A-Z0-9_]{2,})")
NODE_PACKAGE = re.compile(r"^\s*%%\s*(\w+)\s*=\s*([a-z_]+)\s*$")
EDGE = re.compile(
    r'^\s*(\w+)(?:\[[^\]]*\])?\s*--[->]*\|"([A-Za-z_]+)\.([A-Za-z_]+)"\|'
    r"\s*(\w+)"
)
MERMAID_OPEN = "```mermaid"
MERMAID_CLOSE = "```"
RELATIVE_LINK = re.compile(r"\]\(([^)#:]+)(?:#([^)]+))?\)")
HEADING = re.compile(r"^#+\s+(.*?)\s*$", re.MULTILINE)
NOT_A_SLUG_CHARACTER = re.compile(r"[^a-z0-9 _-]")

# Names decsim does not define, written in backticks because they name
# the referent a decision came from.
FOREIGN_NAMES = (
    "BUILT_IN_DECODERS",
    "SIM",
)


def _tool(name: str):
    """One tool of tools/, imported from that folder by path."""
    path = CHECKOUT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    loader.exec_module(module)
    return module


@functools.cache
def _docs_map():
    """The generator, imported once for the whole file."""
    return _tool("docs_map")


@functools.cache
def _module_paths() -> tuple:
    """Every module of decsim/, in path order."""
    tool = _docs_map()
    paths = tool.module_paths(PACKAGE)
    return tuple(paths)


@functools.cache
def _test_module_paths() -> tuple:
    """Every module of tests/, in path order."""
    globbed = TESTS.rglob("*.py")
    return tuple(sorted(globbed))


@functools.cache
def _prose_files() -> tuple:
    """Every page a reader reads: docs/ and the root's own markdown."""
    globbed = DOCS.rglob("*.md")
    found = sorted(globbed)
    for name in ROOT_FILES:
        path = CHECKOUT / name
        found.append(path)
    return tuple(found)


@functools.cache
def _collected_tests() -> frozenset:
    """Every test pytest collects, as `path::name`, from one collection."""
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    finished = subprocess.run(
        command, cwd=CHECKOUT, capture_output=True, text=True, check=True
    )
    lines = finished.stdout.splitlines()
    named = set()
    for line in lines:
        if "::" in line:
            stripped = line.strip()
            named.add(stripped)
    return frozenset(named)


@functools.cache
def _module_level_names() -> frozenset:
    """Every module-level name decsim/ or tests/ assigns.

    A page names a table the tests own as readily as one the package
    owns: ENDS_OF_PATH is the rule the data path page describes, and it
    lives in tests/test_send_ends.py.
    """
    shell = _shell_names()
    named = set(shell)
    for path in _module_paths() + _test_module_paths():
        assigned = _names_assigned(path)
        named.update(assigned)
    return frozenset(named)


@functools.cache
def _shell_names() -> frozenset:
    """Every shell variable the scripts of the tree read.

    A page that shows how to launch a job names the variables that job
    reads, and RUN is one of them: slurm/slurm_run.sh takes it as the
    folder an array task writes its shard into.
    """
    named = set()
    for folder in (SLURM, TOOLS):
        globbed = folder.rglob("*.sh")
        for path in sorted(globbed):
            text = path.read_text()
            found = SHELL_NAME.findall(text)
            named.update(found)
    return frozenset(named)


def _names_assigned(path: pathlib.Path) -> set:
    """The module-level names one module assigns."""
    source = path.read_text()
    tree = ast.parse(source)
    named = set()
    for node in tree.body:
        bound = _assigned_targets(node)
        named.update(bound)
    return named


def _assigned_targets(node: ast.AST) -> set:
    """The names one module-level statement binds."""
    named = set()
    if isinstance(node, ast.ClassDef):
        named.add(node.name)
    if isinstance(node, ast.FunctionDef):
        named.add(node.name)
    if not isinstance(node, ast.Assign):
        return named
    for target in node.targets:
        if isinstance(target, ast.Name):
            named.add(target.id)
    return named


@functools.cache
def _called_methods() -> dict:
    """Which packages of decsim/ call each method name."""
    tool = _docs_map()
    graph = tool.uses_graph_tool()
    calls = {}
    for path in _module_paths():
        package = graph.package_of(path, PACKAGE)
        _record_calls(path, package, calls)
    return calls


def _record_calls(path: pathlib.Path, package: str, calls: dict) -> None:
    """Every method one module calls, filed under its package."""
    source = path.read_text()
    tree = ast.parse(source)
    walked = ast.walk(tree)
    for node in walked:
        name = _called_method(node)
        if name is not None:
            packages = calls.setdefault(name, set())
            packages.add(package)


def _called_method(node: ast.AST):
    """The method name one node calls, or None when it calls no method."""
    if not isinstance(node, ast.Call):
        return None
    called = node.func
    if not isinstance(called, ast.Attribute):
        return None
    return called.attr


@functools.cache
def _port_methods() -> dict:
    """Each port of decsim/ports.py and the method names it declares."""
    path = PACKAGE / "ports.py"
    source = path.read_text()
    tree = ast.parse(source)
    declared = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            declared[node.name] = _method_names(node)
    return declared


def _method_names(node: ast.ClassDef) -> frozenset:
    """The methods one class declares."""
    named = set()
    for child in node.body:
        if isinstance(child, ast.FunctionDef):
            named.add(child.name)
    return frozenset(named)


def _mermaid_blocks(text: str) -> list:
    """Every mermaid block of one page, as its own list of lines."""
    blocks = []
    open_block = None
    for line in text.splitlines():
        open_block = _block_line(line, open_block, blocks)
    return blocks


def _block_line(line: str, open_block, blocks: list):
    """One line of a page, inside a mermaid block or outside every one."""
    stripped = line.strip()
    if open_block is None:
        if stripped == MERMAID_OPEN:
            blocks.append([])
            return blocks[-1]
        return None
    if stripped == MERMAID_CLOSE:
        return None
    open_block.append(line)
    return open_block


def _declared_nodes(block: list) -> dict:
    """The package each node of one diagram stands for."""
    declared = {}
    for line in block:
        match = NODE_PACKAGE.match(line)
        if match is not None:
            declared[match.group(1)] = match.group(2)
    return declared


def _edges(block: list) -> list:
    """Every labelled edge of one diagram, as (tail, port, method, head)."""
    found = []
    for line in block:
        match = EDGE.match(line)
        if match is not None:
            groups = match.groups()
            found.append(groups)
    return found


def test_the_generated_reference_pages_are_what_the_source_generates_now():
    """map, ports, tables and cli are the tool's output, not a stale copy."""
    tool = _docs_map()
    generated = tool.pages(CHECKOUT)
    for name, text in generated.items():
        path = DOCS / "reference" / name
        assert path.exists(), f"{name} is not committed; run tools/docs_map.py"
        assert path.read_text() == text, (
            f"docs/reference/{name} is not what tools/docs_map.py generates "
            "from the source now; run the tool and commit the result"
        )


def test_every_module_opens_with_one_sentence_saying_what_it_is():
    """The map is made of first sentences, so every module has a short one."""
    tool = _docs_map()
    for path in _module_paths():
        source = path.read_text()
        tree = ast.parse(source)
        text = ast.get_docstring(tree)
        assert text is not None, f"{path} has no module docstring"
        sentence = tool.first_sentence(text)
        assert len(sentence) <= 160, (
            f"{path}: the first sentence of the docstring runs "
            f"{len(sentence)} characters; the map wants at most 160"
        )


def test_every_seventeen_plug_in_tables_are_on_the_tables_page():
    """The tables page is the whole set, not a subset that fell behind."""
    tool = _docs_map()
    entries = tool.tables_of(CHECKOUT)
    assert len(entries) == TABLE_COUNT
    page = DOCS / "reference" / "tables.md"
    text = page.read_text()
    for name, _, _ in entries:
        assert f"## `{name}`" in text


def test_every_path_in_backticks_in_the_docs_exists_in_the_tree():
    """A page names a file a reader can open, or the page is wrong."""
    for page in _prose_files():
        text = page.read_text()
        for quoted in BACKTICKED.findall(text):
            _check_one_path(quoted, page)


def _check_one_path(quoted: str, page: pathlib.Path) -> None:
    """One backticked word, when it reads as a path into this tree."""
    if not quoted.startswith(PATH_PREFIXES):
        return
    if "<" in quoted:
        return
    words = quoted.split("::")
    name = words[0]
    path = CHECKOUT / name
    assert path.exists(), (
        f"{page.name} names {quoted}, which is not in the tree"
    )


def test_every_test_named_in_the_docs_is_a_test_pytest_collects():
    """A page that sends a reader to a test sends them to one that exists."""
    collected = _collected_tests()
    for page in _prose_files():
        text = page.read_text()
        for quoted in BACKTICKED.findall(text):
            _check_one_test(quoted, collected, page)


def _check_one_test(quoted: str, collected: frozenset, page) -> None:
    """One backticked word, when it reads as a test identifier."""
    if "::" not in quoted:
        return
    if not quoted.startswith("tests/"):
        return
    assert quoted in collected, (
        f"{page.name} names {quoted}, which pytest does not collect"
    )


def test_every_uppercase_name_in_the_docs_is_a_name_the_package_defines():
    """A page names a table or a constant this tree has, or names a referent."""
    defined = _module_level_names()
    for page in _prose_files():
        text = page.read_text()
        for quoted in BACKTICKED.findall(text):
            _check_one_name(quoted, defined, page)


def _check_one_name(quoted: str, defined: frozenset, page) -> None:
    """One backticked word, when it reads as an uppercase name."""
    if not UPPER_NAME.match(quoted):
        return
    if quoted in FOREIGN_NAMES:
        return
    assert quoted in defined, (
        f"{page.name} names {quoted}, which the tree does not define"
    )


def test_every_diagram_edge_in_the_docs_is_a_port_call_that_happens():
    """A diagram's arrows are the calls the code makes, checked, not drawn."""
    ports = _port_methods()
    calls = _called_methods()
    for page in _prose_files():
        text = page.read_text()
        _check_one_page_of_diagrams(text, ports, calls, page)


def _check_one_page_of_diagrams(text: str, ports, calls, page) -> None:
    """Every mermaid block of one page."""
    for block in _mermaid_blocks(text):
        nodes = _declared_nodes(block)
        _check_one_diagram(block, nodes, ports, calls, page)


def _check_one_diagram(block: list, nodes, ports, calls, page) -> None:
    """Every labelled edge of one diagram."""
    for edge in _edges(block):
        _check_one_edge(edge, nodes, ports, calls, page)


def _check_one_edge(edge: tuple, nodes, ports, calls, page) -> None:
    """One arrow: its port, its method, and the package that calls it."""
    tail, port, method, head = edge
    where = f"{page.name}: {tail} -> {head} labelled {port}.{method}"
    assert port in ports, f"{where}, but decsim/ports.py has no {port}"
    methods = ports[port]
    assert method in methods, f"{where}, but {port} has no {method}"
    assert tail in nodes, f"{where}, but the diagram declares no %% {tail} ="
    assert head in nodes, f"{where}, but the diagram declares no %% {head} ="
    package = nodes[tail]
    callers = calls.get(method, set())
    assert package in callers, (
        f"{where}, but nothing in decsim/{package} calls {method}; "
        f"the packages that call it are {sorted(callers)}"
    )


def test_no_em_dash_in_the_prose_or_in_any_docstring():
    """The owner's rule, held over the pages and over every docstring."""
    for page in _prose_files():
        text = page.read_text()
        assert EM_DASH not in text, f"{page} carries an em dash"
    for path in _docstring_files():
        _check_docstrings(path)


def _docstring_files() -> list:
    """Every source file whose docstrings this rule covers."""
    found = []
    for folder in ("decsim", "tests", "tools"):
        root = CHECKOUT / folder
        globbed = root.rglob("*.py")
        found.extend(sorted(globbed))
    return found


def _check_docstrings(path: pathlib.Path) -> None:
    """Every docstring of one module, the module's own included."""
    source = path.read_text()
    tree = ast.parse(source)
    walked = ast.walk(tree)
    for node in walked:
        _check_one_docstring(node, path)


def _check_one_docstring(node: ast.AST, path: pathlib.Path) -> None:
    """One node that may carry a docstring."""
    if not isinstance(node, DOCUMENTED):
        return
    text = ast.get_docstring(node)
    if text is None:
        return
    assert EM_DASH not in text, f"{path} carries an em dash in a docstring"


DOCUMENTED = (
    ast.Module,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
)


def test_every_relative_link_in_the_docs_opens_a_heading_of_a_page():
    """A link a reader clicks lands on a file, and on the heading it names."""
    for page in _prose_files():
        text = page.read_text()
        for target, anchor in RELATIVE_LINK.findall(text):
            _check_one_link(page, target, anchor)


def _check_one_link(page: pathlib.Path, target: str, anchor: str) -> None:
    """One link, resolved from the folder of the page that carries it."""
    path = page.parent / target
    assert path.exists(), f"{page.name} links to {target}, which is not there"
    if not anchor:
        return
    slugs = _heading_slugs(path)
    assert anchor in slugs, (
        f"{page.name} links to {target}#{anchor}, and {path.name} has no "
        f"such heading; its headings are {sorted(slugs)}"
    )


def _heading_slugs(path: pathlib.Path) -> frozenset:
    """The anchors GitHub gives the headings of one page."""
    text = path.read_text()
    found = HEADING.findall(text)
    slugs = set()
    for heading in found:
        slug = _slug(heading)
        slugs.add(slug)
    return frozenset(slugs)


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading: lowercase, punctuation dropped."""
    text = heading.replace("`", "")
    lowered = text.lower()
    kept = NOT_A_SLUG_CHARACTER.sub("", lowered)
    return kept.replace(" ", "-")


def test_the_front_page_links_to_every_page():
    """docs/README.md is the one page from which every other is reachable."""
    front = DOCS / "README.md"
    text = front.read_text()
    linked = set()
    for target, _anchor in RELATIVE_LINK.findall(text):
        resolved = (DOCS / target).resolve()
        linked.add(resolved)
    for page in DOCS.rglob("*.md"):
        if page.name == "README.md":
            continue
        assert page.resolve() in linked, (
            f"docs/README.md does not link to {page.name}"
        )

"""The documentation answers for itself, against the tree it describes.

A page that is right on the day it is written and wrong a month later is
worse than no page, so the rules below are the ones a machine can hold.
A future writer who adds a page keeps them, and nothing else about a
page is enforced here.

1. The generated page docs/reference/parts.md equals what
   tools/docs_map.py generates from the source now.
2. Every backticked path in docs/ and README.md exists in the tree, and
   every `path::test_name` in them names a test pytest collects.
3. Every uppercase name in backticks in docs/ and README.md is a name
   the tree defines: a module-level name of decsim/, tests/, examples/
   or experiments/, or a shell variable one of the scripts reads, apart
   from the foreign names listed in FOREIGN_NAMES.
4. No em dash in docs/, README.md, STYLE.md, or any docstring of the
   package, the tests or the tools.
5. Every relative link in docs/ and README.md opens a file in the tree,
   its anchor names a heading of that file, and docs/README.md links to
   every page, so a reader can reach any page from the front page.
6. Every page that says which Python to install names the floor
   pyproject.toml's requires-python declares, the one pip enforces.
7. constraints.txt, which the install commands read, pins every package
   the run, test and dev extras name.
8. No page names one maintainer's private environment: the QLX
   container image, the container wrapper around its interpreter, or the
   .pydeps dependency folder. A portable setup (a venv, Apptainer in
   general) is a page's to describe.
9. A Python block whose fence names a file (```python examples/x.py)
   shows lines of that file, in a row, because a page's results come
   from the file, not the page.
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
EXAMPLES = CHECKOUT / "examples"
EXPERIMENTS = CHECKOUT / "experiments"
TOOLS = CHECKOUT / "tools"
SLURM = CHECKOUT / "slurm"
EM_DASH = "—"

PATH_PREFIXES = (
    "decsim/",
    "tests/",
    "tools/",
    "docs/",
    "examples/",
    "experiments/",
    "slurm/",
)
ROOT_FILES = ("README.md", "STYLE.md", "pyproject.toml")
BACKTICKED = re.compile(r"`([^`\n]+)`")
UPPER_NAME = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")
SHELL_NAME = re.compile(r"\$\{?([A-Z][A-Z0-9_]{2,})")
RELATIVE_LINK = re.compile(r"\]\(([^)#:]+)(?:#([^)]+))?\)")
HEADING = re.compile(r"^#+\s+(.*?)\s*$", re.MULTILINE)
NOT_A_SLUG_CHARACTER = re.compile(r"[^a-z0-9 _-]")
REQUIRED_PYTHON = re.compile(r'requires-python = ">=(3\.\d+)"')
NAMED_PYTHON = re.compile(r"Python (3\.\d+)\s+or\s+newer")
PINNED_EXTRA = re.compile(r"^(?:run|test|dev) = \[(.*)\]$", re.MULTILINE)
REQUIREMENT_NAME = re.compile(r'"([A-Za-z0-9_.-]+)')
PINNED_NAME = re.compile(r"^([A-Za-z0-9_.-]+)==", re.MULTILINE)
LISTING_FLAGS = re.MULTILINE | re.DOTALL
FILE_LISTING = re.compile(r"^```python (\S+\.py)\n(.*?)^```$", LISTING_FLAGS)
PRIVATE_ENVIRONMENT = re.compile(
    r"qlx|container wrapper|\.pydeps", re.IGNORECASE
)

# Names decsim does not define that a page writes in backticks: ruff's
# rule families.
FOREIGN_NAMES = ("SIM",)


def _tool(name: str):
    """One tool of tools/, imported from that folder by path."""
    path = CHECKOUT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    loader.exec_module(module)
    return module


@functools.cache
def _module_paths() -> tuple:
    """Every module of decsim/, in path order."""
    globbed = PACKAGE.rglob("*.py")
    return tuple(sorted(globbed))


@functools.cache
def _other_module_paths() -> tuple:
    """Every module of tests/, examples/ and experiments/, in path order."""
    found = []
    for folder in (TESTS, EXAMPLES, EXPERIMENTS):
        globbed = folder.rglob("*.py")
        found.extend(sorted(globbed))
    return tuple(found)


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
    """Every module-level name of decsim/, tests/, examples/, experiments/.

    A page names a constant a run file or a test owns as readily as one
    the package owns.
    """
    shell = _shell_names()
    named = set(shell)
    for path in _module_paths() + _other_module_paths():
        assigned = _names_assigned(path)
        named.update(assigned)
    return frozenset(named)


@functools.cache
def _shell_names() -> frozenset:
    """Every shell variable the scripts of the tree read.

    A page that shows how to launch a job names the variables that job
    reads.
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
        variables = _environment_variables_named(node)
        named.update(variables)
    return named


def _environment_variables_named(node: ast.AST) -> set:
    """The environment variable a *_VARIABLE constant names.

    A page that shows how to launch a job names the variables the job
    reads, as it names a shell script's.
    """
    is_assignment = isinstance(node, ast.Assign)
    if not is_assignment or not isinstance(node.value, ast.Constant):
        return set()
    named = set()
    for target in node.targets:
        if isinstance(target, ast.Name) and target.id.endswith("_VARIABLE"):
            named.add(node.value.value)
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


def test_the_parts_page_is_what_the_source_generates_now():
    """The parts page is the tool's output, not a stale copy."""
    tool = _tool("docs_map")
    generated = tool.parts_page(CHECKOUT)
    path = DOCS / "reference" / tool.PAGE_NAME
    committed = path.read_text()
    assert committed == generated, (
        f"docs/reference/{tool.PAGE_NAME} is not what tools/docs_map.py "
        "generates from the source now; run the tool and commit the result"
    )


def test_every_path_in_backticks_in_the_docs_exists_in_the_tree():
    """A page names a file a reader can open, or the page is wrong."""
    quoted_on_pages = _quoted_on_prose_pages()
    _check_each_quoted(quoted_on_pages, _check_one_path)


def _quoted_on_prose_pages() -> list:
    """(quoted text, page) for every backticked span of every page."""
    quoted_on_pages = []
    for page in _prose_files():
        text = page.read_text()
        for quoted in BACKTICKED.findall(text):
            quoted_on_pages.append((quoted, page))
    return quoted_on_pages


def _check_each_quoted(quoted_on_pages: list, check_one) -> None:
    for quoted, page in quoted_on_pages:
        check_one(quoted, page)


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
    quoted_on_pages = _quoted_on_prose_pages()
    check_one = functools.partial(_check_one_test, collected=collected)
    _check_each_quoted(quoted_on_pages, check_one)


def _check_one_test(quoted: str, page, collected: frozenset) -> None:
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
    quoted_on_pages = _quoted_on_prose_pages()
    check_one = functools.partial(_check_one_name, defined=defined)
    _check_each_quoted(quoted_on_pages, check_one)


def _check_one_name(quoted: str, page, defined: frozenset) -> None:
    """One backticked word, when it reads as an uppercase name."""
    if not UPPER_NAME.match(quoted):
        return
    if quoted in FOREIGN_NAMES:
        return
    assert quoted in defined, (
        f"{page.name} names {quoted}, which the tree does not define"
    )


def test_no_em_dash_in_the_prose_or_in_any_docstring():
    """The owner's rule, held over the pages and over every docstring."""
    pages = _prose_files()
    paths = _docstring_files()
    _check_no_em_dash_on(pages)
    _check_every_docstring(paths)


def _check_no_em_dash_on(pages) -> None:
    for page in pages:
        text = page.read_text()
        assert EM_DASH not in text, f"{page} carries an em dash"


def _check_every_docstring(paths) -> None:
    for path in paths:
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
    pages = _prose_files()
    _check_every_link(pages)


def _check_every_link(pages) -> None:
    for page in pages:
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
    linked = _pages_linked_from(text)
    unlinked = _pages_not_in(linked)
    assert unlinked == [], f"docs/README.md does not link to {unlinked}"


def _pages_linked_from(text: str) -> set:
    linked = set()
    for target, _anchor in RELATIVE_LINK.findall(text):
        resolved = (DOCS / target).resolve()
        linked.add(resolved)
    return linked


def _pages_not_in(linked: set) -> list:
    """The names of the pages other than a README that linked leaves out."""
    unlinked = []
    for page in DOCS.rglob("*.md"):
        if page.name == "README.md":
            continue
        if page.resolve() not in linked:
            unlinked.append(page.name)
    return unlinked


def test_every_page_names_the_python_floor_pyproject_declares():
    """A reader installs on the Python pip will accept, and no older one."""
    pyproject = CHECKOUT / "pyproject.toml"
    declaration = pyproject.read_text()
    required = REQUIRED_PYTHON.search(declaration)
    floor = required.group(1)
    pages = _prose_files()
    _check_every_python_floor(pages, floor)


def _check_every_python_floor(pages, floor: str) -> None:
    for page in pages:
        text = page.read_text()
        named = NAMED_PYTHON.findall(text)
        _check_one_python_floor(named, floor, page)


def _check_one_python_floor(named: list, floor: str, page) -> None:
    """Every "Python 3.N or newer" of one page."""
    for version in named:
        assert version == floor, (
            f"{page.name} says Python {version} or newer, and "
            f"pyproject.toml requires Python {floor} or newer"
        )


def test_the_constraints_file_pins_every_package_the_extras_name():
    """An extra that grew without a new constraints.txt installs unpinned."""
    pyproject = CHECKOUT / "pyproject.toml"
    declaration = pyproject.read_text()
    extras = PINNED_EXTRA.findall(declaration)
    named = _requirement_names(extras)
    constraints = CHECKOUT / "constraints.txt"
    pinned_text = constraints.read_text()
    pinned = PINNED_NAME.findall(pinned_text)
    unpinned = named.difference(pinned)
    assert len(extras) == 3
    assert unpinned == set(), (
        f"constraints.txt pins no version of {sorted(unpinned)}; rerun "
        "the command in its first lines"
    )


def _requirement_names(extras: list) -> set:
    named = set()
    for listed in extras:
        names = REQUIREMENT_NAME.findall(listed)
        named.update(names)
    return named


def test_no_page_names_a_maintainers_private_environment():
    """A reader has the README's install and nothing only one host has."""
    pages = _markdown_pages()
    named_by_page = _private_names_by_page(pages)
    assert named_by_page == {}


def _private_names_by_page(pages) -> dict:
    """Each page that names a private environment: the names it uses."""
    named_by_page = {}
    for page in pages:
        text = page.read_text()
        named = PRIVATE_ENVIRONMENT.findall(text)
        if named:
            named_by_page[page.name] = named
    return named_by_page


def _markdown_pages() -> tuple:
    """The pages a reader reads, without pyproject.toml's tool settings."""
    pages = []
    for path in _prose_files():
        if path.suffix == ".md":
            pages.append(path)
    return tuple(pages)


def test_every_file_listing_in_the_docs_is_lines_of_the_file_it_names():
    """A page that copies a run file shows lines its results come from."""
    pages = _markdown_pages()
    _check_every_file_listing(pages)


def _check_every_file_listing(pages) -> None:
    for page in pages:
        text = page.read_text()
        for name, listing in FILE_LISTING.findall(text):
            path = CHECKOUT / name
            contents = path.read_text()
            assert _holds_in_a_row(contents, listing), (
                f"{page.name} lists lines of {name} the file does not hold "
                "in a row"
            )


def _holds_in_a_row(contents: str, listing: str) -> bool:
    """Whether the listing's lines are a run of the file's lines."""
    file_lines = contents.splitlines()
    listed_lines = listing.splitlines()
    count = len(listed_lines)
    start_count = len(file_lines) - count + 1
    for start in range(start_count):
        if file_lines[start : start + count] == listed_lines:
            return True
    return False

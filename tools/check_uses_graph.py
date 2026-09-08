"""The package uses graph is a partial order (STYLE.md rule 10).

Parnas 1972: "We have a hierarchical structure if a certain relation may
be defined between the modules or programs and that relation is a
partial ordering. The relation we are concerned with is 'uses' or
'depends upon'" (lines 505-511), and a system with the hierarchy can
have its upper levels cut off and still run (518-520). Dijkstra's THE
builds the same order level by level, each level knowing nothing of the
levels above it (dijkstra_the.txt 52-57).

One edge per `import decsim.<package>` line, from the package the file
belongs to. A cycle fails the check; otherwise the levels are printed,
level 0 first, so a reader sees what can be pruned and still run.
"""

import collections
import pathlib
import re
import sys

PACKAGE_IMPORT = re.compile(r"^\s*(?:from|import)\s+decsim\.([a-z_]+)", re.M)


def package_of(path: pathlib.Path, root: pathlib.Path) -> str:
    """The package a file belongs to: its first path part under the root."""
    relative = path.relative_to(root)
    first = relative.parts[0]
    if first.endswith(".py"):
        return first[:-3]
    return first


def source_paths(root: pathlib.Path) -> list:
    """Every module under the root, built sources left out."""
    found = []
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        found.append(path)
    return sorted(found)


def imports_of(path: pathlib.Path, source: str) -> set:
    """The other packages one module imports."""
    text = path.read_text()
    targets = set()
    for match in PACKAGE_IMPORT.finditer(text):
        target = match.group(1)
        if target != source:
            targets.add(target)
    return targets


def read_edges(root: pathlib.Path) -> dict:
    """Every package's imports of other packages."""
    edges = collections.defaultdict(set)
    for path in source_paths(root):
        source = package_of(path, root)
        targets = imports_of(path, source)
        edges[source] |= targets
    return edges


class CycleSearch:
    """A depth-first walk that collects every back edge it meets."""

    def __init__(self, edges: dict) -> None:
        self.edges = edges
        self.state = {}
        self.stack = []
        self.cycles = []

    def visit(self, node: str) -> None:
        """Walk the node's imports, noting a cycle at every back edge."""
        self.state[node] = "open"
        self.stack.append(node)
        imported = self.edges.get(node, ())
        for target in sorted(imported):
            self._follow(target)
        self.stack.pop()
        self.state[node] = "done"

    def _follow(self, target: str) -> None:
        """Note a cycle, or walk a package not yet seen."""
        if self.state.get(target) == "open":
            self._note_cycle(target)
            return
        if target not in self.state:
            self.visit(target)

    def _note_cycle(self, target: str) -> None:
        """The stack from the target back to itself."""
        start = self.stack.index(target)
        walked = self.stack[start:]
        closed = walked + [target]
        self.cycles.append(closed)


def nodes_of(edges: dict) -> list:
    """Every package named as a source or as a target."""
    named = set(edges)
    for targets in edges.values():
        named |= targets
    return sorted(named)


def levels_of(edges: dict, nodes: list) -> dict:
    """Each package's longest path to a leaf; leaves are level 0."""
    level = {}

    def level_of(node: str) -> int:
        if node in level:
            return level[node]
        below = []
        for target in edges.get(node, ()):
            target_level = level_of(target)
            below.append(target_level)
        level[node] = 1 + max(below, default=-1)
        return level[node]

    for node in nodes:
        level_of(node)
    return level


def report_cycles(nodes: list, cycles: list) -> None:
    """Say the graph is not a partial order, and name every cycle."""
    print(f"uses graph: {len(nodes)} packages, NOT a partial order")
    for cycle in cycles:
        joined = " -> ".join(cycle)
        print(f"  cycle: {joined}")


def report_levels(edges: dict, nodes: list) -> None:
    """Say the graph is acyclic, and name the packages of each level."""
    level = levels_of(edges, nodes)
    grouped = collections.defaultdict(list)
    for node, number in level.items():
        grouped[number].append(node)
    print(f"uses graph: {len(nodes)} packages, 0 cycles")
    for number in sorted(grouped):
        named = sorted(grouped[number])
        joined = " ".join(named)
        print(f"  level {number}: {joined}")


def main() -> int:
    """Print the levels, or the cycles and a failure."""
    root = pathlib.Path(sys.argv[1])
    edges = read_edges(root)
    nodes = nodes_of(edges)
    search = CycleSearch(edges)
    for node in nodes:
        if node not in search.state:
            search.visit(node)
    if search.cycles:
        report_cycles(nodes, search.cycles)
        return 1
    report_levels(edges, nodes)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

"""Check that every line of Python does one thing.

The rule, from REWRITE.md: a line calls one function, or reads one
attribute, or does one arithmetic step, or makes one decision. This script
reads each file's syntax tree and reports the statements that break it.

Usage:
    python tools/check_one_action.py <file or directory> ...

Exit status is 1 when anything is reported.

What is reported:
    nested call        a call used as an argument of another call
    call chain         an attribute, index, or call taken from a call's result
    inline conditional an `x if c else y` expression
    busy comprehension a comprehension that both filters and calls
    long condition     an `and` / `or` with more than two operands
    walrus             an assignment hiding inside an expression

Small built-ins that read as part of a phrase are allowed as arguments:
len, str, repr, int, float, bool, tuple, list, set, dict, sorted, range,
enumerate, zip, min, max, abs, isinstance, type, next, iter, reversed, any,
all, sum, getattr, hasattr. Calls inside f-strings and lambdas are
allowed, since a log line or a scheduled action is one thing.
"""

import ast
import pathlib
import sys

ALLOWED_INNER_CALLS = {
    "len", "str", "repr", "int", "float", "bool", "tuple", "list", "set",
    "dict", "sorted", "range", "enumerate", "zip", "min", "max", "abs",
    "isinstance", "type", "next", "iter", "reversed", "any", "all", "sum",
    "getattr", "hasattr",
}


class Finding:
    def __init__(self, path, line, kind, text):
        self.path = path
        self.line = line
        self.kind = kind
        self.text = text

    def __str__(self):
        return f"{self.path}:{self.line}: {self.kind}: {self.text}"


def call_name(node):
    """The plain name a call is made through, or None for anything fancier."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def is_allowed_inner(node):
    name = call_name(node)
    return name in ALLOWED_INNER_CALLS


class Checker(ast.NodeVisitor):
    def __init__(self, path, source_lines):
        self.path = path
        self.source_lines = source_lines
        self.findings = []
        self.inside_fstring = 0
        self.inside_lambda = 0

    def report(self, node, kind):
        text = self.source_lines[node.lineno - 1].strip()
        self.findings.append(Finding(self.path, node.lineno, kind, text))

    def visit_JoinedStr(self, node):
        self.inside_fstring += 1
        self.generic_visit(node)
        self.inside_fstring -= 1

    def visit_Lambda(self, node):
        self.inside_lambda += 1
        self.generic_visit(node)
        self.inside_lambda -= 1

    def quiet(self):
        return self.inside_fstring > 0 or self.inside_lambda > 0

    def visit_Call(self, node):
        if not self.quiet():
            for argument in node.args:
                self.check_argument(node, argument)
            for keyword in node.keywords:
                self.check_argument(node, keyword.value)
            self.check_chain_on_call(node.func, node)
        self.generic_visit(node)

    def check_argument(self, call, argument):
        if isinstance(argument, ast.Starred):
            argument = argument.value
        if isinstance(argument, ast.Call) and not is_allowed_inner(argument):
            self.report(call, "nested call")

    def check_chain_on_call(self, func, call):
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call):
            self.report(call, "call chain")

    def visit_Attribute(self, node):
        if not self.quiet() and isinstance(node.value, ast.Call):
            self.report(node, "call chain")
        self.generic_visit(node)

    def visit_Subscript(self, node):
        if not self.quiet() and isinstance(node.value, ast.Call):
            self.report(node, "call chain")
        self.generic_visit(node)

    def visit_IfExp(self, node):
        if not self.quiet():
            self.report(node, "inline conditional")
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        if not self.quiet() and len(node.values) > 2:
            self.report(node, "long condition")
        self.generic_visit(node)

    def visit_NamedExpr(self, node):
        self.report(node, "walrus")
        self.generic_visit(node)

    def check_comprehension(self, node):
        filters = any(generator.ifs for generator in node.generators)
        calls_in_element = any(
            isinstance(inner, ast.Call) for inner in ast.walk(node.elt))
        if filters and calls_in_element and not self.quiet():
            self.report(node, "busy comprehension")
        self.generic_visit(node)

    visit_ListComp = check_comprehension
    visit_SetComp = check_comprehension
    visit_GeneratorExp = check_comprehension

    def visit_DictComp(self, node):
        filters = any(generator.ifs for generator in node.generators)
        value_nodes = list(ast.walk(node.key)) + list(ast.walk(node.value))
        calls = any(isinstance(inner, ast.Call) for inner in value_nodes)
        if filters and calls and not self.quiet():
            self.report(node, "busy comprehension")
        self.generic_visit(node)


def check_file(path):
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    checker = Checker(path, source.splitlines())
    checker.visit(tree)
    return checker.findings


def python_files(targets):
    for target in targets:
        path = pathlib.Path(target)
        if path.is_dir():
            yield from sorted(path.rglob("*.py"))
        else:
            yield path


def main(arguments):
    findings = []
    for path in python_files(arguments):
        findings.extend(check_file(path))
    for finding in findings:
        print(finding)
    file_count = len({finding.path for finding in findings})
    print(f"{len(findings)} findings in {file_count} files")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

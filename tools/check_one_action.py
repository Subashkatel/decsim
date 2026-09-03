"""Check that every line of Python does one thing (REWRITE.md, rule 1).

Usage:
    python tools/check_one_action.py <file or directory> ...

Exit status is 1 when anything is reported. Vendored files, dependency
folders and scratch folders are skipped (EXCLUDED_PARTS).

What is reported, by kind:
    nested call         a call, other than an allowed built-in, anywhere
                        inside another call's arguments
    busy argument       arithmetic, a comparison, or a boolean inside a
                        call's arguments
    call chain          an attribute, an index, or a call taken from a
                        call's result
    inline conditional  an `x if c else y` expression
    long condition      an `and` or `or` with more than two operands
    mixed condition     `and` and `or` in one expression
    busy comprehension  a comprehension element that does two things, or
                        one that calls while the comprehension filters
    busy lambda         a lambda whose body is more than one call on plain
                        values, or more than one f-string of plain values
    call in f-string    a call, other than an allowed built-in, inside an
                        f-string
    walrus              an assignment hiding inside an expression
    long function       a function longer than MAX_FUNCTION_LINES
    deep nesting        blocks nested deeper than MAX_BLOCK_DEPTH
    wide state          a class whose __init__ sets more attributes than
                        MAX_ATTRIBUTES

Two kinds are reports, not failures: long function and wide state. The
40 lines is Google's prompt to think, not a limit, and the six
attributes is this project's own number (REWRITE.md rule 1); the
checklist row says why a function or class is that size. They are
printed under their own heading and do not set the exit code.
"""

import ast
import pathlib
import sys

ALLOWED_INNER_CALLS = frozenset(
    {
        "len",
        "str",
        "repr",
        "int",
        "float",
        "bool",
        "tuple",
        "list",
        "set",
        "dict",
        "sorted",
        "range",
        "enumerate",
        "zip",
        "min",
        "max",
        "abs",
        "isinstance",
        "next",
        "iter",
        "reversed",
        "any",
        "all",
        "sum",
    }
)
EXCLUDED_PARTS = frozenset(
    {
        ".pydeps",
        ".venv",
        "tmp",
        "archive",
        "__pycache__",
        "stimcircuits",
    }
)
MAX_FUNCTION_LINES = 40
MAX_BLOCK_DEPTH = 2
MAX_ATTRIBUTES = 6

BLOCK_STATEMENTS = (ast.For, ast.While, ast.If, ast.With, ast.Try)
ARITHMETIC = (ast.BinOp, ast.Compare, ast.BoolOp)
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)


class Finding:
    """One reported line."""

    def __init__(self, path, line, kind, text):
        self.path = path
        self.line = line
        self.kind = kind
        self.text = text

    def __str__(self):
        return f"{self.path}:{self.line}: {self.kind}: {self.text}"


def is_allowed_call(node):
    """True for a bare call to one of the allowed built-ins."""
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Name):
        return False
    return node.func.id in ALLOWED_INNER_CALLS


def is_plain_value(node):
    """True for a name, a constant, an attribute walk, or an index of one."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return is_plain_value(node.value)
    if isinstance(node, ast.Subscript):
        return is_plain_value(node.value)
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.operand, ast.Constant)
    return False


def calls_inside(node):
    """Every call in a subtree that is not an allowed built-in."""
    found = []
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call):
            continue
        if is_allowed_call(inner):
            continue
        found.append(inner)
    return found


def arithmetic_inside(node):
    """Every arithmetic, comparison, or boolean node in a subtree."""
    found = []
    for inner in ast.walk(node):
        if isinstance(inner, ARITHMETIC):
            found.append(inner)
    return found


def leaf_operands(node):
    """The operands of a boolean expression, flattened across nesting."""
    if not isinstance(node, ast.BoolOp):
        return [node]
    leaves = []
    for value in node.values:
        if shares_operator(node, value):
            nested = leaf_operands(value)
            leaves.extend(nested)
        else:
            leaves.append(value)
    return leaves


def shares_operator(outer, inner):
    """True when an inner boolean expression uses the outer one's operator."""
    if not isinstance(inner, ast.BoolOp):
        return False
    return type(inner.op) is type(outer.op)


def mixes_operators(node):
    """True when `and` and `or` meet inside one boolean expression."""
    for value in node.values:
        if not isinstance(value, ast.BoolOp):
            continue
        if not shares_operator(node, value):
            return True
    return False


def block_depth(node, depth=0):
    """The deepest block nesting below a node."""
    deepest = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, FUNCTIONS):
            continue
        child_depth = depth
        if isinstance(child, BLOCK_STATEMENTS):
            child_depth = depth + 1
        below = block_depth(child, child_depth)
        deepest = max(deepest, below)
    return deepest


def init_method(class_node):
    """The class's __init__, or None."""
    for statement in class_node.body:
        if not isinstance(statement, FUNCTIONS):
            continue
        if statement.name == "__init__":
            return statement
    return None


def self_attributes_assigned(function_node):
    """The attribute names a function assigns on self."""
    names = set()
    for inner in ast.walk(function_node):
        if not is_self_store(inner):
            continue
        names.add(inner.attr)
    return names


def is_self_store(node):
    """True for `self.<name> = ...`."""
    if not isinstance(node, ast.Attribute):
        return False
    if not isinstance(node.ctx, ast.Store):
        return False
    if not isinstance(node.value, ast.Name):
        return False
    return node.value.id == "self"


def call_arguments(call):
    """Positional and keyword argument expressions of a call."""
    arguments = list(call.args)
    for keyword in call.keywords:
        arguments.append(keyword.value)
    return arguments


def comprehension_elements(node):
    """The expressions a comprehension builds per item."""
    if isinstance(node, ast.DictComp):
        return [node.key, node.value]
    return [node.elt]


def filter_count(node):
    """How many `if` clauses a comprehension carries."""
    count = 0
    for generator in node.generators:
        count += len(generator.ifs)
    return count


def action_count(elements):
    """How many calls and arithmetic steps a list of expressions holds."""
    count = 0
    for element in elements:
        calls = calls_inside(element)
        arithmetic = arithmetic_inside(element)
        count += len(calls)
        count += len(arithmetic)
    return count


def is_simple_fstring(node):
    """True when an f-string holds only plain values and allowed built-ins."""
    for value in node.values:
        if not isinstance(value, ast.FormattedValue):
            continue
        expression = value.value
        if is_plain_value(expression):
            continue
        if is_allowed_call(expression):
            continue
        return False
    return True


def is_simple_lambda(node):
    """True when a lambda is one call on plain values, or one f-string."""
    body = node.body
    if isinstance(body, ast.JoinedStr):
        return is_simple_fstring(body)
    if not isinstance(body, ast.Call):
        return is_plain_value(body)
    for argument in call_arguments(body):
        if not is_plain_value(argument):
            return False
    return True


class Checker(ast.NodeVisitor):
    """Walks one file and collects findings."""

    def __init__(self, path, source_lines):
        self.path = path
        self.source_lines = source_lines
        self.findings = []
        self.lambda_depth = 0

    def report(self, node, kind):
        """Record one finding at the node's line."""
        text = self.source_lines[node.lineno - 1]
        stripped = text.strip()
        finding = Finding(self.path, node.lineno, kind, stripped)
        self.findings.append(finding)

    def visit_FunctionDef(self, node):
        """Report long functions and deep nesting."""
        length = node.end_lineno - node.lineno + 1
        if length > MAX_FUNCTION_LINES:
            self.report(node, "long function")
        depth = block_depth(node)
        if depth > MAX_BLOCK_DEPTH:
            self.report(node, "deep nesting")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """Same rules as a plain function."""
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        """Report a class whose __init__ sets too many attributes."""
        init = init_method(node)
        if init is not None:
            names = self_attributes_assigned(init)
            if len(names) > MAX_ATTRIBUTES:
                self.report(node, "wide state")
        self.generic_visit(node)

    def visit_Call(self, node):
        """Check a call's arguments and what it is called on."""
        if self.lambda_depth == 0:
            self.check_arguments(node)
        if isinstance(node.func, ast.Call):
            self.report(node, "call chain")
        self.generic_visit(node)

    def check_arguments(self, call):
        """Report nested calls and arithmetic among a call's arguments."""
        for argument in call_arguments(call):
            self.check_argument(call, argument)

    def check_argument(self, call, argument):
        """Report one argument that does more than name a value."""
        if isinstance(argument, ast.Starred):
            argument = argument.value
        if isinstance(argument, ast.Lambda):
            return
        if isinstance(argument, COMPREHENSIONS):
            return
        if is_allowed_call(argument):
            self.check_arguments(argument)
            return
        if calls_inside(argument):
            self.report(call, "nested call")
        if arithmetic_inside(argument):
            self.report(call, "busy argument")

    def visit_Attribute(self, node):
        """Report an attribute taken from a call's result."""
        if isinstance(node.value, ast.Call):
            self.report(node, "call chain")
        self.generic_visit(node)

    def visit_Subscript(self, node):
        """Report an index taken from a call's result."""
        if isinstance(node.value, ast.Call):
            self.report(node, "call chain")
        self.generic_visit(node)

    def visit_IfExp(self, node):
        """Report an inline conditional."""
        self.report(node, "inline conditional")
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        """Report long or mixed boolean expressions."""
        leaves = leaf_operands(node)
        if len(leaves) > 2:
            self.report(node, "long condition")
        if mixes_operators(node):
            self.report(node, "mixed condition")
        self.generic_visit(node)

    def visit_NamedExpr(self, node):
        """Report a walrus."""
        self.report(node, "walrus")
        self.generic_visit(node)

    def visit_Lambda(self, node):
        """Report a lambda that does more than one thing."""
        if not is_simple_lambda(node):
            self.report(node, "busy lambda")
        self.lambda_depth += 1
        self.generic_visit(node)
        self.lambda_depth -= 1

    def visit_JoinedStr(self, node):
        """Report calls inside an f-string outside a lambda."""
        if self.lambda_depth == 0:
            for call in calls_inside(node):
                self.report(call, "call in f-string")
        self.generic_visit(node)

    def check_comprehension(self, node):
        """Report a comprehension element that does two things."""
        elements = comprehension_elements(node)
        actions = action_count(elements)
        filters = filter_count(node)
        if actions > 1:
            self.report(node, "busy comprehension")
        elif actions == 1 and filters > 0:
            self.report(node, "busy comprehension")
        self.generic_visit(node)

    def visit_ListComp(self, node):
        """Check a list comprehension."""
        self.check_comprehension(node)

    def visit_SetComp(self, node):
        """Check a set comprehension."""
        self.check_comprehension(node)

    def visit_GeneratorExp(self, node):
        """Check a generator expression."""
        self.check_comprehension(node)

    def visit_DictComp(self, node):
        """Check a dict comprehension."""
        self.check_comprehension(node)


def check_file(path):
    """All findings for one file."""
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    source_lines = source.splitlines()
    checker = Checker(path, source_lines)
    checker.visit(tree)
    return checker.findings


def is_excluded(path):
    """True when any part of the path names a skipped folder or file."""
    for part in path.parts:
        if part in EXCLUDED_PARTS:
            return True
    return False


def files_under(target):
    """The Python files a target names, one file or a whole folder."""
    path = pathlib.Path(target)
    if not path.is_dir():
        return [path]
    found = path.rglob("*.py")
    return sorted(found)


def python_files(targets):
    """Every Python file under the targets, minus the excluded parts."""
    candidates = []
    for target in targets:
        found = files_under(target)
        candidates.extend(found)
    for candidate in candidates:
        if is_excluded(candidate):
            continue
        yield candidate


REPORT_ONLY_KINDS = frozenset({"long function", "wide state"})


def main(arguments):
    """Check every file named, print the findings, return the exit code."""
    findings = []
    for path in python_files(arguments):
        file_findings = check_file(path)
        findings.extend(file_findings)
    failures = []
    reports = []
    for finding in findings:
        if finding.kind in REPORT_ONLY_KINDS:
            reports.append(finding)
        else:
            failures.append(finding)
    for finding in failures:
        print(finding)
    failure_paths = {finding.path for finding in failures}
    print(f"{len(failures)} findings in {len(failure_paths)} files")
    if reports:
        print("reports (size prompts, recorded on the checklist row):")
    for finding in reports:
        print(finding)
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    exit_code = main(sys.argv[1:])
    sys.exit(exit_code)

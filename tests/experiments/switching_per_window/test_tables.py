"""The switching-per-window tables' window classes and the plot rule."""

import importlib.util
import json
import math
import pathlib
import sys

_THIS_FILE = pathlib.Path(__file__)
_TEST_FILE = _THIS_FILE.resolve()
REPOSITORY_ROOT = _TEST_FILE.parents[3]
SCRIPT_FOLDER = REPOSITORY_ROOT / "experiments" / "switching_per_window"


def script_module(name: str):
    """A script of the experiment folder, imported by path."""
    script_path = SCRIPT_FOLDER / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tables = script_module("tables")


def test_neighbouring_wrong_windows_pair_left_to_right():
    spans = [(1, 7, True), (8, 14, True), (15, 21, True), (22, 28, False)]

    classes = tables._answer_classes(spans)

    assert classes == ["wrong_paired", "wrong_paired", "wrong_alone", "right"]


def test_a_window_beside_rounds_union_find_skipped_is_partner_unseen():
    # an escalated window's own rounds end inside its larger window
    spans = [(1, None, True), (22, 28, True), (29, 35, True)]

    classes = tables._answer_classes(spans)

    assert classes == [
        "wrong_partner_unseen",
        "wrong_paired",
        "wrong_paired",
    ]


def test_union_find_alone_pairs_on_switching_spans():
    # union-find alone wrong on 57-63, 64-70 and 71-77; switching
    # escalated 71-91, so 71-77, 78-84 and 85-91 fold into one span
    partition = ((57, 63), (64, 70), (71, 91))
    wrong_windows = {57, 64, 71}
    rows = []
    for commit_lo in (57, 64, 71, 78, 85):
        is_wrong = commit_lo in wrong_windows
        commit_hi = commit_lo + 6
        row = window_row(commit_lo, commit_hi, int(is_wrong))
        rows.append(row)

    spans = tables._folded_spans(partition, rows, ("shot",))
    classes = tables._answer_classes(spans)

    assert classes == ["wrong_paired", "wrong_paired", "wrong_alone"]


def test_a_share_holds_its_total_and_a_failure_rate_its_count():
    plot = script_module("plot")

    share, _low, _high = plot.shown_share((11, 5248), plot.TOTAL_RULE)
    hidden, _low, _high = plot.shown_share((11, 5248), plot.FAILURE_RULE)

    expected = 100 * 11 / 5248
    assert math.isclose(share, expected)
    assert math.isnan(hidden)


def window_row(commit_lo: int, commit_hi: int, answer: int) -> dict:
    """A union-find-alone window_outcomes row with label 0."""
    return {
        "commit_lo": str(commit_lo),
        "commit_hi": str(commit_hi),
        "answer": json.dumps(str(answer)),
        "label": json.dumps("0"),
    }

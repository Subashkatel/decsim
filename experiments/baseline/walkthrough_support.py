"""Helpers for the baseline walkthrough notebook: ticks as microseconds, aligned text tables."""

from decsim.config import TICKS_PER_US


def microseconds(ticks: int) -> float:
    """Simulator ticks as microseconds, three decimals."""
    return round(ticks / TICKS_PER_US, 3)


def row_as_strings(row: dict, columns: list) -> list:
    """One table row: the row's value under each column, as text."""
    values = []
    for column in columns:
        value = row.get(column, "")
        values.append(str(value))
    return values


def column_widths(columns: list, cells: list) -> list:
    """Width of each column: its header or its widest cell."""
    widths = []
    for index, column in enumerate(columns):
        width = len(column)
        for row in cells:
            width = max(width, len(row[index]))
        widths.append(width)
    return widths


def aligned_line(values: list, widths: list) -> str:
    """One printed line: each value padded to its column width."""
    padded = []
    for value, width in zip(values, widths):
        padded.append(value.ljust(width))
    return "  ".join(padded)


def table(rows: list, columns: list) -> None:
    """Print rows (dicts) as an aligned text table with the given columns."""
    cells = []
    for row in rows:
        cells.append(row_as_strings(row, columns))
    widths = column_widths(columns, cells)
    rules = []
    for width in widths:
        rules.append("-" * width)
    print(aligned_line(columns, widths))
    print(aligned_line(rules, widths))
    for row in cells:
        print(aligned_line(row, widths))

"""The plug-in tables' lookup: a name, a row, or a refusal that lists them.

Every package owns a table of the classes its yaml section can name
(DECODERS, ROUND_STORES, WINDOWING_SCHEMES and the rest, each in its own
package's settings module). One function reads them all, so a kind that
is not a row is refused the same way everywhere, with the rows printed.
The table is the last resort, not the authority: sinter resolves a
caller's own object first and its table second
(sinter/_collection/_mux_sampler.py:33-40).

This module holds nothing else, so a settings module can reach the
lookup without importing the root that aggregates the sections.
"""


def row(table: dict, section: str, kind):
    """The table row a section's kind names; a kind off the table is refused."""
    if kind not in table:
        rows = sorted(table)
        raise ValueError(
            f"{section} {kind!r} is not a row of its table; the rows are {rows}"
        )
    return table[kind]

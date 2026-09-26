"""The plug-in tables' lookup: a name, a row, or a refusal that lists them.

Every package owns a table of the classes its yaml section can name
(DECODERS, SYNDROME_BUFFERS, WINDOWING_SCHEMES and the rest, each in its own
package's settings module). One function reads them all, so a kind that
is not a row is refused the same way everywhere, with the rows printed.
The table is the last resort, not the authority: sinter resolves a
caller's own object first and its table second
(sinter/_collection/_mux_sampler.py:33-40).

A row's own yaml keys are split off its section here too, so every
table hands a row its keys the same way.

This module holds nothing else, so a settings module can reach the
lookup without importing the root that aggregates the sections.
"""

import dataclasses
from collections.abc import Mapping


def row(table: dict, section: str, kind):
    """The table row a section's kind names; a kind off the table is refused."""
    if kind not in table:
        rows = sorted(table)
        raise ValueError(
            f"{section} {kind!r} is not a row of its table; the rows are {rows}"
        )
    return table[kind]


def row_settings(
    row_class, section_name: str, section: Mapping, other_keys, *context
):
    """The row's own settings, read from the section's keys it declares.

    A row with keys of its own declares a nested frozen dataclass
    `Settings`: its fields are the keys, its `from_yaml` reads them, and
    the row's constructor takes the record as `settings`. That is gem5's
    shape, a SimObject's parameters declared on its class
    (src/mem/SimpleMemory.py:43-53) and handed to its constructor as one
    Params record (src/mem/simple_mem.cc:53), and CUDA-Q QEC's, a code or
    a decoder built by name with its own options map
    (libs/qec/include/cudaq/qec/code.h:98, get_code at 257). other_keys
    are the keys the section reads for itself or for another row; any
    key that is neither those nor the row's is refused by name, as gem5
    refuses a parameter its class does not declare
    (src/python/m5/SimObject.py:932-936). context rides to from_yaml
    (the decoder tiers hand it the run's clocks). None for a row with no
    Settings, including no row at all.
    """
    settings_class = getattr(row_class, "Settings", None)
    declared_keys = row_keys(row_class)
    _refuse_undeclared_keys(section_name, section, other_keys, declared_keys)
    if settings_class is None:
        return None
    own_section = {}
    for key in declared_keys:
        if key in section:
            own_section[key] = section[key]
    return settings_class.from_yaml(own_section, *context)


def row_keys(row_class) -> tuple:
    """The yaml keys a row declares: its Settings fields, or none."""
    settings_class = getattr(row_class, "Settings", None)
    if settings_class is None:
        return ()
    names = []
    for field in dataclasses.fields(settings_class):
        names.append(field.name)
    return tuple(names)


def _refuse_undeclared_keys(
    section_name: str, section: Mapping, other_keys, declared_keys: tuple
) -> None:
    """A key neither the section nor the row declares is refused by name."""
    known_keys = list(other_keys) + list(declared_keys)
    unknown = set(section) - set(known_keys)
    if not unknown:
        return
    listed = sorted(unknown)
    raise ValueError(
        f"{section_name} does not know {listed}; its keys are {known_keys}"
    )

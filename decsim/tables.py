"""The plug-in tables' lookup: a name, a row, or a refusal that lists them.

Every package owns a table of the classes its yaml section can name
(DECODERS, SYNDROME_BUFFERS, WINDOWING_SCHEMES and the rest, each in its own
package's settings module). One function reads them all, so a kind that
is not a row is refused the same way everywhere, with the rows printed.
The table is the last resort, not the authority: sinter resolves a
caller's own object first and its table second
(sinter/_collection/_mux_sampler.py:33-40).

A row's own yaml keys are split off its section here too, so every
table hands a row its keys the same way, and every section refuses a key
it does not declare, or a required key it lacks, with the same sentence.

This module holds nothing else, so a settings module can reach the
lookup without importing the root that aggregates the sections.
"""

import dataclasses
from collections.abc import Mapping


def row(table: dict, section: str, kind):
    """The table row a section's kind names; a kind off the table is refused.

    A yaml list or mapping in the kind's place is off the table too, and
    is refused with the same sentence rather than by the dict's own
    TypeError, since neither can be a key.
    """
    if isinstance(kind, (list, Mapping)) or kind not in table:
        rows = sorted(table)
        raise ValueError(
            f"{section} {kind!r} is not a row of its table; the rows are {rows}"
        )
    return table[kind]


def row_settings(
    row_class, section_name: str, section: Mapping, other_keys, *context
):
    """The row's own settings, read from the section's keys it declares.

    A row with keys of its own declares them on a nested `Settings`,
    whose contract is decsim/ports.py RowSettings. other_keys are the
    keys the section reads for itself or for another row; any key that
    is neither those nor the row's is refused by name, as gem5 refuses
    a parameter its class does not declare
    (src/python/m5/SimObject.py:932-936). context rides to from_yaml
    (the decoder tiers hand it the run's clocks and their own section
    name, weak_decoder or strong_decoder, which a refusal names, since
    the two tiers share their rows' keys). None for a row with no
    Settings, including no row at all.
    """
    settings_class = getattr(row_class, "Settings", None)
    declared_keys = row_keys(row_class)
    known_keys = list(other_keys) + list(declared_keys)
    refuse_unknown_keys(section_name, section, known_keys)
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


def record_fields(
    record_class, section_name: str, section: Mapping, other_keys
) -> dict:
    """The section's values of a settings record's fields, by field name.

    other_keys are the keys the section reads for itself, its kind; any
    key that is neither those nor a field is refused by name.
    """
    fields = dataclasses.fields(record_class)
    field_names = tuple(field.name for field in fields)
    known_keys = tuple(other_keys) + field_names
    refuse_unknown_keys(section_name, section, known_keys)
    values = {}
    for key in field_names:
        if key in section:
            values[key] = section[key]
    return values


def required_fields(record_class) -> tuple:
    """The names of a settings record's fields that have no default."""
    names = []
    for field in dataclasses.fields(record_class):
        has_default = field.default is not dataclasses.MISSING
        has_factory = field.default_factory is not dataclasses.MISSING
        if not has_default and not has_factory:
            names.append(field.name)
    return tuple(names)


def refuse_unknown_keys(
    section_name: str, section: Mapping, known_keys
) -> None:
    """A key the section does not declare is refused by name.

    gem5 refuses a parameter its class does not declare
    (src/python/m5/SimObject.py:932-936), so a misspelt key is refused
    rather than left at its default.
    """
    unknown = set(section) - set(known_keys)
    if not unknown:
        return
    # A yaml mapping may mix integer and string keys, so they sort as text.
    listed = sorted(unknown, key=str)
    raise ValueError(
        f"{section_name} does not know {listed}; its keys are "
        f"{list(known_keys)}"
    )


def refuse_missing_keys(
    section_name: str, section: Mapping, required_keys
) -> None:
    """A key the section cannot do without is refused by name when absent."""
    missing = set(required_keys) - set(section)
    if not missing:
        return
    listed = sorted(missing)
    raise ValueError(
        f"{section_name} needs the keys {listed}; configs/reference.yaml "
        "holds every key with its unit"
    )

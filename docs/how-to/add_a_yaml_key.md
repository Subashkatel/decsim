[decsim docs](../README.md) › [How-to guides](README.md)

# How to add a yaml key

You want a knob a config file can set. A key belongs to exactly one
settings section, and adding one touches its section's settings module,
`configs/reference.yaml`, and the test of its refusal.

## 1. Add the field to the settings record

Find the package that owns the section. `decsim/settings.py`, `SECTIONS`,
lists every section in the order the root reads them, and each section's
record is built by its own package's `settings.py`.

Add the field to the dataclass, with a default, read it in the
section's `from_yaml`, and add its name to the tuple of keys the
section shares (`WINDOWS_KEYS` here; `DECODER_KEYS`, `QPU_KEYS` and
`WORKLOAD_KEYS` beside the other tables). A key off that tuple is handed to the kind's row, and a row
that does not declare it refuses it:

```python
@dataclasses.dataclass(frozen=True)
class WindowSettings:
    ...
    my_key: int = 0
```

A key only one row reads is that row's, not the section's: it goes on
the row's nested `Settings` instead
([How to add a row to a table](add_a_table_row.md)), as `union_find`'s
`weight_step` does.

Name the field so the name carries the unit: a duration ends in
`_microseconds` or `_ticks`, a count ends in `_count`, a number of
rounds ends in `_rounds`, a cycle count ends in `_cycles`
(`STYLE.md` rule 2). Then a reader never has to guess.

## 2. Check it at the boundary, once

The yaml is where input enters decsim, so this is one of the two places
a check belongs (`STYLE.md` rule 4). The other is a component refusing a
call that would corrupt a result. Check the value in `from_yaml`,
loudly, with a message that reads as a sentence, and raise `ValueError`.
Do not check it again anywhere inside the machine: a component trusts
what its callers send.

Refuse an unknown key in your section too, by name, with the known keys
listed. Some sections already do:

```
decsim: my.yaml: decoder_manager does not know ['nonsense']; its keys
are ['bulk_strong', 'clock', 'dispatch_cycles']
```

The sections with a table of rows (`qpu`, both syndrome buffers,
`windows`, both decoder tiers, `workload`) refuse a key that neither the
section nor the kind's row declares, in that shape, through
`decsim/tables.py`'s `row_settings`. Not every section does yet: the
`controller` and `pauli_frame` sections ignore a key they do not know,
so a typo there runs the default without saying so. A section no
package owns is always refused, with the fifteen sections listed, and
so is a yaml that leaves out a section every run needs.

## 3. Document it in `configs/reference.yaml`, in the same commit

`configs/reference.yaml` is the documentation of the yaml surface: every
key the layer reads, with a comment saying what it means and, where the
value came from a paper or a reference implementation, which one. A key
that is not in that file is a key nobody can find.

This is not optional, and no test does it for you: a key read with a
default and left out of the file passes every test. What
`tests/experiments/test_yaml_surface.py` does catch is the other drift,
a key the file still carries that a section which refuses unknown keys
no longer reads. Its `test_unknown_algorithms_and_stale_keys_fail_loudly`
shows the shape.

## 4. Run the checks

```bash
tools/check.sh
python -m pytest tests
```

A yaml key is a settings change, not a behaviour change, so a default
that keeps the old behaviour moves no result of any existing run.

## Read next

- [The yaml surface](../reference/yaml.md): how the yaml layer is put together.
- [How to add a row to a table](add_a_table_row.md): when the knob is a whole component.
- `configs/reference.yaml`: the file you are adding to.

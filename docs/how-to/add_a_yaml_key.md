[decsim docs](../README.md) › [How-to guides](README.md)

# How to add a yaml key

You want a knob a config file can set. A key belongs to exactly one
settings section, and adding one touches three files.

## 1. Add the field to the settings record

Find the package that owns the section. `decsim/settings.py`, `SECTIONS`,
lists every section in the order the root reads them, and each section's
record is built by its own package's `settings.py`.

Add the field to the dataclass, with a default, and read it in the
section's `from_yaml`:

```python
@dataclasses.dataclass(frozen=True)
class WindowSettings:
    ...
    my_key: int = 0
```

Name the field so the name carries the unit: a duration ends in
`_microseconds` or `_ticks`, a count ends in `_count`, a number of
rounds ends in `_rounds`, a cycle count ends in `_cycles`
(`STYLE.md` rule 2). Then a reader never has to guess.

## 2. Check it at the boundary, once

The yaml is where input enters decsim, so this is one of the two places
a check belongs (`STYLE.md` rule 4). Check the value in `from_yaml`,
loudly, with a message that reads as a sentence, and raise `ValueError`.
Do not check it again anywhere inside the machine: a component trusts
what its callers send.

Refuse an unknown key in your section too, by name, with the known keys
listed. Some sections already do:

```
decsim: my.yaml: decoder_manager does not know ['nonsense']; its keys
are ['bulk_strong', 'clock', 'dispatch_cycles']
```

Not every section does yet. A key the `weak_decoder` section does not
know is currently ignored in silence, so a typo there runs the default
without saying so. A section no package owns is always refused, with the
fifteen sections listed.

## 3. Document it in `configs/reference.yaml`, in the same commit

`configs/reference.yaml` is the documentation of the yaml surface: every
key the layer reads, with a comment saying what it means and, where the
value came from a paper or a reference implementation, which one. A key
that is not in that file is a key nobody can find.

This is not optional and it is not a convention:
`tests/front/test_yaml_surface.py` fails when the file and the readers
drift apart, and it names the drift. Read its
`test_unknown_algorithms_and_stale_keys_fail_loudly` to see the shape.

## 4. Run the checks

```bash
DECSIM_PYTHON=/path/to/decsim/.venv/bin/python \
DECSIM_PYDEPS=/path/to/decsim/.pydeps tools/check.sh
python -m pytest tests
```

A yaml key is a settings change, not a behaviour change, so a default
that keeps the old behaviour moves no result of any existing run.

## Read next

- [The yaml surface](../reference/yaml.md): how the yaml layer is put together.
- [How to add a row to a table](add_a_table_row.md): when the knob is a whole component.
- `configs/reference.yaml`: the file you are adding to.

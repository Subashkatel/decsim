[decsim docs](../README.md) › [How-to guides](README.md)

# How to add a burst detector

You have a way to tell a burst of errors from the usual noise, and you
want decsim to run it beside the decoders. This is the whole job: one
folder and one table line.

## 1. Make the folder

Make one folder under `decsim/burst_detectors/`, named after your row,
the way `event_count/` and `masked_regional_cusum/` are. It holds an
`__init__.py` with a one-line docstring, and a `detector.py` with your
class. Put any part only your row uses, such as its own threshold rule,
in a module beside `detector.py`.

## 2. Write the class

A burst detector fills the `BurstDetector` port (`decsim/ports.py`;
[The ports](../reference/ports.md) lists it). The class has this shape:

```python
class MyBurstDetector:
    """One sentence saying what it fires on, and its source."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The row's own keys."""

        @classmethod
        def from_yaml(cls, section, clocks):
            """Read and check the keys once, at the yaml."""

    def __init__(self, settings, engine, circuits, round_period_microseconds):
        """Calibrate from each operation's circuit."""

    def observe_round(self, operation_id, round_index, events):
        """Read one round's detection events, in formation order."""

    def is_burst_window(self, window):
        """Whether a flag published by now meets the window."""

    def with_burst_priors(self, window, model):
        """The window's model with the flagged region's priors raised."""
```

`circuits` maps each operation id to its circuit and its round count.
The shared modules of `decsim/burst_detectors/` do most of the work:

- `layout.py`: `Layout.from_circuit` gives the check positions, a
  round's events per position (`position_counts`), the usual rates and
  the faults behind them.
- `flag_log.py`: keep one `FlagLog` per operation. Record each round's
  `Flag`, or `None` on a quiet round, with the tick it is published at.
  `publication_tick` charges your row's own cost on its clock.
- `burst_windows.py`: `BurstWindows` answers `is_burst_window` and
  `with_burst_priors` for you. It needs each operation to keep its
  `FlagLog` as `flags` and to name a flag's region with
  `region_of(episode)`, a `BurstRegion` (`burst_region.py`).

Declare `trace` as your own `_TraceSources` holding a `round_flagged`
source (`decsim/trace_source.py`), and fire it on every round you flag.
The run's burst flags and the shot columns read it.

## 3. Add the row and the yaml keys

Add one line to `BURST_DETECTORS` in `decsim/burst_detectors/settings.py`.
Your keys are the fields of `Settings`, written in the `burst_detector`
section beside `kind` (`decsim/tables.py`, `row_settings`). Read a whole
count with `config.whole_count` and an on-off key with `config.boolean`.
Document every key in `configs/reference.yaml` in the same commit.

Nothing else in decsim changes. The build hands your class the
circuits, and the escalation reaches it only through the port.

## 4. Test it

`tests/burst_detectors/burst_rounds.py` builds the d = 5 memory, feeds
it round by round and makes windows. Hold your row against a reference
written straight from the method's rules, as
`tests/burst_detectors/written_rules.py` is for `masked_regional_cusum`,
give each of your modules its own test file, and add your row to
`test_each_row_reports_every_round_it_fires_on` in
`tests/burst_detectors/test_settings.py`.

## 5. Compare it

Add one file to `configs/burst_detectors_compared/` that changes only
`burst_detector`, then follow
[How to compare burst detectors](compare_burst_detectors.md).

## Read next

- [The ports](../reference/ports.md): the `BurstDetector` port in full.
- [How to add a row to a table](add_a_table_row.md): the general recipe.
- [How to add a yaml key](add_a_yaml_key.md): checking a key once at
  the boundary.

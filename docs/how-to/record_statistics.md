[decsim docs](../README.md) › [How-to guides](README.md)

# How to record statistics of your own part

A part keeps its own statistics: it declares them on its class and
updates them where the event happens, as a gem5 SimObject does with its
stats. The machine records them at the part's path, and they appear in
`shot_stats.csv`, `stat_values.csv` and `stat_summary.csv` with no
other change.

## 1. Declare them on the class

```python
import decsim.decoders.decoders as decoders
import decsim.statistics.kinds as statistic_kinds
import decsim.statistics.registry as statistics_registry


class MyDecoder(decoders.PresetLatencyDecoder):
    STATISTICS = (
        statistics_registry.Statistic(
            "iterations", statistic_kinds.Distribution,
            "solver iterations per decode", "steps",
        ),
        statistics_registry.Statistic(
            "gave_up", statistic_kinds.Count, "decodes that hit the cap",
            "decodes",
        ),
    )
    statistics = statistics_registry.unrecorded(STATISTICS)
```

The second line gives the class a group whose every statistic does
nothing, so the part built alone, as a test builds it, records nothing.

Pick the kind by what you measure (`decsim/statistics/kinds.py`):

| Kind | Update | Fields |
| --- | --- | --- |
| `Count` | `add()` or `add(n)` | count |
| `Distribution` | `add(value)`, an integer in the unit | count, total, minimum, maximum, and every value's count |
| `Level` | `set(level)` whenever it changes | area (level times ticks), span, peak, and the ticks at each level |
| `History` | as a `Level` | as a `Level`, and every change kept in `changes` |

Times are integer ticks (picoseconds), so totals stay exact.

## 2. Update them where it happens

```python
    def decode(self, job):
        ...
        self.statistics.iterations.add(steps)
        if steps == cap:
            self.statistics.gave_up.add()
```

A `Level` reads the run's clock itself: call `set` with the new level
and nothing else.

## 3. Where they are recorded

The machine attaches each part after it is built, at its settings path:

| Part | Path |
| --- | --- |
| a decoder row, or a row a wrapper holds | `weak.algorithm`, `strong.algorithm` |
| a syndrome buffer | `weak_syndrome_buffer`, `strong_syndrome_buffer` |
| the frame | `pauli_frame` |
| the strong redecode | `switching` |
| the verdict an edge asks | `switching.verdict` |

A part that declares no statistics takes no path. A new kind of
component in a part (see [How to add a component to a part](add_a_component_to_a_part.md))
is named by that part's `statistics_parts()`, which returns
`(path, component)` pairs the way `seed_roots()` does.

## 4. Read them

`shot_stats.csv` holds every shot's fields, `stat_values.csv` each
task's count of every value, and `stat_summary.csv` the per-task
measures: a count's total and per-shot mean, a level's time mean and
peak, a distribution's mean and extremes, and p50, p90 and p99
(the columns are in [the run folder](../reference/run_folder.md)).

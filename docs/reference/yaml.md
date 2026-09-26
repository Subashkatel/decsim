[decsim docs](../README.md) › [Reference](README.md)

# The yaml surface

`configs/reference.yaml` is the reference for every key a config file may
carry. It is a runnable file with every key present and commented, so
this page does not repeat it: it says how to read it and what the rules
around it are.

## What a config file is

A yaml file describes a **sweep**: a set of points, each of which is one
machine, plus how many shots to run at each point. `decsim collect`
runs the whole sweep. `decsim run` runs one shot of the first point.

The file has one section per component, and the section is read by the
package that owns that component (`decsim/settings.py`, `SECTIONS`). The
sections, in the order the root reads them:

`clocks`, `qpu`, `controller`, `idle_policy`, `detection_events`, `links`,
`weak_syndrome_buffer`, `strong_syndrome_buffer`, `windows`, `weak_decoder`, `strong_decoder`,
`decoder_manager`, `escalation`, `burst_detector`, `pauli_frame`,
`workload`, `magic_state_factory`, `observation`.

A section a component owns carries a `kind` key naming a row of that
component's table, and [The plug-in tables](tables.md) lists every table with
its rows. A `kind` that is not a row is refused when the file is loaded,
with the rows printed, so a typo never runs.

A section no package owns is refused too, with the section list printed.
Both refusals are checked by `tests/machine/test_machine.py`.

## How to read `configs/reference.yaml`

Read it top to bottom as the pipeline: `qpu` is where a readout starts
and `observation` is where the run is watched. Each key carries its unit
in its name (`_microseconds`, `_cycles` for clock cycles, `_rounds` for
rounds, `_count` for a count), and the comment beside it says what the
key means and, where the value came from a paper or a reference
implementation, which one. A key whose comment cites, for
example, `Toshio 2510.25222 Sec. III C`, has that section as its source,
and you can change it knowing what you are departing from.

Three conventions are worth knowing before you read:

- `null` means "the component decides". For example
  `windows.commit_rounds: null` leaves the commit region at the code
  distance.
- Some keys only one `kind` reads are refused for every other kind: the
  switching keys of the `escalation` section, and a row's own keys. A
  row with keys of its own declares them on a nested `Settings` record,
  and they sit in its section beside the keys every row of that table
  shares (`decsim/tables.py`, `row_settings`): `union_find`'s
  `weight_step` and `cycle_count` in a decoder tier, `measured_table`'s
  `device`, `partition` and `bases` in a decoder tier, `dispatch_steps`'s
  `device`, `path` and `workers` there too, and the
  `bivariate_bicycle` code card's `qubit_count` and
  `logical_qubit_count` in the `qpu` section, which names its card
  with `code_card` (default `rotated_surface`, Stim's generated
  `surface_code:rotated_memory_z`). The reference file
  carries both kinds as comments for that reason. Others are read and
  ignored by the kinds that have no use for them (`terminal_policy`),
  and their comments say so.
- The file is the documentation of the yaml surface.
  `tests/experiments/test_yaml_surface.py` loads it and every shipped
  config, so a key the readers stopped knowing fails in a section that
  refuses unknown keys; a key a reader gained is caught by no test.
  Any commit that changes the config surface changes this file in the
  same commit.

## Starting from another file

```yaml
extends: weak_decoder_baseline.yaml
```

`extends` reads the named file from the same folder first, then applies
this file's keys over it (`decsim/experiments/experiment.py`). A
section this file names replaces the base's section whole, so a `sweep`
written here replaces the base's sweep rather than adding to it.
`manifest.json` records the whole chain, nearest first, and `config/`
in the run folder holds a verbatim copy of every file in it.

## Seeing what a file resolves to

```bash
decsim show configs/reference.yaml
```

prints the resolved sections, one line per component, and the sweep
blocks, without running anything. Then, under `values:`, it prints
every value the machine is built with, one per line, gem5's
`config.ini` in one list (`src/python/m5/simulate.py:122-127`). Three
of `decsim show configs/weak_ler.yaml`'s:

```
qpu.distance = [3, 5, 7, 9, 11]  [sweep, configs/weak_ler.yaml:17-33]
controller.decision_to_pulse_cycles = 0  [preset weak_decoder_baseline.yaml, configs/weak_decoder_baseline.yaml:45]
controller.packing_overflow = "STALL"  [default, configs/reference.yaml:513]
```

The bracket names the layer that set the value, `your file`, `preset`
and the file's name for a file your `extends` chain reads, `sweep`, or
`default`, then the line of the yaml key that set it. A default's line
is where `configs/reference.yaml` documents the key. A value that is no
one key's, such as a link's ticks derived from its card or a section's
clock period derived from the domain it names, prints no bracket, since
show names only a source it is sure of. A swept value
lists the sweep's values. This is the fastest way to check that an `extends`
chain says what you meant.

## The shipped configs

| File | What it is for |
| --- | --- |
| `configs/reference.yaml` | every key, commented, with a two-shot sweep so it runs in seconds |
| `configs/weak_decoder_baseline.yaml` | the defaults a weak-tier study starts from |
| `configs/strong_decoder_baseline.yaml` | the same for a strong-tier study |
| `configs/weak_ler.yaml` | the weak tier's logical error rate sweep, 35 points and 10,425,000 shots |
| `configs/strong_ler.yaml` | the strong tier's logical error rate sweep |
| `configs/weak_latency.yaml` | the weak tier's latency sweep |
| `configs/strong_latency.yaml` | the strong tier's latency sweep |
| `configs/strong_latency_preview.yaml` | a short version of it |
| `configs/seam_pinned_switching.yaml` | switching with a seam-pinned strong window |
| `configs/two_tiers.yaml` | switching with both tiers on priced cards, the third tutorial's run |
| `configs/priced_cards_example.yaml` | one tier on a priced card, for a timing study |
| `configs/my_first_sweep.yaml` | three distances at one error rate, the second tutorial's run |
| `configs/cluster_gap_switching.yaml` | switching whose confidence signal is the union find growth's own walk, priced as a card |
| `configs/data_movement.yaml` | the data-movement study: every copy, reference and move counted per hop |
| `configs/data_movement_input_in_place.yaml` | the same with the weak input referenced in place instead of copied |
| `configs/data_movement_fold_in_place.yaml` | the same with the boundary folded in place |
| `configs/data_movement_switching.yaml` | the same under the switching escalation |
| `configs/experiments_2026_09/` | the sixteen decoder experiments: sixteen experiment files that differ only in their decoder rows, one file per experiment and distance for the Slurm arrays, and `configs/experiments_2026_09/PLAN.md` with the shot table, the costs and the submit lines |
| `configs/common/experiments_2026_09_base.yaml` | the shared base those sixteen extend; it names no decoder, so it is not run by itself |

## Read next

- `configs/reference.yaml` itself: every key, with its source.
- [The plug-in tables](tables.md): the rows a `kind` may name.
- [How to add a yaml key](../how-to/add_a_yaml_key.md): adding one.

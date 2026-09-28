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
  `weight_step` and `cycle_count` in a decoder tier, `relay_bp`'s
  `alpha`, `alpha_iteration_scaling_factor`, `gamma0`,
  `pre_iterations`, `relay_set_count`, `iterations_per_set`,
  `gamma_interval`, `converged_solution_count` and `bases` there too
  (relay-bp's own arguments, with the Relay-BP paper's surface code
  values shown beside them, and whether X and Z are decoded apart),
  `tesseract`'s `detector_beam`, `beam_climbing`,
  `no_revisit_detectors`, `priority_queue_limit`,
  `detector_order_method`, `detector_order_count`,
  `detector_order_seed` and `merge_errors` there too, `measured_table`'s
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
extends: ../bases/weak_decoder_baseline.yaml
```

`extends` reads the named file first, its path taken from this file's
folder, then applies this file's keys over it (`decsim/experiments/experiment.py`). A
section this file names replaces the base's section whole, so a `sweep`
written here replaces the base's sweep rather than adding to it.
`manifest.json` records the whole chain, nearest first, and `config/`
in the run folder holds a verbatim copy of every file in it.

## The sweep

```yaml
sweep:
  - axes:
      workload.arguments.physical_error_probability: [0.001, 0.003]
      qpu.distance: [3, 5]
      windows.commit_rounds: [1, 3]
    collection: {max_shots: 400}
```

A block's `axes` map a yaml path to the values the sweep sets there, and
the block is every combination of them in the order written, each
collected as its `collection` says (below): Hydra's multi-run makes one
job per combination of `key=v1,v2` overrides the same way. The blocks are a union, and a point
named twice runs once. Any key can be an axis. Its section must exist,
and a value that is a mapping replaces the node at its path whole, so a
decoder row with keys of its own is one value of a `weak_decoder` axis.

Each point is then read the way a written file is: the axes placed,
every whole-value reference such as `distance: ${qpu.distance}` replaced
by the value at that path (OmegaConf's interpolation, kept to whole
values; inside a flow mapping `{...}` it is quoted, `'${qpu.distance}'`,
since a brace opens a mapping there), and the sections read by the
packages that own them
(`decsim/experiments/experiment.py`, `ExperimentConfig.point_task`). A
path that does not resolve and a reference that leads back to itself
are refused with the path named. A point's metadata is its
`{path: value}`, and its id is a hash of that and every setting it
resolves to (`decsim/collect.py`, `Task.strong_id`); the run folder
names its files by the id, and each csv row carries the id and then one
column per swept path ([The run folder](run_folder.md)).

## The collection

```yaml
collection:
  max_failures: 100
  max_shots: 1000000
  max_core_seconds: null
  min_shots: 0
  piece_rounds: 20000
```

How a point's shots are cut and when they stop, sinter's
`CollectionOptions` as yaml (`decsim/experiments/collection.py`). A
point runs seeds 0, 1, 2 and on, in pieces of `piece_rounds` QEC rounds
each saved whole the moment it ends, so a killed collect run again runs
only the pieces it lacks. It stops at the first shot where its scored
failures reach `max_failures` with at least `min_shots` scored shots
behind them, or where its shots reach `max_shots` or its shots' own
seconds reach `max_core_seconds`, whichever comes first; a point needs
one of the two caps. No piece past the stop is started. The section
goes at the top of a file or in a sweep block, whose keys override the
top's one by one, and no key of it enters a point's id.

## Seeing what a file resolves to

```bash
decsim show configs/reference.yaml
```

prints the resolved sections, one line per component, and the sweep
blocks, without running anything. Then, under `values:`, it prints
every value the machine is built with, one per line, gem5's
`config.ini` in one list (`src/python/m5/simulate.py:122-127`). Three
of `decsim show configs/examples/my_first_sweep.yaml`'s:

```
qpu.distance = [3, 5, 7]  [sweep, configs/examples/my_first_sweep.yaml:16-21]
controller.decision_to_pulse_cycles = 0  [preset weak_decoder_baseline.yaml, configs/bases/weak_decoder_baseline.yaml:51]
controller.packing_overflow = "STALL"  [default, configs/reference.yaml:658]
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

`configs/` holds three kinds of file, and `configs/reference.yaml`
beside them. A base under `bases/` is the defaults a study starts from
and is read through `extends`; an example under `examples/` teaches one
feature at small cost; an experiment under `experiments/` answers one
question, one folder per question, its file named for the study, since
the run folder is named for the file. An experiment extends a base or
writes every section itself, never extending an example or another
experiment, and it holds its grid whole: no copy per distance and no
preview copy. A run too long for one job is cut into pieces by
`collect`, not by more files.

| File | What it is for |
| --- | --- |
| `configs/reference.yaml` | every key, commented, with a two-shot sweep so it runs in seconds |
| `configs/bases/weak_decoder_baseline.yaml` | the defaults a weak-tier study starts from |
| `configs/bases/strong_decoder_baseline.yaml` | the same for a strong-tier study |
| `configs/examples/my_first_sweep.yaml` | three distances at one error rate, the second tutorial's run |
| `configs/examples/two_tiers.yaml` | switching with both tiers on priced cards, the third tutorial's run |
| `configs/examples/priced_cards_example.yaml` | one tier on a priced card, for a timing study |
| `configs/experiments/switching/seam_pinned_switching.yaml` | switching with a seam-pinned strong window |
| `configs/experiments/switching/cluster_gap_switching.yaml` | switching whose confidence signal is the union find growth's own walk, priced as a card |
| `configs/experiments/data_movement/data_movement.yaml` | every copy, reference and move of the data path counted per hop, in four blocks: every hop copying, the weak input read in place, the boundary folded in place, and the switching escalation |
| `configs/experiments/decoder_baseline/decoder_baseline.yaml` | the paper's decoder baseline: Union-Find, MWPM, XYZ-Relay-BP-5 and Tesseract on the same samples of the rotated surface code memory, d = 5 to 15, six error rates, both bases, 100 rounds, each point stopped at 100 failures or 24 core-hours |

## Read next

- `configs/reference.yaml` itself: every key, with its source.
- [The plug-in tables](tables.md): the rows a `kind` may name.
- [How to add a yaml key](../how-to/add_a_yaml_key.md): adding one.

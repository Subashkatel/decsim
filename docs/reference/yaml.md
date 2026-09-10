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

`clocks`, `qpu`, `controller`, `idle_policy`, `links`, `round_store`,
`strong_round_store`, `windows`, `weak_decoder`, `strong_decoder`,
`decoder_manager`, `escalation`, `pauli_frame`, `workload`,
`observation`.

A section a component owns carries a `kind` key naming a row of that
component's table, and [`docs/reference/tables.md`](tables.md) lists every table with
its rows. A `kind` that is not a row is refused when the file is loaded,
with the rows printed, so a typo never runs.

A section no package owns is refused too, with the section list printed.
Both refusals are checked by `tests/machine/test_machine.py`.

## How to read `configs/reference.yaml`

Read it top to bottom as the pipeline: `qpu` is where a readout starts
and `observation` is where the run is watched. Each key carries its unit
in its name (`_us` for microseconds, `_cycles` for clock cycles,
`_rounds` for rounds, `_count` for a count), and the comment beside it
says what the key means and, where the value came from a paper or a
reference implementation, which one. A key whose comment cites, for
example, `Toshio 2510.25222 Sec. III C`, has that section as its source,
and you can change it knowing what you are departing from.

Three conventions are worth knowing before you read:

- `null` means "the component decides". For example
  `windows.commit_rounds: null` leaves the commit region at the code
  distance.
- A key that only one `kind` reads is refused for every other kind. The
  switching keys of the `escalation` section are commented out in the
  reference file for that reason, with the comment saying which config
  runs them.
- The file is the documentation of the yaml surface, and
  `tests/front/test_yaml_surface.py` fails when the file and the readers
  drift apart. Any commit that changes the config surface changes this
  file in the same commit.

## Starting from another file

```yaml
extends: weak_decoder_baseline.yaml
```

`extends` reads the named file from the same folder first, then applies
this file's keys over it (`decsim/front/experiment.py`). A section this
file names replaces the base's section whole, so a `sweep` written here
replaces the base's sweep rather than adding to it. `manifest.json`
records the whole chain, nearest first, and `config/` in the run folder
holds a verbatim copy of every file in it.

## Seeing what a file resolves to

```bash
decsim show configs/reference.yaml
```

prints the resolved sections, one line per component, and the sweep
blocks, without running anything. This is the fastest way to check that
an `extends` chain says what you meant.

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
| `configs/campaigns_2026_09/` | the six 2026-09 decoder campaigns: one shared base, six campaign files that differ only in their decoder rows, one file per campaign and distance for the Slurm arrays, and `configs/campaigns_2026_09/PLAN.md` with the shot table, the costs and the submit lines |

## Read next

- `configs/reference.yaml` itself: every key, with its source.
- [`docs/reference/tables.md`](tables.md): the rows a `kind` may name.
- [`docs/how-to/add_a_yaml_key.md`](../how-to/add_a_yaml_key.md): adding one.

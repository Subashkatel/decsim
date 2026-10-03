# decsim

decsim is a discrete-event simulator for the classical control and
decoding path of a quantum error corrected computer. You describe one
system, run a workload through it, and measure the reaction time and the
logical error rate the system achieves.

The simulated path is the whole loop: QPU rounds, controller readout,
links, the syndrome buffers, window creation, decoder memory, decoder
units, the Pauli frame, and the instruction back to the QPU. A hop is
one link between two of those parts. Every hop charges its configured
latency and bandwidth, so a run says where the time went and which
component set the reaction time. Decoding is real: windows of a Stim
circuit are decoded by PyMatching (weighted or unweighted), BP-OSD,
belief matching, union find, Relay-BP or Tesseract. A run may also price
a decoder with a number instead of measuring one.

## Install

Python 3.10 or newer, and the `run` extra: the root imports Stim, the
yaml layer reads PyYAML, the decoder table imports ldpc, and a run on
real syndrome data needs PyMatching, numpy and scipy as well.

```bash
python -m pip install -e ".[run]" -c constraints.txt
```

`constraints.txt` pins every package the `run`, `test` and `dev` extras
install, at versions tested together on Python 3.10, 3.12 and 3.13; pip
installs only what the extras ask for, at those versions. It is uv's
universal resolution, one file for every platform, and its first lines
are the command that regenerates it after an extra changes.
beliefmatching caps numpy at 2.2.6, which ships no wheel past Python
3.13.

The `bb-decoders` extra adds the three optional backends (Relay-BP,
Tesseract, BP-OSD through quits). decsim keeps its choices in tables: a
table lists the names a yaml may write, and a row is one name and the
class built for it. Each optional backend is one row, and nothing else
needs it.

The union find row decodes in C, and the cluster gap that reads its
growth walks in C beside it, in one library. Build it once, and again
whenever `decsim/decoders/union_find/union_find.c` or its
`cluster_gap.c` changes:

```bash
tools/build_union_find.sh
```

Run it where a C compiler is. The suite builds the library for you when
it is missing and a compiler is reachable, and otherwise stops with one
sentence naming this command.

## One run

```bash
decsim run configs/reference.yaml --seed 0 --trace --out results/first_shot
```

That is one shot of the reference config, the one that documents every
key, with a Chrome trace of every round and window written into the run
folder it names.
[Your first run](docs/tutorials/first_run.md) walks through it and its
output.

## The documentation

[decsim documentation](docs/README.md) is the front door: tutorials to
learn from, how-to guides for one task each, reference to look things up
in, explanation for the design and its sources, and a link to every page.

## The tests

The `test` extra adds pytest and the reference decoders the suite checks
decsim against:

```bash
python -m pip install -e ".[run,test]" -c constraints.txt
python -m pytest tests
```

A test whose optional backend is absent (the `bb-decoders` and Deltakit
extras) is skipped and says which module it could not import.

The tutorials show what their commands print.
`tools/check_tutorial_runs.py` runs each page's commands and holds what
the page shows to what they print: every line on a page whose decoders
are priced by cards, and the lines no decoder's wall clock moves on the
others. It needs only the `run` extra and takes about ten minutes on
four cores:

```bash
python tools/check_tutorial_runs.py
```

`tools/check.sh` runs the style and structure checks that `STYLE.md`
describes, with the ruff the `dev` extra installs:

```bash
python -m pip install -e ".[run,test,dev]" -c constraints.txt
tools/check.sh
```

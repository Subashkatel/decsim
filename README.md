# decsim

decsim is a discrete-event simulator for the classical control and
decoding path of a quantum error corrected computer. You describe one
system, run a workload through it, and measure the reaction time and the
logical error rate the system achieves.

The simulated path is the whole loop: QPU rounds, controller readout,
links, the syndrome buffers, window creation, decoder memory, decoder
units, the Pauli frame, and the instruction back to the QPU. Every hop
charges its configured latency and bandwidth, so a run says where the
time went and which component set the reaction time. Decoding is real:
windows of a Stim circuit are decoded by PyMatching, BP-OSD, belief
matching, union find, Relay-BP or Tesseract. A run may also price a
decoder with a number instead of measuring one.

## Install

Python 3.9 or newer, and the `run` extra: the root imports Stim, and a
run on real syndrome data needs PyMatching, numpy, scipy and matplotlib
as well.

```bash
python -m pip install -e ".[run]"
```

The `bb-decoders` extra adds the three optional backends (Relay-BP,
Tesseract, BP-OSD through quits); each one is one row of a table and
nothing else needs it.

## One run

```bash
decsim run configs/reference.yaml --seed 0 --trace
```

That is one shot of the smallest shipped config, with a Chrome trace of
every round and window written into the run folder it names.
`docs/tutorials/first_run.md` walks through it and its output.

## The documentation

`docs/index.md` is the front door. It is arranged in the Diataxis
scheme: tutorials to learn from, how-to guides for one task each,
reference to look things up in, and explanation for the design and its
sources.

## The tests

```bash
python -m pytest tests
```

`tools/check.sh` runs the style and structure checks that `STYLE.md`
describes.

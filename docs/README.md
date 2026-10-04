# decsim documentation

decsim is a simulator of the classical machinery that keeps a quantum
computer's errors under control. A quantum error corrected computer
measures its qubits over and over, and each measurement round produces a
**syndrome**, a pattern of bits saying where something looks wrong. A
classical **decoder** turns those bits into a correction. The whole loop,
readout out of the fridge, the wires, the buffers, the decoder, the
correction, and the instruction back to the machine, has to finish fast
enough that the quantum computation can go on. decsim builds that loop
out of parts you choose, runs a real workload through it, charges every
hop its configured time, and reports where the time went and how often
the answer was wrong.

Two things a simulator can measure are here at once. The **reaction
time** is how long the loop takes, end to end. It is charged out of
cards: a card is a small record of numbers that prices one part, such
as a link's latency or a decoder's time per decode. The **logical error
rate** is how often the corrected answer is wrong, and it is real:
windows of a Stim circuit are decoded by PyMatching, union find, BP-OSD
or another of the decoders the [parts page](reference/parts.md) lists,
so an accuracy number is a measurement and not a model.

**New here?** Start with [Your first run](tutorials/first_run.md). It
takes ten minutes and needs nothing but a terminal.

## How the pages are arranged

There are three kinds of page, after [Diátaxis](https://diataxis.fr),
and each kind answers one kind of need:

- **Tutorials** teach by doing. Follow one at the keyboard when you are
  new to decsim.
- **How-to guides** get one task done. Use them when you know what you
  want and need the steps.
- **Reference** states what the pieces are. Look things up here while
  you work.

Every page opens with a line that brings you back here, and GitHub's
outline button at the top of any page lists that page's sections.

## Tutorials

Four lessons, to do in order.

- [Your first run](tutorials/first_run.md): a run file, one shot, the
  results folder, and one round followed through the machine.
- [Your first sweep](tutorials/first_sweep.md): a small sweep, its error
  bars, and pieces folded back into one report.
- [Two tiers](tutorials/two_tiers.md): a machine with two decoders, and
  one window decoded twice.
- [Build a machine step by step](tutorials/build_a_machine.md): the
  parts, built and connected by hand.

## How-to guides

Each guide adds one kind of thing to the machine.

- [How to add a decoder backend](how-to/add_a_decoder_backend.md)
- [How to add a component to a part](how-to/add_a_component_to_a_part.md)
- [How to plug in a workload maker](how-to/plug_in_a_workload_maker.md)

## Reference

- [The parts](reference/parts.md): every settings record a machine is
  built from, generated from the source.
- [The run folder](reference/run_folder.md): every file a run writes and
  every column of every csv.
- [Glossary](reference/glossary.md): decsim's names against the papers'
  names, with where to read each.

## Where everything lives

| Folder | What is in it |
| --- | --- |
| `decsim/` | the simulator, in packages layered so that a lower one never imports a higher one |
| `decsim/ports.py` | the ports, the only way two packages talk |
| `experiments/` | the experiments' run files, one folder each |
| `examples/` | runnable examples, each run by a test |
| `results/` | what a run writes, one folder per run. Not tracked by git. |
| `tests/` | the test suite, one folder per package |
| `tools/` | the checks `tools/check.sh` runs, the tutorials' check `tools/check_tutorial_runs.py`, and the parts page's generator |
| `STYLE.md` | the rules every line of the package is written to |

## How to cite decsim

There is no paper for decsim yet. Cite the repository and the commit you
ran, which every results folder records for you: `run.json` holds the
git commit and the library versions, each point's `machine.json` every
value it ran with, and `code_state.patch` any uncommitted change.
Quoting the commit from `run.json` is enough for someone else to reproduce the run.

The decoders, the circuits and the models decsim runs are other people's
work and are cited where they are used, in the docstrings and comments
beside each value, and the papers behind each name are in
[Glossary](reference/glossary.md).

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
time** is how long the loop takes, end to end, and it is charged out of
component cards and link latencies. The **logical error rate** is how
often the corrected answer is wrong, and it is real: windows of a Stim
circuit are decoded by PyMatching, union find, belief matching, BP-OSD,
Relay-BP or Tesseract, so an accuracy number is a measurement and not a
model.

**New here?** Start with [Your first run](tutorials/first_run.md). It
takes ten minutes and needs nothing but a terminal.

## How the pages are arranged

There are four kinds of page, after [Diátaxis](https://diataxis.fr),
and each kind answers one kind of need:

- **Tutorials** teach by doing. Follow one at the keyboard when you are
  new to decsim.
- **How-to guides** get one task done. Use them when you know what you
  want and need the steps.
- **Reference** states what the pieces are. Look things up here while
  you work.
- **Explanation** says why the machine is shaped the way it is. Read it
  away from the keyboard.

Every page opens with a line that brings you back here, and GitHub's
outline button at the top of any page lists that page's sections.

## Tutorials

Three lessons, to do in order. Each was run end to end, and every output
on its page is from that run.

- [Your first run](tutorials/first_run.md): one shot, the run folder,
  and one round followed through the machine.
- [Your first sweep](tutorials/first_sweep.md): a small sweep, its error
  bars, and shards folded back into one report.
- [Two tiers](tutorials/two_tiers.md): a machine with two decoders, and
  one window decoded twice.

## How-to guides

Each guide is one task for a reader who has done the tutorials.

**Plug something in.** decsim is built so that a new part is one class
and one table row. These guides are the recipe and its variants.

- [How to add a row to a table](how-to/add_a_table_row.md)
- [How to add a decoder backend](how-to/add_a_decoder_backend.md)
- [How to plug a component in without a table row](how-to/plug_in_without_a_table_row.md)
- [How to add a yaml key](how-to/add_a_yaml_key.md)

**Run and read an experiment.** Getting a sweep through a cluster,
and reading what it wrote.

- [How to run a sweep on Slurm](how-to/run_a_sweep_on_slurm.md)
- [How to read a trace and follow one round or one window](how-to/read_a_trace.md)
- [How to compare two runs](how-to/compare_two_runs.md)
- [How to run a timing study whose numbers do not depend on your computer](how-to/run_a_timing_only_study.md)
- [How to run a workload whose next operation waits on a decision](how-to/run_a_feedback_workload.md)

## Reference

Facts to look up. Four of these pages are generated from the source by
`tools/docs_map.py` and checked by `tests/test_docs.py`, so they cannot
drift from the code.

**What goes in and what comes out.**

- [The yaml surface](reference/yaml.md): how to read
  `configs/reference.yaml`, and every shipped config.
- [The run folder](reference/run_folder.md): every file a run writes and
  every column of every csv.
- [The commands](reference/cli.md): every subcommand and flag.

**The shape of the package.**

- [The plug-in tables](reference/tables.md): the seventeen tables and
  every row a yaml may name.
- [The ports](reference/ports.md): every port, its methods and members,
  in the order a readout travels.
- [The map of the package](reference/map.md): every package and module,
  in uses order.
- [Glossary](reference/glossary.md): decsim's names against the papers'
  names, with where to read each.

## Explanation

Read these away from the keyboard.

**The machine.** What the components are and what a readout costs on
its way through them.

- [Architecture](explanation/architecture.md)
- [The data path, hop by hop](explanation/data_path.md)
- [Time](explanation/time.md)

**Decoding.** Why decoding is cut into windows, and what a second,
slower decoder buys.

- [Windows and boundaries](explanation/windows_and_boundaries.md)
- [Two tiers](explanation/two_tiers.md)

**Why it is shaped this way.** The modelling decisions with their
sources, and the ideas behind the tree.

- [The design decisions](explanation/decisions.md)
- [The principles behind the shape](explanation/principles.md)

## Where everything lives

| Folder | What is in it |
| --- | --- |
| `decsim/` | the simulator: twenty-five packages on eleven uses levels, mapped in [The map of the package](reference/map.md) |
| `decsim/ports.py` | the thirty-eight ports, the only way two packages talk |
| `configs/` | the yaml experiments. `configs/reference.yaml` documents every key. |
| `results/` | what a run writes, one folder per run. Not tracked by git. |
| `tests/` | the test suite, one folder per package |
| `tools/` | the checks `tools/check.sh` runs, and the documentation generator |
| `slurm/` | the array scripts for running a sweep on a cluster |
| `STYLE.md` | the rules every line of the package is written to |

## How to cite decsim

There is no paper for decsim yet. Cite the repository and the commit you
ran, which every run folder records for you: `manifest.json` holds the
git commit, the resolved config and the library versions, and
`code_state.patch` holds any uncommitted change. Quoting the commit from
the manifest is enough for someone else to reproduce the run.

The decoders, the circuits and the models decsim runs are other people's
work and are cited where they are used: the papers behind each design
decision are in [The design decisions](explanation/decisions.md), and
the papers behind each name are in [Glossary](reference/glossary.md).

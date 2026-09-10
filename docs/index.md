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

**New here?** Start with `docs/tutorials/first_run.md`.

## How this documentation is arranged

These pages follow **Diataxis**, Daniele Procida's scheme
(https://diataxis.fr). It sorts documentation by two questions about the
reader: is this about action or cognition, and is the reader acquiring a
skill or applying one (https://diataxis.fr/compass/). The four answers
are the four sections below, and they are kept apart on purpose: a page
that teaches and a page that states facts are read differently and fail
differently.

### Tutorials

"A tutorial is an experience that takes place under the guidance of a
tutor. A tutorial is always learning-oriented"
(https://diataxis.fr/tutorials/). Read these in order, at the keyboard.
Each one was run end to end and every output on the page is from that
run.

| Page | What you do |
| --- | --- |
| `docs/tutorials/first_run.md` | run one shot with a trace, read the run folder, follow one round |
| `docs/tutorials/first_sweep.md` | run a small sweep, read its error bars, cut it into shards and fold it back |
| `docs/tutorials/two_tiers.md` | run a two-decoder machine and watch one window get decoded twice |

### How-to guides

"How-to guides are directions that guide the reader through a problem or
towards a result. How-to guides are goal-oriented"
(https://diataxis.fr/how-to-guides/). Each page is one task, for a
reader who has done the tutorials.

| Page | The task |
| --- | --- |
| `docs/how-to/add_a_table_row.md` | plug any new part in: the general recipe |
| `docs/how-to/add_a_decoder_backend.md` | plug a decoder in |
| `docs/how-to/plug_in_without_a_table_row.md` | use a Python object directly, with no table row |
| `docs/how-to/add_a_yaml_key.md` | add a key to a settings section |
| `docs/how-to/run_a_sweep_on_slurm.md` | run a sweep as a Slurm array and combine the shards |
| `docs/how-to/read_a_trace.md` | open a trace and follow a round or a window |
| `docs/how-to/compare_two_runs.md` | put two runs side by side |
| `docs/how-to/run_a_timing_only_study.md` | price the decoders instead of measuring them |

### Reference

"Reference guides are technical descriptions of the machinery and how to
operate it. Reference material is information-oriented"
(https://diataxis.fr/reference/). Look things up here. Four of these
pages are generated from the source by `tools/docs_map.py` and checked
by `tests/test_docs.py`, so they cannot drift from the code.

| Page | What it lists |
| --- | --- |
| `docs/reference/map.md` | every package and module, in uses order, with one sentence each. Generated. |
| `docs/reference/ports.md` | every port, its methods and its members, in pipeline order. Generated. |
| `docs/reference/tables.md` | all seventeen tables and every row of each. Generated. |
| `docs/reference/cli.md` | every subcommand and flag. Generated. |
| `docs/reference/yaml.md` | how to read `configs/reference.yaml`, which is the key reference |
| `docs/reference/run_folder.md` | every file a run writes, and every column of every csv |
| `docs/reference/glossary.md` | every name decsim uses, and what the papers call it |
| `docs/reference/frozen_gate.md` | the behaviour gate: what it pins and how to run it |

### Explanation

"Explanation is a discursive treatment of a subject, that permits
reflection. Explanation is understanding-oriented"
(https://diataxis.fr/explanation/). Read these away from the keyboard.

| Page | The subject |
| --- | --- |
| `docs/explanation/architecture.md` | the components in pipeline order, and the diagram |
| `docs/explanation/data_path.md` | the readout hop by hop, and what each hop costs |
| `docs/explanation/time.md` | ticks, clocks, priced cards, and what reproducibility means here |
| `docs/explanation/windows_and_boundaries.md` | why decoding is cut into windows, and what happens at a seam |
| `docs/explanation/two_tiers.md` | weak and strong decoders, confidence, thresholds, escalation |
| `docs/explanation/decisions.md` | the ten modelling decisions, their reasons and their sources |
| `docs/explanation/principles.md` | the eleven architecture ideas the tree is built on |

## Where everything lives

| Folder | What is in it |
| --- | --- |
| `decsim/` | the simulator: twenty-five packages on eleven uses levels, mapped in `docs/reference/map.md` |
| `decsim/ports.py` | the thirty-four ports, the only way two packages talk |
| `configs/` | the yaml experiments. `configs/reference.yaml` documents every key. |
| `results/` | what a run writes, one folder per run. Not tracked by git. |
| `tests/` | the test suite, one folder per package |
| `tools/` | the checks `tools/check.sh` runs, and the documentation generator |
| `slurm/` | the array script for running a sweep on a cluster |
| `STYLE.md` | the rules every line of the package is written to |

## How to cite decsim

There is no paper for decsim yet. Cite the repository and the commit you
ran, which every run folder records for you: `manifest.json` holds the
git commit, the resolved config and the library versions, and
`code_state.patch` holds any uncommitted change. Quoting the commit from
the manifest is enough for someone else to reproduce the run.

The decoders, the circuits and the models decsim runs are other people's
work and are cited where they are used: the papers behind each design
decision are in `docs/explanation/decisions.md`, and the papers behind
each name are in `docs/reference/glossary.md`.

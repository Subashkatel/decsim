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

**New here?** Start with [Your first run](tutorials/first_run.md).

This page lists every page and every section, so you can find a thing
without opening folders. The pages are arranged in the four kinds of
[Diataxis](https://diataxis.fr): tutorials to learn from, how-to guides
for one task each, reference to look things up in, and explanation to
read away from the keyboard. Every page carries a line at the top that
brings you back here.

## Tutorials

Lessons to do in order, at the keyboard. Each one was run end to end and
every output on the page is from that run.

**[Your first run](tutorials/first_run.md)**. Run one shot with a trace,
read the run folder, follow one round through the machine. Ten minutes.
Sections: [What decsim is, in one paragraph](tutorials/first_run.md#what-decsim-is-in-one-paragraph) · [Step 1. Install](tutorials/first_run.md#step-1-install) · [Step 2. Run one shot](tutorials/first_run.md#step-2-run-one-shot) · [Step 3. Run the sweep and get a run folder](tutorials/first_run.md#step-3-run-the-sweep-and-get-a-run-folder) · [Step 4. Open the run folder](tutorials/first_run.md#step-4-open-the-run-folder) · [Step 5. Read one row and one figure](tutorials/first_run.md#step-5-read-one-row-and-one-figure) · [Step 6. Follow one round](tutorials/first_run.md#step-6-follow-one-round).

**[Your first sweep](tutorials/first_sweep.md)**. Run a small sweep, read
its error bars, cut it into shards and fold it back. Fifteen minutes.
Sections: [Step 1. Copy a shipped config and make a sweep of it](tutorials/first_sweep.md#step-1-copy-a-shipped-config-and-make-a-sweep-of-it) · [Step 2. Run it, on four processes](tutorials/first_sweep.md#step-2-run-it-on-four-processes) · [Step 3. Read the error bars](tutorials/first_sweep.md#step-3-read-the-error-bars) · [Step 4. Draw it](tutorials/first_sweep.md#step-4-draw-it) · [Step 5. Cut it into shards and put it back together](tutorials/first_sweep.md#step-5-cut-it-into-shards-and-put-it-back-together).

**[Two tiers](tutorials/two_tiers.md)**. Run a two-decoder machine and
watch one window get decoded twice. Ten minutes.
Sections: [What escalation is](tutorials/two_tiers.md#what-escalation-is) · [Step 1. Read the config](tutorials/two_tiers.md#step-1-read-the-config) · [Step 2. Run the sweep](tutorials/two_tiers.md#step-2-run-the-sweep) · [Step 3. Trace one shot](tutorials/two_tiers.md#step-3-trace-one-shot) · [Step 4. A window that was kept](tutorials/two_tiers.md#step-4-a-window-that-was-kept) · [Step 5. A window that escalated](tutorials/two_tiers.md#step-5-a-window-that-escalated).

## How-to guides

One task per page, for a reader who has done the tutorials.

**[Add a row to a table](how-to/add_a_table_row.md)**. Plug any new part
in: the general recipe, the refusal a typo gets, the worked examples.
Sections: [1. Find the port and fill it](how-to/add_a_table_row.md#1-find-the-port-and-fill-it) · [2. Add the row](how-to/add_a_table_row.md#2-add-the-row) · [3. Name it in the yaml, and in the reference](how-to/add_a_table_row.md#3-name-it-in-the-yaml-and-in-the-reference) · [What a typo gets](how-to/add_a_table_row.md#what-a-typo-gets) · [The worked examples](how-to/add_a_table_row.md#the-worked-examples) · [Two rules your class has to keep](how-to/add_a_table_row.md#two-rules-your-class-has-to-keep) · [What will tell you what you forgot](how-to/add_a_table_row.md#what-will-tell-you-what-you-forgot).

**[Add a decoder backend](how-to/add_a_decoder_backend.md)**. The decoder
case of that recipe, with the fault model contract and the referent check.
Sections: [1. Write the class](how-to/add_a_decoder_backend.md#1-write-the-class) · [2. Say what fault model you need](how-to/add_a_decoder_backend.md#2-say-what-fault-model-you-need) · [3. Add the row and the yaml key](how-to/add_a_decoder_backend.md#3-add-the-row-and-the-yaml-key) · [4. Run it and check it against a referent](how-to/add_a_decoder_backend.md#4-run-it-and-check-it-against-a-referent) · [5. Read the worked example](how-to/add_a_decoder_backend.md#5-read-the-worked-example).

**[Plug a component in without a table row](how-to/plug_in_without_a_table_row.md)**.
Use a Python object directly while it is still changing.
Sections: [Pass the instance](how-to/plug_in_without_a_table_row.md#pass-the-instance) · [What you give up](how-to/plug_in_without_a_table_row.md#what-you-give-up) · [When to add the row](how-to/plug_in_without_a_table_row.md#when-to-add-the-row) · [The worked examples](how-to/plug_in_without_a_table_row.md#the-worked-examples).

**[Add a yaml key](how-to/add_a_yaml_key.md)**. A knob a config file can
set: the field, the boundary check, the reference file, the tests.
Sections: [1. Add the field to the settings record](how-to/add_a_yaml_key.md#1-add-the-field-to-the-settings-record) · [2. Check it at the boundary, once](how-to/add_a_yaml_key.md#2-check-it-at-the-boundary-once) · [3. Document it in configs/reference.yaml, in the same commit](how-to/add_a_yaml_key.md#3-document-it-in-configsreferenceyaml-in-the-same-commit) · [4. Run the checks](how-to/add_a_yaml_key.md#4-run-the-checks).

**[Run a sweep on Slurm](how-to/run_a_sweep_on_slurm.md)**. A sweep as an
array job, and how the shards fold back into one report.
Sections: [1. Understand the two knobs](how-to/run_a_sweep_on_slurm.md#1-understand-the-two-knobs) · [2. Submit the array](how-to/run_a_sweep_on_slurm.md#2-submit-the-array) · [3. Fold the shards](how-to/run_a_sweep_on_slurm.md#3-fold-the-shards) · [4. Plot](how-to/run_a_sweep_on_slurm.md#4-plot) · [If a task dies](how-to/run_a_sweep_on_slurm.md#if-a-task-dies).

**[Read a trace](how-to/read_a_trace.md)**. Open a trace in a viewer, or
print one round's or one window's path with `decsim trace follow`.
Sections: [1. Ask for one](how-to/read_a_trace.md#1-ask-for-one) · [2. For the whole shot, use a viewer](how-to/read_a_trace.md#2-for-the-whole-shot-use-a-viewer) · [3. For one round, use trace follow](how-to/read_a_trace.md#3-for-one-round-use-trace-follow) · [4. For one window, the same command](how-to/read_a_trace.md#4-for-one-window-the-same-command) · [5. Write the path as a page](how-to/read_a_trace.md#5-write-the-path-as-a-page) · [What is in the file](how-to/read_a_trace.md#what-is-in-the-file).

**[Compare two runs](how-to/compare_two_runs.md)**. Add shards, read two
rows side by side, or plot both folders; what to check before believing
a difference.
Sections: [If they are shards of one sweep, add them](how-to/compare_two_runs.md#if-they-are-shards-of-one-sweep-add-them) · [If they are different configurations, read the rows](how-to/compare_two_runs.md#if-they-are-different-configurations-read-the-rows) · [If you want a picture, hand plot both folders](how-to/compare_two_runs.md#if-you-want-a-picture-hand-plot-both-folders) · [What to check before you believe a difference](how-to/compare_two_runs.md#what-to-check-before-you-believe-a-difference).

**[Run a timing-only study](how-to/run_a_timing_only_study.md)**. Price
the decoders with cards instead of measuring them, so the ticks do not
depend on your computer.
Sections: [1. Write the config](how-to/run_a_timing_only_study.md#1-write-the-config) · [2. Run it](how-to/run_a_timing_only_study.md#2-run-it) · [3. Check that it is deterministic](how-to/run_a_timing_only_study.md#3-check-that-it-is-deterministic) · [What still runs for real](how-to/run_a_timing_only_study.md#what-still-runs-for-real).

## Reference

Look things up here. Four of these pages are generated from the source by
`tools/docs_map.py` and checked by `tests/test_docs.py`, so they cannot
drift from the code.

**[The yaml surface](reference/yaml.md)**. How to read
`configs/reference.yaml`, the `extends` rule, `decsim show`, and every
shipped config.
Sections: [What a config file is](reference/yaml.md#what-a-config-file-is) · [How to read configs/reference.yaml](reference/yaml.md#how-to-read-configsreferenceyaml) · [Starting from another file](reference/yaml.md#starting-from-another-file) · [Seeing what a file resolves to](reference/yaml.md#seeing-what-a-file-resolves-to) · [The shipped configs](reference/yaml.md#the-shipped-configs).

**[The run folder](reference/run_folder.md)**. Every file a run writes
and every column of every csv, including the latency points.
Sections: [What a folder holds](reference/run_folder.md#what-a-folder-holds) · [The columns of each file](reference/run_folder.md#the-columns-of-each-file).

**[The commands](reference/cli.md)**. Every subcommand and flag. Generated.
Sections: [decsim collect](reference/cli.md#decsim-collect) · [decsim combine](reference/cli.md#decsim-combine) · [decsim plot](reference/cli.md#decsim-plot) · [decsim run](reference/cli.md#decsim-run) · [decsim show](reference/cli.md#decsim-show) · [decsim trace](reference/cli.md#decsim-trace).

**[The plug-in tables](reference/tables.md)**. All seventeen tables and
every row of each. Generated.
Sections: [BOUNDARY_PAYLOADS](reference/tables.md#boundary_payloads) · [BOUNDARY_POLICIES](reference/tables.md#boundary_policies) · [CONFIDENCE_SIGNALS](reference/tables.md#confidence_signals) · [DECODERS](reference/tables.md#decoders) · [DETECTION_EVENT_FORMATION](reference/tables.md#detection_event_formation) · [ESCALATIONS](reference/tables.md#escalations) · [FRAMES](reference/tables.md#frames) · [IDLE_POLICIES](reference/tables.md#idle_policies) · [LINK_FABRICS](reference/tables.md#link_fabrics) · [MAGIC_STATE_FACTORIES](reference/tables.md#magic_state_factories) · [ROUND_STORES](reference/tables.md#round_stores) · [STRONG_WINDOW_SHAPES](reference/tables.md#strong_window_shapes) · [SYNDROME_SOURCES](reference/tables.md#syndrome_sources) · [THRESHOLD_SOURCES](reference/tables.md#threshold_sources) · [WINDOWING_SCHEMES](reference/tables.md#windowing_schemes) · [WINDOW_CHECKS](reference/tables.md#window_checks) · [WORKLOADS](reference/tables.md#workloads).

**[The ports](reference/ports.md)**. Every port, its methods and its
members, in pipeline order. Generated.
Sections: [the QPU emits a readout](reference/ports.md#the-qpu-emits-a-readout) · [the store holds the packed round](reference/ports.md#the-store-holds-the-packed-round) · [the window manager closes a window](reference/ports.md#the-window-manager-closes-a-window) · [the decoder manager schedules a decode](reference/ports.md#the-decoder-manager-schedules-a-decode) · [the decoder returns a result](reference/ports.md#the-decoder-returns-a-result) · [the frame commits the correction](reference/ports.md#the-frame-commits-the-correction) · [the frame releases the controller](reference/ports.md#the-frame-releases-the-controller) · [the controller instructs the QPU](reference/ports.md#the-controller-instructs-the-qpu) · [every hop rides a link](reference/ports.md#every-hop-rides-a-link) · [the pluggable policies off the path](reference/ports.md#the-pluggable-policies-off-the-path).

**[The map of the package](reference/map.md)**. Every package and module,
in uses order, with one sentence each. Generated.
Sections: [The package root](reference/map.md#the-package-root) · [Level 0: config, records, tables, trace_source](reference/map.md#level-0-config-records-tables-trace_source) · [Level 1: engine, pauli_frame, ports, seeding, syndrome_buffer](reference/map.md#level-1-engine-pauli_frame-ports-seeding-syndrome_buffer) · [Level 2: detector_error_model, escalation, links, windows](reference/map.md#level-2-detector_error_model-escalation-links-windows) · [Level 3: confidence, controller, decoders, qpu](reference/map.md#level-3-confidence-controller-decoders-qpu) · [Level 4: frontends, observe](reference/map.md#level-4-frontends-observe) · [Level 5: settings](reference/map.md#level-5-settings) · [Level 6: build](reference/map.md#level-6-build) · [Level 7: machine](reference/map.md#level-7-machine) · [Level 8: collect](reference/map.md#level-8-collect) · [Level 9: front](reference/map.md#level-9-front) · [Level 10: __main__](reference/map.md#level-10-__main__).

**[Glossary](reference/glossary.md)**. Every name decsim uses, what the
papers call it, and where to read it.
Sections: [The words this code means precisely](reference/glossary.md#the-words-this-code-means-precisely) · [The decoding regions](reference/glossary.md#the-decoding-regions) · [The two tiers and the confidence](reference/glossary.md#the-two-tiers-and-the-confidence) · [The two stores](reference/glossary.md#the-two-stores) · [The link paths](reference/glossary.md#the-link-paths).

## Explanation

Read these away from the keyboard.

**[Architecture](explanation/architecture.md)**. The components in
pipeline order, what each owns, the diagram, and the package order.
Sections: [The shape, and where it comes from](explanation/architecture.md#the-shape-and-where-it-comes-from) · [The components, in the order a readout travels](explanation/architecture.md#the-components-in-the-order-a-readout-travels) · [What each component owns](explanation/architecture.md#what-each-component-owns) · [The pluggable parts](explanation/architecture.md#the-pluggable-parts) · [The package order](explanation/architecture.md#the-package-order) · [The line where a call stops being local](explanation/architecture.md#the-line-where-a-call-stops-being-local).

**[The data path, hop by hop](explanation/data_path.md)**. The readout
across the eleven hops: what crosses, how many bits, and what it costs.
Sections: [Three words for what a transfer does](explanation/data_path.md#three-words-for-what-a-transfer-does) · [Every hop rides a link](explanation/data_path.md#every-hop-rides-a-link) · [The eleven hops](explanation/data_path.md#the-eleven-hops) · [Why the copies are where they are](explanation/data_path.md#why-the-copies-are-where-they-are) · [Reading a real path](explanation/data_path.md#reading-a-real-path).

**[Time](explanation/time.md)**. Ticks, the engine, clocks, priced cards,
the wall-clock decoder, and where the reaction time is measured.
Sections: [The tick](explanation/time.md#the-tick) · [The engine](explanation/time.md#the-engine) · [Clocks](explanation/time.md#clocks) · [Priced cards, and what a card is](explanation/time.md#priced-cards-and-what-a-card-is) · [The wall-clock decoder, and what it costs you](explanation/time.md#the-wall-clock-decoder-and-what-it-costs-you) · [Where the reaction time is measured](explanation/time.md#where-the-reaction-time-is-measured).

**[Windows and boundaries](explanation/windows_and_boundaries.md)**. Why
decoding is cut into windows, what a boundary is and costs, the four
windowing schemes, the seam.
Sections: [Why a window exists](explanation/windows_and_boundaries.md#why-a-window-exists) · [What a boundary is](explanation/windows_and_boundaries.md#what-a-boundary-is) · [How the boundary is priced](explanation/windows_and_boundaries.md#how-the-boundary-is-priced) · [When the boundary ships](explanation/windows_and_boundaries.md#when-the-boundary-ships) · [The four windowing schemes](explanation/windows_and_boundaries.md#the-four-windowing-schemes) · [The seam](explanation/windows_and_boundaries.md#the-seam).

**[Two tiers](explanation/two_tiers.md)**. Weak and strong decoders,
confidence, thresholds, the strong window's shape, restart, and the bill.
Sections: [The problem](explanation/two_tiers.md#the-problem) · [The two tiers](explanation/two_tiers.md#the-two-tiers) · [The confidence](explanation/two_tiers.md#the-confidence) · [The threshold](explanation/two_tiers.md#the-threshold) · [The strong window's shape](explanation/two_tiers.md#the-strong-windows-shape) · [Restart](explanation/two_tiers.md#restart) · [What it costs](explanation/two_tiers.md#what-it-costs).

**[The design decisions](explanation/decisions.md)**. The ten modelling
decisions, their reasons and sources, and what is not modelled yet.
Sections: [D1. Whoever executes a send is an end of that hop](explanation/decisions.md#d1-whoever-executes-a-send-is-an-end-of-that-hop) · [D2. The complementary gap is two forced-class jobs of one window](explanation/decisions.md#d2-the-complementary-gap-is-two-forced-class-jobs-of-one-window) · [D3. The complementary gap is the exact reference; the cluster gap is the real-time default](explanation/decisions.md#d3-the-complementary-gap-is-the-exact-reference-the-cluster-gap-is-the-real-time-default) · [D4. One decoder manager over both pools, and its work is not free](explanation/decisions.md#d4-one-decoder-manager-over-both-pools-and-its-work-is-not-free) · [D5. The strong buffer hop is a priced link like every other](explanation/decisions.md#d5-the-strong-buffer-hop-is-a-priced-link-like-every-other) · [D6. Pair placement is deferred](explanation/decisions.md#d6-pair-placement-is-deferred) · [D7. A priced weak card is charged once per forced solve](explanation/decisions.md#d7-a-priced-weak-card-is-charged-once-per-forced-solve) · [D8. The cluster gap's own walk is charged on the weak unit's clock](explanation/decisions.md#d8-the-cluster-gaps-own-walk-is-charged-on-the-weak-units-clock) · [D9. The boundary payload is priced dense by default](explanation/decisions.md#d9-the-boundary-payload-is-priced-dense-by-default) · [D10. The policy instance is the authority, not the settings row](explanation/decisions.md#d10-the-policy-instance-is-the-authority-not-the-settings-row) · [What is not modelled yet](explanation/decisions.md#what-is-not-modelled-yet).

**[The principles behind the shape](explanation/principles.md)**. The
eleven architecture ideas the tree is built on, each quoted from its
source.
Sections: [1. A module hides one decision that is likely to change](explanation/principles.md#1-a-module-hides-one-decision-that-is-likely-to-change) · [2. The uses relation is a partial order](explanation/principles.md#2-the-uses-relation-is-a-partial-order) · [3. An interface reveals as little as it can, and never an order it does not need](explanation/principles.md#3-an-interface-reveals-as-little-as-it-can-and-never-an-order-it-does-not-need) · [4. An interface is the set of assumptions two programs make about each other](explanation/principles.md#4-an-interface-is-the-set-of-assumptions-two-programs-make-about-each-other) · [5. An update is a function of its arguments, so a run can be replayed](explanation/principles.md#5-an-update-is-a-function-of-its-arguments-so-a-run-can-be-replayed) · [6. Model objects, a separate configuration script, a port API, timing apart from function](explanation/principles.md#6-model-objects-a-separate-configuration-script-a-port-api-timing-apart-from-function) · [7. A record that crosses a boundary lives in one shared place](explanation/principles.md#7-a-record-that-crosses-a-boundary-lives-in-one-shared-place) · [8. Different parts change at different rates, and the fast must not force the slow](explanation/principles.md#8-different-parts-change-at-different-rates-and-the-fast-must-not-force-the-slow) · [9. Complexity is essential, so remove the accidental and grow the rest](explanation/principles.md#9-complexity-is-essential-so-remove-the-accidental-and-grow-the-rest) · [10. The system copies the organisation that builds it](explanation/principles.md#10-the-system-copies-the-organisation-that-builds-it) · [11. Local and remote calls differ in kind, and the interface must say which](explanation/principles.md#11-local-and-remote-calls-differ-in-kind-and-the-interface-must-say-which) · [What is not here](explanation/principles.md#what-is-not-here).

## Where everything lives

| Folder | What is in it |
| --- | --- |
| `decsim/` | the simulator: twenty-five packages on eleven uses levels, mapped in [the map](reference/map.md) |
| `decsim/ports.py` | the thirty-four ports, the only way two packages talk |
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

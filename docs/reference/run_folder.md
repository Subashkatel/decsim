[decsim docs](../README.md) › [Reference](README.md)

# The run folder

Every `decsim run` writes one results folder,
`results/<date>_<name>/`, named for the day it started and the
experiment it ran, and a second one that day gets `_2`
(`decsim/experiments/run_folder.py`, `new_run_dir`), unless `--out`
names one. A launcher fixes the folder once and hands it to every task.
`results/` is output, not code, and git does not track it, which is
gem5's `m5out/`. `examples/deltakit_example.py` and
`examples/live_memory_example.py` write the same records into their
`--output` folder. A sinter experiment, `experiments/decoder_baseline/run.py`,
writes the same `run.json`, code state and script copy, and its points'
counts to `stats.csv` in sinter's columns (`shots`, `errors`,
`discards`, `decoder`, the labels in `json_metadata`); its rate per
round is sinter's `shot_error_rate_to_piece_error_rate` over the 100
rounds its script sets, as `sweep.csv`'s per-round columns are.

A results folder holds one experiment: its pieces, one record per
point, and the files folded from the pieces at its root. It holds one
tree's results, so a run or a launch into a folder whose `run.json`
names another commit, the same commit clean against dirty, or dirty
with another `code_state.patch`, is refused:

| Name | Written by | What it is |
| --- | --- | --- |
| `pieces/<id>/<first>-<last>/` | `decsim/experiments/pieces.py`, `write` | one piece: seeds `first` to `last` of one point, its additive files (`shots.csv`, `shot_links.csv`, `window_samples.csv`, `latency_samples.csv`, `shot_data_movement.csv`, `window_confidence.csv`, `confidence_histogram.csv`) without the swept columns, an online point's calibrator as the piece left it in `state.pickle`, which the point's next piece starts from, and `piece.json`. Its files are written into a hidden staging folder of the writer's own beside it and the folder is renamed into place last, so a piece folder exists only whole; of two writers of one piece the first to rename wins and the other drops its copy; a run skips a piece whose folder exists, and a staging folder a killed writer left is passed over |
| `piece.json` | `decsim/experiments/pieces.py`, `write` | the piece's `point_id`, `first_seed` and `count` (its other counts are its own `shots.csv` columns summed: `is_scored`, `logical_failure`, `sim_wall_seconds`, `executed_rounds` and the `*_windows` status columns), its `confidence_shot_count` (the shots its confidence rows cover, `all`, or null when no confidence signal ran; a fold refuses a point whose pieces differ in it, a piece without it included), the `state_sha256` of an online point's `state.pickle`, its `peak_memory_mb` (the peak resident memory of the process that ran it, read when the piece ended), and the `commit`, `dirty`, `patch_sha256` (the sha256 of the tree's `code_state.patch`, null when it has none), `python`, `packages` (each third-party top-level module the process that ran the shots had imported, and its version: the module's own `__version__`, else its installed distribution's; read in that process, so a pooled worker names the decoder package it loaded), `host`, `processor_model`, `slurm_job_id`, `slurm_array_job_id` and `slurm_array_task_id` of the process that ran it (a run and a fold refuse a folder holding a piece whose tree is not the one its `run.json` names: another commit, the same commit clean against dirty, or another patch, since its rows would pool two simulators under one `run.json`) |
| `run.sbatch`, `fold.sbatch`, `logs/` | `decsim/experiments/plan_command.py`, `launch` | what `decsim run --slurm` writes: the array whose task `i` runs point `i` of the run file where it stands, so a file beside it is found, the job that folds after it, and their Slurm logs, `logs/<i>.log` and `logs/fold.log` |
| `points/<name>/machine.json`, `points/<name>/inputs/` | `decsim/experiments/run_folder.py`, `write_point_record` | each point's record and workload, below, written before any shot and only once every point of the experiment has built and been accepted, so a refused one changes no record. A name the folder holds for a point of another id is refused. The record also holds `rounds_per_shot`, the QEC rounds the plan gives a shot, each patch's rounds added up over every operation that sends detector data, which sizes the point's pieces (a live stream also idles through its feedback wait, rounds only its run knows), and `experiment`: its `collection`, whether its threshold is `adaptive`, and the `algorithm` that decodes its windows, which `decsim run --fold` folds it by, whatever its run file says later |

The root files are the fold: a run and `decsim run --fold` build the whole
fold in a staging folder beside it and move it in in place of the
earlier fold's files, so a refused fold leaves the last one as it was.

A run folder holds facts that add up and nothing else. No summary is
stored: `sweep.csv` is computed from the additive files when it is
written, and a run folds it over every piece, so a sweep cut into
pieces, stopped and picked up again, gives the numbers one uncut run
gives (`decsim/experiments/report.py`, `fold_pieces`).

## What a folder holds

| Name | Written by | What it is |
| --- | --- | --- |
| `shots.csv` | `decsim/experiments/report.py`, `shot_rows` | one row per shot |
| `shot_links.csv` | `decsim/experiments/report.py`, `shot_link_rows` | one row per shot per link |
| `window_samples.csv` | `decsim/experiments/report.py`, `window_sample_rows` | one row per sweep point, latency point, committing tier and distinct microsecond value |
| `latency_samples.csv` | `decsim/experiments/report.py`, `latency_sample_rows` | one row per decoded window of a decoder named by a table row, written only when one ran |
| `window_confidence.csv` | `decsim/experiments/report.py`, `window_confidence_rows` | one row per committed window of the scored shots among the first `confidence_shot_count` shots of a point (its `record_options`, a `decsim.collect.RecordOptions`, 100 by default), written only when a confidence signal decides the escalation |
| `confidence_histogram.csv` | `decsim/experiments/report.py`, `confidence_histogram_rows` | counts of every scored shot's window gaps and smallest gap per 0.1 dB bin, written only when a confidence signal decides the escalation |
| `sweep.csv` | `decsim/experiments/report.py`, `fold_pieces` | one row per sweep point, in the sweep's task order, summarized from `shots.csv` and `window_samples.csv` |
| `shot_data_movement.csv` | `decsim/experiments/report.py`, `shot_data_movement_rows` | one row per shot per path: that shot's copy and move counters and the memory class the path crosses, written only when `observation.data_movement` is on. A point's mean per shot on a path or a memory class is a counter's sum over those rows divided by the point's distinct seeds; `references` and `referenced_rounds` repeat on every row of a shot, so they count once per seed |
| `run.json` | `decsim/experiments/run_folder.py`, `write_run_record` | one object: what ran, where, and with which library versions |
| `<run file>.py` | `decsim/experiments/run_folder.py`, `snapshot_code_state` | a verbatim copy of the run file, under its own name beside the results. The copy is written once: a later run into the folder must bring the same text, or a run file whose points have the ids `run.json` recorded, which differs only in how far they run (a pilot's caps raised) and replaces the copy; any other run file is refused, and so is an array task's run file that is not the copy, since its task `i` may name another point |
| `code_state.patch` | `decsim/experiments/run_folder.py`, `snapshot_code_state` | `git diff HEAD`, and a patch creating each untracked file git does not ignore, written only when there is either; its sha256 is `run.json`'s `git.patch_sha256` |
| `points/<name>/machine.json` | `decsim/experiments/run_folder.py`, `record_point` | one per point, in a folder named by the point's name: its content `id`, its `name`, its metadata, the seeds this folder ran of it (ranges of first and how many, joined from its pieces), every setting, each record under its `class` (module and qualified name) beside its fields, as gem5's `config.json` writes each object's type, so two records with the same fields are two points, and the values the build derives from them (`built`: the code card, the window sizes a null resolves to, the rows the plan built, the run plan) |
| `points/<name>/inputs/` | `decsim/experiments/run_folder.py`, `record_point` | the workload the point ran, as the `files` workload row reads it (`operations.json`, and `circuit.stim` with `measurement_rounds.json` or `fragments/`), and `hashes.json`, each file's sha256 |
| `result.json` | `decsim/experiments/run_folder.py`, `write_shot` | `decsim run --seed` and the two Deltakit examples only: every field of the shot's result |
| `trace/<id>_seed<seed>.trace.json` | `decsim/observe/trace_writer.py` | one Chrome trace per traced shot, named by its point's id and its seed (`decsim/experiments/measure.py`, `shot_label`), so two points never share a file. How long the data sat is read from it: a structure's stays are the complete events on its lane whose `cat` holds `residence`, each from `args.tick` for `dur` microseconds, a link path's waits are the `args.queue_wait_ticks` of the complete events on its lane, whose `cat` holds `link`, and a store's accesses are the complete events on its `<store> port <i>` lanes whose `cat` holds `access`, a write or a read each, with the tick it asked for the port in `args.arrival` and its wait for the port in `args.waited_ticks`. A decode's time on a unit is the complete event `W<n> service` on that unit's lane, and each verdict the escalation gives is an instant `verdict` on the `Window planner` lane, its `args.verdict` `keep` or `escalate` |
| `log/<id>_seed<seed>.log` | `decsim/experiments/measure.py`, and `decsim/experiments/run_folder.py` for one shot | the engine narrator's lines, written when the `observation` section asks for a log |
| `online_threshold_<id>.csv` | `decsim/experiments/collect_command.py` | the online threshold's trajectory at one point, written when the point's `switching.threshold` is an `OnlineThreshold.Settings`: `point_id`, the swept paths and `algorithm`, then `window_count`, `threshold_db` and `event` per audit, target move and hundredth window, and an `end` row |
| `threshold_summary.csv` | `decsim/experiments/collect_command.py`, `_threshold_summary_row` | one row per online point, the counters its stderr line prints and the rest of the calibrator's summary at the end of its prefix: `point_id`, the swept paths and `algorithm`, then `windows`, `escalated`, `escalation_rate`, `target_escalation_rate`, `audited`, `audited_bad`, `kept_bad_rate`, `raises`, `relaxes`, `pending_audits` and `threshold_db` |

`run.json`, the run file's copy and the patch together are the whole
experiment: the commit plus the patch is the code, and the run file is
the input. `machine.json` says what every value came to at each point,
and `inputs/` holds each point's workload, so a point reruns with the
`files` row pointed at `points/<name>/inputs/operations.json` (and its
circuit keys) without the maker installed.

## The columns of each file

The additive files first, because the two summaries are made of them.

### `shots.csv`

One row per shot. The first columns are the shot's own scalars, and then
every latency point appears twice, once as that shot's mean and once as
its maximum.

A point is named by `point_id`, its content id, as sinter's csv names
a task by its `strong_id` (`sinter/_data/_csv_out.py:69-77`); it is
also the `id` in its `machine.json`. Right after it come the
point's swept values, one column per swept path, a name the points'
metadata holds (`qpu.distance`,
`workload.arguments.physical_error_probability`), in the order the
points first set them (`decsim/experiments/run_folder.py`,
`swept_values`). Every other file below names its point by the same
columns, and `algorithm` follows them: the `name` of the decoder record
of the tier that decodes the plan's windows, such as `pymatching`, or a
priced row's latency in microseconds in its place
(`decsim/experiments/measure.py`, `active_decoder_kind`). The values the design fixed come
first and what was measured after, one variable per column, which is
Wickham's tidy table (Tidy Data, J. Stat. Softw. 59(10), 2014, section
2.3), so a reader groups, filters and plots by a column with no parsing.

Every cell is the point's metadata value at that name, and a point
whose metadata does not hold the name has an empty cell. A string or a
number is written as itself, and any other value (a flag, a null, a
whole record) as one cell of compact json with its keys sorted, as
sinter writes `json_metadata` (`sinter/_data/_csv_out.py:35-37`). The
typed value is in the point's `machine.json`. A column's unit is in its
name (`_us` microseconds, `_bits`, `_per_shot`).

| Column | What it is |
| --- | --- |
| `point_id`, the swept paths, `algorithm`, `seed` | the sweep point and the seed, which together name the shot |
| `decoded_windows` | how many windows this shot decoded |
| `logical_failure` | `True` when any scored owner's decoded observable did not match its truth, else `False`; a scored owner is an operation the source sampled a truth for, and a live stream's segments and the operations that only hold or resume its patch are scored through the stream's owner. A `memory_patches` shot fails when any patch does. An unscored shot is never a failure, as sinter never counts an error on a discarded shot |
| `load` | service time per window divided by the interval between windows arriving; above 1 the decoder cannot keep up |
| `throughput_windows_per_us`, `throughput_rounds_per_us` | what the machine got through |
| `max_queued_windows` | the most jobs that waited in the ready queue at once; a depth counts only when time passes at it, so a job that joins and leaves in one tick never waited |
| `weak_queue_max`, `strong_queue_max` | the most jobs that waited in each tier's ready queue at once, by the same rule. The tier that decodes the planned windows owns the default pool's number, so under `strong_only` that number is in the strong column. A tier the run does not build reads zero. |
| `weak_busy_fraction`, `strong_busy_fraction` | the time-weighted fraction of each tier's units whose compute was busy |
| `escalated_windows`, `strong_decoded_rounds`, `strong_service_sum_us` | the windows the strong tier committed, the rounds its decodes read, and their service added up; a sweep point divides the sum by the windows |
| `commit_rounds` | r_com, the rounds a window commits, as the code card the QPU ran sizes it: `windows.scheme.commit_rounds`, or the card's own when it is null (the code distance for both shipped cards) |
| `window_period_us` | a window's inter-arrival, `commit_rounds` times the round period the QPU ran (the card's own when it has one): what `load` divides by, and the deadline a window's decode must beat |
| `parallel_processes_needed` | Skoric's least count of parallel decoding processes for no backlog, ceil(2 tau_W / ((n_com + n_W) tau_rd)) from this shot's mean service (2209.08552 lines 429-438) |
| `weak_syndrome_weight_mean`, `weak_syndrome_weight_max` | the set bits of each weak decode's input, its detection events when they are formed ahead of the decoder; only when `observation.record_switching_windows` is on |
| `weak_service_mean_us` | each weak decode's compute, its first stage's start to its last stage's end; the same switch |
| `strong_wait_mean_us`, `strong_wait_max_us` | each strong decode's wait from its enqueue to its compute start, for a unit and for the unit's compute; the same switch |
| `strong_held_in_units_max` | the most strong decodes held in the units' memory at once, landed and free to compute but waiting for a unit's compute, by the rule of the queue peaks. A unit takes the next decode into its memory while it computes, so this wait never shows in `strong_queue_max`; the same switch |
| `backlog_peak_rounds` | the most rounds produced and not yet decoded at once; only when `observation.backlog_trace` is on |
| `referee_windows_checked`, `referee_window_disagreements` | the referee's count, when the decoder's record is wrapped in `TesseractCheckedDecoder.Settings` |
| `sim_wall_seconds` | how long the simulation itself took to run, on the host |
| `is_scored` | whether every decode a window committed, provisional or final, got a correction from its backend. A backend that produced none (it raised, returned a vector that is not a correction, or found no correction at all) commits an empty correction in its place, and its shot is unscored. An escalated window's weak answer is committed provisionally before the strong one replaces it, and the replacement does not undo what the provisional commit fed forward: its boundary, when `windows.boundary_policy` ships provisional boundaries (a shipped one is never revised), and its crossing commit, which the strong result keeps. decsim does not trace which of those reached a later decode, so a replaced provisional decode with no correction unscores the shot too. A provisional result never reaches the Pauli frame |
| `provisional_no_correction_windows` | how many windows committed a provisional decode with no correction that the strong answer then replaced; the status columns below count final decodes and do not show these |
| `unscored_reason` | the backends' reasons for the windows committed with no correction, each once, sorted and joined by `;` (`BackendFailureReason` in `decsim/records/decoding.py`: `upstream_exception`, `correction_not_binary`, `correction_wrong_arity`, `nonzero_syndrome_without_faults`, `no_perfect_matching`); empty on a scored shot |
| `executed_rounds` | the rounds the scored owners read out this shot, each patch's rounds added up; a live stream's include the rounds it idled through while its feedback waited |
| `scored_outputs`, `rounds_per_output` | the scored owners, each an independent output the shot fails on when it is wrong, and the patch-rounds each ran, 0 when they ran different counts; the per-round rate reads them |
| `sample_digest` | the sha256 of every scored owner's sampled detection events and observable truth, one byte a bit, in operation order. Two points that differ only in their decoder hold the same digest at the same seed, so pairing their shots can be checked rather than assumed |
| `predictions` | every operation's predicted observables as the loop decoded them, compact sorted json keyed by operation id, each value one character a bit in observable order (Stim's `01` format), null for an operation left unanswered: `{"1":"0","2":"1"}`. Two paired shots compare answer by answer here; with several observables or operations, two failures can be two different answers |
| `<status>_windows` | one count per status a window's decode may carry besides success, `low_confidence_windows`, `nonconverged_windows`, `invalid_correction_windows`, `empty_model_unsatisfiable_windows` and `backend_error_windows` (`BackendDecodeStatus` in `decsim/records/decoding.py`): how many of the shot's windows committed a decode with that status. A window counts its final decode, so an escalated window counts the strong answer's status and not the weak one it replaced |
| `<point>_mean_us`, `<point>_max_us` | one pair per latency point below |

The latency points are the tuple `POINTS` in `decsim/experiments/measure.py`,
and they are the same names in `shots.csv`, `window_samples.csv` and
`sweep.csv`:

| Point | From, to |
| --- | --- |
| `cwb_per_round` | the controller to the weak syndrome buffer, one round: latency, serialization and queue |
| `cwb_stall_per_round` | the packed round finding the weak syndrome buffer full, to the freed slot that admitted it: the store's back-pressure on the controller, zero for a round that found room |
| `csb_stall_per_round` | the same wait in front of the strong syndrome buffer, one sample per round that reached it |
| `buffer_fill` | the first round of a window arriving, to the last: the wait on the QPU |
| `admission_wait` | the window's data complete in the weak syndrome buffer, to its decode job entering the queue: the window side's decision (`windows.decision_cycles`), and any earlier request of the window that was withdrawn, as a restart window's is when a `double_window` strong window re-slices it; zero otherwise |
| `dep_block` | the input landing in the unit's memory, to the first tick the decode may compute: the dependency wait, for the predecessor's boundary and for the escalation message, and zero when nothing was owed at the landing |
| `compute_wait` | that first startable tick, to the compute starting: the wait for the unit's own compute, busy with another decode |
| `queue_wait` | queued, to a unit assigned |
| `input_link_per_window` | a unit assigned, to the input in that unit's memory |
| `fetch` | the unit reading the window out of its own memory |
| `algorithm` | the decoding algorithm itself |
| `backend_queue_wait` | inside `algorithm`: the time a strong backend's decode waited for its dispatcher or a worker (`decsim/decoders/strong_backend.py`). `algorithm` less this is the GPU's own steps: launch, copies and kernel. Zero for a decoder with no queue of its own |
| `release` | the unit writing the correction out |
| `service` | the compute start, to the decode done: the fetch, the algorithm and the release, and nothing the decode waited for outside the unit. A strong backend's queue holds the unit, so `backend_queue_wait` is inside |
| `confidence` | the committing decode's end, to the verdict on the window's answer: the confidence signal's own computation, which is the walk under `cluster_gap` and the other forced-class solve's remaining time under `complementary_gap`, and zero for an escalated window whose strong decode began after the verdict |
| `selection_wait` | the verdict, or the committing decode's end when later, to its answer leaving for the frame: a finished strong answer kept in its unit's output slot until the verdict's selection has crossed `weak_decoder_to_strong_decoder`, which a speculative strong decode under `run_both_at_once` waits for, and zero for every other window |
| `weak_attempt` | a unit taking an escalated window's weak job, to the verdict that escalated it: the attempt whose result did not commit, zero when the first decode committed |
| `escalation_link_per_window` | the weak decoder to the strong decoder: the escalation hop, from the selection's send to the landing of the rounds the strong store lacked, zero for a window that did not escalate |
| `dd_per_window` | one decoder to the next: the boundary handoff |
| `output_link_per_window` | the decoder to the Pauli frame |
| `frame_commit` | the frame accepting a correction, to it being committed |
| `buffer0_ready_to_frame` | the window complete in the syndrome buffer its first decode reads (the weak one, or the strong one in a run with no weak decoder), to the frame |
| `buffer0_first_round_to_frame` | the window's first round in that syndrome buffer, to the frame |
| `qpu_last_round_to_frame` | the last round the window needs leaving the QPU, to the frame |
| `qpu_first_round_to_frame` | the window's first round leaving the QPU, to the frame |

The two `buffer0` totals start the clock where the decoder could first
have started. The two `qpu` totals start it where the physics did, so
they also carry the link out of the fridge, the controller's own work
and the write into the weak syndrome buffer.

Every point of a window describes the decode whose result the frame
committed, which the frame's own record names by tier and by request
ordinal: for a window the strong tier recovered, the two link points are
the strong tier's hops and the stage points are the strong decode's. A
window can be decoded more than once, and then every tick still belongs
to one decode.

- `admission_wait` runs from the window's data complete to its job
  entering the queue.
- `queue_wait` ends where a unit took the window's first decode.
- `weak_attempt` runs from there to the verdict that sent the window to
  the strong tier. It is zero when the first decode is the one that
  committed.
- `input_link_per_window` is the committing decode's own hop in.
- `dep_block` and `compute_wait` are the two halves of its park, the
  wait between its input landing and its compute starting.
- `service` is its compute, and `output_link_per_window` its way home.
- `confidence` is the signal the verdict needs, computed after that
  decode ended.
- `selection_wait` is a strong answer that finished first waiting for
  the verdict's selection to reach it.
- `frame_commit` closes it.

On a serial path those eleven add
up to `buffer0_ready_to_frame` to the tick, on every window of every
config this repository ships.

The park is two points because it has two causes.
A run whose windows wait on the seam reports the park in `dep_block` and
zero in `compute_wait`; two decodes of one window sharing a unit, which
is what a complementary gap's forced-class pair is, report it in the
second. A strong decode whose rounds crossed with the escalation waits
for them before its input hop can start, and that wait is `dep_block`
too: the point runs from the verdict to the first startable tick, less
the input hop itself.

`input_link_per_window` is that decode's own hop, so it is zero when the
decode read an input that was already in its unit's memory: a tier
reading in place, and the second forced-class solve of a complementary
gap reading what the first solve brought. The rounds still crossed the
link once and `shot_links.csv` still counts that crossing; `dep_block`
then starts where the decode's own path started, at the dispatch or at
the verdict, and runs to the tick that input was readable.

One run is outside that sum, and knowingly: under
`switching.run_both_at_once` the weak attempt and the strong decode
overlap rather than follow each other, so adding both would count the
same wall time twice. `tests/experiments/test_measure.py` asserts the
identity window by window on `decsim.settings.weak_decoder_baseline`,
`examples/two_tiers.py`, and the `redo_window_d3` and `cluster_gap_d3`
points of `experiments/switching/run.py`, which are a run with no signal to compute, a run whose signal is a second
forced-class solve, the same on a host-clock strong tier, and a run
whose signal is a priced walk.

`escalation_link_per_window` is a hop that is measured and not summed,
like `dd_per_window`: one span from the selection's send to the landing
of the rounds the strong store lacked, the two transfers side by side,
and what it makes the strong decode wait for is in `dep_block`.

### `shot_links.csv`

One row per shot per link path, straight off that shot's traffic
counters.

| Column | What it is |
| --- | --- |
| `point_id`, the swept paths, `algorithm`, `seed` | the shot |
| `link` | the link path's name, one of the values in `decsim/records/transfers.py` |
| `transfers` | how many transfers crossed that path |
| `payload_bits` | how many bits they carried |
| `unknown_payload_transfers` | transfers whose payload size the sender could not state |
| `queue_wait_us`, `serialization_us`, `propagation_us` | the three parts of the time the path charged |

The link set is data, not schema: a topology with another link adds rows
here and moves no column of any file. A link's mean per shot at a point
is the mean of a column over that point's rows of the link, and its bits
per transfer is the mean `payload_bits` over the mean `transfers`.

### `window_samples.csv`

One row per sweep point, latency point, tier and distinct microsecond
value.

| Column | What it is |
| --- | --- |
| `point_id`, the swept paths, `algorithm` | the sweep point |
| `name` | which latency point, from the list above |
| `tier` | the tier whose decode the frame committed for the windows counted here: `weak` for a kept window and `strong` for an escalated one, when the weak tier decodes the run's windows. Empty for the three `_per_round` points, whose samples are rounds, not windows |
| `value_us` | one microsecond value that occurred |
| `count` | how many windows carried it |

This is the multiset of a point's window samples, one per tier. A median
and a p99 need nothing more, and one piece records nothing more for
another process to reach the same numbers. The columns in `sweep.csv`
add the tiers together, except the formed-to-commit split below.

### `latency_samples.csv`

One row per decoded window, written only for a decoder named by a table
row: the time that held the unit, its measured wall clock or its own
cycle count. A decoder priced by a number produces no rows, and a run
with no rows writes no file.

| Column | What it is |
| --- | --- |
| `point_id`, the swept paths, `algorithm`, `seed` | the shot |
| `tier` | the tier whose decode the frame committed for the window, as in `window_samples.csv` |
| `algorithm_us` | the time the algorithm stage held the unit for one decode: its wall clock, or its cycle count |
| `window_period_us` | the shot's window inter-arrival, the deadline a window's decode must beat |

### `window_confidence.csv`

One row per window whose confidence the escalation verdict read, for
the shots of seed 0 up to the point's `confidence_shot_count`, 100 by
default; `decsim.Point(..., record_options=decsim.collect.RecordOptions(
confidence_shot_count=None))` writes every scored shot's (`all`). A run whose escalation reads no confidence writes no file.
A window has no truth of its own, so a row carries its shot's failure
and whether the strong decode changed the window's answer. An unscored
shot writes no row: it is sinter's discard, neither a failure nor a
success, and sweep.csv's `unscored_shots` counts it.

| Column | What it is |
| --- | --- |
| `point_id`, the swept paths, `algorithm`, `seed` | the shot |
| `signal` | the confidence the verdict read, the `name` of the run's `switching.confidence` record (the signals are under `decsim/confidence` in [parts.md](parts.md)) |
| `operation_id`, `window_index` | the window: its operation and its index within it |
| `gap_nats` | the window's gap, ln of the likelihood ratio; empty when the signal gave none, a window the escalation then escalates |
| `escalated` | whether the verdict sent the window to the strong tier |
| `strong_revised` | for an escalated window, whether the strong decode predicted other observables than the weak one; empty for a kept window |
| `shot_failed` | whether the shot ended in a logical failure |

### `confidence_histogram.csv`

Counts over every scored shot of a point, whatever `confidence_shot_count`
says, so the counts of pieces add, as sinter's `custom_counts` do. An
unscored shot is counted in no cell, as sinter keeps a discard out of
every count it conditions on failure. A gap is
binned in decibels, `dB = nats x 10 / ln 10`, to the tenth below it; an
infinite gap has its own bin.

| Column | What it is |
| --- | --- |
| `point_id`, the swept paths, `algorithm` | the sweep point |
| `signal` | the confidence the verdict read |
| `histogram` | `window`, every window's gap, or `shot_minimum`, each shot's smallest window gap |
| `gap_low_decibels` | the bin's lower edge in decibels; empty for a window with no gap, and for a shot one of whose windows had none |
| `escalated` | for a `window` row, whether the window escalated; empty for a `shot_minimum` row |
| `shot_failed` | whether the shot ended in a logical failure |
| `count` | how many windows, or shots, fell in that cell |

The `shot_minimum` rows give Toshio et al.'s gap density and the
failure rate at each gap (2510.25222), and Gidney et al.'s minimum over
a span's draws (2312.04522).

### `sweep.csv`

One row per sweep point, summarized from `shots.csv` and
`window_samples.csv` by `summarize_point` in `decsim/experiments/report.py`.
The scalar columns come first, then four columns for every latency
point the run held:

| Column | What it is |
| --- | --- |
| `point_id`, the swept paths, `algorithm` | the point |
| `shots` | how many shots the point ran |
| `windows_per_shot` | the mean over those shots |
| `logical_failures` | the failures among every shot the point holds |
| `scored_shots`, `unscored_shots` | how many of the point's shots were scored, and how many were not (`is_scored`) |
| `state` | why the point's contiguous prefix of seeds stopped, by its collection's rule (`decsim/experiments/collection.py`): `target` (its scored failures reached `max_failures` past `min_shots`), `minimum` (the target was reached by `min_shots`, which stopped it), `cap` (the shot cap, before the target; incomplete), `time cap` (`max_core_seconds`, before the target and the shot cap; a cap's limits, exact only when a shot's run time does not depend on whether it failed), `running` (not stopped: a collect that ended early, or a missing seed that holds the stop), `adaptive` (a threshold that learns online; its shots are not independent draws), `no data`. A point a Python caller ran with its shots fixed in advance is a `cap` |
| `logical_error_rate_estimate` | the prefix's failures over its scored shots, sinter's errors over shots less discards; empty with no scored shot, at a cap with no failure, and for an adaptive point |
| `logical_error_rate_low`, `logical_error_rate_high` | its 95 percent limits, exact for the rule the prefix stopped by (`estimate` in `decsim/experiments/failure_statistics.py`): Clopper and Pearson's at a cap or minimum, Jennison and Turnbull's beta quantiles at a target; at a cap with no failure the upper limit alone, 1 - 0.025^(1/n) |
| `logical_error_rate_plan_unbiased` | Girshick, Mosteller and Savage's estimate, unbiased over every outcome of a stopping plan fixed in advance in counts: (r - 1)/(n - 1) at a target stop, x/n otherwise. It is not unbiased among the outcomes that reached the target, and it is empty under a time cap, for a prefix still running and for an adaptive point |
| `logical_error_rate_per_round`, `_per_round_low`, `_per_round_high` | the estimate and its limits as one output's rate for one round: sinter's shot_error_rate_to_piece_error_rate with `values` the `scored_outputs` (a shot fails when any output does, so each survives with the shot's survival to the power one over the outputs) and `pieces` the `rounds_per_output`, a two-code circuit's rounds added up as Tesseract 2503.10988 counts them; empty when the prefix's shots ran different shapes, or its outputs different counts, which no one shape converts |
| `is_shot_rate_above_half` | whether the estimate is past one half; with one output, there the per-round rate goes through its complement and, for an even round count, no round-flip probability gives the shot rate |
| `prefix_shots`, `prefix_scored_shots`, `prefix_failures` | the prefix's counts, which the estimate and its limits read; a pool may run pieces past the stop, which count in `shots` and not here |
| `logical_error_rate_unscored_as_failures` | the failures and the unscored shots together over every shot: the rate this sample would read if every unscored shot had failed, a bound on the sample and not a confidence bound. Beside the estimate, which is conditional on scoring, it shows how much a backend that failed on hard syndromes could hide |
| `<status>_windows`, `provisional_no_correction_windows` | the sums over the point's shots |
| `throughput_windows_per_us`, `throughput_rounds_per_us` | the means |
| `max_queued_windows` | the deepest queue over the point |
| `weak_queue_max`, `strong_queue_max` | the deepest each tier's own queue over the point |
| `weak_busy_fraction`, `strong_busy_fraction` | the mean busy fractions |
| `escalated_windows`, `strong_service_mean_us` | the strong tier's windows over the point, and the point's summed `strong_service_sum_us` over those windows: every strong decode weighs the same, whichever shot ran it. Empty when nothing escalated |
| `strong_service_bound_us` | Toshio's Theorem 1 bound on one strong decode's time, the unit of `strong_service_mean_us`: tau_gen times the point's generated rounds (its shots' `executed_rounds`) over its escalated windows, which is d tau_gen over the switching rate per d rounds, with tau_gen the shots' `window_period_us` over their `commit_rounds` (2510.25222 eq. (6)); generated rounds and not committed windows, since a double window absorbs windows whose rounds were still generated; infinite when nothing escalated |
| `parallel_processes_needed` | the largest over the point's shots |
| `weak_syndrome_weight_mean`, `weak_service_mean_us`, `strong_wait_mean_us` | the means over the point's shots, when they kept the switching records |
| `weak_syndrome_weight_max`, `strong_wait_max_us`, `strong_held_in_units_max`, `backlog_peak_rounds` | the largest over the point's shots, when they kept the records |
| `escalated_fraction` | the windows the strong tier committed over the windows decoded, beside the columns above |
| `referee_windows_checked`, `referee_window_disagreements` | the referee's totals |
| `load` | the mean load |
| `sim_wall_seconds_per_shot` | what the simulation cost to run |

Then four columns per latency point, in the order of `POINTS`:
`<point>_mean_us`, `<point>_median_us`, `<point>_p99_us` and
`<point>_max_us`. The mean and the max fold over the shot rows; the
median and the p99 come from the sample counts, through
`percentile_of_counts`.

Then `buffer0_ready_to_frame_<tier>_median_us` and
`buffer0_ready_to_frame_<tier>_p99_us` for each tier that committed a
window at the point: the formed-to-commit time of the kept windows
(`weak`) and of the escalated ones (`strong`), each over that tier's own
sample counts. A tier with no window has no column.

### `run.json`

One object. Its keys, from `write_run_record` in
`decsim/experiments/run_folder.py`:

| Key | What it is |
| --- | --- |
| `run_files` | the run file; empty for a run no file describes |
| `points` | the experiment's point ids in its order: the order a fold writes its rows in |
| `git` | the `commit`, whether the checkout was `dirty`, and `patch_sha256`, the sha256 of its `code_state.patch` (null when it has none), read once when the process started; a run or a fold stops on a `run.json` or `piece.json` without `patch_sha256`, which an older tree wrote |
| `container` | the container image, when one was in use |
| `versions` | the Python version, and `packages`: every installed package and its version |
| `compiled_libraries` | every compiled library a loader the run imported names, keyed by its absolute path, each its sha256, since a library is built and not tracked and the commit does not name it; a loader may take its file from outside the package (the environment variable the Union-Find row reads, `LIBRARY_VARIABLE` in `decsim/decoders/union_find/compiled_decoder.py`), and a named file not built is left out |
| `host`, `slurm_job_id` | where it ran |
| `argv` | the command line as it was invoked |
| `started_utc`, `finished_utc` | when |

`run.json` is written twice, once when the run starts and once when
it ends with `finished_utc` filled in, and both writes name the same
tree: the reading is taken once, before the first shot, and reused. A
tree that moves while a run is going, which is what an array running
for hours out of a checkout somebody commits to does, would otherwise
leave every folder naming code that no part of the run read.

## Read next

- [Your first sweep](../tutorials/first_sweep.md): a sweep, its pieces and its error bars.

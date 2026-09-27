[decsim docs](../README.md) › [Reference](README.md)

# The run folder

Every `decsim collect` and every `decsim run` writes one folder under
`results/`, named for the UTC time it started and the config it ran, and
never reused (`decsim/experiments/run_folder.py`, `new_run_dir`), unless
`--out` names one. `results/` is output, not code, and git does not
track it, which is gem5's `m5out/`. `tools/deltakit_example.py` and
`tools/live_memory_example.py` write the same records into their
`--output` folder.

A `decsim collect` folder is an experiment folder. It holds the
experiment's pieces, and one run folder per configuration folded from
them:

| Name | Written by | What it is |
| --- | --- | --- |
| `pieces/<id>/<first>-<last>/` | `decsim/experiments/pieces.py`, `write` | one piece: seeds `first` to `last` of one sweep point, its additive files (`shots.csv`, `shot_links.csv`, `window_samples.csv`, `latency_samples.csv`, `shot_data_movement.csv`) without the swept columns, the `residence.csv` rows of its traced shots when it traced any, an online point's calibrator as the piece left it in `state.pickle`, which the point's next piece starts from, and `piece.json`. Its files are written into a hidden staging folder of the writer's own beside it and the folder is renamed into place last, so a piece folder exists only whole; of two writers of one piece the first to rename wins and the other drops its copy; a collect skips a piece whose folder exists, and a staging folder a killed writer left is passed over |
| `piece.json` | `decsim/experiments/pieces.py`, `write` | the piece's `point_id`, `first_seed` and `count`, its `scored_shots`, `failures`, `unscored_shots` and `core_seconds` (its shots' own wall time), its window status counts (the summary's `*_windows` status columns and `provisional_no_correction_windows`), its `configuration_id` and its `rounds` (its shots times the point's rounds per shot), the `state_sha256` of an online point's `state.pickle`, its `peak_memory_mb` (the peak resident memory of the process that ran it, read when the piece ended, which a later round's memory request is sized by), and the `commit`, `dirty`, `host` and `slurm_job_id` of the process that ran it |
| `resolved/<id>.json`, `inputs/<id>/` | `decsim/experiments/run_folder.py`, `record_point` | each point's record and workload, written before any shot, as a run folder holds them below; the record also holds `rounds_per_shot`, the QEC rounds of a shot, which sizes the point's pieces |
| `configurations.csv` | `decsim/experiments/run_folder.py`, `record_configuration` | one row per configuration: its `configuration_id` (the sha256 of its sections, the sweep and the collection left out), its `name` and its `config_chain` |
| `combined/<name>-<id8>/` | `decsim/experiments/collect_command.py`, `run_experiment` | the configuration's run folder, named by its yaml and the first eight characters of its configuration id |

A run folder holds facts that add up and nothing else. No summary is
stored: `sweep.csv` and `links.csv` are computed from the additive files
when they are written, and a collect folds them over every piece, so a
sweep cut into pieces, stopped and picked up again, gives the numbers
one uncut run gives (`decsim/experiments/report.py`, `fold_pieces`).

## What a folder holds

| Name | Written by | What it is |
| --- | --- | --- |
| `shots.csv` | `decsim/experiments/report.py`, `shot_rows` | one row per shot |
| `shot_links.csv` | `decsim/experiments/report.py`, `shot_link_rows` | one row per shot per link |
| `window_samples.csv` | `decsim/experiments/report.py`, `window_sample_rows` | one row per sweep point, latency point and distinct microsecond value |
| `latency_samples.csv` | `decsim/experiments/report.py`, `latency_sample_rows` | one row per decoded window of a decoder named by a table row, written only when one ran |
| `sweep.csv` | `decsim/experiments/report.py`, `summarize` | one row per sweep point, in the sweep's task order, summarized from `shots.csv` and `window_samples.csv` |
| `links.csv` | `decsim/experiments/report.py`, `link_rows` | one row per sweep point per link, averaged over that point's shots |
| `shot_data_movement.csv` | `decsim/experiments/report.py`, `shot_data_movement_rows` | one row per shot per path: that shot's copy and move counters and the memory class the path crosses, written only when `observation.data_movement` is on |
| `data_movement.csv` | `decsim/experiments/report.py`, `data_movement_rows` | one row per sweep point per path, then per memory class, averaged over the point's shots |
| `residence.csv` | `decsim/experiments/residence.py`, `write_residence` | one row per traced shot per structure, then per link path: how long a round or window sat there, and how long a move waited on the wire |
| `manifest.json` | `decsim/experiments/run_folder.py`, `write_manifest` | one object: what ran, where, and with which library versions |
| `config/` | `decsim/experiments/run_folder.py`, `snapshot_code_state` | a verbatim copy of every yaml file in the config chain, each at its place relative to the others, so every `extends` still resolves |
| `code_state.patch` | `decsim/experiments/run_folder.py`, `snapshot_code_state` | `git diff HEAD`, and a patch creating each untracked file git does not ignore, written only when there is either |
| `resolved/<id>.json` | `decsim/experiments/run_folder.py`, `record_point` | one per sweep point, named by the point's content id: its metadata, the seeds this folder ran of it (ranges of first and how many, joined from its pieces), `sections`, the yaml the point resolved to with its axes placed and its references resolved (null for a point a Python caller built), `maker`, what the workload's row says made it for this point (the `producer` row answers its `function`, the point's own `arguments` and its package's `version`; the `files` row and a Python-built workload answer null), every setting, and the values the build derives from them (`built`: the code card, the window sizes a null resolves to, the rows the plan built, the run plan) |
| `inputs/<id>/` | `decsim/experiments/run_folder.py`, `record_point` | the workload the point ran, as the `files` workload row reads it (`operations.json`, and `circuit.stim` with `measurement_rounds.json` or `fragments/`), and `hashes.json`, each file's sha256 |
| `result.json` | `decsim/experiments/run_command.py`, `write_shot` | `decsim run` and the two tools only: every field of the shot's result |
| `commands.json` | `decsim/experiments/run_command.py`, `write_shot` | `decsim run` and the two tools only: when each QPU command arrived and when it started |
| `trace/<id>_seed<seed>.trace.json` | `decsim/observe/trace_writer.py` | one Chrome trace per traced shot, named by its point's id and its seed (`decsim/experiments/measure.py`, `shot_label`), so two points never share a file |
| `log/<id>_seed<seed>.log` | `decsim/experiments/measure.py`, and `decsim/experiments/run_command.py` for one shot | the engine narrator's lines, written when the `observation` section asks for a log |
| `online_threshold_<id>.csv` | `decsim/experiments/collect_command.py` | the online threshold's trajectory at one point, written when `escalation.threshold_source` is `online`: `point_id`, the swept paths and `algorithm`, then `window_count`, `threshold_db` and `event` per audit, target move and hundredth window, and an `end` row |
| `timeline.png` | `decsim/experiments/plots.py`, `plots` | the figure `decsim collect` draws itself, when a shot was traced: the lowest traced seed of the first point, in the sweep's order |
| `timeline.png`, `stage_breakdown.png` | `decsim/experiments/plots.py`, `FIGURES` | one figure per `decsim plot --figure` name, the two that read decsim's own records: a trace, and the stage columns in pipeline order, one bar per point. A figure of the sweep's numbers is the reader's to draw from the files above; what a figure computes from them (a bar's length, a median, a log scale) is computed when it is drawn, not stored |

The manifest, the config copy and the patch together are the whole
experiment: the commit plus the patch is the code, and the config chain
is the input. `resolved/` says what every value came to at each point,
and `inputs/` holds each point's workload, so a point reruns with the
`files` row pointed at `inputs/<id>/operations.json` (and its circuit
keys) without the maker installed.

## The columns of each file

The additive files first, because the two summaries are made of them.

### `shots.csv`

One row per shot. The first columns are the shot's own scalars, and then
every latency point appears twice, once as that shot's mean and once as
its maximum.

A point is named by `point_id`, its content id, as sinter's csv names
a task by its `strong_id` (`sinter/_data/_csv_out.py:69-77`); it is
also the name of its `resolved/` record. Right after it come the
point's swept values, one column per yaml path the sweep sets, named
by that path (`qpu.distance`,
`workload.arguments.physical_error_probability`), in the order the
sweep first sets them (`decsim/experiments/run_folder.py`,
`swept_values`). Every other file below names its point by the same
columns, and `algorithm` follows them. The values the design fixed come
first and what was measured after, one variable per column, which is
Wickham's tidy table (Tidy Data, J. Stat. Softw. 59(10), 2014, section
2.3), so a reader groups, filters and plots by a column with no parsing.

Every cell is the value the point ran with, read from its resolved
yaml: a swept reference such as `${windows.commit_rounds}` is written
as the value it names, a mapping as its child axes changed it, and a
point whose block did not set a path holds the value its yaml resolved
to there. A path its yaml does not hold is an empty cell. A string or a number is written as itself, and
any other value (a flag, a null, a whole decoder row an axis set) as one
cell of compact json with its keys sorted, as sinter writes
`json_metadata` (`sinter/_data/_csv_out.py:35-37`). The typed value is
in the point's `resolved/` record. A column's unit is in its name
(`_us` microseconds, `_bits`, `_per_shot`), and a swept path's meaning
and unit are its key's in `configs/reference.yaml`.

| Column | What it is |
| --- | --- |
| `point_id`, the swept paths, `algorithm`, `seed` | the sweep point and the seed, which together name the shot |
| `decoded_windows` | how many windows this shot decoded |
| `logical_failure` | 1 when any operation's decoded observable did not match its truth, else 0; a `memory_patches` shot fails when any patch does. An unscored shot is never a failure, as sinter never counts an error on a discarded shot |
| `load` | service time per window divided by the interval between windows arriving; above 1 the decoder cannot keep up |
| `throughput_windows_per_us`, `throughput_rounds_per_us` | what the machine got through |
| `max_queued_windows` | the most jobs that waited in the ready queue at once; a depth counts only when time passes at it, so a job that joins and leaves in one tick never waited |
| `weak_queue_max`, `strong_queue_max` | the most jobs that waited in each tier's ready queue at once, by the same rule. The tier that decodes the planned windows owns the default pool's number, so under `strong_only` that number is in the strong column. A tier the run does not build reads zero. |
| `weak_busy_fraction`, `strong_busy_fraction` | the time-weighted fraction of each tier's units whose compute was busy |
| `escalated_windows`, `strong_decoded_rounds`, `strong_service_mean_us` | the windows the strong tier committed, the rounds its decodes read, and their mean service |
| `commit_rounds` | r_com, the rounds a window commits: `windows.commit_rounds`, or the code distance when it is null |
| `window_period_us` | a window's inter-arrival, `commit_rounds` times the QPU's round period: what `load` divides by, and the deadline a window's decode must beat |
| `parallel_processes_needed` | Skoric's least count of parallel decoding processes for no backlog, ceil(2 tau_W / ((n_com + n_W) tau_rd)) from this shot's mean service (2209.08552 lines 429-438) |
| `weak_syndrome_weight_mean`, `weak_syndrome_weight_max` | the set bits of each weak decode's input, its detection events when they are formed ahead of the decoder; only when `observation.record_switching_windows` is on |
| `weak_service_mean_us` | each weak decode's compute, its first stage's start to its last stage's end; the same switch |
| `strong_wait_mean_us`, `strong_wait_max_us` | each strong decode's wait from its enqueue to its compute start, for a unit and for the unit's compute; the same switch |
| `strong_held_in_units_max` | the most strong decodes held in the units' memory at once, landed and free to compute but waiting for a unit's compute, by the rule of the queue peaks. A unit takes the next decode into its memory while it computes, so this wait never shows in `strong_queue_max`; the same switch |
| `backlog_peak_rounds` | the most rounds produced and not yet decoded at once; only when `observation.backlog_trace` is on |
| `referee_windows_checked`, `referee_window_disagreements` | the referee's count, when `observation.check_windows_with` asked for one |
| `sim_wall_seconds` | how long the simulation itself took to run, on the host |
| `burst_first_flag_round` | the first round at or after the burst's onset that the burst detector fired on, counted from round 1 on a shot with no burst, and 0 when it fired on none; only when `burst_detector.kind` is not `none` |
| `burst_caught_in_time` | whether that round came at most `burst_detector.catch_deadline_rounds` after the onset; only on a `burst_stim` shot whose burst probability is above 0, with a detector |
| `is_scored` | whether every decode a window committed, provisional or final, got a correction from its backend. A backend that produced none (it raised, returned a vector that is not a correction, or found no correction at all) commits an empty correction in its place, and its shot is unscored. An escalated window's weak answer is committed provisionally before the strong one replaces it, and the replacement does not undo what the provisional commit fed forward: its boundary, when `windows.boundaries` ships provisional boundaries (a shipped one is never revised), and its crossing commit, which the strong result keeps. decsim does not trace which of those reached a later decode, so a replaced provisional decode with no correction unscores the shot too. A provisional result never reaches the Pauli frame |
| `provisional_no_correction_windows` | how many windows committed a provisional decode with no correction that the strong answer then replaced; the status columns below count final decodes and do not show these |
| `unscored_reason` | the backends' reasons for the windows committed with no correction, each once, sorted and joined by `;` (`BackendFailureReason` in `decsim/records/decoding.py`: `upstream_exception`, `correction_not_binary`, `correction_wrong_arity`, `nonzero_syndrome_without_faults`, `no_perfect_matching`); empty on a scored shot |
| `sample_digest` | the sha256 of every operation's sampled detection events and observable truth, one byte a bit, in operation order. Two points that differ only in their decoder hold the same digest at the same seed, so pairing their shots can be checked rather than assumed |
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
| `dep_block` | the input landing in the unit's memory, to the first tick the decode may compute: the dependency wait, for the predecessor's boundary and for the escalation message, and zero when nothing was owed at the landing |
| `compute_wait` | that first startable tick, to the compute starting: the wait for the unit's own compute, busy with another decode |
| `queue_wait` | queued, to a unit assigned |
| `input_link_per_window` | a unit assigned, to the input in that unit's memory |
| `fetch` | the unit reading the window out of its own memory |
| `algorithm` | the decoding algorithm itself |
| `release` | the unit writing the correction out |
| `service` | the compute start, to the decode done: the fetch, the algorithm and the release, and nothing the decode waited for |
| `confidence` | the committing decode's end, to the verdict on the window's answer: the confidence signal's own computation, which is the walk under `cluster_gap` and the sibling forced-class solve's remaining time under `complementary_gap`, and zero for a window that escalated |
| `weak_attempt` | a unit taking an escalated window's weak job, to the verdict that escalated it: the attempt whose result did not commit, zero when the first decode committed |
| `escalation_link_per_window` | the weak decoder to the strong decoder: the escalation hop, from the selection's send to the landing of the rounds the strong store lacked, zero for a window that did not escalate |
| `dd_per_window` | one decoder to the next: the boundary handoff |
| `output_link_per_window` | the decoder to the Pauli frame |
| `frame_commit` | the frame accepting a correction, to it being committed |
| `buffer0_ready_to_frame` | the window complete in the weak syndrome buffer, to the frame |
| `buffer0_first_round_to_frame` | the window's first round in the weak syndrome buffer, to the frame |
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
- `frame_commit` closes it.

On a serial path those nine add
up to `buffer0_ready_to_frame` to the tick, on every window of every
config this repository ships.

The park is two points because it has two causes
([D15](../explanation/decisions.md#d15-the-park-before-a-decode-is-two-points-by-what-it-waited-for)).
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
`escalation.run_both_at_once` the weak attempt and the strong decode
overlap rather than follow each other, so adding both would count the
same wall time twice. `tests/experiments/test_measure.py` asserts the
identity window by window on `configs/weak_decoder_baseline.yaml`,
`configs/two_tiers.yaml`, `configs/seam_pinned_switching.yaml` and
`configs/cluster_gap_switching.yaml`, which are a run with no signal to
compute, a run whose signal is a second forced-class solve, the same on
a host-clock strong tier, and a run whose signal is a priced walk.

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
here and moves no column of any file.

### `window_samples.csv`

One row per sweep point, latency point and distinct microsecond value.

| Column | What it is |
| --- | --- |
| `point_id`, the swept paths, `algorithm` | the sweep point |
| `name` | which latency point, from the list above |
| `value_us` | one microsecond value that occurred |
| `count` | how many windows carried it |

This is the multiset of a point's window samples. A median and a p99
need nothing more, and one piece records nothing more for another
process to reach the same numbers.

### `latency_samples.csv`

One row per decoded window, written only for a decoder named by a table
row: the time that held the unit, its measured wall clock or its own
cycle count. A decoder priced by a number produces no rows, and a run
with no rows writes no file.

| Column | What it is |
| --- | --- |
| `point_id`, the swept paths, `algorithm`, `seed` | the shot |
| `algorithm_us` | the time the algorithm stage held the unit for one decode: its wall clock, or its cycle count |
| `window_period_us` | the shot's window inter-arrival, the deadline a window's decode must beat |

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
| `logical_error_rate_per_round`, `_per_round_low`, `_per_round_high` | the estimate and its limits as a round's rate, sinter's shot_error_rate_to_piece_error_rate over the shot's QEC rounds; empty when the rounds were not recorded |
| `is_shot_rate_above_half` | whether the estimate is past one half, where the per-round rate goes through its complement and, for an even round count, no round-flip probability gives the shot rate |
| `prefix_shots`, `prefix_scored_shots`, `prefix_failures` | the prefix's counts, which the estimate and its limits read; a pool may run pieces past the stop, which count in `shots` and not here |
| `logical_error_rate_unscored_as_failures` | the failures and the unscored shots together over every shot: the rate this sample would read if every unscored shot had failed, a bound on the sample and not a confidence bound. Beside the estimate, which is conditional on scoring, it shows how much a backend that failed on hard syndromes could hide |
| `<status>_windows`, `provisional_no_correction_windows` | the sums over the point's shots |
| `throughput_windows_per_us`, `throughput_rounds_per_us` | the means |
| `max_queued_windows` | the deepest queue over the point |
| `weak_queue_max`, `strong_queue_max` | the deepest each tier's own queue over the point |
| `weak_busy_fraction`, `strong_busy_fraction` | the mean busy fractions |
| `escalated_windows`, `strong_service_mean_us` | the strong tier's windows over the point and their mean service |
| `strong_service_bound_us` | Toshio's Theorem 1 bound on one strong decode's time, the unit of `strong_service_mean_us`: tau_gen r_com windows / escalated windows over the point, with tau_gen r_com the shots' `window_period_us` (2510.25222 eq. (6)); infinite when nothing escalated |
| `parallel_processes_needed` | the largest over the point's shots |
| `weak_syndrome_weight_mean`, `weak_service_mean_us`, `strong_wait_mean_us` | the means over the point's shots, when they kept the switching records |
| `weak_syndrome_weight_max`, `strong_wait_max_us`, `strong_held_in_units_max`, `backlog_peak_rounds` | the largest over the point's shots, when they kept the records |
| `escalated_fraction` | the windows the strong tier committed over the windows decoded, beside the columns above |
| `referee_windows_checked`, `referee_window_disagreements` | the referee's totals |
| `flagged_share` | the share of the point's shots whose `burst_first_flag_round` is not 0. On shots with no burst it is the share holding a false alarm, and dividing it by one shot's time gives the false-alarm rate per second |
| `caught_in_time_share` | the share of the point's shots with `burst_caught_in_time` true |
| `load` | the mean load |
| `sim_wall_seconds_per_shot` | what the simulation cost to run |

Then four columns per latency point, in the order of `POINTS`:
`<point>_mean_us`, `<point>_median_us`, `<point>_p99_us` and
`<point>_max_us`. The mean and the max fold over the shot rows; the
median and the p99 come from the sample counts, through
`percentile_of_counts`.

### `links.csv`

One row per sweep point per link path, averaged over that point's shots.

| Column | What it is |
| --- | --- |
| `point_id`, the swept paths, `algorithm` | the point |
| `link` | the path's name |
| `transfers_per_shot`, `payload_bits_per_shot` | the means |
| `bits_per_transfer` | the payload bits divided by the transfers |
| `unknown_payload_transfers_per_shot` | the mean count of transfers with no stated size |
| `queue_wait_us_per_shot`, `serialization_us_per_shot`, `propagation_us_per_shot` | the mean time each part of the path charged |

### `residence.csv`

One row per traced shot per structure, then one per link path. The two
data movement files, `shot_data_movement.csv` and `data_movement.csv`,
are described in the table at the top of this page and are not tabulated
here.

| Column | What it is |
| --- | --- |
| `point_id`, the swept paths, `algorithm`, `seed` | the traced shot |
| `counting` | what the row counts: `residence` for stays in a structure, `link_path` for a path's moves |
| `name` | the structure (a store, a decoder unit, the controller, the frame) or the link path |
| `samples` | how many stays or moves the shot had there |
| `mean_us`, `longest_us` | the mean and the longest: a stay's length, or a move's wait on the wire |

### `manifest.json`

One object. Its keys, from `write_manifest` in
`decsim/experiments/run_folder.py`:

| Key | What it is |
| --- | --- |
| `config_files` | the yaml chain, in the order it was read |
| `experiment_config` | the config as its files write it, as json: its sections after every `extends` was folded in, with each `${...}` reference as written, their folders, the sweep blocks and the files; what each point resolves to is in its `resolved/` record |
| `points` | the sweep's point ids in task order, a point two blocks name listed once: the order a fold writes its rows in |
| `git` | the commit and whether the checkout was dirty, read once when the process started |
| `container` | the container image, when one was in use |
| `versions` | the Python version, and `packages`: every installed package and its version |
| `compiled_libraries` | every compiled library a loader the run imported names, keyed by its absolute path, each its sha256, since a library is built and not tracked and the commit does not name it; a loader may take its file from outside the package (the environment variable the Union-Find row reads, `LIBRARY_VARIABLE` in `decsim/decoders/union_find/compiled_decoder.py`), and a named file not built is left out |
| `host`, `slurm_job_id` | where it ran |
| `argv` | the command line as it was invoked |
| `started_utc`, `finished_utc` | when |

The manifest is written twice, once when the run starts and once when
it ends with `finished_utc` filled in, and both writes name the same
tree: the reading is taken once, before the first shot, and reused. A
tree that moves while a run is going, which is what an array running
for hours out of a checkout somebody commits to does, would otherwise
leave every folder naming code that no part of the run read.

## Read next

- [How to compare two runs](../how-to/compare_two_runs.md): read two folders side by side.
- [The commands](cli.md): the commands that write and read these files.
- [Your first sweep](../tutorials/first_sweep.md): a sweep, its pieces and its error bars.

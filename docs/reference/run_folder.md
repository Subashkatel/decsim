[decsim docs](../README.md) › [Reference](README.md)

# The run folder

Every `decsim collect` writes one folder under `results/`, named for the
UTC time it started and the config it ran, and never reused
(`decsim/front/run_folder.py`, `new_run_dir`). `results/` is output, not
code, and git does not track it, which is gem5's `m5out/`.

A run folder holds facts that add up and nothing else. No summary is
stored: `sweep.csv` and `links.csv` are computed from the additive files
when they are written, and `decsim combine` recomputes them over several
folders, so a sweep may be cut into shards and folded back together
without any number changing (`decsim/front/report.py`).

## What a folder holds

| Name | Written by | What it is |
| --- | --- | --- |
| `shots.csv` | `decsim/front/report.py`, `shot_rows` | one row per shot |
| `shot_links.csv` | `decsim/front/report.py`, `shot_link_rows` | one row per shot per link |
| `window_samples.csv` | `decsim/front/report.py`, `window_sample_rows` | one row per sweep point, latency point and distinct microsecond value |
| `latency_samples.csv` | `decsim/front/report.py`, `latency_sample_rows` | one row per decoded window of a wall-clock decoder |
| `sweep.csv` | `decsim/front/report.py`, `summarize` | one row per sweep point, summarized from `shots.csv` and `window_samples.csv` |
| `links.csv` | `decsim/front/report.py`, `link_rows` | one row per sweep point per link, averaged over that point's shots |
| `shot_data_movement.csv` | `decsim/front/report.py`, `shot_data_movement_rows` | one row per shot per path: that shot's copy and move counters and the memory class the path crosses, written only when `observation.data_movement` is on |
| `data_movement.csv` | `decsim/front/report.py`, `data_movement_rows` | one row per sweep point per path, then per memory class, averaged over the point's shots |
| `residence.csv` | `decsim/front/residence.py`, `write_residence` | one row per traced shot per structure, then per link path: how long a round or window sat there, and how long a move waited on the wire |
| `manifest.json` | `decsim/front/run_folder.py`, `write_manifest` | one object: what ran, where, and with which library versions |
| `config/` | `decsim/front/run_folder.py`, `snapshot_code_state` | a verbatim copy of every yaml file in the config chain |
| `code_state.patch` | `decsim/front/run_folder.py`, `snapshot_code_state` | `git diff HEAD`, written only when the checkout was dirty |
| `trace/<shot>.trace.json` | `decsim/observe/trace_writer.py` | one Chrome trace per traced shot |
| `log/<shot>.log` | `decsim/front/measure.py`, and `decsim/front/run_command.py` for one shot | the engine narrator's lines, written when the `observation` section asks for a log |
| `timeline.png`, `ler.png`, `latency.png` | `decsim/front/plots.py`, `plots` | the figures `decsim collect` draws itself, each one when its input is there: a timeline when a shot was traced, an error-rate figure when the sweep has more than one physical error rate, and a latency figure when a wall-clock decoder ran at more than one distance |
| `timeline.png`, `stage_breakdown.png`, `latency_combined.png`, `ler_vs_distance.png`, `data_movement.png` | `decsim/front/plots.py`, `FIGURES` | one figure per `decsim plot --figure` name, written beside the first run folder given |

The manifest, the config copy and the patch together are the whole
experiment: the commit plus the patch is the code, and the config chain
is the input.

## The columns of each file

The additive files first, because the two summaries are made of them.

### `shots.csv`

One row per shot. The first columns are the shot's own scalars, and then
every latency point appears twice, once as that shot's mean and once as
its maximum.

| Column | What it is |
| --- | --- |
| `physical_error_probability`, `distance`, `round_period_us`, `algorithm`, `seed` | the sweep point and the seed, which together name the shot |
| `windows` | how many windows this shot decoded |
| `logical_failure` | 1 when the decoded observable did not match the truth, else 0 |
| `load` | service time per window divided by the interval between windows arriving; above 1 the decoder cannot keep up |
| `direct_failure`, `direct_mismatch` | the same shot decoded straight through PyMatching outside the machine, and whether the machine disagreed with it |
| `throughput_windows_per_us`, `throughput_rounds_per_us` | what the machine got through |
| `max_queued_windows` | the deepest the ready queue ever got |
| `tesseract_windows_checked`, `tesseract_window_disagreements` | the referee's count, when `observation.check_windows_with` asked for one |
| `sim_wall_seconds` | how long the simulation itself took to run, on the host |
| `<point>_mean_us`, `<point>_max_us` | one pair per latency point below |

The latency points are the tuple `POINTS` in `decsim/front/measure.py`,
and they are the same names in `shots.csv`, `window_samples.csv` and
`sweep.csv`:

| Point | From, to |
| --- | --- |
| `cwb_per_round` | the controller to Buffer 0, one round: latency, serialization and queue |
| `cwb_stall_per_round` | the packed round finding Buffer 0 full, to the freed slot that admitted it: the store's back-pressure on the controller, zero for a round that found room |
| `csb_stall_per_round` | the same wait in front of Buffer 1, one sample per round that reached it |
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
| `escalation_link_per_window` | the weak decoder to the strong decoder: the escalation hop, zero for a window that did not escalate |
| `dd_per_window` | one decoder to the next: the boundary handoff |
| `output_link_per_window` | the decoder to the Pauli frame |
| `frame_commit` | the frame accepting a correction, to it being committed |
| `buffer0_ready_to_frame` | the window complete in Buffer 0, to the frame |
| `buffer0_first_round_to_frame` | the window's first round in Buffer 0, to the frame |
| `qpu_last_round_to_frame` | the last round the window needs leaving the QPU, to the frame |
| `qpu_first_round_to_frame` | the window's first round leaving the QPU, to the frame |

The two `buffer0` totals start the clock where the decoder could first
have started. The two `qpu` totals start it where the physics did, so
they also carry the link out of the fridge, the controller's own work
and the write into Buffer 0.

Every point of a window describes the decode whose result the frame
committed, which the frame's own record names by tier and by request
ordinal: for a window the strong tier recovered, the two link points are
the strong tier's hops and the stage points are the strong decode's. A
window can be decoded more than once, and then every tick still belongs
to one decode. `queue_wait` ends where a unit took the window's first
decode; `weak_attempt` runs from there to the verdict that sent the
window to the strong tier, and is zero when the first decode is the one
that committed; `input_link_per_window`, `dep_block`, `compute_wait`,
`service` and `output_link_per_window` are the committing decode's own
hop, the two halves of its park, its compute and its way home;
`confidence` is the signal the verdict needs, computed after that
decode ended; `frame_commit` closes it. On a serial path those nine add
up to `buffer0_ready_to_frame` to the tick, on every window of every
config this repository ships.

The park is two points because it has two causes. `dep_block` is what
the decode waited for its window's last boundary, and it ends at the
first tick the decode may compute, whether or not a unit is free then;
`compute_wait` is the rest, the unit's compute still busy with another
decode. A run whose windows wait on the seam reports the park in the
first and zero in the second; two decodes of one window sharing a unit,
which is what a complementary gap's forced-class pair is, report it in
the second.

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
same wall time twice. `tests/front/test_measure.py` asserts the
identity window by window on `configs/weak_decoder_baseline.yaml`,
`configs/two_tiers.yaml`, `configs/seam_pinned_switching.yaml` and
`configs/cluster_gap_switching.yaml`, which are a run with no signal to
compute, a run whose signal is a second forced-class solve, the same on
a host-clock strong tier, and a run whose signal is a priced walk.

`escalation_link_per_window` is a hop that is measured and not summed,
like `dd_per_window`: the escalation hop runs beside the strong input
hop, and what it makes the strong decode wait for is in `dep_block`.

### `shot_links.csv`

One row per shot per link path, straight off that shot's traffic
counters.

| Column | What it is |
| --- | --- |
| `distance`, `physical_error_probability`, `algorithm`, `round_period_us`, `seed` | the shot |
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
| `distance`, `physical_error_probability`, `algorithm`, `round_period_us` | the sweep point |
| `name` | which latency point, from the list above |
| `value_us` | one microsecond value that occurred |
| `count` | how many windows carried it |

This is the multiset of a point's window samples. A median and a p99
need nothing more, and one shard records nothing more for another
process to reach the same numbers.

### `latency_samples.csv`

One row per decoded window, written only for a decoder named by a table
row, because only those measure a real wall clock. A decoder priced by a
number produces no rows here.

| Column | What it is |
| --- | --- |
| `distance`, `physical_error_probability`, `round_period_us`, `algorithm`, `seed` | the shot |
| `algorithm_us` | the wall clock that one decode took |

### `sweep.csv`

One row per sweep point, 83 columns, summarized from `shots.csv` and
`window_samples.csv` by `summarize_point` in `decsim/front/report.py`.
The scalars come first:

| Column | What it is |
| --- | --- |
| `distance`, `physical_error_probability`, `algorithm`, `round_period_us` | the point |
| `shots` | how many shots the point ran |
| `windows_per_shot` | the mean over those shots |
| `logical_failures`, `logical_error_rate` | the count and the fraction |
| `ler_wilson_low`, `ler_wilson_high` | the Wilson interval of that fraction at z = 1.96, from `wilson_interval` |
| `direct_pymatching_failures`, `prediction_mismatches_vs_direct` | the same shots decoded outside the machine, and the disagreements |
| `throughput_windows_per_us`, `throughput_rounds_per_us` | the means |
| `max_queued_windows` | the deepest queue over the point |
| `tesseract_windows_checked`, `tesseract_window_disagreements` | the referee's totals |
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
| `distance`, `physical_error_probability`, `algorithm`, `round_period_us` | the point |
| `link` | the path's name |
| `transfers_per_shot`, `payload_bits_per_shot` | the means |
| `bits_per_transfer` | the payload bits divided by the transfers |
| `unknown_payload_transfers_per_shot` | the mean count of transfers with no stated size |
| `queue_wait_us_per_shot`, `serialization_us_per_shot`, `propagation_us_per_shot` | the mean time each part of the path charged |

### `manifest.json`

One object. Its keys, from `write_manifest` in
`decsim/front/run_folder.py`:

| Key | What it is |
| --- | --- |
| `config_files` | the yaml chain, in the order it was read |
| `resolved_config` | the whole config after every `extends` was folded in, as json |
| `shard`, `shots_per_unit` | the `--shard` and `--shots-per-unit` this process ran with, or null |
| `git` | the commit and whether the checkout was dirty, read once when the process started |
| `container` | the container image, when one was in use |
| `versions` | the Python, Stim, PyMatching and numpy versions |
| `host`, `slurm_job_id` | where it ran |
| `argv` | the command line as it was invoked |
| `started_utc`, `finished_utc` | when |

`decsim combine` writes the same shape through `write_combined_manifest`,
with the resolved config of the folders it folded.

The manifest is written twice, once when the run starts and once when
it ends with `finished_utc` filled in, and both writes name the same
tree: the reading is taken once, before the first shot, and reused. A
tree that moves while a run is going, which is what an array running
for hours out of a checkout somebody commits to does, would otherwise
leave every folder naming code that no part of the run read.

## Read next

- [How to compare two runs](../how-to/compare_two_runs.md): read two folders side by side.
- [The commands](cli.md): the commands that write and read these files.
- [Your first sweep](../tutorials/first_sweep.md): a sweep, its shards and its error bars.

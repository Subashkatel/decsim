[decsim docs](../README.md) › [How-to guides](README.md)

# How to compare burst detectors

A burst detector is judged on two numbers: how often it flags a shot
that has no burst (its false alarms), and how often it flags a burst
soon enough after it starts (its catches). The grid
`configs/experiments/burst_detection/burst_detection.yaml` measures both
for four detectors:

| Detector | Its `burst_detector` value in the grid |
| --- | --- |
| event count | `kind: event_count`, the patch's detection events over a window, and each position's |
| whole-patch CUSUM | `kind: masked_regional_cusum` with `mask_count: null` and `region_radii: []`, one CUSUM bank on the whole patch, no mask |
| regional CUSUM | `kind: masked_regional_cusum` with `mask_count: null`, CUSUMs on discs around each check and on the whole patch, no mask |
| masked regional CUSUM | `kind: masked_regional_cusum` at its defaults, the same with checks that act like defects masked out |

The grid's sweep has two axes beyond the point itself: `burst_detector`
takes those four values, and `qpu` takes two, quiet shots
(`kind: stim_device`) and a burst at d = 5 (`kind: burst_stim`). Every
other section is the same at every point, so two rows differ only by
the detector or the burst.

## Run the grid

Each of the eight points is 20 shots, 400 rounds each:

```bash
decsim collect configs/experiments/burst_detection/burst_detection.yaml --processes 8 --out results/burst_detection
```

A quiet shot takes about four minutes on one core. A burst shot takes
ten to twenty-five, most of it strong decodes: every window a flag
meets is decoded again by belief matching. So a burst point is several
hours on one core. [How to run a sweep on Slurm](run_a_sweep_on_slurm.md)
runs the grid as a job that picks up where it stopped when submitted
again.

## Read the results

The run folder's `sweep.csv`, under `results/burst_detection/combined/`,
has two columns for this ([the run folder](../reference/run_folder.md)):

- `flagged_share`: the share of shots on which the detector flagged a
  round at or after the burst's onset. On a quiet point every flag is a
  false alarm.
- `caught_in_time_share`: the share of shots flagged no later than
  `burst_detector.catch_deadline_rounds` (300 by default) after the
  onset. A quiet point has no burst, so it has no value there.

`shots.csv` holds the per-shot column behind both,
`burst_first_flag_round` (0 when the detector flagged nothing).

Read the rows with `decsim.results`. Each row's `burst_detector` column
is the detector's value in the grid, and its `settings.qpu.kind` says
whether the burst was on. The false-alarm rate per second is the quiet
point's flagged share over one shot's time, the shot's rounds times its
round period:

```python
import glob

import decsim.results as results

folders = glob.glob("results/burst_detection/combined/*")
quiet_rows = {}
burst_rows = {}
for row in results.load(*folders):
    detector = row["burst_detector"]
    if row["settings.qpu.kind"] == "stim_device":
        quiet_rows[detector] = row
    else:
        burst_rows[detector] = row
for detector, quiet in quiet_rows.items():
    burst = burst_rows[detector]
    rounds = quiet["settings.workload.row_settings.arguments.rounds_per_shot"]
    period = quiet["settings.qpu.round_period_microseconds"]
    shot_seconds = rounds * period * 1e-6
    false_alarms_per_second = quiet["flagged_share"] / shot_seconds
    print(
        detector,
        burst["caught_in_time_share"],
        false_alarms_per_second,
    )
```

A good detector has a high caught share and a false-alarm rate near
its budget. The CUSUM rows set their thresholds for
`false_alarms_per_second` (0.03 by default); `event_count` sets them
for `false_alarms_per_round`.

## Know what the numbers can say

Twenty shots place a caught share within about 0.2 either way (its
exact 95 percent interval). The quiet points check that false alarms
are not far above budget, and no more: at 0.03 per second a 400 us
shot alarms about once in 80,000 shots, so a quiet point sees none
unless the detector is badly off. To measure the rate itself, give the
quiet points a sweep block of their own whose `collection.max_shots`
is that much larger, or raise the budget.

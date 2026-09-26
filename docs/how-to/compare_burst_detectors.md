[decsim docs](../README.md) › [How-to guides](README.md)

# How to compare burst detectors

A burst detector is judged on two numbers: how often it flags a shot
that has no burst (its false alarms), and how often it flags a burst
soon enough after it starts (its catches). The folder
`configs/burst_detectors_compared/` measures both for four detectors:

| File stem | Detector |
| --- | --- |
| `event_count` | the flipped-bit count: the patch's detection events over a window, and each position's |
| `whole_patch_cusum` | one CUSUM bank on the whole patch, no mask (`mask_count: null`, `region_radii: []`) |
| `regional_cusum` | CUSUMs on discs around each check and on the whole patch, no mask (`mask_count: null`) |
| `masked_regional_cusum` | the same, with checks that act like defects masked out |

Each detector has two files. `<stem>_burst.yaml` runs it on a burst
at d = 5; `<stem>_quiet.yaml` runs it on the same shots with the burst
off. Every burst file extends
`configs/common/burst_detectors_compared_base.yaml` and changes only
its `burst_detector` section, and every quiet file extends its burst
file and changes only `qpu`, so two rows differ only by the detector
or the burst.

## Run the folder

Each file is one sweep point of 20 shots, 400 rounds each:

```bash
for config in configs/burst_detectors_compared/*.yaml; do
  decsim collect "$config" --out "results/$(basename "$config" .yaml)"
done
```

A quiet shot takes about four minutes on one core. A burst shot takes
ten to twenty-five, most of it strong decodes: every window a flag
meets is decoded again by belief matching. So one burst file is several
hours on one core. [How to run a sweep on Slurm](run_a_sweep_on_slurm.md)
runs the files side by side and splits a file's shots with `--shard`;
`decsim combine` folds the shards back into one folder.

## Read the results

Each run folder's `sweep.csv` has two columns for this
([the run folder](../reference/run_folder.md)):

- `flagged_share`: the share of shots on which the detector flagged a
  round at or after the burst's onset. On a quiet file every flag is a
  false alarm.
- `caught_in_time_share`: the share of shots flagged no later than
  `burst_detector.catch_deadline_rounds` (300 by default) after the
  onset. Quiet files have no burst, so they have no such column.

`shots.csv` holds the per-shot column behind both,
`burst_first_flag_round` (0 when the detector flagged nothing).

Read the folders into one table with `decsim.results`. The false-alarm
rate per second is the quiet file's flagged share over one shot's
time, the shot's rounds times its round period:

```python
import decsim.results as results

detectors = [
    "event_count",
    "whole_patch_cusum",
    "regional_cusum",
    "masked_regional_cusum",
]
for detector in detectors:
    burst, = results.load(f"results/{detector}_burst")
    quiet, = results.load(f"results/{detector}_quiet")
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
Wilson interval). The quiet files check that false alarms are not far
above budget, and no more: at 0.03 per second a 400 us shot alarms
about once in 80,000 shots, so a quiet run sees none unless the
detector is badly off. To measure the rate itself, raise the quiet
files' `shots` by that much, or raise the budget.

r"""Sliding windows against one window over the whole shot.

Skoric et al. 2209.08552 check that windowed decoding loses nothing:
the same shots decoded in windows and globally fail as often, "within
the numerical error of each other" (Sec. I C; App. C Fig. 6 a for
sliding windows). This decodes pilot.py's shots, seeds 0 to 19,999 of
each saved dataset, as one window over the whole shot: a sliding window
whose commit is the shot's 10 d rounds, so the switching run reads one
gap per shot as Toshio et al. 2510.25222 Sec. IV does. The three runs
are the pilot's; window_check_plots.py pairs them with the pilot's
sliding windows of d rounds shot by shot.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md):

```
export PYTHONPATH=$PWD/../..
python -m decsim run window_check.py --slurm \
  --out $PWD/../../results/basic_case_window_check
```
"""

import dataclasses

import decsim
import decsim.experiments.sweeps as sweeps
import decsim.windows.schemes.sliding as sliding

config = sweeps.load_beside(__file__, "config.py")
pilot = sweeps.load_beside(__file__, "pilot.py")
DECODINGS = config.DECODINGS


def whole_shot_machine(
    decoding: str, distance: int, physical_error_probability: float
):
    """The pilot's machine with one window over the whole shot."""
    machine = config.machine(decoding, distance, physical_error_probability)
    windows = machine.held("windows")
    rounds = config.constants.rounds_per_distance * distance
    whole = sliding.SlidingWindowScheme.Settings(commit_rounds=rounds)
    one_window = dataclasses.replace(windows, scheme=whole)
    return machine.with_held("windows", one_window)


def metadata(cell: dict, task: decsim.Task) -> dict:
    """The pilot's columns, and the windowing."""
    pilot_columns = pilot.metadata(cell, task)
    return {**pilot_columns, "windowing": "whole_shot"}


stop = decsim.CollectionSettings(max_shots=pilot.SHOTS)
sweep = decsim.Sweep(
    "basic_case_window_check",
    whole_shot_machine,
    grid={
        "decoding": DECODINGS,
        "distance": pilot.DISTANCES,
        "physical_error_probability": config.PHYSICAL_ERROR_PROBABILITIES,
    },
    task_name="whole_{decoding}_d{distance}_p{physical_error_probability}",
    stop=stop,
    metadata=metadata,
    constants=config.constants,
)
experiment = sweep.experiment()

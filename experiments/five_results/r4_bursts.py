r"""R4 on streams: a fixed threshold against adaptive ones, in a burst.

The double window on 2 strong cores at d = 9, the pilot design that
kept up at every share, each strong decode priced by the fitted time
(B1 stage 2). Three threshold rules: fixed at the 18.4 dB that
escalated 16.6% of calibration windows; RateThreshold holding an 18%
share from the gaps alone; and RateThreshold with TargetSteps, which
steps that share down to 2% while the waiting line rises and back up
when it is calm, checked every 20 windows so it can act inside a fade
of 550 rounds. The burst is r2_bursts.py's. Task names:
{rule}_peak{peak_factor}_d{distance}_p{physical_error_probability}.

Usage
-----

```
export PYTHONPATH=$PWD/../..
python -m decsim stream r4_bursts.py --only steps_peak10.0_d9_p0.003 \
  --rounds 8000 --warmup 1000 --seed 1 \
  --set observation.series_every_rounds=100 --out steps.json
```
"""

import decsim
import decsim.escalation.target_steps as target_steps
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.sweeps as sweeps
import decsim.presets.machines as machines
import decsim.topology.graph as graph

config = sweeps.load_beside(__file__, "config.py")
calibration = sweeps.load_beside(__file__, "calibration.py")
bursts = sweeps.load_beside(__file__, "r2_bursts.py")
RULES = ("fixed", "rate", "steps")
PEAK_FACTORS = (1.0, 4.0, 10.0)
SHARE = 0.2
RATE_TARGET = 0.18
STEP_LEVELS = (0.02, 0.05, 0.1, 0.18)
STEP_CHECK_EVERY_WINDOWS = 20


def machine(
    rule: str,
    peak_factor: float,
    distance: int,
    physical_error_probability: float,
    rate_target: float = RATE_TARGET,
) -> graph.Machine:
    """The double window on 2 cores under one threshold rule."""
    point = (distance, physical_error_probability)
    threshold_decibels = calibration.CALIBRATED[point][SHARE]
    design = config.switching(
        "double", distance, physical_error_probability, threshold_decibels, 2
    )
    timed = config.fitted_strong_time(design, distance, "double")
    ruled = with_rule(timed, rule, threshold_decibels, rate_target)
    return bursts.with_burst(config.streaming(ruled), peak_factor)


def with_rule(
    machine: graph.Machine,
    rule: str,
    threshold_decibels: float,
    rate_target: float = RATE_TARGET,
) -> graph.Machine:
    """The machine's threshold source for the rule; fixed keeps its own.

    The step rule's levels are STEP_LEVELS below the target, then the
    target itself.
    """
    if rule == "fixed":
        return machine
    steps = None
    if rule == "steps":
        levels = tuple(level for level in STEP_LEVELS if level < rate_target)
        steps = target_steps.TargetSteps.Settings(
            start=rate_target,
            levels=levels + (rate_target,),
            check_every_windows=STEP_CHECK_EVERY_WINDOWS,
        )
    threshold = threshold_sources.RateThreshold.Settings(
        threshold_decibels=threshold_decibels,
        target_escalation_rate=rate_target,
        steps=steps,
    )
    return machine.with_settings(machines.SWITCHING, threshold=threshold)


# decsim stream runs one shot; an experiment still names its collection
stop = decsim.CollectionSettings(max_shots=1)
sweep = decsim.Sweep(
    "five_results_r4_bursts",
    machine,
    grid={
        "rule": RULES,
        "peak_factor": PEAK_FACTORS,
        "distance": (9,),
        "physical_error_probability": (0.003,),
    },
    task_name="{rule}_peak{peak_factor}_d{distance}_p{physical_error_probability}",
    stop=stop,
    metadata=config.metadata,
)
experiment = sweep.experiment()

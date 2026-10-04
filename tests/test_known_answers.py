"""Known answers and invariants of whole runs, through the Python front end.

Each test builds its machine as a user does, from decsim.settings' bases on
Stim's rotated surface code memory at distance 3, and checks one law whose
answer is known without running the machine:

- zero noise draws no detection event (stim.Circuit.generated with every
  noise channel at 0), so no tier flips a logical observable.

Three more laws sit beside the parts they exercise: switching that never
escalates is the weak tier alone (tests/escalation/test_switching_mode.py),
strong-only predicts what its decoder predicts offline on the same events
(tests/decoders/test_decoder_tiers.py), and the frame is the XOR of its
committed corrections (tests/pauli_frame/test_pauli_frame.py).
"""

import dataclasses

import pytest

import decsim.collect as collect
import decsim.confidence.complementary as complementary
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.settings as machine_settings
import decsim.windows.settings as window_settings

DISTANCE = 3
ROUND_PERIOD_MICROSECONDS = 1.0
# no complementary gap reaches it, so every window escalates
UNREACHABLE_DECIBELS = 1e6
REDO_WINDOW = strong_window_shapes.RedoWindow.Settings()


def run(settings, seed: int = 0) -> collect.Shot:
    """One seeded shot of the settings, run."""
    task = collect.Task(settings, {})
    return collect.run_shot(task, seed)


def weak_only(physical_error_probability: float):
    """The weak decoder baseline at distance 3 on 1 us rounds."""
    return machine_settings.weak_decoder_baseline(
        DISTANCE, physical_error_probability, ROUND_PERIOD_MICROSECONDS
    )


def strong_only(physical_error_probability: float):
    """The strong decoder baseline at distance 3 on 1 us rounds."""
    return machine_settings.strong_decoder_baseline(
        DISTANCE, physical_error_probability, ROUND_PERIOD_MICROSECONDS
    )


def switching(physical_error_probability: float, strong_window=REDO_WINDOW):
    """The weak base escalating every window to the strong base's tier.

    The strong side's four hops are one room cycle each.
    """
    base = weak_only(physical_error_probability)
    strong_base = strong_only(physical_error_probability)
    confidence = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=UNREACHABLE_DECIBELS
    )
    slot = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold, strong_window=strong_window
    )
    windows = window_settings.switching_windows(base.windows, strong_window)
    links = machine_settings.one_cycle_strong_side(base.links)
    return dataclasses.replace(
        base,
        links=links,
        windows=windows,
        strong_decoder=strong_base.strong_decoder,
        switching=slot,
    )


def logical_failures(shot: collect.Shot) -> list:
    """Each operation's logical failure: a predicted bit off the truth."""
    failures = []
    for operation_result in shot.result.operation_results:
        failures.append(operation_result.logical_failure)
    return failures


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize(
    "shape",
    (weak_only, strong_only, switching),
    ids=("weak_only", "strong_only", "switching"),
)
def test_a_noiseless_run_makes_no_logical_error(shape, seed):
    """Zero noise gives zero logical errors on every tier that commits."""
    settings = shape(0.0)

    shot = run(settings, seed)

    assert logical_failures(shot) == [False]

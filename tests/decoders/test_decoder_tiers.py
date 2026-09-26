"""The two decoder tiers as units: the mode picks one and it decodes.

The weak tier's real MWPM must reach whole-circuit PyMatching's answer
on the same events; the strong tier's belief matching must run and
decode a d=3 memory shot. Toshio arXiv 2510.25222: lightweight decoders
decode constantly, a separate accurate decoder is invoked on demand.
"""

import pytest

import decsim.decoders.staged_decoder as staged_decoder
import decsim.experiments.experiment as experiment
import decsim.machine as machine_module
from tests.experiments.yaml_configs import (
    MINIMAL_CONFIG,
    measure_point_shot,
    strong_unit,
    write_config,
)


def _algorithm_records(machine) -> list:
    """The stage records of every decode's algorithm stage."""
    records = []
    for record in machine.observation.stages.records:
        if record.stage == staged_decoder.ALGORITHM_STAGE:
            records.append(record)
    return records


@pytest.mark.parametrize("seed", range(3))
def test_weak_unit_loop_matches_direct_pymatching(tmp_path, seed):
    # The functional gate: the loop with the weak unit's real MWPM reaches
    # the same prediction as whole-circuit PyMatching on the same events.
    config_path = write_config(
        tmp_path,
        {
            "weak_decoder": {
                **MINIMAL_CONFIG["weak_decoder"],
                "kind": "pymatching",
            }
        },
    )
    config = experiment.load_experiment(config_path)
    measurement = measure_point_shot(
        config,
        physical_error_probability=0.005,
        distance=3,
        round_period_microseconds=1.0,
        seed=seed,
    )
    assert measurement.algorithm == "pymatching"
    assert measurement.windows > 0
    assert not measurement.direct_mismatch


def test_strong_unit_runs_belief_matching(tmp_path):
    strong_decoder = strong_unit("belief_matching")
    card = {"escalation": {"kind": "strong_only"}}
    card.update(strong_decoder)
    config_path = write_config(tmp_path, card)
    config = experiment.load_experiment(config_path)
    measurement = measure_point_shot(
        config,
        physical_error_probability=0.001,
        distance=3,
        round_period_microseconds=1.0,
        seed=0,
    )
    assert measurement.algorithm == "belief_matching"
    assert measurement.windows > 0
    assert not measurement.logical_failure


def test_a_union_find_tier_with_a_cycle_count_is_held_by_the_count(tmp_path):
    """Every algorithm stage ends on the count's clock edge, past its setup.

    A tier charged its host wall clock ends on no edge; one under a
    cycle_count block holds its unit for whole cycles of the named
    clock and never fewer than the eleven the quiet machine costs.
    """
    config_path = write_config(
        tmp_path,
        {
            "clocks": {**MINIMAL_CONFIG["clocks"], "helios": 100.0},
            "weak_decoder": {
                **MINIMAL_CONFIG["weak_decoder"],
                "kind": "union_find",
                "cycle_count": {"clock": "helios", "delay_cycles": 3},
            },
        },
    )
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.001,
        distance=3,
        round_period_microseconds=1.0,
    )
    machine = machine_module.Machine.build(settings)
    machine.run()
    cycle_count = settings.weak_decoder.row_settings.cycle_count
    period_ticks = cycle_count.clock.period_ticks
    algorithm = _algorithm_records(machine)
    ticks_past_an_edge = {
        record.end_ticks % period_ticks for record in algorithm
    }
    held_ticks = [record.end_ticks - record.start_ticks for record in algorithm]
    assert len(algorithm) > 0
    assert ticks_past_an_edge == {0}
    assert min(held_ticks) >= 11 * period_ticks


def test_a_measured_table_tier_is_held_by_the_measured_line(tmp_path):
    """Every decode holds its unit 78.957 us plus 9.762 us an iteration.

    The gh200 and a100 rows differ, so the device key read from the
    yaml is the one the unit prices by: a100, whole, the 360-detector
    region nearest a d = 5 window (decoders/measured_table).
    """
    pytest.importorskip("relay_bp")
    strong_decoder = strong_unit("measured_table")
    strong_decoder["strong_decoder"]["device"] = "a100"
    card = {"escalation": {"kind": "strong_only"}}
    card.update(strong_decoder)
    config_path = write_config(tmp_path, card)
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.001,
        distance=5,
        round_period_microseconds=1.0,
    )
    machine = machine_module.Machine.build(settings)
    machine.run()
    algorithm = _algorithm_records(machine)
    intercept_ticks = 78_957_000
    ticks_per_iteration = 9_762_000
    held_ticks = [record.end_ticks - record.start_ticks for record in algorithm]
    iteration_ticks = [held - intercept_ticks for held in held_ticks]
    remainders = {ticks % ticks_per_iteration for ticks in iteration_ticks}
    assert len(algorithm) > 0
    assert min(iteration_ticks) >= 0
    assert remainders == {0}

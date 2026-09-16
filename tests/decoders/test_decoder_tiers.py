"""The two decoder tiers as units: the mode picks one and it decodes.

The weak tier's real MWPM must reach whole-circuit PyMatching's answer
on the same events; the strong tier's belief matching must run and
decode a d=3 memory shot. Toshio arXiv 2510.25222: lightweight decoders
decode constantly, a separate accurate decoder is invoked on demand.
"""

import decsim.decoders.staged_decoder as staged_decoder
import decsim.experiments.experiment as experiment
import decsim.machine as machine_module
from tests.experiments.yaml_configs import (
    MINIMAL_CONFIG,
    measure_point_shot,
    strong_unit,
    write_config,
)


def test_weak_unit_loop_matches_direct_pymatching(tmp_path):
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
    for seed in range(3):
        measurement = measure_point_shot(
            config,
            physical_error_probability=0.005,
            distance=3,
            round_period_us=1.0,
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
        round_period_us=1.0,
        seed=0,
    )
    assert measurement.algorithm == "belief_matching"
    assert measurement.windows > 0
    assert not measurement.logical_failure


def test_a_union_find_tier_with_a_cycle_count_is_held_by_the_count(tmp_path):
    """Every algorithm stage ends on the count's clock edge, past its setup.

    A tier charged its host wall clock ends on no edge; one under a
    cycle_count block holds its unit for whole cycles of the named
    clock and never fewer than the setup cycles.
    """
    config_path = write_config(
        tmp_path,
        {
            "clocks": {**MINIMAL_CONFIG["clocks"], "helios": 100.0},
            "weak_decoder": {
                **MINIMAL_CONFIG["weak_decoder"],
                "kind": "union_find",
                "cycle_count": {
                    "clock": "helios",
                    "setup_cycles": 11,
                    "cycles_per_step": 4,
                    "cycles_per_hop": 3,
                },
            },
        },
    )
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.001, distance=3, round_period_us=1.0
    )
    machine = machine_module.Machine.build(settings)
    machine.run()
    period_ticks = settings.weak_decoder.cycle_count.clock.period_ticks
    records = machine.observation.stages.records
    algorithm = [
        record
        for record in records
        if record.stage == staged_decoder.ALGORITHM_STAGE
    ]
    ticks_past_an_edge = {
        record.end_ticks % period_ticks for record in algorithm
    }
    held_ticks = [record.end_ticks - record.start_ticks for record in algorithm]
    assert len(algorithm) > 0
    assert ticks_past_an_edge == {0}
    assert min(held_ticks) >= 11 * period_ticks

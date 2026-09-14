"""The two decoder tiers as units: the mode picks one and it decodes.

The weak tier's real MWPM must reach whole-circuit PyMatching's answer
on the same events; the strong tier's belief matching must run and
decode a d=3 memory shot. Toshio arXiv 2510.25222: lightweight decoders
decode constantly, a separate accurate decoder is invoked on demand.
"""

import decsim.front.experiment as experiment
from tests.front.yaml_configs import (
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

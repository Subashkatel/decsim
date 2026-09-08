"""What a window's hand-off costs on decoder_to_decoder.

The referents are the implementations that write the update: quits
updates one check layer (`syn_update` sized by `hz.shape[0]`,
sliding_window.py:164-174), cuda-q QEC writes `syndrome_mods` only
between the next window's round bounds (sliding_window.cpp:325-344), and
Tan et al. 2209.09219 lines 936-946 state the rule. A rotated surface
code's bulk layer carries d*d-1 detectors, so a dense mask over the seam
layer is 8 bits at d=3 and 24 at d=5. The sparse row is Skoric's
artificial defect list (2209.08552 lines 265-269) and Bombin's small
set of updated checks (2303.04846 lines 784-786), one index per flip.
"""

import pathlib

import decsim.front.experiment as experiment
import decsim.machine as machine_module
import decsim.records.windows as window_records
import decsim.windows.boundary_payloads as boundary_payloads
import decsim.windows.window_interactions as window_interactions
import tests.declared_run as declared_run

THIS_FILE = pathlib.Path(__file__)
CONFIGS = THIS_FILE.parents[2] / "configs"


def boundary_transfers(link_traffic: dict) -> list:
    """Every decoder_to_decoder transfer of a run's traffic report."""
    found = []
    for transfer in link_traffic["transfers"]:
        if transfer["path"] == "decoder_to_decoder":
            found.append(transfer)
    return found


def reference_run(distance: int):
    """One shot of the weak baseline at that distance."""
    config_path = CONFIGS / "weak_decoder_baseline.yaml"
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.003,
        distance=distance,
        round_period_us=1.0,
    )
    machine = machine_module.Machine.build(settings, 0)
    return machine.run()


def test_the_dense_row_charges_the_seam_layer_at_distance_three():
    result = reference_run(3)
    transfers = boundary_transfers(result.link_traffic)
    assert transfers
    for transfer in transfers:
        assert transfer["payload_bits"] == 8
        assert transfer["payload_source"] == (
            "DependencyResidual seam-layer detectors"
        )


def test_the_dense_row_charges_the_seam_layer_at_distance_five():
    result = reference_run(5)
    transfers = boundary_transfers(result.link_traffic)
    assert transfers
    for transfer in transfers:
        assert transfer["payload_bits"] == 24


def test_the_sparse_row_charges_one_index_per_flipped_detector():
    row = boundary_payloads.SparseSeamList()
    seam = window_records.BoundarySeam(detector_count=24, flip_count=3)
    assert row.bits(seam) == 15


def test_the_sparse_row_charges_nothing_when_the_message_flips_nothing():
    row = boundary_payloads.SparseSeamList()
    seam = window_records.BoundarySeam(detector_count=24, flip_count=0)
    assert row.bits(seam) == 0


def test_the_interaction_counts_the_flips_on_the_destinations_oldest_layer():
    positions = {
        10: (4, 0),
        11: (4, 1),
        12: (4, 2),
        13: (5, 0),
    }
    destination = window_records.WindowInfo(
        operation_id=1,
        window_index=1,
        commit_lo=4,
        commit_hi=6,
        buffer_hi=9,
        round_count=15,
        buffer_lo=4,
        deps=(),
        dependents=(),
        detector_positions=positions,
    )
    residual = window_records.DependencyResidual(detector_ids=(10, 13, 99))
    payload = boundary_payloads.SparseSeamList()
    interaction = window_interactions.DefaultWindowInteraction(0, payload)
    assert interaction.boundary_payload_bits(residual, destination) == 2
    dense = boundary_payloads.DenseSeamMask()
    dense_interaction = window_interactions.DefaultWindowInteraction(0, dense)
    assert dense_interaction.boundary_payload_bits(residual, destination) == 3


def test_a_run_with_one_window_sends_no_boundary():
    machine = declared_run.weak_only_run(rounds=6)
    report = machine.observation.traffic.traffic_json_value()
    assert boundary_transfers(report) == []

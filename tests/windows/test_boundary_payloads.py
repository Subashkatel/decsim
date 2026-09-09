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

import dataclasses
import pathlib

import decsim.front.experiment as experiment
import decsim.machine as machine_module
import decsim.ports as ports
import decsim.records.windows as window_records
import decsim.windows.boundary_payloads as boundary_payloads
import decsim.windows.settings as window_settings
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


def test_every_row_of_the_table_answers_a_width_for_a_seam():
    """The table's contract, as ports.BoundaryPayload states it.

    A row's whole job is to turn one BoundarySeam into the bits the wire
    carries, so every row satisfies the port and answers a count for a
    d=3 bulk layer. The shipped table says nothing about a direction: a
    hand-off is written the same way whichever face it lands on, since
    both are the destination's own layer (Tan 2209.09219 lines 936-946).
    """
    seam = window_records.BoundarySeam(detector_count=8, flip_count=2)
    for name, row_class in window_settings.BOUNDARY_PAYLOADS.items():
        row = row_class()
        assert isinstance(row, ports.BoundaryPayload), name
        bits = row.bits(seam)
        assert isinstance(bits, int), name
        assert bits >= 0, name


def _window_reading(round_count: int) -> window_records.WindowInfo:
    """A window over rounds 4-9 whose planned round count is the caller's."""
    positions = {10: (4, 0), 11: (4, 1), 12: (5, 0)}
    return window_records.WindowInfo(
        operation_id=1,
        window_index=1,
        commit_lo=4,
        commit_hi=6,
        buffer_hi=9,
        round_count=round_count,
        buffer_lo=4,
        deps=(),
        dependents=(),
        detector_positions=positions,
    )


def test_the_width_is_the_rows_arithmetic_on_the_seam_not_the_round_count():
    """A hand-off is one layer wide, however long the window is.

    quits sizes `syn_update` by `hz.shape[0]`, one check layer
    (sliding_window.py:164-174), and cuda-q QEC bounds `syndrome_mods`
    to the next window's first round (sliding_window.cpp:325-344).
    Neither scales with the window's rounds, so two windows that read a
    different number of rounds over the same seam layer are charged the
    same, and the dense charge is the layer's detectors.
    """
    residual = window_records.DependencyResidual(detector_ids=(10, 11))
    dense = boundary_payloads.DenseSeamMask()
    interaction = window_interactions.DefaultWindowInteraction(0, dense)
    six_rounds = _window_reading(6)
    forty_rounds = _window_reading(40)
    short = interaction.boundary_payload_bits(residual, six_rounds)
    long = interaction.boundary_payload_bits(residual, forty_rounds)
    assert short == long
    # rounds 4-9 hold three detectors, two of them on the seam layer 4
    assert short == 2


def test_a_destination_with_no_window_model_is_priced_by_its_card():
    """No layer to count is not a hand-off that costs nothing.

    A window whose detector positions are unknown has no seam to size,
    so the interaction answers None and the wire prices the transfer by
    its link card. Answering zero would make the hand-off free.
    """
    known = _window_reading(6)
    unknown = dataclasses.replace(known, detector_positions=None)
    residual = window_records.DependencyResidual(detector_ids=(10, 11))
    dense = boundary_payloads.DenseSeamMask()
    interaction = window_interactions.DefaultWindowInteraction(0, dense)
    assert interaction.boundary_payload_bits(residual, unknown) is None


def _pinned_faces(result) -> list:
    """(strong window index, payload bits) of every pinned face on the wire.

    A weak delivery is attributed to the window that produced it; a
    pinned face is attributed to the strong window that reads it.
    """
    faces = []
    for transfer in result.link_traffic["transfers"]:
        if transfer["path"] != "decoder_to_decoder":
            continue
        attribution = transfer["attribution"]
        source = attribution["relation"]["request_key"]["window_id"]
        if attribution["window_id"] == source:
            continue
        faces.append((attribution["window_id"], transfer["payload_bits"]))
    return faces


def _pinned_run(strong_window: str, distance: int):
    """The switching point of configs/seam_pinned_switching.yaml, one shot.

    The yaml names near_seam_pinned; the row under test replaces it in
    the escalation card, which is where the build reads it from.
    """
    config_path = CONFIGS / "seam_pinned_switching.yaml"
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.008,
        distance=distance,
        round_period_us=1.0,
    )
    escalation = dataclasses.replace(
        settings.escalation, strong_window=strong_window
    )
    settings = dataclasses.replace(settings, escalation=escalation)
    machine = machine_module.Machine.build(settings, 0)
    return machine.run()


def test_a_pinned_face_is_charged_the_dense_width_of_its_seam_layer():
    """The pinned hop is priced by the same table row as a weak hand-off.

    A pinned face is a boundary message like any other: Skoric
    2209.08552 lines 1038-1040 ships the defects block to block. The
    dense row charges the seam layer, d*d-1 detectors on a bulk layer of
    a rotated surface code, so 8 bits at d=3 and 24 at d=5.
    """
    result = _pinned_run("near_seam_pinned", 3)
    at_three = _pinned_faces(result)
    assert at_three
    for _window_id, payload_bits in at_three:
        assert payload_bits == 8
    wider = _pinned_run("near_seam_pinned", 5)
    at_five = _pinned_faces(wider)
    assert at_five
    for _window_id, payload_bits in at_five:
        assert payload_bits == 24


def test_a_two_faced_window_costs_the_sum_of_its_two_one_sided_halves():
    """Toshio 2510.25222 lines 1248-1259: both ends of the strong window.

    The forward pinned row pins a near face and a far face, and each
    face is its own message, so the window pays the two one-sided
    charges added up. A window that pins one face on the same run pays
    one of them.
    """
    result = _pinned_run("forward_seam_pinned", 3)
    faces = _pinned_faces(result)
    charged = _charge_by_window(faces)
    counts = _face_counts(faces)
    two_faced = [window_id for window_id, count in counts if count == 2]
    one_faced = [window_id for window_id, count in counts if count == 1]
    assert two_faced
    assert one_faced
    for window_id in two_faced:
        assert charged[window_id] == 2 * 8
    for window_id in one_faced:
        assert charged[window_id] == 8


def _charge_by_window(faces: list) -> dict:
    """What each strong window paid for the faces it pinned."""
    charged = {}
    for window_id, payload_bits in faces:
        running = charged.get(window_id, 0)
        charged[window_id] = running + payload_bits
    return charged


def _face_counts(faces: list) -> list:
    """(window index, how many faces it pinned) over the run's pins."""
    counts = {}
    for window_id, _payload_bits in faces:
        counts[window_id] = counts.get(window_id, 0) + 1
    listed = counts.items()
    return sorted(listed)

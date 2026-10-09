"""What a window's hand-off costs on the wire.

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

import pytest

import decsim.machine as machine_module
import decsim.ports as ports
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.windows.boundary_payloads as boundary_payloads
import decsim.windows.schemes.parallel as parallel_scheme
import decsim.windows.settings as window_settings
import decsim.windows.window_interactions as window_interactions
import tests.declared_run as declared_run
import tests.escalation.test_strong_window_shapes as shape_tests

# the payload rows the tree ships
PAYLOAD_ROWS = (
    boundary_payloads.DenseSeamMask,
    boundary_payloads.SparseSeamList,
)


def boundary_transfers(link_traffic: dict) -> list:
    """Every decoder_to_decoder transfer of a run's traffic report."""
    found = []
    for transfer in link_traffic["transfers"]:
        if transfer["path"] == "decoder_to_decoder":
            found.append(transfer)
    return found


def _source_committing(commit_lo: int, commit_hi: int):
    """A neighbour of operation 1 committing over those rounds."""
    return window_records.WindowInfo(
        operation_id=1,
        window_index=0,
        commit_lo=commit_lo,
        commit_hi=commit_hi,
        buffer_hi=commit_hi,
        round_count=15,
        buffer_lo=commit_lo,
        deps=(),
        dependents=(),
    )


def reference_run(distance: int):
    """One shot of the weak baseline at that distance."""
    settings = machine_settings.weak_decoder_baseline(distance, 0.003, 1.0)
    machine = machine_module.Machine.build(settings, 0)
    return machine.run()


def test_the_dense_row_charges_the_seam_layer_at_distance_three():
    result = reference_run(3)
    transfers = boundary_transfers(result.link_traffic)
    widths = {transfer["payload_bits"] for transfer in transfers}
    sources = {transfer["payload_source"] for transfer in transfers}
    assert transfers
    assert widths == {8}
    assert sources == {"DependencyResidual seam-layer detectors"}


def test_the_dense_row_charges_the_seam_layer_at_distance_five():
    result = reference_run(5)
    transfers = boundary_transfers(result.link_traffic)
    widths = {transfer["payload_bits"] for transfer in transfers}
    assert transfers
    assert widths == {24}


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
    source = _source_committing(1, 3)
    sparse_bits = interaction.boundary_payload_bits(
        residual, destination, source
    )
    assert sparse_bits == 2
    dense = boundary_payloads.DenseSeamMask()
    dense_interaction = window_interactions.DefaultWindowInteraction(0, dense)
    dense_bits = dense_interaction.boundary_payload_bits(
        residual, destination, source
    )
    assert dense_bits == 3


@pytest.mark.parametrize("row_class", PAYLOAD_ROWS)
def test_every_row_of_the_table_answers_a_width_for_a_seam(row_class):
    """The rows' contract, as ports.BoundaryPayload states it.

    A row's whole job is to turn one BoundarySeam into the bits the wire
    carries, so every row satisfies the port and answers a count for a
    d=3 bulk layer. No shipped row says anything about a direction: a
    hand-off is written the same way whichever face it lands on, since
    both are the destination's own layer (Tan 2209.09219 lines 936-946).
    """
    seam = window_records.BoundarySeam(detector_count=8, flip_count=2)
    row = row_class()
    bits = row.bits(seam)
    assert isinstance(row, ports.BoundaryPayload)
    assert isinstance(bits, int)
    assert bits >= 0


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
    source = _source_committing(1, 3)
    short = interaction.boundary_payload_bits(residual, six_rounds, source)
    long = interaction.boundary_payload_bits(residual, forty_rounds, source)
    assert short == long
    # rounds 4-9 hold three detectors, two of them on the seam layer 4
    assert short == 2


def test_a_destination_with_no_window_model_is_priced_by_its_card():
    """No layer to count is not a hand-off that costs nothing.

    A window whose detector positions are unknown is sized on one round
    of the code card's checks, 24 at d=5, and no error was drawn to flip
    any of them: the dense row pays the layer, the sparse row nothing.
    With no card count either, the interaction answers None.
    """
    known = _window_reading(6)
    card_only = dataclasses.replace(
        known, detector_positions=None, layer_detector_count=24
    )
    residual = window_records.DependencyResidual(detector_ids=(10, 11))
    dense_row = boundary_payloads.DenseSeamMask()
    sparse_row = boundary_payloads.SparseSeamList()
    dense = window_interactions.DefaultWindowInteraction(0, dense_row)
    sparse = window_interactions.DefaultWindowInteraction(0, sparse_row)
    source = _source_committing(1, 3)
    dense_bits = dense.boundary_payload_bits(residual, card_only, source)
    sparse_bits = sparse.boundary_payload_bits(residual, card_only, source)
    unknown = dataclasses.replace(card_only, layer_detector_count=None)
    unknown_bits = dense.boundary_payload_bits(residual, unknown, source)
    assert dense_bits == 24
    assert sparse_bits == 0
    assert unknown_bits is None


def _pinned_faces(result) -> list:
    """(strong window index, seam bits) of every pinned face on the wire.

    A weak delivery is attributed to the window that produced it; a
    pinned face is attributed to the strong window that reads it. The
    seam leaves out the request's name a hop across the wall adds.
    """
    faces = []
    for window_id, _source, seam in _pins_with_sources(result):
        faces.append((window_id, seam))
    return faces


def _pinned_run(strong_window, distance: int, seed: int = 0):
    """One shot of the redo window switching experiment's switching task.

    The row under test replaces the redo window in the switching slot,
    which is where the build reads it from.
    """
    settings = shape_tests.weak_base_switching(distance, 0.008, 1.0)
    switching = dataclasses.replace(
        settings.switching, strong_window=strong_window
    )
    windows = window_settings.switching_windows(
        settings.windows, switching.strong_window
    )
    settings = dataclasses.replace(
        settings, switching=switching, windows=windows
    )
    machine = machine_module.Machine.build(settings, seed)
    return machine.run()


def test_a_pinned_face_is_charged_the_dense_width_of_its_seam_layer():
    """The pinned hop is priced by the same table row as a weak hand-off.

    A pinned face is a boundary message like any other: Skoric
    2209.08552 lines 1038-1040 ships the defects block to block. The
    dense row charges the seam layer, d*d-1 detectors on a bulk layer of
    a rotated surface code, so 8 bits at d=3 and 24 at d=5.
    """
    result = _pinned_run(declared_run.REDO_WINDOW, 3)
    at_three = _pinned_faces(result)
    wider = _pinned_run(declared_run.REDO_WINDOW, 5)
    at_five = _pinned_faces(wider)
    widths_at_three = {payload_bits for _window_id, payload_bits in at_three}
    widths_at_five = {payload_bits for _window_id, payload_bits in at_five}
    assert widths_at_three == {8}
    assert widths_at_five == {24}


def test_a_two_faced_window_costs_the_sum_of_its_two_one_sided_halves():
    """Toshio 2510.25222 lines 1248-1259: both ends of the strong window.

    The forward pinned row pins a near face and a far face, and each
    face is its own message on the layer where its neighbour meets the
    strong window, so the window pays the two one-sided charges added
    up. A window that pins one face on the same run pays one of them.
    The two layers are the same width on that run, so the halves are
    also priced on a window whose oldest layer holds two detectors and
    whose newest holds one: 2 and 1, a sum no single layer of it gives.
    """
    result = _pinned_run(declared_run.DOUBLE_WINDOW, 3, 1)
    faces = _pinned_faces(result)
    charged = _charge_by_window(faces)
    counts = _face_counts(faces)
    two_faced = _windows_with_faces(counts, 2)
    one_faced = _windows_with_faces(counts, 1)
    two_faced_charges = {charged[window_id] for window_id in two_faced}
    one_faced_charges = {charged[window_id] for window_id in one_faced}
    assert two_faced_charges == {2 * 8}
    assert one_faced_charges == {8}
    near, far = _one_sided_halves()
    assert near == 2
    assert far == 1


def _one_sided_halves() -> tuple:
    """What each face of a window with two layer widths is charged.

    The window reads 4-9 and commits 4-6, with two detectors on round 4
    and one on round 9. The near face pins on the window before it, the
    far face on the window after it, so the two messages land on the two
    ends (Tan 2209.09219 lines 936-946).
    """
    positions = {10: (4, 0), 11: (4, 1), 12: (9, 0)}
    strong = window_records.WindowInfo(
        operation_id=1,
        window_index=1,
        commit_lo=4,
        commit_hi=6,
        buffer_hi=9,
        round_count=6,
        buffer_lo=4,
        deps=(),
        dependents=(),
        detector_positions=positions,
    )
    residual = window_records.DependencyResidual(detector_ids=())
    dense = boundary_payloads.DenseSeamMask()
    interaction = window_interactions.DefaultWindowInteraction(0, dense)
    earlier = _source_committing(1, 3)
    later = _source_committing(7, 9)
    near = interaction.boundary_payload_bits(residual, strong, earlier)
    far = interaction.boundary_payload_bits(residual, strong, later)
    return (near, far)


def test_a_far_pin_is_charged_the_layer_its_mask_lands_on():
    """The far face lands on the strong window's newest read layer.

    On double_window at d=3, seed 0, the escalated window (1,0)
    reads rounds 1 to 9 and pins its far face on (1,3), which commits
    from round 10: the message updates round 9, a bulk layer of
    d*d-1 = 8 detectors, while round 1 carries only (d*d-1)/2 = 4.
    Bombin 2303.04846 lines 775-788 makes that update the input the
    strong task reads, and it is one layer.
    """
    result = _pinned_run(declared_run.DOUBLE_WINDOW, 3, 0)
    near, far = _pins_by_direction(result)
    assert far == [8]
    assert near == [8]


def _pins_by_direction(result) -> tuple:
    """The bits of the run's near pins and of its far pins.

    A pin is attributed to the strong window that reads it, so a
    transfer whose attributed window is not its source is a pinned face;
    an earlier source is the near face and a later one the far face.
    """
    near = []
    far = []
    for window_id, source, seam in _pins_with_sources(result):
        if source < window_id:
            near.append(seam)
        else:
            far.append(seam)
    return (near, far)


def _pins_with_sources(result) -> list:
    """(strong window, source window, seam bits) of every pinned face."""
    pins = []
    for transfer in result.link_traffic["transfers"]:
        if not shape_tests.is_boundary(transfer):
            continue
        attribution = transfer["attribution"]
        source = attribution["relation"]["request_key"]["window_id"]
        window_id = attribution["window_id"]
        if window_id == source:
            continue
        seam = shape_tests.seam_bits(transfer)
        pins.append((window_id, source, seam))
    return pins


def test_a_backward_weak_hand_off_is_charged_the_layer_it_lands_on():
    """A later window's hand-off lands on its neighbour's newest layer.

    Skoric's parallel windows (2209.08552 lines 398-401) give the middle
    window a neighbour on each side: at d=5 window (1,1) commits and
    reads 11-25, and window (1,2), which commits from round 26, hands
    its boundary back onto round 25. Two of that layer's detectors are
    flipped at seed 62, and the sparse row charges one index per flip,
    ceil(log2(24)) = 5 bits each. Priced on round 11, where the message
    lands nothing, it would be free.
    """
    result = _parallel_run(5, 62)
    charged = _hand_offs(result)
    assert charged[(2, 1)] == 10
    assert charged[(2, 3)] == 5
    assert charged[(0, 1)] == 0


def _parallel_run(distance: int, seed: int):
    """One shot of the weak baseline cut into parallel windows.

    The sparse row prices the flips the message carries, so which layer
    is counted is visible in the bits.
    """
    settings = machine_settings.weak_decoder_baseline(distance, 0.005, 1.0)
    windows = declared_run.windows_on(
        settings.windows, parallel_scheme.ParallelWindowScheme
    )
    sparse = boundary_payloads.SparseSeamList.Settings()
    windows = dataclasses.replace(windows, boundary_payload=sparse)
    settings = dataclasses.replace(settings, windows=windows)
    machine = machine_module.Machine.build(settings, seed)
    return machine.run()


def _hand_offs(result) -> dict:
    """(source window, destination window) to the bits its message took."""
    charged = {}
    for transfer in result.link_traffic["transfers"]:
        if transfer["path"] != "decoder_to_decoder":
            continue
        relation = transfer["attribution"]["relation"]
        source = _window_key(relation["source_window_key"])
        destination = _window_key(relation["destination_window_key"])
        charged[(source[1], destination[1])] = transfer["payload_bits"]
    return charged


def _window_key(recorded: dict) -> tuple:
    """The (operation, window) key a traffic record writes as a tuple."""
    parts = []
    for item in recorded["items"]:
        parts.append(int(item["value"]))
    return tuple(parts)


def _charge_by_window(faces: list) -> dict:
    """What each strong window paid for the faces it pinned."""
    charged = {}
    for window_id, payload_bits in faces:
        running = charged.get(window_id, 0)
        charged[window_id] = running + payload_bits
    return charged


def _windows_with_faces(counts: list, face_count: int) -> list:
    """The windows that pinned exactly face_count faces."""
    windows = []
    for window_id, count in counts:
        if count == face_count:
            windows.append(window_id)
    return windows


def _face_counts(faces: list) -> list:
    """(window index, how many faces it pinned) over the run's pins."""
    counts = {}
    for window_id, _payload_bits in faces:
        counts[window_id] = counts.get(window_id, 0) + 1
    listed = counts.items()
    return sorted(listed)

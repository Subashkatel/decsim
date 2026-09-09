"""What crosses every hop, derived by hand first and then measured.

The oracle is the circuit and the window plan, not the run: a rotated
surface code memory reads d*d-1 ancilla bits per round and d*d data
bits on its last round, and its detector layers are (d*d-1)/2 on the
first round, d*d-1 in the bulk and 3(d*d-1)/2 on the last (Tan et al.
2209.09219 lines 936-946 for the seam layer; note 14 sections 2.2 and
5.2 derive the three counts). Every expected number below is that
arithmetic over the extents the plan lays, so a change in the data path
moves a number a reader can recompute rather than a guess.

Three shapes of the escalation policy, at d=3 and d=5, 15 rounds, on the
logical_reference card, where every one of the eleven paths is priced.
The switching shape runs with an unreachable gap threshold, so every
window escalates and the strong tier's traffic is the whole plan rather
than a sample. The feedback half of the loop needs a second workload: a
memory experiment never releases a conditional operation, so nothing
crosses frame_to_controller or controller_to_qpu until an operation
waits on another's result.

One subtlety the table states rather than hides: the two
controller-to-store hops price the raw readout bits, counted before the
detection events are formed (controller/round_assembly.py wire_bits),
while the stores then hold the formed events. The two counts differ on
the first and last rounds of a stream.
"""

import collections
import dataclasses

import pytest

import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.machine as machine_module
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.records.program as program_records
import decsim.settings as machine_settings_module

ROUNDS = 15
PROBABILITY = 0.008
SEED = 0
CODE_TASK = "surface_code:rotated_memory_z"
ENGINE_MEGAHERTZ = 250.0
# every window escalates at this threshold, so the strong tier's traffic
# is the plan rather than a sample of the weak decoder's confidence
UNREACHABLE_GAP_DECIBELS = 1000.0
DECODER_INPUT_PATHS = (
    "weak_buffer_to_weak_decoder",
    "strong_buffer_to_strong_decoder",
)
READOUT_PATH = "qpu_to_controller"
STORE_PATHS = (
    "controller_to_weak_buffer",
    "controller_to_strong_buffer",
)
CASES = (
    ("weak", 3),
    ("weak", 5),
    ("strong", 3),
    ("strong", 5),
    ("switching", 3),
    ("switching", 5),
)


@dataclasses.dataclass(frozen=True)
class Traffic:
    """One path's traffic: its transfers, its bits, its unpriced payloads."""

    transfers: int
    known_bits: int
    unknown_payloads: int = 0


# Derived, path by path, from the plan each case lays (the plan is
# asserted below beside the traffic):
#   d=3, R=15: layers 4, 8, 12; raw 8 per round and 8+9 on round 15.
#   d=5, R=15: layers 12, 24, 36; raw 24 per round and 24+25 on round 15.
# These cards form the detection events at the controller, so the
# readout hop carries the raw outcomes (129 and 385) and the hops into
# the two stores carry the layers they formed (4 + 13*8 + 12 = 120,
# 12 + 13*24 + 36 = 360).
EXPECTED = {
    ("weak", 3): {
        "qpu_to_controller": Traffic(15, 129),
        "controller_to_weak_buffer": Traffic(15, 120),
        "weak_buffer_to_weak_decoder": Traffic(4, 192),
        "decoder_to_decoder": Traffic(3, 24),
        "weak_decoder_to_frame": Traffic(4, 4),
    },
    ("weak", 5): {
        "qpu_to_controller": Traffic(15, 385),
        "controller_to_weak_buffer": Traffic(15, 360),
        "weak_buffer_to_weak_decoder": Traffic(2, 480),
        "decoder_to_decoder": Traffic(1, 24),
        "weak_decoder_to_frame": Traffic(2, 2),
    },
    ("strong", 3): {
        "qpu_to_controller": Traffic(15, 129),
        "controller_to_strong_buffer": Traffic(15, 120),
        "strong_buffer_to_strong_decoder": Traffic(4, 192),
        "decoder_to_decoder": Traffic(3, 24),
        "strong_decoder_to_frame": Traffic(4, 4),
    },
    ("strong", 5): {
        "qpu_to_controller": Traffic(15, 385),
        "controller_to_strong_buffer": Traffic(15, 360),
        "strong_buffer_to_strong_decoder": Traffic(2, 480),
        "decoder_to_decoder": Traffic(1, 24),
        "strong_decoder_to_frame": Traffic(2, 2),
    },
    ("switching", 3): {
        "qpu_to_controller": Traffic(15, 129),
        "controller_to_weak_buffer": Traffic(15, 120),
        "controller_to_strong_buffer": Traffic(15, 120),
        "weak_buffer_to_weak_decoder": Traffic(5, 220),
        "weak_decoder_to_strong_decoder": Traffic(5, 0, 5),
        "strong_buffer_to_strong_decoder": Traffic(5, 312),
        "decoder_to_decoder": Traffic(4, 32),
        "strong_decoder_to_frame": Traffic(5, 5),
    },
    ("switching", 5): {
        "qpu_to_controller": Traffic(15, 385),
        "controller_to_weak_buffer": Traffic(15, 360),
        "controller_to_strong_buffer": Traffic(15, 360),
        "weak_buffer_to_weak_decoder": Traffic(3, 612),
        "weak_decoder_to_strong_decoder": Traffic(3, 0, 3),
        "strong_buffer_to_strong_decoder": Traffic(3, 840),
        "decoder_to_decoder": Traffic(2, 48),
        "strong_decoder_to_frame": Traffic(3, 3),
    },
}
# The commit extents the sliding scheme lays for the two distances, read
# back so the traffic above is read against a stated plan.
EXPECTED_COMMITS = {
    ("weak", 3): ((1, 3), (4, 6), (7, 9), (10, 15)),
    ("weak", 5): ((1, 5), (6, 15)),
    ("strong", 3): ((1, 3), (4, 6), (7, 9), (10, 15)),
    ("strong", 5): ((1, 5), (6, 15)),
    ("switching", 3): ((1, 3), (4, 6), (7, 9), (10, 12), (13, 15)),
    ("switching", 5): ((1, 5), (6, 10), (11, 15)),
}

_RUNS = {}


def layer_detectors(round_index: int, total_rounds: int, distance: int) -> int:
    """The detectors one round layer of the memory contributes."""
    bulk = distance * distance - 1
    if round_index == 1:
        return bulk // 2
    if round_index == total_rounds:
        return bulk + bulk // 2
    return bulk


def window_detectors(
    round_lo: int, round_hi: int, total_rounds: int, distance: int
) -> int:
    """The detectors of every layer a window reads, clipped to the stream."""
    first = max(1, round_lo)
    last = min(total_rounds, round_hi)
    past_last = last + 1
    layers = range(first, past_last)
    bits = 0
    for round_index in layers:
        bits += layer_detectors(round_index, total_rounds, distance)
    return bits


def raw_readout_bits(round_index: int, total_rounds: int, distance: int) -> int:
    """The measurement bits one round puts on the wire, before detection."""
    ancilla = distance * distance - 1
    if round_index == total_rounds:
        return ancilla + distance * distance
    return ancilla


def machine_settings(shape: str, distance: int):
    """One memory operation on the Stim device, with the reference card."""
    circuit = workload_settings.memory_circuit(
        CODE_TASK, ROUNDS, distance, PROBABILITY
    )
    operation = program_records.Operation(
        id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit
    )
    rounds_policy = round_policies.FixedRounds(ROUNDS)
    workload = workload_settings.WorkloadSettings(
        operations=(operation,), rounds_policy=rounds_policy
    )
    device = stim_device.StimDevice()
    qpu = qpu_settings.QpuSettings(distance=distance, device=device)
    links = link_profiles.logical_reference_profile()
    weak = decoder_settings.DecoderSettings(
        kind="pymatching", engine_megahertz=ENGINE_MEGAHERTZ
    )
    strong = decoder_settings.DecoderSettings(
        kind="pymatching", engine_megahertz=ENGINE_MEGAHERTZ
    )
    if shape == "weak":
        escalation = escalation_settings.EscalationSettings(
            kind="weak_baseline"
        )
        return machine_settings_module.MachineSettings(
            workload=workload,
            qpu=qpu,
            weak_decoder=weak,
            escalation=escalation,
            links=links,
        )
    if shape == "strong":
        escalation = escalation_settings.EscalationSettings(kind="strong_only")
        return machine_settings_module.MachineSettings(
            workload=workload,
            qpu=qpu,
            strong_decoder=strong,
            escalation=escalation,
            links=links,
        )
    nats = escalation_settings.decibels_to_nats(UNREACHABLE_GAP_DECIBELS)
    escalation = escalation_settings.EscalationSettings(
        kind="switching",
        confidence="complementary_gap",
        gap_threshold_decibels=UNREACHABLE_GAP_DECIBELS,
        gap_threshold_nats=nats,
    )
    return machine_settings_module.MachineSettings(
        workload=workload,
        qpu=qpu,
        weak_decoder=weak,
        strong_decoder=strong,
        escalation=escalation,
        links=links,
    )


def run_case(shape: str, distance: int):
    """One run per case, kept: every test below reads the same traffic."""
    case = (shape, distance)
    done = _RUNS.get(case)
    if done is not None:
        return done
    settings = machine_settings(shape, distance)
    machine = machine_module.Machine.build(settings, SEED)
    result = machine.run()
    _RUNS[case] = (machine, result)
    return machine, result


def transfers_by_path(result) -> dict:
    """Every booked transfer, grouped by the path that carried it."""
    grouped = collections.defaultdict(list)
    for transfer in result.link_traffic["transfers"]:
        grouped[transfer["path"]].append(transfer)
    return dict(grouped)


def traffic_of(transfers: list) -> Traffic:
    """One path's transfers, bits and unpriced payloads."""
    known_bits = 0
    unknown = 0
    for transfer in transfers:
        payload_bits = transfer["payload_bits"]
        if payload_bits is None:
            unknown += 1
            continue
        known_bits += payload_bits
    return Traffic(len(transfers), known_bits, unknown)


def commit_extents(machine) -> tuple:
    """The commit region of every window the run laid, in plan order."""
    windows = machine.window_manager.planner.windows_by_key
    planned = windows.items()
    in_plan_order = sorted(planned)
    extents = []
    for _key, window in in_plan_order:
        extents.append((window.commit_lo, window.commit_hi))
    return tuple(extents)


@pytest.mark.parametrize("shape,distance", CASES)
def test_every_path_carries_the_traffic_the_plan_derives(shape, distance):
    """The eleven-path table: what fired, how often, and for how many bits."""
    machine, result = run_case(shape, distance)
    grouped = transfers_by_path(result)
    measured = {}
    for path, transfers in grouped.items():
        measured[path] = traffic_of(transfers)

    assert commit_extents(machine) == EXPECTED_COMMITS[(shape, distance)]
    assert measured == EXPECTED[(shape, distance)]


@pytest.mark.parametrize("shape,distance", CASES)
def test_a_decoder_input_carries_its_windows_detectors(shape, distance):
    """DecodeJob.payload_bits() is the layer sum over the window's rounds."""
    _machine, result = run_case(shape, distance)
    grouped = transfers_by_path(result)
    checked = 0
    for path in DECODER_INPUT_PATHS:
        for transfer in grouped.get(path, ()):
            attribution = transfer["attribution"]
            expected = window_detectors(
                attribution["round_lo"],
                attribution["round_hi"],
                ROUNDS,
                distance,
            )
            assert transfer["payload_bits"] == expected, attribution
            checked += 1
    assert checked > 0


def _checked_round_hops(transfers, width_of, distance) -> int:
    """Hold every single-round transfer to the width its sender let go of."""
    checked = 0
    for transfer in transfers:
        attribution = transfer["attribution"]
        round_index = attribution["round_lo"]
        expected = width_of(round_index, ROUNDS, distance)
        assert attribution["round_hi"] == round_index
        assert transfer["payload_bits"] == expected, attribution
        checked += 1
    return checked


@pytest.mark.parametrize("shape,distance", CASES)
def test_a_round_hop_carries_the_width_its_sender_let_go_of(shape, distance):
    """The readout hop prices the outcomes, the store hops the layer.

    These cards form the detection events at the controller
    (controller.detection_events_formed_at), so the round narrows at the
    assembler: what leaves the QPU is the measurement outcomes, and what
    leaves the controller for either store is the layer the store then
    holds.
    """
    _machine, result = run_case(shape, distance)
    grouped = transfers_by_path(result)
    readout = grouped.get(READOUT_PATH, ())
    checked = _checked_round_hops(readout, raw_readout_bits, distance)
    for path in STORE_PATHS:
        stored = grouped.get(path, ())
        checked += _checked_round_hops(stored, layer_detectors, distance)
    assert checked > 0


@pytest.mark.parametrize("distance", (3, 5))
def test_the_boundary_hop_pays_one_bulk_layer_per_hand_off(distance):
    """A hand-off updates the destination's oldest layer and nothing else.

    Tan et al. 2209.09219 lines 936-946; the layer is d*d-1 detectors, 8
    at d=3 and 24 at d=5, whatever the window's own extent is.
    """
    _machine, result = run_case("switching", distance)
    grouped = transfers_by_path(result)
    bulk_layer = distance * distance - 1
    sizes = []
    for transfer in grouped["decoder_to_decoder"]:
        sizes.append(transfer["payload_bits"])
    assert set(sizes) == {bulk_layer}


@pytest.mark.parametrize("distance", (3, 5))
def test_the_selection_carries_no_payload(distance):
    """The switching decision names a request; it moves no syndrome."""
    _machine, result = run_case("switching", distance)
    grouped = transfers_by_path(result)
    for transfer in grouped["weak_decoder_to_strong_decoder"]:
        assert transfer["payload_bits"] is None


@pytest.mark.parametrize("distance", (3, 5))
def test_the_wire_prices_raw_bits_where_the_store_holds_formed_events(
    distance,
):
    """The subtlety of note 14 section 5.4 item 5, stated as a number.

    wire_bits is computed from the raw fragments before
    _form_detection_events runs (controller/round_assembly.py), so the
    first round puts d*d-1 bits on the wire while its layer contributes
    (d*d-1)/2 detectors to the window that reads it. A test that
    asserted "the store holds what the wire carried" would be wrong.
    """
    _machine, result = run_case("weak", distance)
    grouped = transfers_by_path(result)
    wire_by_round = {}
    for transfer in grouped["qpu_to_controller"]:
        attribution = transfer["attribution"]
        wire_by_round[attribution["round_lo"]] = transfer["payload_bits"]
    first_layer = layer_detectors(1, ROUNDS, distance)
    last_layer = layer_detectors(ROUNDS, ROUNDS, distance)
    wire_bits = wire_by_round.values()
    wire_total = sum(wire_bits)
    detector_total = window_detectors(1, ROUNDS, ROUNDS, distance)

    assert wire_by_round[1] == distance * distance - 1
    assert wire_by_round[1] == 2 * first_layer
    assert wire_by_round[ROUNDS] > last_layer
    assert wire_total != detector_total


def test_the_feedback_hops_fire_when_an_operation_waits_on_a_result():
    """The second workload: a conditional release closes the loop.

    A memory experiment releases nothing, so frame_to_controller and
    controller_to_qpu never fire on it. With one operation blocked on
    another's result the frame's decision crosses to the controller as
    one 32-bit bus word and the released command reaches the QPU as one
    128-bit instruction word (link_profiles.py BUS_WORD_BITS,
    INSTRUCTION_WORD_BITS).
    """
    distance = 3
    rounds = 6
    circuit = workload_settings.memory_circuit(
        CODE_TASK, rounds, distance, PROBABILITY
    )
    first = program_records.Operation(
        id=1, name="mem0", qubits=(0,), patches=(0,), circuit=circuit
    )
    second = program_records.Operation(
        id=2,
        name="mem1",
        qubits=(1,),
        patches=(1,),
        circuit=circuit,
        blocked_by=1,
    )
    rounds_policy = round_policies.FixedRounds(rounds)
    workload = workload_settings.WorkloadSettings(
        operations=(first, second), rounds_policy=rounds_policy
    )
    device = stim_device.StimDevice()
    qpu = qpu_settings.QpuSettings(distance=distance, device=device)
    weak = decoder_settings.DecoderSettings(
        kind="pymatching", engine_megahertz=ENGINE_MEGAHERTZ
    )
    links = link_profiles.logical_reference_profile()
    settings = machine_settings_module.MachineSettings(
        workload=workload, qpu=qpu, weak_decoder=weak, links=links
    )
    machine = machine_module.Machine.build(settings, SEED)
    result = machine.run()
    grouped = transfers_by_path(result)
    decisions = traffic_of(grouped["frame_to_controller"])
    commands = traffic_of(grouped["controller_to_qpu"])
    released = [
        line
        for line in machine.observation.log.lines
        if "conditional release for op#2" in line
    ]

    assert decisions == Traffic(1, link_profiles.BUS_WORD_BITS)
    assert commands == Traffic(1, link_profiles.INSTRUCTION_WORD_BITS)
    assert len(released) == 1

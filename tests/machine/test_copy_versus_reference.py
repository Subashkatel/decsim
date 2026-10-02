"""The copy-versus-reference settings: each default is today's behaviour.

Every hop of this tree copies today, and each of these keys says whether
one hop may be a reference instead, so the two can be measured against
each other.
The sources are classical and quantum both: AFS's on-chip access
(2001.06598 lines 528-531), Collision Clustering's Init unit
(2309.05558 lines 268-271), Toshio's transfer of the assigned data
(2510.25222 lines 1248-1250), Helios's single-writer shared memory
(2301.08419 lines 632-640), Chen's non-blocking frame manager
(2605.30765 lines 1618-1620), Riverlane's polled status register
(2410.05202 lines 1256-1259) and Google's detections formed at the
workstation (2408.13687 lines 474-476), against IBM's detector window
processing on the decoder's own chip (2510.21600 lines 235-237).
"""

import dataclasses
import functools
import pathlib

import numpy
import pytest
import stim

import decsim.build.decoders as decoder_build
import decsim.confidence.cluster as cluster
import decsim.decoders.minimum_weight_perfect_matching.decoder as mwpm
import decsim.decoders.union_find.decoder as union_find
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.experiment as experiment
import decsim.machine as machine_module
import decsim.records.program as program_records
import decsim.records.transfers as transfer_records
import decsim.records.workload as workload_records
import tests.declared_run as declared_run

CONFIGS = pathlib.Path("configs")
WEAK_INPUT_PATH = transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER
# the seats each placement named here forms at
SEATS = {
    "controller": ("controller",),
    "weak_syndrome_buffer": ("weak_syndrome_buffer",),
    "decoder": ("weak_decoder", "strong_decoder"),
}
WEAK_SEATS = ("weak_decoder",)
BOTH_DECODERS = ("weak_decoder", "strong_decoder")
CHIP_THEN_HOST_DECODER = ("weak_decoder", "strong_syndrome_buffer")
# (config, windows.kind, escalation.strong_window, formed_at): every seat
# on every path a round takes, on the window shapes that read a round
# whose predecessor the seat never formed (Skoric's B blocks, and the
# strong side's regions of both shapes)
STIM_GRID = (
    (
        "bases/weak_decoder_baseline.yaml",
        "sliding",
        "redo_window",
        ("controller",),
    ),
    (
        "bases/weak_decoder_baseline.yaml",
        "sliding",
        "redo_window",
        ("weak_syndrome_buffer",),
    ),
    (
        "bases/weak_decoder_baseline.yaml",
        "sliding",
        "redo_window",
        WEAK_SEATS,
    ),
    (
        "bases/weak_decoder_baseline.yaml",
        "parallel",
        "redo_window",
        WEAK_SEATS,
    ),
    (
        "experiments/switching/redo_window_switching.yaml",
        "sliding",
        "redo_window",
        ("weak_syndrome_buffer",),
    ),
    (
        "experiments/switching/redo_window_switching.yaml",
        "sliding",
        "redo_window",
        BOTH_DECODERS,
    ),
    (
        "experiments/switching/redo_window_switching.yaml",
        "sliding",
        "redo_window",
        CHIP_THEN_HOST_DECODER,
    ),
    (
        "experiments/switching/redo_window_switching.yaml",
        "sliding",
        "double_window",
        BOTH_DECODERS,
    ),
    (
        "experiments/switching/redo_window_switching.yaml",
        "sliding",
        "double_window",
        CHIP_THEN_HOST_DECODER,
    ),
    (
        "bases/strong_decoder_baseline.yaml",
        "sliding",
        "redo_window",
        ("strong_syndrome_buffer",),
    ),
    (
        "bases/strong_decoder_baseline.yaml",
        "sliding",
        "redo_window",
        ("strong_decoder",),
    ),
)
# a decoder seat priced as Yang et al. price theirs: 5 cycles, then one
# round a clock (2605.04892 lines 1273-1275)
YANG_CYCLES = {"decoder": (5, 1)}


def _machine_formed_at(where: str, distance: int = 3):
    """The weak baseline, forming its detection events there."""
    config_path = CONFIGS / "bases/weak_decoder_baseline.yaml"
    config = experiment.load_experiment(config_path)
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": distance,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    detection_events = _formed_at(settings, where)
    observation = dataclasses.replace(settings.observation, data_movement=True)
    settings = dataclasses.replace(
        settings, detection_events=detection_events, observation=observation
    )
    return machine_module.Machine.build(settings, 0)


def _parallel_machine_formed_at(where: str):
    """The weak baseline on Skoric's A/B blocks, formed there."""
    formed_there = _machine_formed_at(where)
    settings = formed_there.settings
    windows = dataclasses.replace(settings.windows, kind="parallel")
    settings = dataclasses.replace(settings, windows=windows)
    return machine_module.Machine.build(settings, 0)


def _seated_machine(
    config_name, windows_kind, strong_window, formed_at, latency_cycles=0
):
    """A d=3 run of the config on that shape, formed at those seats.

    The seats form in latency_cycles on the section's clock, none by
    default, the reference card's cost.
    """
    config_path = CONFIGS / config_name
    config = experiment.load_experiment(config_path)
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.008,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    detection_events = dataclasses.replace(
        settings.detection_events,
        formed_at=formed_at,
        latency_cycles=latency_cycles,
    )
    windows = dataclasses.replace(settings.windows, kind=windows_kind)
    switching = settings.switching
    if switching is not None:
        named = declared_run.strong_window_settings(strong_window)
        switching = dataclasses.replace(switching, strong_window=named)
    settings = dataclasses.replace(
        settings,
        detection_events=detection_events,
        windows=windows,
        switching=switching,
    )
    return machine_module.Machine.build(settings, 0)


def _landed_rounds(machine) -> list:
    """Every round each decoder unit consumed, as its memory took it.

    The landing is where the unit's input is deposited; the masked
    duplicate a boundary fold makes afterwards is not a landing.
    """
    landed = []

    def record(job, bits, source_name, memory_name) -> None:
        del bits, source_name
        if memory_name == "masked view":
            return
        for fragment in job.decoder_input.fragments():
            key = (fragment.operation_id, fragment.round_index)
            landed.append((key, fragment.bits))

    for service in _services(machine):
        service.staging.trace.copy_made.connect(record)
    return landed


def _raw_rounds(machine) -> dict:
    """Every round's raw bits as the QPU emits them, and each operation."""
    device = machine.qpu.syndrome_source
    emitted = {"bits": {}, "operations": {}}
    emit = device.round_payloads

    def recording(operation, round_index):
        readouts = emit(operation, round_index)
        bits = []
        for readout in readouts:
            bits.extend(readout.bits)
        emitted["bits"][(operation.id, round_index)] = bits
        emitted["operations"][operation.id] = operation
        return readouts

    device.round_payloads = recording
    return emitted


def _mismatched_rounds(landed: list, expected: dict) -> list:
    """The keys of the landed rounds whose bits are not Stim's."""
    mismatched = []
    for key, bits in landed:
        if bits != expected[key]:
            mismatched.append(key)
    return mismatched


def _stims_events_by_round(machine, emitted) -> dict:
    """Stim's own converter's events on the emitted rows, by round."""
    device = machine.qpu.syndrome_source
    events_by_round = {}
    for operation_id, operation in emitted["operations"].items():
        table = device.formation_table(operation_id)
        by_round = _stims_events_of(operation, table, emitted["bits"])
        events_by_round.update(by_round)
    return events_by_round


def _stims_events_of(operation, table, bits_by_round) -> dict:
    """One operation's events from Stim's converter, keyed by round."""
    after_last_round = table.round_count + 1
    round_indices = range(1, after_last_round)
    row = []
    for round_index in round_indices:
        row.extend(bits_by_round[(operation.id, round_index)])
    converter = operation.circuit.compile_m2d_converter()
    measurements = numpy.array([row], dtype=numpy.bool_)
    converted = converter.convert(
        measurements=measurements, append_observables=False
    )
    shot_events = converted[0]
    by_round = {}
    for round_index in round_indices:
        recipes = table.detectors_of_round(round_index)
        bits = []
        for recipe in recipes:
            bits.append(int(shot_events[recipe.detector_index]))
        by_round[(operation.id, round_index)] = tuple(bits)
    return by_round


def _timing_only_machine_formed_at(where: str):
    """The weak baseline on a source that carries no outcomes to form."""
    formed_there = _machine_formed_at(where)
    settings = formed_there.settings
    qpu = dataclasses.replace(settings.qpu, kind="timing_only")
    settings = dataclasses.replace(settings, qpu=qpu)
    return machine_module.Machine.build(settings, 0)


def _decode_spans(machine) -> list:
    """Each window's ticks from its dispatch to its decode's end."""
    spans = []
    planner = machine.windows.window_manager.planner
    for window in planner.windows_by_key.values():
        span = window.t_done - window.t_dispatch
        spans.append(span)
    return spans


def _switching_machine_formed_at(where: str):
    """A switching run at d=3, where both tiers decode the same rounds."""
    config_path = CONFIGS / "experiments/switching/redo_window_switching.yaml"
    config = experiment.load_experiment(config_path)
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.008,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    detection_events = _formed_at(settings, where)
    settings = dataclasses.replace(settings, detection_events=detection_events)
    return machine_module.Machine.build(settings, 0)


def _double_window_switching_at_the_decoder():
    """A switching run whose strong region absorbs weak windows.

    The double window (Toshio 2510.25222 Sec. III C) rewrites the
    windows ahead of the seam, so the run withdraws weak decodes that
    were already submitted.
    """
    config_path = CONFIGS / "experiments/switching/redo_window_switching.yaml"
    config = experiment.load_experiment(config_path)
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.008,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    detection_events = _formed_at(settings, "decoder")
    double_window = strong_window_shapes.DoubleWindow.Settings()
    switching = dataclasses.replace(
        settings.switching, strong_window=double_window
    )
    settings = dataclasses.replace(
        settings, detection_events=detection_events, switching=switching
    )
    return machine_module.Machine.build(settings, 0)


def _formed_at(settings, where: str):
    """The detection_events card forming at the seats where names."""
    seats = SEATS[where]
    latency_cycles, cycles_per_round = YANG_CYCLES.get(where, (0, 0))
    return dataclasses.replace(
        settings.detection_events,
        formed_at=seats,
        latency_cycles=latency_cycles,
        cycles_per_round=cycles_per_round,
    )


def _services(machine) -> list:
    """The decode service of each side's manager, the chip's then the host's."""
    services = [machine.decoders.decoder_manager.service]
    if machine.decoders.strong_decoder_manager is not None:
        services.append(machine.decoders.strong_decoder_manager.service)
    return services


def _rounds_read_by_tier(machine) -> dict:
    """Per tier, every round a decode that started read."""
    read = {}

    def record(job, unit) -> None:
        del unit
        decoder_input = job.decoder_input
        if decoder_input is None:
            return
        keys = read.setdefault(job.pool, set())
        for round_input in decoder_input.rounds:
            keys.add((round_input.operation_id, round_input.round_index))

    for service in _services(machine):
        service.trace.job_started.connect(record)
    return read


def _rounds_charged_by_tier(machine) -> dict:
    """Per tier, every round a decode's formation stage was charged for."""
    charged = {}

    def record(job, unit) -> None:
        del unit
        claimed = job.detection_event_rounds
        if claimed is None:
            return
        keys = charged.setdefault(job.pool, set())
        keys.update(claimed)

    for service in _services(machine):
        service.trace.job_finished.connect(record)
    return charged


def _decoder_inputs(machine) -> list:
    """Every decode's landed input, bit by bit, in the order they started."""
    inputs = []

    def record(job, unit) -> None:
        del unit
        if job.decoder_input is None:
            return
        rounds = []
        for fragment in job.decoder_input.fragments():
            rounds.append(
                (fragment.operation_id, fragment.round_index, fragment.bits)
            )
        inputs.append((job.label, tuple(rounds)))

    for service in _services(machine):
        service.trace.job_started.connect(record)
    return inputs


def _formation_stages(machine) -> list:
    """Every event-detection stage recorded, with its cycles and rounds."""
    stages = []
    for record in machine.observation.stages.records:
        if record.stage != decoder_build.FORMATION_STAGE:
            continue
        ticks = record.end_ticks - record.start_ticks
        stages.append((record.cycles, len(record.round_keys), ticks))
    return stages


def _formed_round_keys(machine) -> list:
    """Every round key an event-detection stage was charged for."""
    keys = []
    for record in machine.observation.stages.records:
        if record.stage != decoder_build.FORMATION_STAGE:
            continue
        keys.extend(record.round_keys)
    return keys


def _store_copy_bits(machine) -> int:
    """The bits the assembler copied into the weak syndrome buffer per run."""
    copies = machine.observation.data_movement.copies.by_path
    for path, counts in copies.items():
        if path == "controller assembler -> weak syndrome buffer":
            return counts.bits
    return 0


def _machine(**weak_changes):
    """The weak baseline at d=3, counting its data movement."""
    config_path = CONFIGS / "bases/weak_decoder_baseline.yaml"
    config = experiment.load_experiment(config_path)
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    weak = dataclasses.replace(settings.weak_decoder, **weak_changes)
    observation = dataclasses.replace(settings.observation, data_movement=True)
    settings = dataclasses.replace(
        settings, weak_decoder=weak, observation=observation
    )
    return machine_module.Machine.build(settings, 0)


def _unit_input_copies(machine) -> int:
    """The copies into a decoder unit's own memory."""
    return _copies_on_paths_ending_with(machine, "memory")


def _masked_view_copies(machine) -> int:
    """The masked duplicates the boundary fold made of those inputs."""
    return _copies_on_paths_ending_with(machine, "masked view")


def _copies_on_paths_ending_with(machine, ending: str) -> int:
    copies = 0
    for (
        path,
        counts,
    ) in machine.observation.data_movement.copies.by_path.items():
        if "unit" in path and path.endswith(ending):
            copies += counts.events
    return copies


def _weak_input_transfers(machine) -> int:
    snapshot = machine.observation.traffic.snapshot()
    for path_snapshot in snapshot.paths:
        if path_snapshot.path is WEAK_INPUT_PATH:
            return path_snapshot.counters.transfer_count
    return 0


def _busy_unit_ticks(machine) -> int:
    """The time-integrated ticks the pool's units held their compute."""
    utilization = machine.observation.decoder_utilization.result()
    return utilization["busy_unit_ticks"]


def _weak_input_bits(machine) -> int:
    """The bits the weak tier's input link carried."""
    snapshot = machine.observation.traffic.snapshot()
    for path_snapshot in snapshot.paths:
        if path_snapshot.path is WEAK_INPUT_PATH:
            return path_snapshot.counters.known_payload_bits
    return 0


def _observables(result) -> list:
    """Each operation's committed logical observables, in order."""
    observables = []
    for row in result.operation_results:
        observables.append((row.operation_id, row.logical_observables))
    return observables


def test_the_input_default_copies_the_rounds_into_the_unit():
    """Copy is today's behaviour: one copy and one link move per window."""
    machine = _machine()
    machine.run()
    copies = _unit_input_copies(machine)
    transfers = _weak_input_transfers(machine)
    windows = len(machine.observation.windows.windows)
    assert copies == windows
    assert transfers == windows


def test_the_boundary_fold_default_leaves_the_run_as_it_is():
    """Copy is today's behaviour: a masked view per window past the first.

    tests/windows/test_decode_requests.py exercises both rows against a
    real unit memory; here the shipped baseline shows the count the fold
    adds, which is the windows that have a predecessor and not the seams
    that carried a defect.
    """
    copied = _machine()
    copied_result = copied.run()
    folded = _machine(copies_boundary_fold=True)
    folded_result = folded.run()
    windows = len(copied.observation.windows.windows)
    assert _observables(folded_result) == _observables(copied_result)
    assert _unit_input_copies(folded) == _unit_input_copies(copied)
    assert _masked_view_copies(copied) == windows - 1


def test_an_in_place_input_references_the_rounds_and_moves_nothing():
    """In place: no deposit, no link move, the store's hold kept instead."""
    copied = _machine()
    copied_result = copied.run()
    in_place = _machine(copies_input=False)
    in_place_result = in_place.run()
    assert _unit_input_copies(in_place) == 0
    assert _weak_input_transfers(in_place) == 0
    copied_references = copied.observation.data_movement.references.events
    in_place_references = in_place.observation.data_movement.references.events
    windows = len(in_place.observation.windows.windows)
    assert in_place_references == copied_references + windows
    copied_observables = _observables(copied_result)
    in_place_observables = _observables(in_place_result)
    assert in_place_observables == copied_observables


def test_an_input_rule_that_is_not_a_flag_is_refused_by_name():
    with pytest.raises(ValueError, match="copies_input 'in-place'"):
        _machine(copies_input="in-place")


def test_a_boundary_fold_rule_that_is_not_a_flag_is_refused_by_name():
    with pytest.raises(ValueError, match="copies_boundary_fold 'in place'"):
        _machine(copies_boundary_fold="in place")


def test_the_result_default_frees_the_unit_at_the_decodes_end():
    """False is today's behaviour: the compute is back when the decode ends."""
    default = _machine()
    default.run()
    non_blocking = _machine(result_blocks_unit=False)
    non_blocking.run()
    assert _busy_unit_ticks(non_blocking) == _busy_unit_ticks(default)


def test_a_blocking_result_holds_the_unit_until_the_window_commits():
    """True keeps the unit busy over the output hop and the frame write."""
    default = _machine()
    default_result = default.run()
    blocking = _machine(result_blocks_unit=True)
    blocking_result = blocking.run()
    assert _busy_unit_ticks(blocking) > _busy_unit_ticks(default)
    assert _observables(blocking_result) == _observables(default_result)
    default_windows = default.observation.windows.windows
    blocking_windows = blocking.observation.windows.windows
    assert len(blocking_windows) == len(default_windows)


def test_the_formation_default_sends_the_events_from_the_controller():
    """Controller is today's behaviour: the input link carries the events."""
    default = _machine()
    default.run()
    at_the_controller = _machine_formed_at("controller")
    at_the_controller.run()
    assert _weak_input_bits(at_the_controller) == _weak_input_bits(default)


def test_a_source_with_no_outcomes_pays_the_tiers_formation_stage():
    """The stage is the tier's hardware, whatever the source sends it."""
    with_outcomes = _machine_formed_at("decoder")
    with_outcomes.run()
    without_outcomes = _timing_only_machine_formed_at("decoder")
    without_outcomes.run()
    spans_without_outcomes = _decode_spans(without_outcomes)
    spans_with_outcomes = _decode_spans(with_outcomes)
    assert spans_without_outcomes == spans_with_outcomes


def test_events_formed_at_the_decoder_widen_the_tiers_input_link():
    """The store and the input link then carry the raw outcomes."""
    at_the_controller = _machine_formed_at("controller")
    controller_result = at_the_controller.run()
    at_the_decoder = _machine_formed_at("decoder")
    decoder_result = at_the_decoder.run()
    assert _weak_input_bits(at_the_decoder) > _weak_input_bits(
        at_the_controller
    )
    assert _observables(decoder_result) == _observables(controller_result)


@pytest.mark.parametrize("distance", (3, 5))
def test_the_same_events_are_decoded_wherever_they_were_formed(distance):
    """The seat moves the width and the time, never a bit of the syndrome."""
    at_the_controller = _machine_formed_at("controller", distance)
    controller_inputs = _decoder_inputs(at_the_controller)
    controller_result = at_the_controller.run()
    at_the_decoder = _machine_formed_at("decoder", distance)
    decoder_inputs = _decoder_inputs(at_the_decoder)
    decoder_result = at_the_decoder.run()
    assert decoder_inputs == controller_inputs
    assert _observables(decoder_result) == _observables(controller_result)


def test_the_widths_the_two_seats_send_are_the_events_and_the_outcomes():
    """d=3: 240 event bits into the weak syndrome buffer, 249 outcome bits."""
    at_the_controller = _machine_formed_at("controller")
    at_the_controller.run()
    at_the_decoder = _machine_formed_at("decoder")
    at_the_decoder.run()
    assert _store_copy_bits(at_the_controller) == 240
    assert _store_copy_bits(at_the_decoder) == 249
    assert _weak_input_bits(at_the_controller) == 432
    assert _weak_input_bits(at_the_decoder) == 441


def test_events_formed_on_the_weak_chip_cross_raw_and_leave_its_store_formed():
    """d=3: 249 outcome bits into the chip, 432 event bits out of its store."""
    at_the_controller = _machine_formed_at("controller")
    controller_inputs = _decoder_inputs(at_the_controller)
    controller_result = at_the_controller.run()
    on_the_weak_chip = _machine_formed_at("weak_syndrome_buffer")
    chip_inputs = _decoder_inputs(on_the_weak_chip)
    chip_result = on_the_weak_chip.run()
    chip_observables = _observables(chip_result)
    controller_observables = _observables(controller_result)
    assert _store_copy_bits(on_the_weak_chip) == 249
    assert _weak_input_bits(on_the_weak_chip) == 432
    assert chip_inputs == controller_inputs
    assert chip_observables == controller_observables


def test_a_region_escalated_from_the_weak_chip_is_formed_by_no_tier():
    """The strong side decodes the events the chip stored, bit for bit."""
    at_the_controller = _switching_machine_formed_at("controller")
    controller_inputs = _decoder_inputs(at_the_controller)
    controller_result = at_the_controller.run()
    on_the_weak_chip = _switching_machine_formed_at("weak_syndrome_buffer")
    chip_inputs = _decoder_inputs(on_the_weak_chip)
    chip_result = on_the_weak_chip.run()
    chip_observables = _observables(chip_result)
    controller_observables = _observables(controller_result)
    assert _formation_stages(on_the_weak_chip) == []
    assert chip_inputs == controller_inputs
    assert chip_observables == controller_observables


def test_a_tier_pays_yangs_latency_then_one_round_a_clock():
    """Six rounds at the first window, three at each window after it.

    The stage is pipelined (Yang 2605.04892 line 1273), so the cycles
    are 5 + (r - 1) and 5 + (r - b - 1), not 5 per round.
    """
    at_the_decoder = _machine_formed_at("decoder")
    at_the_decoder.run()
    charged = _formation_stages(at_the_decoder)
    assert charged[0] == (10, 6, 40000)
    assert charged[1] == (7, 3, 28000)


def test_the_controller_seat_charges_no_tier_for_a_formation_it_did():
    at_the_controller = _machine_formed_at("controller")
    at_the_controller.run()
    assert _formation_stages(at_the_controller) == []


def test_one_tier_forms_and_charges_each_round_it_reads_once():
    """Windows of one tier overlap; the rounds they share are formed once."""
    at_the_decoder = _machine_formed_at("decoder")
    at_the_decoder.run()
    charged = _formed_round_keys(at_the_decoder)
    assert len(charged) == len(set(charged))


def test_the_second_tier_forms_the_rounds_it_reads_out_of_its_own_store():
    """Both syndrome buffers hold the raw rounds, so both tiers pay."""
    switching = _switching_machine_formed_at("decoder")
    switching.run()
    charged = _formed_round_keys(switching)
    charged_twice = len(charged) - len(set(charged))
    assert charged_twice > 0


def test_no_round_a_tier_read_goes_uncharged_on_that_tier():
    """A withdrawn weak window leaves its rounds for whoever reads them.

    The double window withdraws weak decodes when the strong region
    absorbs their windows, and the strong tier then reads the same
    rounds out of the strong syndrome buffer; every round a started decode read
    is charged once on the tier that read it.
    """
    forward = _double_window_switching_at_the_decoder()
    read = _rounds_read_by_tier(forward)
    charged = _rounds_charged_by_tier(forward)

    forward.run()

    assert read["default"] - charged["default"] == set()
    assert read["strong"] - charged["strong"] == set()


def test_disjoint_ranges_formed_at_the_decoder_decode_the_same_events():
    """Each block's decoder reads the raw round before the block.

    Skoric A/B blocks decode disjoint round ranges (2209.08552 sec. I.C),
    so a B block starts at a round whose predecessor its tier never
    formed; the store sends that round with the block, and every decode
    reads what it reads when the controller forms the events.
    """
    at_the_decoder = _parallel_machine_formed_at("decoder")
    decoder_inputs = _decoder_inputs(at_the_decoder)
    at_the_controller = _parallel_machine_formed_at("controller")
    controller_inputs = _decoder_inputs(at_the_controller)

    at_the_decoder.run()
    at_the_controller.run()

    assert decoder_inputs == controller_inputs


@pytest.mark.parametrize(
    "config_name, windows_kind, strong_window, formed_at", STIM_GRID
)
def test_every_decoder_unit_consumes_stims_events_from_every_seat(
    config_name, windows_kind, strong_window, formed_at
):
    """What each unit's memory takes is Stim's events, bit for bit.

    The referent is stim.Circuit.compile_m2d_converter on the raw rows
    the QPU emitted, so the check reads what the decoder consumes after
    every hop, whichever seat formed it.
    """
    machine = _seated_machine(
        config_name, windows_kind, strong_window, formed_at
    )
    emitted = _raw_rounds(machine)
    landed = _landed_rounds(machine)

    machine.run()

    expected = _stims_events_by_round(machine, emitted)
    mismatched = _mismatched_rounds(landed, expected)
    assert landed
    assert mismatched == []


# a seed that escalates a window whose predecessor stayed weak
LOOKBACK_SEED = 5


@pytest.mark.parametrize("formed_at", [BOTH_DECODERS, CHIP_THEN_HOST_DECODER])
def test_a_switching_run_on_a_two_round_lookback_consumes_stims_events(
    formed_at,
):
    """A region after a weak window lands Stim's events at every unit.

    The strong side joins that region mid-stream and is given the two
    raw rounds its first round reads (stim.Circuit.compile_m2d_converter
    on the rows the QPU emitted is the referent).
    """
    machine = _lookback_switching_machine(formed_at, LOOKBACK_SEED)
    emitted = _raw_rounds(machine)
    landed = _landed_rounds(machine)

    machine.run()

    expected = _stims_events_by_round(machine, emitted)
    mismatched = _mismatched_rounds(landed, expected)
    assert mismatched == []


@pytest.mark.parametrize("unit_count", [1, 3])
def test_a_seam_formed_after_a_later_block_reads_the_rounds_it_kept(
    unit_count: int,
):
    """Round 4 reads round 2, formed two blocks before; Stim agrees.

    stim.Circuit.compile_m2d_converter on the rows the QPU emitted is
    the referent for every unit's landed rounds.
    """
    machine = _parallel_lookback_machine(unit_count)
    emitted = _raw_rounds(machine)
    landed = _landed_rounds(machine)

    machine.run()

    expected = _stims_events_by_round(machine, emitted)
    mismatched = _mismatched_rounds(landed, expected)
    assert landed
    assert mismatched == []


@pytest.mark.parametrize("formed_at", [BOTH_DECODERS, CHIP_THEN_HOST_DECODER])
def test_a_strong_region_is_given_every_round_its_later_rounds_read(
    formed_at,
):
    """Round 4 of the region from round 3 reads round 1; Stim agrees.

    stim.Circuit.compile_m2d_converter on the rows the QPU emitted is
    the referent for every unit's landed rounds.
    """
    machine = _reach_growing_machine(formed_at)
    emitted = _raw_rounds(machine)
    landed = _landed_rounds(machine)

    machine.run()

    expected = _stims_events_by_round(machine, emitted)
    mismatched = _mismatched_rounds(landed, expected)
    assert mismatched == []


@pytest.mark.parametrize("round_count", [24, 48])
def test_a_strong_seat_lets_go_of_rounds_only_the_weak_side_forms(
    round_count: int,
):
    """The rounds that read a region's rounds form at the weak side alone.

    They leave the store once formed there, so the strong seat lets go
    of the packets they read: it holds at most five one-bit packets at
    48 rounds as at 24, the regions in flight and the rounds they read,
    and none at the end.
    """
    machine = _every_third_escalating_machine(round_count)
    held_by_seat = {}
    note = functools.partial(_note_packets, held_by_seat)
    machine.readout.detection_events.trace.state_held.connect(note)

    machine.run()

    held = held_by_seat["strong_decoder"]
    assert max(held) <= 5
    assert held[-1] == 0


@pytest.mark.parametrize("round_count", [24, 48])
def test_no_seat_or_store_keeps_a_round_for_an_observable(round_count: int):
    """Every round's bit is in the observable, which folds as it arrives.

    Each seat holds the two rounds back its detectors read, and the weak
    store the rounds in flight, at 48 rounds as at 24.
    """
    machine = _every_third_escalating_machine(round_count, EVERY_ROUND_OBSERVED)
    held_by_seat = {}
    note = functools.partial(_note_packets, held_by_seat)
    machine.readout.detection_events.trace.state_held.connect(note)
    store = machine.readout.weak_syndrome_buffer
    stored_rounds = []
    count = functools.partial(_note_stored_rounds, store, stored_rounds)
    store.trace.round_stored.connect(count)

    machine.run()

    assert max(held_by_seat["weak_decoder"]) <= 3
    assert max(held_by_seat["strong_decoder"]) <= 3
    assert max(stored_rounds) <= 3


@pytest.mark.parametrize("round_count", [24, 48])
def test_no_seat_keeps_a_record_of_rounds_once_the_run_ends(round_count: int):
    """Every round has retired: no events, claims or done rounds stay."""
    machine = _every_third_escalating_machine(round_count)

    machine.run()

    weak = _bookkeeping_of(machine, "weak_decoder")
    strong = _bookkeeping_of(machine, "strong_decoder")
    assert weak == {"events": 0, "claims": 0, "done_above": 0}
    assert strong == {"events": 0, "claims": 0, "done_above": 0}


# an observable record of the round just measured, after its detector
EVERY_ROUND_OBSERVED = "OBSERVABLE_INCLUDE(0) rec[-1]\n"


def test_a_strong_seat_forming_in_cycles_stores_each_escalated_round_once():
    """Five cycles at the strong buffer: formed after it lands, sent once.

    A window woken between the landing and the store sends nothing
    again; each unit still consumes Stim's events, and the run ends.
    """
    machine = _seated_machine(
        "experiments/switching/redo_window_switching.yaml",
        "sliding",
        "redo_window",
        CHIP_THEN_HOST_DECODER,
        latency_cycles=5,
    )
    emitted = _raw_rounds(machine)
    landed = _landed_rounds(machine)

    machine.run()

    expected = _stims_events_by_round(machine, emitted)
    mismatched = _mismatched_rounds(landed, expected)
    assert landed
    assert mismatched == []


@pytest.mark.parametrize(
    "where, seat",
    [
        ("controller", "controller"),
        ("weak_syndrome_buffer", "weak_syndrome_buffer"),
        ("decoder", "weak_decoder"),
    ],
)
def test_the_forming_seat_reports_the_two_raw_rounds_it_holds(where, seat):
    """d=3: a bulk detector reads the round before, so a seat holds two.

    The most is the last round's 8 check bits and 9 data bits beside the
    8 of the round before it.
    """
    machine = _machine_formed_at(where)

    machine.run()

    data_movement = machine.observation.data_movement
    assert data_movement.formation_state_bits_by_seat == {seat: 25}


def _lookback_workload(
    settings, noisy_round: str = "X_ERROR(0.05) 0\nM(0.05) 0\n"
):
    """Ten rounds of one qubit whose detector reads two rounds back.

    Round r's detector is rec[-1] ^ rec[-3], so a seat that joins at a
    region reads two raw rounds before it.
    """
    circuit = stim.Circuit(
        f"R 0\n{noisy_round}DETECTOR rec[-1]\n"
        f"{noisy_round}DETECTOR rec[-1]\n"
        f"REPEAT 8 {{\n{noisy_round}DETECTOR rec[-1] rec[-3]\n}}\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n"
    )
    measurement_rounds = {index: index + 1 for index in range(10)}
    physical = workload_records.FiniteCircuit(circuit, measurement_rounds)
    operation = program_records.Operation(1, "memory", (0,), patches=(0,))
    workload = workload_records.Workload((operation,), {1: 10}, physical)
    return settings.workload.running(workload)


def _lookback_switching_machine(formed_at, seed):
    """Union-find switching on the lookback circuit, windows of 2 + 2."""
    machine = _seated_machine(
        "experiments/switching/redo_window_switching.yaml",
        "sliding",
        "redo_window",
        formed_at,
    )
    settings = machine.settings
    workload = _lookback_workload(settings)
    windows = dataclasses.replace(
        settings.windows, commit_rounds=2, buffer_rounds=2
    )
    union_find_settings = union_find.UnionFindDecoder.Settings()
    weak_decoder = dataclasses.replace(
        settings.weak_decoder, algorithm=union_find_settings
    )
    confidence = cluster.ClusterGap.Settings()
    escalation = dataclasses.replace(settings.switching, confidence=confidence)
    settings = dataclasses.replace(
        settings,
        workload=workload,
        windows=windows,
        weak_decoder=weak_decoder,
        switching=escalation,
    )
    return machine_module.Machine.build(settings, seed)


def _parallel_lookback_machine(unit_count: int):
    """Skoric's blocks of one round plus one on the two-round lookback.

    The weak decoder forms, on unit_count units: rounds 1 to 3, then 5
    to 7, then the seam 3 to 5, whose round 4 reads round 2.
    """
    machine = _seated_machine(
        "experiments/switching/redo_window_switching.yaml",
        "sliding",
        "redo_window",
        BOTH_DECODERS,
    )
    settings = machine.settings
    # a flipped readout alone, so no fault straddles two blocks' commits
    workload = _lookback_workload(settings, "M(0.15) 0\n")
    windows = dataclasses.replace(
        settings.windows, kind="parallel", commit_rounds=1, buffer_rounds=1
    )
    # a declared decoder decodes no syndrome, so no fault's ownership is
    # asked of the blocks; the landed events are the check
    weak = declared_run.DeclaredConfidenceDecoder.Settings(0.028, _is_no_window)
    weak_decoder = dataclasses.replace(
        settings.weak_decoder,
        algorithm=weak,
        engine=declared_run.DECLARED_ENGINE,
        unit_count=unit_count,
    )
    settings = dataclasses.replace(
        settings,
        workload=workload,
        windows=windows,
        weak_decoder=weak_decoder,
        strong_decoder=None,
        switching=None,
    )
    return machine_module.Machine.build(settings, 0)


def _is_no_window(job) -> bool:
    del job
    return False


def _reach_growing_machine(formed_at):
    """The second window of a reach-growing circuit escalates, alone.

    Ten rounds of rec[-1], then rec[-1] ^ rec[-4] from round 4 on, in
    windows of two plus two, so the strong region from round 3 reads
    round 3 alone first and round 1 for its round 4. The weak decoder
    declares the confidence, so only window 1 escalates.
    """
    machine = _seated_machine(
        "experiments/switching/redo_window_switching.yaml",
        "sliding",
        "redo_window",
        formed_at,
    )
    settings = machine.settings
    noisy_round = "M(0.15) 0\n"
    circuit = stim.Circuit(
        f"R 0\nREPEAT 3 {{\n{noisy_round}DETECTOR rec[-1]\n}}\n"
        f"REPEAT 7 {{\n{noisy_round}DETECTOR rec[-1] rec[-4]\n}}\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n"
    )
    measurement_rounds = {index: index + 1 for index in range(10)}
    physical = workload_records.FiniteCircuit(circuit, measurement_rounds)
    operation = program_records.Operation(1, "memory", (0,), patches=(0,))
    workload = workload_records.Workload((operation,), {1: 10}, physical)
    windows = dataclasses.replace(
        settings.windows, commit_rounds=2, buffer_rounds=2
    )
    weak = declared_run.DeclaredConfidenceDecoder.Settings(
        0.028, _is_window_one
    )
    weak_decoder = dataclasses.replace(
        settings.weak_decoder,
        algorithm=weak,
        engine=declared_run.DECLARED_ENGINE,
    )
    matching = mwpm.PyMatchingDecoder.Settings(preset_latency_microseconds=0.2)
    strong_decoder = dataclasses.replace(
        settings.strong_decoder,
        algorithm=matching,
    )
    switching = _declared_confidence(settings.switching)
    running = settings.workload.running(workload)
    settings = dataclasses.replace(
        settings,
        workload=running,
        windows=windows,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        switching=switching,
    )
    return machine_module.Machine.build(settings, 0)


def _declared_confidence(switching):
    """The yaml's switching slot, deciding on the weak tier's declared gap."""
    confidence = declared_run.DeclaredConfidence.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_nats=declared_run.ESCALATION_THRESHOLD
    )
    return dataclasses.replace(
        switching, confidence=confidence, threshold=threshold
    )


def _is_window_one(job) -> bool:
    return job.window_id == 1


def _every_third_escalating_machine(round_count: int, observed: str = ""):
    """Rounds reading two back; windows of one round; every third escalates.

    The strong side forms the escalated windows' rounds, and the weak
    side every round, on three units each. observed follows each
    round's detector: an observable's record of the round, or nothing.
    """
    machine = _seated_machine(
        "experiments/switching/redo_window_switching.yaml",
        "sliding",
        "redo_window",
        BOTH_DECODERS,
    )
    settings = machine.settings
    noisy_round = "M(0.15) 0\n"
    later_rounds = round_count - 2
    circuit = stim.Circuit(
        f"R 0\nREPEAT 2 {{\n{noisy_round}DETECTOR rec[-1]\n{observed}}}\n"
        f"REPEAT {later_rounds} {{\n{noisy_round}DETECTOR rec[-1] rec[-3]\n"
        f"{observed}}}\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n"
    )
    measurement_rounds = {index: index + 1 for index in range(round_count)}
    physical = workload_records.FiniteCircuit(circuit, measurement_rounds)
    operation = program_records.Operation(1, "memory", (0,), patches=(0,))
    workload = workload_records.Workload(
        (operation,), {1: round_count}, physical
    )
    windows = dataclasses.replace(
        settings.windows, commit_rounds=1, buffer_rounds=0
    )
    weak = declared_run.DeclaredConfidenceDecoder.Settings(
        0.028, _is_every_third
    )
    weak_decoder = dataclasses.replace(
        settings.weak_decoder,
        algorithm=weak,
        engine=declared_run.DECLARED_ENGINE,
        unit_count=3,
    )
    matching = mwpm.PyMatchingDecoder.Settings(preset_latency_microseconds=0.2)
    strong_decoder = dataclasses.replace(
        settings.strong_decoder,
        algorithm=matching,
        unit_count=3,
    )
    switching = _declared_confidence(settings.switching)
    running = settings.workload.running(workload)
    settings = dataclasses.replace(
        settings,
        workload=running,
        windows=windows,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        switching=switching,
    )
    return machine_module.Machine.build(settings, 0)


def _is_every_third(job) -> bool:
    return job.window_id % 3 == 0


def _note_packets(held_by_seat: dict, seat, operation_id, bits) -> None:
    del operation_id
    held = held_by_seat.setdefault(seat, [])
    held.append(bits)


def _bookkeeping_of(machine, seat: str) -> dict:
    placement = machine.readout.detection_events
    history = placement.history_by_seat[seat]
    done_rounds = history.memory.done_by_operation[1]
    return {
        "events": len(history.memory.events_by_round),
        "claims": len(history.memory.claimed_keys),
        "done_above": len(done_rounds.above),
    }


def _note_stored_rounds(store, stored_rounds: list, *stored) -> None:
    del stored
    stored_rounds.append(len(store.round_by_key))

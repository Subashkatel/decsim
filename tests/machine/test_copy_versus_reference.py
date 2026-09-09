"""The copy-versus-reference settings: each default is today's behaviour.

Design audit note 14 section 8.2 (recommendation R4-12): every hop of
this tree copies today, and each of these keys says whether one hop may
be a reference instead, so the two can be measured against each other.
The sources are classical and quantum both: AFS's on-chip access
(2001.06598 lines 528-531), Collision Clustering's Init unit
(2309.05558 lines 268-271), Toshio's transfer of the assigned data
(2510.25222 lines 1248-1250), Helios's single-writer shared memory
(2301.08419 lines 632-640), Chen's non-blocking frame manager
(2605.30765 lines 1618-1620), Riverlane's polled status register
(2410.05202 lines 1256-1259) and Google's detections formed at the
workstation (2408.13687 lines 474-476).
"""

import dataclasses
import pathlib

import pytest

import decsim.build.decoders as decoder_build
import decsim.front.experiment as experiment
import decsim.machine as machine_module
import decsim.records.transfers as transfer_records

CONFIGS = pathlib.Path("configs")
WEAK_INPUT_PATH = transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER


def _machine_formed_at(where: str, distance: int = 3):
    """The weak baseline, forming its detection events there."""
    config_path = CONFIGS / "weak_decoder_baseline.yaml"
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.001,
        distance=distance,
        round_period_us=1.0,
    )
    controller = dataclasses.replace(
        settings.controller, detection_events_formed_at=where
    )
    observation = dataclasses.replace(settings.observation, data_movement=True)
    settings = dataclasses.replace(
        settings, controller=controller, observation=observation
    )
    return machine_module.Machine.build(settings, 0)


def _switching_machine_formed_at(where: str):
    """A switching run at d=3, where both tiers decode the same rounds."""
    config_path = CONFIGS / "seam_pinned_switching.yaml"
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.008, distance=3, round_period_us=1.0
    )
    controller = dataclasses.replace(
        settings.controller, detection_events_formed_at=where
    )
    settings = dataclasses.replace(settings, controller=controller)
    return machine_module.Machine.build(settings, 0)


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

    machine.decoder_manager.service.trace.job_started.connect(record)
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
    """The bits the assembler copied into Buffer 0 over the run."""
    copies = machine.observation.data_movement.copies.by_path
    for path, counts in copies.items():
        if path == "controller assembler -> Buffer 0":
            return counts.bits
    return 0


def _machine(**weak_changes):
    """The weak baseline at d=3, counting its data movement."""
    config_path = CONFIGS / "weak_decoder_baseline.yaml"
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.001, distance=3, round_period_us=1.0
    )
    weak = dataclasses.replace(settings.weak_decoder, **weak_changes)
    observation = dataclasses.replace(
        settings.observation, data_movement=True, decoder_utilization=True
    )
    settings = dataclasses.replace(
        settings, weak_decoder=weak, observation=observation
    )
    return machine_module.Machine.build(settings, 0)


def _unit_input_copies(machine) -> int:
    """The copies into a decoder unit's own memory."""
    copies = 0
    for (
        path,
        counts,
    ) in machine.observation.data_movement.copies.by_path.items():
        if "unit" in path:
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
    """Copy is today's behaviour; the fold itself is tested on the gate.

    tests/windows/test_decode_requests.py exercises both rows against a
    real unit memory, because no shipped config blocks a window long
    enough to fold a boundary at its start.
    """
    copied = _machine()
    copied_result = copied.run()
    folded = _machine(boundary_fold="copy")
    folded_result = folded.run()
    assert _observables(folded_result) == _observables(copied_result)
    assert _unit_input_copies(folded) == _unit_input_copies(copied)


def test_an_in_place_input_references_the_rounds_and_moves_nothing():
    """In place: no deposit, no link move, the store's hold kept instead."""
    copied = _machine()
    copied_result = copied.run()
    in_place = _machine(input="in_place")
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


def test_an_input_kind_that_is_not_a_row_is_refused_by_name():
    with pytest.raises(ValueError, match="weak_decoder.input 'in-place'"):
        _machine(input="in-place")


def test_a_boundary_fold_that_is_not_a_row_is_refused_by_name():
    with pytest.raises(
        ValueError, match="weak_decoder.boundary_fold 'in place'"
    ):
        _machine(boundary_fold="in place")


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


def test_the_same_events_are_decoded_wherever_they_were_formed():
    """The row moves the width and the time, never a bit of the syndrome."""
    for distance in (3, 5):
        at_the_controller = _machine_formed_at("controller", distance)
        controller_inputs = _decoder_inputs(at_the_controller)
        controller_result = at_the_controller.run()
        at_the_decoder = _machine_formed_at("decoder", distance)
        decoder_inputs = _decoder_inputs(at_the_decoder)
        decoder_result = at_the_decoder.run()
        assert decoder_inputs == controller_inputs
        assert _observables(decoder_result) == _observables(controller_result)


def test_the_widths_the_two_rows_send_are_the_events_and_the_outcomes():
    """d=3: 240 bits of events into Buffer 0, 249 bits of outcomes."""
    at_the_controller = _machine_formed_at("controller")
    at_the_controller.run()
    at_the_decoder = _machine_formed_at("decoder")
    at_the_decoder.run()
    assert _store_copy_bits(at_the_controller) == 240
    assert _store_copy_bits(at_the_decoder) == 249
    assert _weak_input_bits(at_the_controller) == 432
    assert _weak_input_bits(at_the_decoder) == 441


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


def test_the_controller_row_charges_no_tier_for_a_formation_it_did():
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
    """Buffer 0 and Buffer 1 hold the raw rounds, so both tiers pay."""
    switching = _switching_machine_formed_at("decoder")
    switching.run()
    charged = _formed_round_keys(switching)
    charged_twice = len(charged) - len(set(charged))
    assert charged_twice > 0


def test_a_shape_that_reads_disjoint_ranges_cannot_form_at_the_decoder():
    """A tier asked out of order refuses.

    Skoric A/B blocks decode disjoint round ranges (2209.08552 sec. I.C),
    so a tier is asked for a round whose predecessor its table has
    forgotten, and it refuses rather than form the wrong syndrome.
    """
    config_path = CONFIGS / "weak_decoder_baseline.yaml"
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.001, distance=3, round_period_us=1.0
    )
    controller = dataclasses.replace(
        settings.controller, detection_events_formed_at="decoder"
    )
    windows = dataclasses.replace(settings.windows, kind="parallel")
    settings = dataclasses.replace(
        settings, controller=controller, windows=windows
    )
    machine = machine_module.Machine.build(settings, 0)

    with pytest.raises(RuntimeError) as refusal:
        machine.run()

    sentence = str(refusal.value)
    assert "the rounds of an operation are formed in order" in sentence
    assert "form its events at the controller" in sentence


def test_a_formation_place_that_is_not_a_row_is_refused_by_name():
    with pytest.raises(
        ValueError, match="controller.detection_events_formed_at 'workstation'"
    ):
        _machine_formed_at("workstation")

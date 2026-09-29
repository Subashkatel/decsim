"""The assembly file describes the machine the root builds.

One thing is asked of this file: the root builds the same objects in
one fixed order. The order is checked here against a short expected
list for one declared run, and both golden hashes of the frozen suite
check the rest of it on every run of the gate.

The other laws are the ones a table of rows can break on its own: a run
the machine has no use for a seat in has no row for it, and a wire that
names a seat the run did not build, a port its class does not declare,
or a peer that does not answer that port is refused rather than
silently bound, which is gem5's PortRef.connect refusing by name
(gem5 src/python/m5/params/port_params.py:109-114).
"""

import dataclasses

import pytest

import decsim.assembly as assembly
import decsim.build.controller_side as controller_side
import decsim.build.decoders as decoder_build
import decsim.build.escalation as escalation_build
import decsim.build.parts as build_parts
import decsim.build.plan as plan_build
import decsim.controller.controller as controller_module
import decsim.controller.syndrome_round_sender as syndrome_round_sender
import decsim.decoders.decoder_manager as decoder_manager_module
import decsim.decoders.schedulers as schedulers
import decsim.decoders.settings as decoder_settings
import decsim.decoders.strong_requests as strong_requests_module
import decsim.engine as engine_module
import decsim.escalation.pending_strong_windows as pending_strong_windows
import decsim.escalation.strong_redecode as strong_redecode_module
import decsim.frontends.execution_runtime as execution_runtime_module
import decsim.machine as machine_module
import decsim.qpu.cycle_clock as cycle_clock
import decsim.records.seeds as seed_records
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import decsim.windows.window_manager as window_manager_module
import tests.declared_run as declared_run
from decsim.syndrome_buffer import (
    strong_syndrome_round_receiver as strong_syndrome_round_receiver_module,
)

# ten seats of a switching run, in the order the root builds them
EXPECTED_SEATS = (
    ("held_rounds", syndrome_round_sender.HeldRounds),
    ("weak_syndrome_buffer", syndrome_buffer_module.SyndromeBuffer),
    ("strong_requests", strong_requests_module.StrongRequests),
    ("decoder_manager", decoder_manager_module.DecoderManager),
    ("strong_decoder_manager", decoder_manager_module.DecoderManager),
    ("pending_strong_windows", pending_strong_windows.PendingStrongWindows),
    ("window_manager", window_manager_module.WindowManager),
    (
        "strong_syndrome_round_receiver",
        strong_syndrome_round_receiver_module.StrongSyndromeRoundReceiver,
    ),
    ("syndrome_round_sender", syndrome_round_sender.SyndromeRoundSender),
    ("qpu", cycle_clock.QPUDevice),
    ("controller", controller_module.Controller),
    ("execution_runtime", execution_runtime_module.ExecutionRuntime),
)


def _names_among(seats: dict, names: list) -> list:
    """The seats' names that are among names, in the seats' own order."""
    among = []
    for name in seats:
        if name in names:
            among.append(name)
    return among


def _seats_of_another_kind(seats: dict) -> list:
    """The expected seats whose built object is not the expected class."""
    other_kinds = []
    for name, class_built in EXPECTED_SEATS:
        if not isinstance(seats[name], class_built):
            other_kinds.append(name)
    return other_kinds


def test_the_root_builds_the_same_seats_in_the_same_order():
    settings = _switching_settings()
    parts = _parts_of(settings)
    seats = assembly.build_seats(parts)
    expected_names = [name for name, _class_built in EXPECTED_SEATS]
    built_names = _names_among(seats, expected_names)
    other_kinds = _seats_of_another_kind(seats)
    assert built_names == expected_names
    assert other_kinds == []


def test_a_run_that_never_escalates_has_no_room_side_rows():
    settings = _weak_settings()
    parts = _parts_of(settings)
    names = _seat_names(parts)
    assert "strong_syndrome_buffer" not in names
    assert "strong_syndrome_round_receiver" not in names
    assert "strong_output" not in names
    assert "pending_strong_windows" not in names
    assert "strong_redecode" not in names
    assert "strong_decoder_manager" not in names
    assert "weak_syndrome_buffer" in names


def test_a_run_that_decides_on_no_confidence_has_no_gap_join_row():
    settings = _switching_settings()
    parts = _parts_of(settings)
    names = _seat_names(parts)
    assert "gap_join" not in names
    assert "confidence_signal" not in names
    assert "strong_redecode" in names


def test_a_run_whose_burst_detector_is_none_has_no_detector_row():
    settings = _switching_settings()
    parts = _parts_of(settings)
    names = _seat_names(parts)
    assert "burst_detector" not in names


def test_an_escalating_run_builds_the_room_side():
    settings = _switching_settings()
    parts = _parts_of(settings)
    seats = assembly.build_seats(parts)
    writer = seats["strong_syndrome_round_receiver"]
    redecode = seats["strong_redecode"]
    assert isinstance(
        writer,
        strong_syndrome_round_receiver_module.StrongSyndromeRoundReceiver,
    )
    assert isinstance(redecode, strong_redecode_module.StrongRedecode)


def test_each_side_has_its_own_manager_decoder_pool_and_one_ledger():
    settings = _switching_settings()
    parts = _parts_of(settings)
    seats = assembly.build_seats(parts)
    wires = assembly.wires_for(parts)
    assembly.bind(wires, seats)
    chip = seats["decoder_manager"]
    host = seats["strong_decoder_manager"]
    assert sorted(chip.pool.units_by_pool) == ["default"]
    assert sorted(host.pool.units_by_pool) == ["strong"]
    assert chip.strong_requests is seats["strong_requests"]
    assert host.strong_requests is seats["strong_requests"]
    assert host.escalation_policy is seats["escalation_policy"]
    assert chip.decoder is seats["primary_decoder"]
    assert host.decoder is seats["strong_decoder"]
    assert seats["models"].strong_decoder is seats["strong_decoder"]
    assert chip.queue is not host.queue
    assert chip.queue.scheduler is not host.queue.scheduler
    assert chip.service.staging is not host.service.staging


class _SeedRecordingScheduler(schedulers.FifoScheduler):
    """A FIFO that keeps the seed the run hands it."""

    def __init__(self):
        self.reserved_seeds = []

    def reserve_run_seed(self, seed):
        self.reserved_seeds.append(seed)
        return seed_records.RunSeedReservation("derived", seed, None)

    def commit_run_seed(self, reservation):
        del reservation

    def cancel_run_seed(self, reservation):
        del reservation


def test_each_managers_scheduler_is_seeded_on_its_own_path():
    weak, strong = declared_run.switching_decoders(0.0, None)
    weak_decoder = decoder_settings.DecoderSettings(decoder=weak)
    strong_decoder = decoder_settings.DecoderSettings(decoder=strong)
    manager_settings = decoder_settings.DecoderManagerSettings(
        scheduler=_SeedRecordingScheduler
    )
    switching = _switching_settings()
    settings = dataclasses.replace(
        switching,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        decoder_manager=manager_settings,
    )
    machine = machine_module.Machine.build(settings, 7)
    chip_seeds = machine.decoder_manager.queue.scheduler.reserved_seeds
    host_seeds = machine.strong_decoder_manager.queue.scheduler.reserved_seeds
    assert len(chip_seeds) == 1
    assert len(host_seeds) == 1
    assert chip_seeds != host_seeds


def test_the_strong_side_submits_to_the_hosts_manager():
    settings = _switching_settings()
    parts = _parts_of(settings)
    seats = assembly.build_seats(parts)
    wires = assembly.wires_for(parts)
    assembly.bind(wires, seats)
    host = seats["strong_decoder_manager"]
    assert seats["strong_redecode"].decode_queue is host
    assert seats["requester"].strong_decode_queue is host
    assert seats["requester"].decode_queue is seats["decoder_manager"]


def test_a_wire_that_names_a_seat_the_run_did_not_build_is_refused():
    settings = _switching_settings()
    parts = _parts_of(settings)
    seats = assembly.build_seats(parts)
    wires = (("transmitter.link", "no_such_seat"),)
    with pytest.raises(ValueError, match="no seat for"):
        assembly.bind(wires, seats)


def test_a_wire_that_names_no_port_of_its_seat_is_refused():
    settings = _switching_settings()
    parts = _parts_of(settings)
    seats = assembly.build_seats(parts)
    wires = (("transmitter.no_such_port", "links"),)
    with pytest.raises(ValueError, match="names no port"):
        assembly.bind(wires, seats)


def test_a_wire_whose_peer_does_not_answer_its_port_is_refused():
    settings = _switching_settings()
    parts = _parts_of(settings)
    seats = assembly.build_seats(parts)
    wires = (("transmitter.link", "held_rounds"),)
    with pytest.raises(ValueError, match="does not answer"):
        assembly.bind(wires, seats)


def _switching_settings():
    """The declared switching run's settings, which read the room side."""
    machine = declared_run.switching_run(escalation_probability=1.0)
    return machine.settings


def _weak_settings():
    """The declared weak-only run: one tier, the weak syndrome buffer only."""
    machine = declared_run.weak_only_run()
    return machine.settings


def _seat_names(parts: build_parts.Parts) -> list:
    """The names of the rows one run builds, in order."""
    names = []
    for name, _build in assembly.seats_for(parts):
        names.append(name)
    return names


def _parts_of(settings) -> build_parts.Parts:
    """The fixtures one run compiles before any seat, as the root does."""
    engine = engine_module.Engine()
    escalation_policy = escalation_build.build_escalation_policy(
        settings.escalation, settings.weak_decoder
    )
    plan = plan_build.build_plan(settings, escalation_policy)
    detection_events = controller_side.build_detection_events(
        settings, plan.device, escalation_policy
    )
    pool = decoder_build.build_decoder_pool(
        settings, plan, escalation_policy, detection_events
    )
    return build_parts.Parts(
        settings=settings,
        engine=engine,
        plan=plan,
        escalation_policy=escalation_policy,
        pool=pool,
        detection_events=detection_events,
    )

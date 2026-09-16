"""The assembly file describes the machine the root used to build by hand.

Note 36 section 5 item C4 asks one thing of this file: the root builds
the same objects in the same order it did when that order was written
out in machine.py. The order is checked here against a short expected
list for one declared run, and both golden hashes of the frozen suite
check the rest of it on every run of the gate.

The other laws are the ones a table of rows can break on its own: a run
the machine has no use for a seat in has no row for it, and a wire that
names a seat the run did not build, a port its class does not declare,
or a peer that does not answer that port is refused rather than
silently bound, which is gem5's PortRef.connect refusing by name
(tmp/resources/gem5/src/python/m5/params/port_params.py:109-114).
"""

import pytest

import decsim.assembly as assembly
import decsim.build.controller_side as controller_side
import decsim.build.decoders as decoder_build
import decsim.build.escalation as escalation_build
import decsim.build.plan as plan_build
import decsim.controller.controller as controller_module
import decsim.controller.round_sender as round_sender
import decsim.decoders.decoder_manager as decoder_manager_module
import decsim.engine as engine_module
import decsim.escalation.strong_redecode as strong_redecode_module
import decsim.frontends.execution_runtime as execution_runtime_module
import decsim.qpu.cycle_clock as cycle_clock
import decsim.syndrome_buffer.strong_round_receiver as strong_receiver_module
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import decsim.windows.window_manager as window_manager_module
import tests.declared_run as declared_run

# nine seats of a switching run, in the order the root builds them
EXPECTED_SEATS = (
    ("held_rounds", round_sender.HeldRounds),
    ("weak_syndrome_buffer", syndrome_buffer_module.SyndromeBuffer),
    ("decoder_manager", decoder_manager_module.DecoderManager),
    ("window_manager", window_manager_module.WindowManager),
    (
        "strong_round_receiver",
        strong_receiver_module.StrongRoundReceiver,
    ),
    ("round_sender", round_sender.RoundSender),
    ("qpu", cycle_clock.QPUDevice),
    ("controller", controller_module.Controller),
    ("execution_runtime", execution_runtime_module.ExecutionRuntime),
)


def test_the_root_builds_the_same_seats_in_the_same_order():
    settings = _switching_settings()
    parts = _parts_of(settings)
    seats = assembly.build_seats(parts)
    expected_names = []
    for name, _class_built in EXPECTED_SEATS:
        expected_names.append(name)
    built_names = []
    for name in seats:
        if name in expected_names:
            built_names.append(name)
    assert built_names == expected_names
    for name, class_built in EXPECTED_SEATS:
        assert isinstance(seats[name], class_built), name


def test_a_run_that_never_escalates_has_no_room_side_rows():
    settings = _weak_settings()
    parts = _parts_of(settings)
    names = _seat_names(parts)
    assert "strong_syndrome_buffer" not in names
    assert "strong_round_receiver" not in names
    assert "strong_output" not in names
    assert "strong_redecode" not in names
    assert "weak_syndrome_buffer" in names


def test_a_run_that_decides_on_no_confidence_has_no_gap_join_row():
    settings = _switching_settings()
    parts = _parts_of(settings)
    names = _seat_names(parts)
    assert "gap_join" not in names
    assert "confidence_signal" not in names
    assert "strong_redecode" in names


def test_an_escalating_run_builds_the_room_side():
    settings = _switching_settings()
    parts = _parts_of(settings)
    seats = assembly.build_seats(parts)
    writer = seats["strong_round_receiver"]
    redecode = seats["strong_redecode"]
    assert isinstance(writer, strong_receiver_module.StrongRoundReceiver)
    assert isinstance(redecode, strong_redecode_module.StrongRedecode)


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


def _seat_names(parts: assembly.Parts) -> list:
    """The names of the rows one run builds, in order."""
    names = []
    for name, _build in assembly.seats_for(parts):
        names.append(name)
    return names


def _parts_of(settings) -> assembly.Parts:
    """The fixtures one run compiles before any seat, as the root does."""
    engine = engine_module.Engine()
    escalation_policy = escalation_build.build_escalation_policy(
        settings.escalation
    )
    plan = plan_build.build_plan(settings, escalation_policy)
    detection_events = controller_side.build_detection_events(
        settings, plan.device
    )
    pool = decoder_build.build_decoder_pool(
        settings, plan, escalation_policy, detection_events
    )
    return assembly.Parts(
        settings=settings,
        engine=engine,
        plan=plan,
        escalation_policy=escalation_policy,
        pool=pool,
        detection_events=detection_events,
    )

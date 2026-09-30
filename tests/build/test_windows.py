"""The windows part: the components it holds and the wires it binds.

STYLE.md rule 7: a component declares each neighbour as a port, its
constructor takes settings only, and the part that holds it binds each
wire by assignment. The windows part binds its own wires when it is
built and takes the stores and the decoder managers when the machine
connects the parts. This file pins the binds the window side makes
late: the courier names the facade it tells of a landing, the committer
and the verdict name the strong redecode built after them, which a run
that never escalates leaves unbound, and the strong side submits to the
host's manager.
"""

import tests.declared_run as declared_run
import tests.escalation.declared_fabric as fabric


def test_the_courier_names_the_facade_it_tells_a_landing():
    machine = declared_run.weak_only_run(rounds=6)
    courier = machine.windows.window_manager.courier

    assert courier.windows is machine.windows.window_manager


def test_the_committer_and_the_verdict_name_the_strong_redecode():
    machine = fabric.switching_machine(rounds=9, escalated_windows={1})
    redecode = machine.windows.window_manager.strong_redecode
    verdict = machine.windows.window_manager.requester.verdict

    assert verdict.strong_redecode is redecode
    assert verdict.committer.strong_redecode is redecode


def test_a_run_that_never_escalates_leaves_the_redecode_unbound():
    """An optional port nobody binds reads as None; nothing binds None."""
    machine = declared_run.weak_only_run(rounds=6)
    verdict = machine.windows.window_manager.requester.verdict

    assert machine.windows.pending_strong_windows is None
    assert machine.windows.window_manager.strong_redecode is None
    assert verdict.strong_redecode is None
    assert verdict.committer.strong_redecode is None


def test_a_switching_run_builds_only_the_window_side_it_names():
    """It may escalate, so it re-decodes on the strong side.

    Its policy decides on no confidence and it names no burst detector,
    so it has no confidence signal, no gap join and no detector.
    """
    machine = declared_run.switching_run(escalation_probability=1.0)
    windows = machine.windows

    assert windows.strong_redecode is not None
    assert windows.confidence_signal is None
    assert windows.gap_join is None
    assert windows.burst_detector is None


def test_the_strong_side_submits_to_the_hosts_manager():
    machine = declared_run.switching_run(escalation_probability=1.0)
    chip = machine.decoders.decoder_manager
    host = machine.decoders.strong_decoder_manager
    requester = machine.windows.requester

    assert machine.windows.strong_redecode.decode_queue is host
    assert requester.strong_decode_queue is host
    assert requester.decode_queue is chip

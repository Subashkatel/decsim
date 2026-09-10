"""The root's callback law, and what each callback holds together.

decsim/machine.py's docstring (lines 15-29) states the law: "Every
component is wired by constructor; nothing is bound to a component after
it is built. Where two components refer to each other the per-job
callback law breaks the cycle (SimPy's callback on the event,
simpy/core.py step()): the submitting side carries the return path with
the job."  Each test below removes exactly one of the callbacks that
docstring names from a run that otherwise completes, and shows the run
then never completes: either it stops on the root's own refusal
(decsim/machine.py:346-362, :379-380) or it never settles at all.

A run that never settles is caught by a bounded runner rather than by
waiting: a listener on the engine's own action_done source
(decsim/engine.py:98) stops the run after more actions than the wired
run needs.  The rule pinned is the root's sentence, not a tick.
"""

import dataclasses

import pytest

import decsim.build.stores as store_build
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.escalation.policies as escalation_policies
import decsim.escalation.settings as escalation_settings
import decsim.machine as machine_module
import decsim.qpu.cycle_clock as cycle_clock
import decsim.settings as machine_settings
import tests.declared_run as declared_run

WEAK_MICROSECONDS = declared_run.DECLARED_MICROSECONDS["weak"]
STRONG_MICROSECONDS = declared_run.DECLARED_MICROSECONDS["strong"]
ACTION_BUDGET = 4000


class NeverSettlesError(RuntimeError):
    """The bounded runner's stop: the run ran past the budget."""


def run_bounded(machine):
    """Run the machine, stopping if it runs past the action budget."""
    counter = _ActionCounter(ACTION_BUDGET)
    machine.engine.action_done.connect(counter.action_done)
    return machine.run()


class _ActionCounter:
    """A listener on action_done that stops a run that will not end."""

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.count = 0

    def action_done(self, tick) -> None:
        del tick
        self.count += 1
        if self.count > self.budget:
            raise NeverSettlesError(f"more than {self.budget} actions")


def _weak():
    """The declared weak-only settings, named on their own line."""
    return weak_settings()


def _strong():
    """The strong-primary settings, named on their own line."""
    return strong_settings()


def weak_settings(**changes):
    """The declared weak-only run, the one every test here starts from."""
    decoder = decoders.PresetLatencyDecoder(WEAK_MICROSECONDS)
    operation = declared_run.memory_operation(1)
    workload = declared_run.declared_workload([operation], 6)
    weak = decoder_settings.DecoderSettings(decoder=decoder)
    qpu = declared_run.declared_qpu()
    links = declared_run.declared_profile()
    controller = declared_run.declared_controller()
    frame = declared_run.declared_frame()
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        weak_decoder=weak,
        links=links,
        controller=controller,
        pauli_frame=frame,
    )
    return dataclasses.replace(settings, **changes)


def strong_settings():
    """The strong-primary run, whose rounds land in the room-side store."""
    decoder = decoders.PresetLatencyDecoder(STRONG_MICROSECONDS)
    operation = declared_run.memory_operation(1)
    workload = declared_run.declared_workload([operation], 6)
    strong = decoder_settings.DecoderSettings(decoder=decoder)
    policy = escalation_policies.StrongOnly(escalation_policies.NO_CONFIDENCE)
    escalation = escalation_settings.EscalationSettings(policy=policy)
    qpu = declared_run.declared_qpu()
    links = declared_run.declared_profile()
    controller = declared_run.declared_controller()
    frame = declared_run.declared_frame()
    return machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        strong_decoder=strong,
        escalation=escalation,
        links=links,
        controller=controller,
        pauli_frame=frame,
    )


def test_both_runs_this_file_breaks_complete_when_every_callback_is_wired():
    """The control: the two runs finish well inside the action budget."""
    weak_settings_record = _weak()
    weak = machine_module.Machine.build(weak_settings_record, 0)
    weak_result = run_bounded(weak)
    strong_settings_record = _strong()
    strong = machine_module.Machine.build(strong_settings_record, 0)
    strong_result = run_bounded(strong)

    assert weak_result.terminal_status == "complete"
    assert strong_result.terminal_status == "complete"


def test_without_the_qpus_completion_callback_the_run_never_settles(
    monkeypatch,
):
    """The cycle: the QPU ends a body, the runtime owns the operation's life.

    machine.py:252 binds it (`qpu.connect_completion_receiver(
    execution_runtime.body_done)`); the runtime is built after the QPU,
    so the QPU cannot take it by constructor.
    """

    def drop_the_receiver(self, receiver) -> None:
        del receiver
        self.receivers.completion = _hear_nothing

    monkeypatch.setattr(
        cycle_clock.QPUDevice, "connect_completion_receiver", drop_the_receiver
    )
    settings = _weak()
    machine = machine_module.Machine.build(settings, 0)
    with pytest.raises(NeverSettlesError):
        run_bounded(machine)


def test_without_the_strong_writers_landing_callback_the_room_side_deadlocks(
    monkeypatch,
):
    """The cycle: the writer lands a round, the window manager waits for it.

    build/stores.py gives the writer `window_manager.accept_room_round`;
    the window manager is built before the writer, so the writer takes
    the callback and the manager never names the writer.
    """
    real = store_build.build_strong_round_writer

    def writer_without_the_callback(engine, store, window_manager):
        writer = real(engine, store, window_manager)
        writer.on_round_stored = _hear_nothing
        return writer

    monkeypatch.setattr(
        store_build, "build_strong_round_writer", writer_without_the_callback
    )
    settings = _strong()
    machine = machine_module.Machine.build(settings, 0)
    with pytest.raises(RuntimeError) as raised:
        run_bounded(machine)
    assert "still holds" in str(raised.value)
    assert machine.engine.idle


def test_without_the_qpus_readout_callback_the_run_ends_with_nothing_decoded(
    monkeypatch,
):
    """The cycle: the QPU emits a readout, the controller receives it.

    machine.py:251 binds it (`qpu.connect_readout_receiver(controller)`);
    the controller reaches the window manager through the assembler and
    the round writer, and the window manager is built from the plan the
    QPU device came out of, so the QPU cannot take the controller by
    constructor.  Removing it does not deadlock: the run ends early and
    silently, with no round on any hop and no window decoded, which is
    the misordering half of the law.
    """
    wired_settings = _weak()
    wired = machine_module.Machine.build(wired_settings, 0)
    wired_result = run_bounded(wired)

    def drop_the_receiver(self, receiver) -> None:
        del receiver
        self.receivers.readout = _SilentReadout()

    monkeypatch.setattr(
        cycle_clock.QPUDevice, "connect_readout_receiver", drop_the_receiver
    )
    broken_settings = _weak()
    broken = machine_module.Machine.build(broken_settings, 0)
    broken_result = run_bounded(broken)
    assert broken_result.terminal_status == "complete"
    assert broken_result.fully_done_ticks < wired_result.fully_done_ticks
    assert wired.observation.windows.contribution_by_key != {}
    assert broken.observation.windows.contribution_by_key == {}
    assert _transfers(wired_result, "qpu_to_controller") == 6
    assert _transfers(broken_result, "qpu_to_controller") == 0


class _SilentReadout:
    """A receiver that hears every readout and tells nobody."""

    def accept_qpu_readout(self, *values) -> None:
        del values


def test_the_bounded_run_of_the_wired_weak_machine_needs_far_fewer_actions():
    """The budget is honest: the wired run uses a small part of it."""
    settings = _weak()
    machine = machine_module.Machine.build(settings, 0)
    counter = _ActionCounter(ACTION_BUDGET)
    machine.engine.action_done.connect(counter.action_done)
    machine.run()
    assert counter.count < ACTION_BUDGET // 4


def _transfers(result, path: str) -> int:
    """One path's transfer count out of the run result's link ledger."""
    for edge in result.link_traffic["semantic_edges"]:
        if edge["path"] == path:
            return edge["counters"]["transfer_count"]
    return 0


def _hear_nothing(*values) -> None:
    """A receiver that is called and tells nobody: the dropped callback."""
    del values

"""controller.packing_rounds_in_flight bounds the whole packing stage.

A round counts against the bound from its first fragment until the
windows have heard of it, not only while it is in assembly. On the
declared fabric (1 us rounds, 2 us qpu_to_controller, 3 us readout to
bits, 4 us controller_to_weak_buffer) round r's fragment lands at r + 5
us and its publication completes at r + 9 us, so a bound of b stops the
run when round 1 + b lands at 6 + b us, with round 1 still in flight; a
bound of six clears a 12-round memory experiment.
"""

import pytest

from decsim.controller.settings import ControllerSettings
from decsim.decoders.decoders import PresetLatencyDecoder
from decsim.decoders.settings import DecoderSettings
from decsim.frontends.settings import WorkloadSettings
from decsim.machine import Machine, MachineSettings
from decsim.pauli_frame.pauli_frame import PauliFrameConfig
from decsim.qpu.round_policies import FixedRounds

STOP_TICK_BY_BOUND = {
    1: 7_000_000,
    2: 8_000_000,
    3: 9_000_000,
    4: 10_000_000,
}
RUN_END_TICK = 54_000_000


def bounded_settings(fabric, bound):
    """A 12-round weak-only memory experiment with the bound set."""
    declared = fabric["DECLARED_US"]
    controller = ControllerSettings(
        readout_to_bits_microseconds=declared["binary"],
        packing_microseconds_per_round=declared["pack"],
        packing_rounds_in_flight=bound,
    )
    memory = fabric["memory_op"](1)
    twelve_rounds = FixedRounds(12)
    workload = WorkloadSettings(
        operations=[memory], rounds_policy=twelve_rounds
    )
    weak = PresetLatencyDecoder(declared["weak"])
    weak_decoder = DecoderSettings(decoder=weak)
    links = fabric["declared_profile"](
        controller_to_weak_buffer=True, controller_to_strong_buffer=False
    )
    frame = PauliFrameConfig(commit_microseconds=declared["frame"])
    qpu = fabric["declared_qpu"]()
    return MachineSettings(
        workload=workload,
        qpu=qpu,
        weak_decoder=weak_decoder,
        links=links,
        controller=controller,
        pauli_frame=frame,
    )


@pytest.mark.parametrize("bound", sorted(STOP_TICK_BY_BOUND))
def test_a_round_counts_until_the_windows_hear_of_it(fabric, bound):
    settings = bounded_settings(fabric, bound)
    machine = Machine.build(settings, 0)
    stop_tick = STOP_TICK_BY_BOUND[bound]

    sentence = f"the packing workspace is full at tick {stop_tick}: "
    with pytest.raises(RuntimeError, match=sentence):
        machine.run()

    assert machine.engine.now == stop_tick


def test_a_bound_of_six_clears_twelve_rounds(fabric):
    settings = bounded_settings(fabric, 6)

    completed = fabric["run_machine"](settings, 0)

    assert completed.observation.round_events.packing_drops == 0
    assert completed.engine.now == RUN_END_TICK

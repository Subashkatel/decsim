"""Feedback boundary modes for blocking operations."""

import pytest

from conftest import fixed_latency_link_config

from decsim.codes import SurfaceCodeModel
from decsim.controllers import ModularController
from decsim.decoders import PresetLatencyDecoder
from decsim.devices import TimingOnlyDevice
from decsim.frontends.circuit import CircuitFrontend
from decsim.message import Operation
from decsim.metrics import ConditionalReactionTime
from decsim.planner import PerOpRounds
from decsim.schemes import SlidingWindowScheme
from decsim.run_spec import RunSpec, simulate
from decsim.planner import FixedRounds
from decsim.policies import from_mode


def _zero_link_controller(engine, links):
    """Controller whose fabric links take no simulated time."""
    return ModularController(engine, links=links, log_syndromes=False)


def _feedback_chain():
    """Two T operations where the second waits for the first decode."""
    return CircuitFrontend([
        Operation(0, "T0", (0,), clifford=False, consumes_magic_state=False),
        Operation(1, "T1", (0,), clifford=False, consumes_magic_state=False,
                  blocked_by=0),
    ]).build()


def _reaction_rows(result):
    """Return conditional reaction rows for a completed run."""
    return ConditionalReactionTime(result.chip).rows()


def test_trailing_buffer_boundary_keeps_existing_static_wait():
    """The default SWIPER-baseline mode waits for the final look-ahead buffer."""
    result = simulate(RunSpec(
                 ops=_feedback_chain(),
                 num_units=1,
                 rounds_policy=FixedRounds(3),
                 round_us=1.0,
                 code=SurfaceCodeModel(d=3),
                 scheme=SlidingWindowScheme(),
                 decoder=PresetLatencyDecoder(2.0),
                 links=fixed_latency_link_config(),
                 make_controller=_zero_link_controller,
                 make_metrics=lambda _engine, _cluster, chip, _factory: [
            ConditionalReactionTime(chip)
        ],
             ), verbose=False)

    rows = _reaction_rows(result)
    assert len(rows) == 1
    assert rows[0]["wait_rounds"] == 5.0
    assert result.cluster.memory_rounds[0] >= 3


def test_measurement_closed_boundary_removes_only_static_buffer_wait():
    """DecLat-style closed boundaries submit the final window without the +d buffer wait."""
    result = simulate(RunSpec(
                 ops=_feedback_chain(),
                 num_units=1,
                 rounds_policy=FixedRounds(3),
                 round_us=1.0,
                 code=SurfaceCodeModel(d=3),
                 scheme=SlidingWindowScheme(),
                 decoder=PresetLatencyDecoder(2.0),
                 links=fixed_latency_link_config(),
                 make_controller=_zero_link_controller,
                 make_metrics=lambda _engine, _cluster, chip, _factory: [
            ConditionalReactionTime(chip)
        ],
                 feedback_boundary_mode="measurement_closed",
             ), verbose=False)

    rows = _reaction_rows(result)
    assert len(rows) == 1
    assert rows[0]["wait_rounds"] == 2.0
    assert result.cluster.memory_rounds[0] < 3


def _live_stream_pair():
    """A stream-backed T pair whose second segment waits on the first."""
    stream = Operation(0, "live-stream", (0,), clifford=True, patches=(0,))
    first = Operation(
        1,
        "A:T(q0)",
        (0,),
        clifford=False,
        consumes_magic_state=False,
        patches=(0,),
        stream_id=stream.id,
        has_successor=True,
    )
    second = Operation(
        2,
        "B:T(q0)",
        (0,),
        clifford=False,
        consumes_magic_state=False,
        patches=(0,),
        predecessors=(first.id,),
        stream_id=stream.id,
        blocked_by=first.id,
    )
    return stream, [first, second]


def _run_live_stream_pair(mode: str):
    """Run the small timing-only live-stream pair under one boundary mode."""
    stream, operations = _live_stream_pair()
    rounds = {
        operations[0].id: 2,
        operations[1].id: 2,
    }
    return simulate(RunSpec(
               ops=operations,
               dynamic_streams=[stream],
               idle_policy=from_mode("extend_stream"),
               device=TimingOnlyDevice(),
               code=SurfaceCodeModel(d=3, commit_rounds_override=2, buffer_rounds_override=1),
               rounds_policy=PerOpRounds(rounds),
               scheme=SlidingWindowScheme(),
               decoder=PresetLatencyDecoder(0.0),
               num_units=1,
               round_us=1.0,
               links=fixed_latency_link_config(),
               make_controller=_zero_link_controller,
               feedback_boundary_mode=mode,
           ), verbose=False)


def test_measurement_closed_boundary_closes_live_stream_without_idle_buffer():
    """Live stream boundaries close at the body measurement in DecLat mode."""
    trailing = _run_live_stream_pair("trailing_buffer")
    closed = _run_live_stream_pair("measurement_closed")

    trailing_offsets = trailing.result.stream_offsets()
    closed_offsets = closed.result.stream_offsets()

    assert (
        trailing_offsets[2] > trailing_offsets[1] + 2
    )
    assert (
        closed_offsets[2] == closed_offsets[1] + 2
    )
    assert trailing.cluster.rounds_arrived[0] > closed.cluster.rounds_arrived[0]
    assert len(closed.cluster.committed_windows) == closed.cluster.total_windows


def test_real_syndrome_measurement_closed_finite_operation_uses_stim_circuit():
    """Finite Stim operations work when the operation measurement closes the boundary."""
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")

    from decsim.adapters.stim_device import StimDevice
    from decsim.mwpm_decoder import PyMatchingDecoder
    from decsim.stimcircuits import NoiseModel

    code = SurfaceCodeModel(d=3)
    circuit = NoiseModel.circuit_level(0.003).circuit(
        distance=code.distance,
        rounds=code.commit_rounds(),
    )

    operations = _feedback_chain()
    for operation in operations:
        operation.circuit = circuit

    result = simulate(RunSpec(
                 ops=operations,
                 device=StimDevice(),
                 code=code,
                 scheme=SlidingWindowScheme(),
                 decoder=PyMatchingDecoder(PresetLatencyDecoder(0.0)),
                 num_units=1,
                 rounds_policy=FixedRounds(code.commit_rounds()),
                 round_us=1.0,
                 links=fixed_latency_link_config(),
                 make_controller=_zero_link_controller,
                 feedback_boundary_mode="measurement_closed",
                 seed=19,
             ), verbose=False)

    assert result.cluster.rounds_arrived[operations[0].id] == code.commit_rounds()
    assert operations[0].id in result.cluster.op_results
    assert operations[1].id in result.chip.decode_release_time


def test_real_syndrome_measurement_closed_internal_stream_boundary_rejected():
    """A continuous Stim stream is not silently reused as an internal closed boundary."""
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")

    from decsim.adapters.stim_device import StimDevice
    from decsim.mwpm_decoder import PyMatchingDecoder
    from decsim.stimcircuits import NoiseModel

    code = SurfaceCodeModel(d=3, commit_rounds_override=2, buffer_rounds_override=1)
    circuit = NoiseModel.circuit_level(0.003).circuit(distance=code.distance, rounds=4)
    stream, operations = _live_stream_pair()
    stream.circuit = circuit
    for operation in operations:
        operation.circuit = circuit

    rounds = {
        stream.id: 4,
        operations[0].id: 2,
        operations[1].id: 2,
    }
    with pytest.raises(RuntimeError, match="destructive boundary"):
        simulate(RunSpec(
            ops=operations,
            dynamic_streams=[stream],
            idle_policy=from_mode("extend_stream"),
            device=StimDevice(),
            code=code,
            rounds_policy=PerOpRounds(rounds),
            scheme=SlidingWindowScheme(),
            decoder=PyMatchingDecoder(PresetLatencyDecoder(0.0)),
            num_units=1,
            round_us=1.0,
            links=fixed_latency_link_config(),
            make_controller=_zero_link_controller,
            feedback_boundary_mode="measurement_closed",
            seed=19,
        ), verbose=False)

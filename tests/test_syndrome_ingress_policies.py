"""Classical-buffer policy tests grounded in new primary sources."""
import pytest

from decsim.engine import Engine
from decsim.links import LinkModelConfig
from decsim.message import SyndromePayload, WINDOW_INPUT_ROUTE
from decsim.syndrome_ingress import (
    IngressOverflowPolicy,
    ReassemblyQueueAdmission,
    SyndromeReassemblyTimeout,
    SyndromeIngress,
    SyndromeIngressPolicy,
)


class _WindowSink:
    def __init__(self): self.packets = []
    def accept_window_input(self, packet): self.packets.append(packet); return True


class _MemorySink:
    def accept_feedback_memory_round(self, operation_id): pass


def _ingress(engine, *, capacity=4, policy=SyndromeIngressPolicy()):
    return SyndromeIngress(
        engine, links=LinkModelConfig.logical_reference_profile().resolve(),
        ingress_context_capacity=capacity, window_input_receiver=_WindowSink(),
        feedback_memory_receiver=_MemorySink(), log_syndromes=False,
        policy=policy,
    )


def _payload(round_index, *, count=1, index=0):
    return SyndromePayload(0, 0, round_index, n_fragments=count,
                           fragment_index=index)


def test_completion_admission_removes_reassembly_induced_hol_blocking():
    engine = Engine()
    ingress = _ingress(engine, policy=SyndromeIngressPolicy(
        queue_admission=ReassemblyQueueAdmission.ON_COMPLETION))
    sink = ingress.window_input_receiver
    ingress.relay_syndrome(_payload(1, count=2), WINDOW_INPUT_ROUTE)
    ingress.relay_syndrome(_payload(2), WINDOW_INPUT_ROUTE)
    engine.run()
    assert [packet.round_index for packet in sink.packets] == [2]
    assert len(ingress.ingress_snapshot().partial_identities) == 1


def test_allocation_admission_preserves_explicit_hol_baseline():
    engine = Engine()
    ingress = _ingress(engine, policy=SyndromeIngressPolicy(
        queue_admission=ReassemblyQueueAdmission.ON_ALLOCATION))
    sink = ingress.window_input_receiver
    ingress.relay_syndrome(_payload(1, count=2), WINDOW_INPUT_ROUTE)
    ingress.relay_syndrome(_payload(2), WINDOW_INPUT_ROUTE)
    engine.run()
    assert sink.packets == []


def test_drop_policy_is_observable_and_does_not_overwrite_live_context():
    engine = Engine()
    ingress = _ingress(engine, capacity=1, policy=SyndromeIngressPolicy(
        overflow=IngressOverflowPolicy.DROP_ROUND))
    ingress.relay_syndrome(_payload(1, count=2), WINDOW_INPUT_ROUTE)
    ingress.relay_syndrome(_payload(2), WINDOW_INPUT_ROUTE)
    engine.run()
    assert ingress.ingress_drops == 1
    assert len(ingress.ingress_snapshot().partial_identities) == 1


def test_configured_reassembly_timeout_fails_loudly_and_frees_slot():
    engine = Engine()
    ingress = _ingress(engine, capacity=1, policy=SyndromeIngressPolicy(
        reassembly_timeout_ticks=3))
    ingress.relay_syndrome(_payload(1, count=2), WINDOW_INPUT_ROUTE)
    with pytest.raises(SyndromeReassemblyTimeout, match="received 1/2"):
        engine.run()
    assert ingress.ingress_snapshot().ingress_contexts == 0
    assert ingress.reassembly_timeouts == 1



def test_timeout_does_not_expire_a_complete_packet_during_packing():
    engine = Engine()
    ingress = _ingress(engine, capacity=1, policy=SyndromeIngressPolicy(
        reassembly_timeout_ticks=3))
    ingress.t_pack = 5
    ingress.relay_syndrome(_payload(1, count=2, index=0), WINDOW_INPUT_ROUTE)
    ingress.relay_syndrome(_payload(1, count=2, index=1), WINDOW_INPUT_ROUTE)
    engine.run()
    assert ingress.reassembly_timeouts == 0
    assert [p.round_index for p in ingress.window_input_receiver.packets] == [1]


def test_run_spec_exposes_policy_without_making_ingress_global_state():
    from decsim import RunSpec
    completed = RunSpec(
        ops=[],
        syndrome_ingress_policy=SyndromeIngressPolicy(
            queue_admission=ReassemblyQueueAdmission.ON_COMPLETION),
    ).build()
    assert completed.syndrome_ingress.policy.queue_admission is ReassemblyQueueAdmission.ON_COMPLETION
    assert not hasattr(completed.window_manager, "ingress")

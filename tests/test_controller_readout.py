"""Controller-owned QPU readout boundary and timing."""

import pytest

from decsim import RunSpec, TimingConfig
from decsim.engine import Engine
from decsim.message import (
    Operation, QPUReadout, SyndromePacketRoute, SyndromePayload, WINDOW_INPUT_ROUTE,
)
from decsim.planner import FixedRounds
from decsim.decoders import PerRoundDecoder
from decsim.links import LinkModelConfig, LinkPath
from dataclasses import replace


class _Ingress:
    def __init__(self, engine):
        self.engine = engine
        self.received = []

    def relay_qpu_readout(self, payload, route, *, processing_ticks):
        self.engine.schedule(
            processing_ticks,
            lambda: self.received.append((self.engine.now, payload, route)),
        )


def _controller(engine, ingress, delay):
    controller = object.__new__(__import__("decsim.controller", fromlist=["Controller"]).Controller)
    controller.engine = engine
    controller.syndrome_ingress = ingress
    controller.binary_availability_ticks = delay
    return controller


def test_controller_converts_readout_to_immutable_binary_after_configured_cost():
    engine = Engine(verbose=False)
    ingress = _Ingress(engine)
    controller = _controller(engine, ingress, 7)
    source = [True, 0, 1]
    readout = QPUReadout(3, "p", 2, bits=source, size_bits=3)

    controller.accept_qpu_readout(readout, WINDOW_INPUT_ROUTE)
    source[0] = False
    assert ingress.received == []
    engine.run()

    tick, payload, route = ingress.received[0]
    assert tick == 7
    assert type(payload) is SyndromePayload
    assert payload.bits == (1, 0, 1)
    assert route is WINDOW_INPUT_ROUTE


def test_controller_rejects_nonbinary_qpu_values_before_ingress():
    engine = Engine(verbose=False)
    controller = _controller(engine, _Ingress(engine), 0)
    with pytest.raises(TypeError, match="binary"):
        controller.accept_qpu_readout(
            QPUReadout(1, 0, 1, bits=[0, 2]), WINDOW_INPUT_ROUTE)


def test_discrimination_cost_shifts_qc_without_duplicating_qc_or_cwd():
    def run(delay_us):
        return RunSpec(
            ops=[Operation(0, "memory", (0,))],
            rounds_policy=FixedRounds(3),
            decoder=PerRoundDecoder(0.0),
            links=replace(
                LinkModelConfig.logical_reference_profile(),
                profile_name="qc-excludes-controller-processing",
                qc_excludes_controller_processing=True,
            ),
            timing=TimingConfig(t_binary_availability_us=delay_us),
        ).build(False)

    baseline = run(0.0)
    delayed = run(0.4)
    base_qc = [record for record in baseline.result.link_traffic["transfers"]
               if record["path"] == LinkPath.QC.value]
    delayed_qc = [record for record in delayed.result.link_traffic["transfers"]
                  if record["path"] == LinkPath.QC.value]
    base_cwd = [record for record in baseline.result.link_traffic["transfers"]
                if record["path"] == LinkPath.CWD.value]
    delayed_cwd = [record for record in delayed.result.link_traffic["transfers"]
                   if record["path"] == LinkPath.CWD.value]

    assert len(base_qc) == len(delayed_qc) == 3
    assert len(base_cwd) == len(delayed_cwd) == 1
    assert [record["send_ticks"] for record in delayed_qc] == [
        record["send_ticks"] for record in base_qc]
    assert delayed.result.fully_done_ticks == baseline.result.fully_done_ticks + 400_000
    assert [record["payload_bits"] for record in delayed_qc] == [
        record["payload_bits"] for record in base_qc]
    assert [record["attribution"] for record in delayed_qc] == [
        record["attribution"] for record in base_qc]


def test_ingress_rejects_preconversion_readout():
    completed = RunSpec(ops=[]).build(False)
    with pytest.raises(TypeError, match="SyndromePayload"):
        completed.syndrome_ingress.relay_syndrome(
            QPUReadout(0, 0, 1), WINDOW_INPUT_ROUTE)


def test_reference_qc_profile_rejects_separate_readout_cost():
    with pytest.raises(ValueError, match="QC latency excludes"):
        RunSpec(
            ops=[Operation(0, "memory", (0,))],
            rounds_policy=FixedRounds(3),
            decoder=PerRoundDecoder(0.0),
            timing=TimingConfig(t_binary_availability_us=0.4),
        ).build(False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"operation_id": object()},
        {"patch_id": object()},
        {"round_index": 0},
        {"size_bits": -1},
    ],
)
def test_qpu_readout_rejects_invalid_wire_metadata(kwargs):
    fields = {"operation_id": 1, "patch_id": 0, "round_index": 1}
    fields.update(kwargs)
    with pytest.raises(TypeError):
        QPUReadout(**fields)

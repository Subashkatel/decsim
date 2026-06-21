"""Continuous-stream support in StimDevice (3b/5-real foundation).

A continuous patch is one circuit sampled ONCE per shot; its operation segments each serve
their local rounds from the shared sample at the right GLOBAL round (stream_offset + r), so the
whole stream is one detection record with one observable. Standalone ops are unchanged.

Requires stim."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")

from decsim.message import Operation
from decsim.adapters.stim_device import StimDevice
from decsim.stimcircuits import NoiseModel

D = 3
R1, R2 = 12, 12
R = R1 + R2


def _single_payload(device, operation, round_index):
    payloads = device.round_payloads(operation, round_index)
    assert len(payloads) == 1
    return payloads[0]


def _continuous_circuit():
    return NoiseModel.circuit_level(0.01).circuit(distance=D, rounds=R)


def test_stream_segments_cover_the_whole_record_once():
    circ = _continuous_circuit()
    segA = Operation(0, "segA", (0,), circuit=circ, stream_id="s", stream_offset=0)
    segB = Operation(1, "segB", (0,), circuit=circ, stream_id="s", stream_offset=R1)
    dev = StimDevice(seed=4)
    dev.begin_operation(segA)
    dev.begin_operation(segB)                      # must NOT re-sample the stream
    # concatenate every segment's per-round bits in global order == the full detection record
    bits = []
    for r in range(1, R1 + 1):
        bits += list(_single_payload(dev, segA, r).bits)
    for r in range(1, R2 + 1):
        bits += list(_single_payload(dev, segB, r).bits)
    full = dev._dets["s"]
    assert len(bits) == len(full)
    assert np.array_equal(np.asarray(bits, np.uint8), np.asarray(full, np.uint8))
    # one observable truth shared across the stream; mirrored onto each segment id
    assert dev._truth[0] is dev._truth["s"] and dev._truth[1] is dev._truth["s"]


def test_stream_segment_serves_its_global_rounds():
    circ = _continuous_circuit()
    segA = Operation(0, "segA", (0,), circuit=circ, stream_id="s", stream_offset=0)
    segB = Operation(1, "segB", (0,), circuit=circ, stream_id="s", stream_offset=R1)
    dev = StimDevice(seed=7)
    dev.begin_operation(segA)
    dev.begin_operation(segB)
    # segB local round r serves global round R1+r: compare against a single-circuit reference
    ref = StimDevice(seed=7)
    whole = Operation(9, "whole", (0,), circuit=circ)     # standalone: local==global
    ref.begin_operation(whole)
    for r in range(1, R2 + 1):
        got = np.asarray(_single_payload(dev, segB, r).bits, np.uint8)
        want = np.asarray(_single_payload(ref, whole, R1 + r).bits, np.uint8)
        assert np.array_equal(got, want), r


def test_standalone_op_unchanged():
    circ = NoiseModel.circuit_level(0.01).circuit(distance=D, rounds=R1)
    op = Operation(3, "mem", (0,), circuit=circ)          # no stream_id
    dev = StimDevice(seed=2)
    dev.begin_operation(op)
    total = sum(len(_single_payload(dev, op, r).bits) for r in range(1, R1 + 1))
    assert total == circ.num_detectors                   # every detector emitted exactly once

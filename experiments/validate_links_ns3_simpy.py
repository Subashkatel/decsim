"""Gate 6: decsim's link timing against ns-3's point-to-point link and an
independent SimPy store-and-forward model.

What a decsim link does (links.py Link.reserve): a payload sent at time t on
a channel waits until the channel's serializer is free, is serialized for
bits / bandwidth, then propagates for the channel's latency; sends on one
channel are served first come first served; two paths bound to the same
LinkConfig share that one serializer; a channel with no capacity card has no
serializer (queue wait 0, serialization 0). Every reservation is counted per
path and per channel and the report refuses counts that do not reconcile.

The classical reference is ns-3's point-to-point link (pinned source under
tmp/references/code/ns-3): PointToPointNetDevice::TransmitStart sets the
device BUSY and schedules TransmitComplete after
DataRate::CalculateBitsTxTime(bits) = bits / bps
(point-to-point-net-device.cc:226-254, data-rate.cc:226-231);
TransmitComplete dequeues the next packet from the device's drop-tail FIFO
queue and starts it (net-device.cc:256-286); the channel delivers each packet
to the far device at txTime + m_delay (point-to-point-channel.cc:78-97).
That is: serialize one packet at a time in FIFO order at the data rate, then
one fixed propagation delay. Ethernet's inter-frame gap is a constant added
to txTime; decsim has none, so it is set to zero here.

The second reference is a SimPy model written from that rule (SimPy pinned at
tmp/references/code/simpy): a channel is a Resource of capacity one; a send
requests it at its send time, holds it for bits / bandwidth, releases it, then
waits the propagation delay and is delivered.

Cases: (1) one channel, unlimited bandwidth: pure propagation, no queueing;
(2) one channel with finite bandwidth and bursty sends so packets queue;
(3) two semantic paths sharing one channel: their sends interleave in one
FIFO; (4) many channels of a full reference fabric with a random but fixed
send schedule. Compared per transfer: serializer start, serializer end and
delivery tick, exactly (integer ticks; the references compute in exact
rationals and are rounded to ticks the same way decsim's us() rounds).
(5) A whole closed loop on real Stim data with the bandwidth-limited card:
every transfer in the ledger carries the real payload (the round's detector
bits, a window's summed bits), its timing arithmetic holds, and its queue
wait equals ns-3's FIFO replayed over its channel.

Usage: python -m experiments.validate_links_ns3_simpy
"""

from __future__ import annotations

import random
import sys
from fractions import Fraction
from pathlib import Path

from decsim.config import TICKS_PER_US
from decsim.links.link_traffic_report import traffic_json_value
from decsim.links.links import (LinkCapacityConfig, LinkConfig, LinkEdgeConfig,
                                LinkModelConfig, LinkPath, LinkQuantityBasis,
                                TrafficAttribution)

sys.path.insert(0, str(Path("tmp/references/code/simpy/src")))
import simpy  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results" / "validation" / "links_ns3_simpy.md"


def ticks_of(us_value: Fraction) -> int:
    """decsim's us(): round(microseconds * 1e6) to an integer tick."""
    return int(round(us_value * TICKS_PER_US))


# ---- reference 1: ns-3 point-to-point, transcribed -------------------------

def ns3_point_to_point(sends, *, bits_per_us, propagation_ticks):
    """(send_tick, bits) per packet, in send order, on one device/channel.
    Returns (tx_start, tx_complete, receive) ticks per packet.

    TransmitStart: if READY, transmit now; else the packet waits in the
    device queue and TransmitComplete starts it when the wire frees. txTime =
    bits / bps (CalculateBitsTxTime). Receive at txTime + delay
    (PointToPointChannel::TransmitStart). Inter-frame gap zero."""
    tx_ready_at = 0
    out = []
    for send_tick, bits in sends:
        tx_start = send_tick if tx_ready_at <= send_tick else tx_ready_at
        tx_time = 0 if bits_per_us is None else ticks_of(Fraction(bits) / Fraction(bits_per_us))
        tx_complete = tx_start + tx_time
        tx_ready_at = tx_complete
        out.append((tx_start, tx_complete, tx_complete + propagation_ticks))
    return out


# ---- reference 2: SimPy store-and-forward --------------------------------

def simpy_store_and_forward(sends, *, bits_per_us, propagation_ticks):
    env = simpy.Environment()
    wire = simpy.Resource(env, capacity=1)
    out = [None] * len(sends)

    def send(index, send_tick, bits):
        yield env.timeout(send_tick)
        with wire.request() as grant:           # first come first served
            yield grant
            tx_start = env.now
            tx_time = 0 if bits_per_us is None else ticks_of(Fraction(bits) / Fraction(bits_per_us))
            yield env.timeout(tx_time)
            tx_complete = env.now
        yield env.timeout(propagation_ticks)
        out[index] = (tx_start, tx_complete, env.now)

    for index, (send_tick, bits) in enumerate(sends):
        env.process(send(index, send_tick, bits))
    env.run()
    return out


# ---- decsim ---------------------------------------------------------------

def _edge(channel, actual="payload bits"):
    return LinkEdgeConfig(channel, None, actual)


def _channel(propagation_us, bits_per_us, source):
    capacity = (None if bits_per_us is None else LinkCapacityConfig(
        bits_per_us, LinkQuantityBasis.DIRECT_AGGREGATE, None, source))
    return LinkConfig(ticks_of(Fraction(propagation_us)), capacity, source)


def _fabric(channel_by_path):
    """A card wiring the nine required paths; paths not in the map get their
    own unbounded zero-latency channel."""
    edges = {}
    for path in LinkPath:
        if path is LinkPath.C2B:
            continue
        channel = channel_by_path.get(path) or _channel(0, None, "unused")
        edges[path.value] = _edge(channel)
    return LinkModelConfig(**edges, profile_name="gate6")


def _round_attribution(path, op, round_index):
    return TrafficAttribution(op, (), None, round_index, round_index)


def decsim_sends(model, path, sends):
    """Reserve each (send_tick, bits) on `path`; return the same triple as the references."""
    out = []
    for round_index, (send_tick, bits) in enumerate(sends, start=1):
        r = model.reserve(path, payload_bits=bits, now_ticks=send_tick,
                          attribution=_round_attribution(path, 1, round_index))
        out.append((r.serializer_start_ticks, r.serializer_end_ticks,
                    r.send_ticks + r.total_delay_ticks))
    return out


# ---- cases ----------------------------------------------------------------

def case_propagation_only():
    sends = [(0, 24), (100, 24), (150, 96), (150, 8)]
    channel = _channel(Fraction(1, 2), None, "0.5 us, unbounded")
    model = _fabric({LinkPath.QC: channel}).resolve()
    ours = decsim_sends(model, LinkPath.QC, sends)
    ns3 = ns3_point_to_point(sends, bits_per_us=None, propagation_ticks=500_000)
    sp = simpy_store_and_forward(sends, bits_per_us=None, propagation_ticks=500_000)
    return "one channel, unbounded bandwidth, 0.5 us propagation", sends, ours, ns3, sp


def case_bursty_fifo():
    # 24 bits per send at 24 bits/us: 1 us of serialization each; three arrive
    # in the same microsecond, so they queue behind one another.
    sends = [(0, 24), (100_000, 24), (200_000, 24), (5_000_000, 48), (5_100_000, 12)]
    channel = _channel(Fraction(3, 20), 24, "24 bits/us, 0.15 us")
    model = _fabric({LinkPath.QC: channel}).resolve()
    ours = decsim_sends(model, LinkPath.QC, sends)
    ns3 = ns3_point_to_point(sends, bits_per_us=24, propagation_ticks=150_000)
    sp = simpy_store_and_forward(sends, bits_per_us=24, propagation_ticks=150_000)
    return "one channel, 24 bits/us, bursty sends queue FIFO", sends, ours, ns3, sp


def case_shared_channel():
    # WDO and DO ride one physical channel: their sends interleave in one FIFO.
    channel = _channel(1, 100, "shared 100 bits/us, 1 us")
    model = _fabric({LinkPath.WDO: channel, LinkPath.DO: channel}).resolve()
    interleaved = [(0, "wdo", 500), (1_000_000, "do", 500), (2_000_000, "wdo", 200),
                   (2_100_000, "do", 900), (2_200_000, "wdo", 100)]
    from decsim.links.links import RequestTransferRelation
    from decsim.message import DecoderRequestKey, DecoderTier
    ours = []
    for k, (send_tick, path_name, bits) in enumerate(interleaved):
        path = LinkPath(path_name)
        tier = DecoderTier.WEAK if path is LinkPath.WDO else DecoderTier.STRONG
        key = DecoderRequestKey(1, k, tier, k)
        attribution = TrafficAttribution(1, (), k, 1, 3, RequestTransferRelation(key))
        r = model.reserve(path, payload_bits=bits, now_ticks=send_tick, attribution=attribution)
        ours.append((r.serializer_start_ticks, r.serializer_end_ticks, r.send_ticks + r.total_delay_ticks))
    sends = [(t, b) for t, _, b in interleaved]
    ns3 = ns3_point_to_point(sends, bits_per_us=100, propagation_ticks=1_000_000)
    sp = simpy_store_and_forward(sends, bits_per_us=100, propagation_ticks=1_000_000)
    report = traffic_json_value(model.snapshot())
    counters = {edge["path"]: edge["counters"]["transfer_count"] for edge in report["semantic_edges"]}
    assert counters["wdo"] == 3 and counters["do"] == 2, counters
    assert all(row["reconciles"] for row in report["reconciliation"])
    return "two paths (WDO, DO) on one 100 bits/us channel, one FIFO", sends, ours, ns3, sp


def case_random_fabric():
    rng = random.Random(7)
    channels = {}
    for path in LinkPath:
        if path is LinkPath.C2B:
            continue
        channels[path] = _channel(Fraction(rng.randint(1, 30), 10), rng.choice([None, 16, 64, 256]),
                                  f"{path.value} card")
    model = _fabric(channels).resolve()
    per_path = {}
    for path in channels:
        sends = []
        t = 0
        for _ in range(12):
            t += rng.randint(0, 800_000)
            sends.append((t, rng.randint(1, 512)))
        per_path[path] = sends
    ok = True
    rows = []
    for path, sends in per_path.items():
        channel = channels[path]
        bits_per_us = None if channel.capacity is None else channel.capacity.aggregate_bits_per_us
        if path in (LinkPath.QC, LinkPath.CWD):
            ours = decsim_sends(model, path, sends)
        else:
            continue          # window paths need request relations; QC and CWD suffice for the fabric case
        ns3 = ns3_point_to_point(sends, bits_per_us=bits_per_us, propagation_ticks=channel.propagation_latency_ticks)
        sp = simpy_store_and_forward(sends, bits_per_us=bits_per_us, propagation_ticks=channel.propagation_latency_ticks)
        agree = ours == ns3 == sp
        ok = ok and agree
        rows.append((path.value, len(sends), agree))
    return "reference fabric, random fixed schedule on QC and CWD", rows, ok


def case_real_run_payloads():
    """A whole closed loop (Stim rotated memory d=3, 12 rounds, the bandwidth
    limited card at a quarter of its rates so every channel queues) and, from
    its traffic ledger, three checks per transfer: (a) the payload the link
    priced is the real one (QC: the detector bits of that round; CWD per
    window: the sum of its rounds' bits; C2B: the packed round's bits);
    (b) serialization = round(bits / bandwidth) and delivery = send + wait +
    serialization + propagation; (c) the queue waits equal ns-3's FIFO rule
    replayed over that channel's transfers in send order."""
    import stim
    from decsim.links.link_profiles import bandwidth_limited_profile, with_controller_to_buffer_edge
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.message import Operation
    from decsim.qpu.round_policies import FixedRounds
    from decsim.qpu.stim_device import StimDevice
    from decsim.run_spec import RunSpec
    rounds = 12
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=rounds, distance=3,
                                     after_clifford_depolarization=0.003)
    op = Operation(id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit)
    links = with_controller_to_buffer_edge(bandwidth_limited_profile(capacity_scale=0.25),
                                           latency_us=0.1, aggregate_bits_per_us=8.0, source="gate6 c2b")
    done = RunSpec(ops=[op], d=3, rounds_policy=FixedRounds(rounds), device=StimDevice(),
                   decoder=PresetLatencyDecoder(1.0), num_units=1, links=links, seed=3).build()
    report = done.result.link_traffic
    transfers = report["transfers"]
    # detectors per emitted round: the circuit's own chronology (the final
    # data-qubit detectors belong to the last round; Gate 5 checked the bits)
    from decsim.detector_error_model.detector_chronology import resolve_detector_rounds
    bits_of_round = {}
    for round_index in resolve_detector_rounds(circuit, None, rounds).values():
        bits_of_round[round_index] = bits_of_round.get(round_index, 0) + 1
    windows = done.window_manager.windows
    checks = {"payload equals the real bits": 0, "timing arithmetic": 0, "FIFO waits equal ns-3": 0}
    problems = []
    for row in transfers:
        bits = row["payload_bits"]
        if row["path"] == "qc":
            round_index = row["attribution"]["round_lo"]
            expected = bits_of_round[round_index]
            if bits != expected:
                problems.append(("qc bits", round_index, bits, expected))
            checks["payload equals the real bits"] += 1
        elif row["path"] == "cwd" and row["attribution"]["window_id"] is not None:
            window = windows[(1, row["attribution"]["window_id"])]
            expected = sum(bits_of_round[r]
                           for r in range(window.start_round, min(window.buffer_hi, rounds) + 1))
            if bits != expected:
                problems.append(("cwd bits", row["attribution"]["window_id"], bits, expected))
            checks["payload equals the real bits"] += 1
        elif row["path"] == "c2b":
            round_index = row["attribution"]["round_lo"]
            expected = bits_of_round[round_index]
            if bits != expected:
                problems.append(("c2b bits", round_index, bits, expected))
            checks["payload equals the real bits"] += 1
        arithmetic = (row["delivery_ticks"] == row["send_ticks"] + row["queue_wait_ticks"]
                      + row["serialization_ticks"] + row["propagation_ticks"])
        if not arithmetic:
            problems.append(("arithmetic", row["path"], row["send_ticks"]))
        checks["timing arithmetic"] += 1
    # (c) per physical channel: replay ns-3 over the transfers in send order
    channel_of = {edge["path"]: edge["physical_alias"] for edge in report["semantic_edges"]}
    fabric = links.resolve().snapshot()
    rate_of_alias = {ch.alias: (None if ch.config.capacity is None else ch.config.capacity.aggregate_bits_per_us,
                                ch.config.propagation_latency_ticks) for ch in fabric.channels}
    by_channel = {}
    for row in transfers:
        by_channel.setdefault(channel_of[row["path"]], []).append(row)
    for alias, rows in by_channel.items():
        rows.sort(key=lambda r: (r["send_ticks"], r["physical_sequence"]))
        rate, propagation = rate_of_alias[alias]
        sends = [(r["send_ticks"], r["payload_bits"] if r["payload_bits"] is not None else 0) for r in rows]
        expected = ns3_point_to_point(sends, bits_per_us=rate, propagation_ticks=propagation)
        for r, (tx_start, tx_complete, receive) in zip(rows, expected):
            ours = (r["serializer_start_ticks"], r["serializer_end_ticks"], r["delivery_ticks"])
            if ours != (tx_start, tx_complete, receive):
                problems.append(("fifo", alias, r["path"], ours, (tx_start, tx_complete, receive)))
            checks["FIFO waits equal ns-3"] += 1
    return checks, problems, len(transfers)


def main(argv) -> None:
    lines = ["# Gate 6: decsim links vs ns-3 point-to-point and a SimPy store-and-forward model", "",
             "Per transfer: (serializer start, serializer end, delivery) in ticks (1 us = 1,000,000). "
             "ns-3 rule: FIFO device queue, txTime = bits / bps, receive at txTime + delay "
             "(point-to-point-net-device.cc:226-286, data-rate.cc:226-231, point-to-point-channel.cc:78-97). "
             "SimPy: one Resource per channel, hold bits / bandwidth, then the propagation delay.", ""]
    all_ok = True
    for case in (case_propagation_only, case_bursty_fifo, case_shared_channel):
        title, sends, ours, ns3, sp = case()
        agree = ours == ns3 == sp
        all_ok = all_ok and agree
        lines += [f"## {title}", "", "| send tick | bits | decsim | ns-3 | SimPy | equal |", "|---|---|---|---|---|---|"]
        for (t, b), o, n, s in zip(sends, ours, ns3, sp):
            lines.append(f"| {t} | {b} | {o} | {n} | {s} | {'yes' if o == n == s else 'NO'} |")
        lines.append("")
    title, rows, ok = case_random_fabric()
    all_ok = all_ok and ok
    lines += [f"## {title}", "", "| path | sends | decsim = ns-3 = SimPy |", "|---|---|---|"]
    lines += [f"| {p} | {n} | {'yes' if a else 'NO'} |" for p, n, a in rows]
    checks, problems, transfer_count = case_real_run_payloads()
    all_ok = all_ok and not problems
    lines += ["", "## a real closed loop: Stim rotated memory d=3, 12 rounds, bandwidth-limited card at 0.25, priced C2B",
              "", f"{transfer_count} transfers in the ledger.", "", "| check | transfers checked | problems |", "|---|---|---|"]
    kind_of = {"payload equals the real bits": ("qc bits", "cwd bits", "c2b bits"),
               "timing arithmetic": ("arithmetic",), "FIFO waits equal ns-3": ("fifo",)}
    lines += [f"| {name} | {count} | {sum(1 for p in problems if p[0] in kind_of[name])} |"
              for name, count in checks.items()]
    if problems:
        lines += ["", "Problems: " + "; ".join(map(str, problems[:10]))]
    lines += ["", f"Verdict: {'PASS' if all_ok else 'FAIL'}. decsim's Link is ns-3's point-to-point link "
              "with an unbounded drop-tail queue and no inter-frame gap: one serializer per channel, "
              "first come first served, bits / bandwidth, then a fixed propagation delay; two semantic "
              "paths on one channel share that serializer, and the per-path counters reconcile with the "
              "channel's. What decsim adds on top is the attribution ledger (whose transfer this was, "
              "on which path, for which window or round), which ns-3 has no counterpart for."]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)

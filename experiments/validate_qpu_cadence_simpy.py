"""Gate 4: decsim's QPU cycle clock against an independent SimPy model of the
reference rule (Q-064).

The rule, from Google 2207.06431 / 2408.13687 (every measure qubit is read out
every cycle), Skoric 2209.08552 App. D and SWIPER device_manager.py: one global
QEC cycle; every live patch yields one syndrome round per cycle whether or not
an operation uses it; an operation starts on the first cycle boundary at or
after it becomes ready and holds its patches for whole cycles; an operation
blocked on a decode becomes ready when that decode completes.

The model here is written from that rule on SimPy (pinned at
tmp/references/code/simpy, f438164), not from decsim, and covers cases SWIPER's
integer-round device cannot: decode latencies that are not whole cycles, a
1.1 us Willow cycle, and two patches with one waiting. Links are zero and each
operation is one window, so decode completes exactly one decoder latency after
the operation's last round. Compared per case: every operation's start, body
done and decode release time, and the number of idle rounds emitted.

Usage: python -m experiments.validate_qpu_cadence_simpy
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import stim

from decsim.config import TICKS_PER_US, TimingConfig
from decsim.decoder_engine import DecoderEngine, DecoderTiming
from decsim.decoders import PresetLatencyDecoder
from decsim.message import Operation
from decsim.mwpm_decoder.decoder import PyMatchingDecoder
from decsim.pauli_frame import PauliFrameConfig
from decsim.program.round_policies import FixedRounds
from decsim.run_spec import RunSpec
from experiments.validate_timing_swiper import zero_latency_links

sys.path.insert(0, str(Path("tmp/references/code/simpy/src")))
import simpy  # noqa: E402

REPORT = Path("experiments/results/validation/qpu_cadence_simpy.md")
NS = 1000                                     # SimPy runs in integer nanoseconds

# (name, cycle_us, decode_us, ops); an op is (id, patch, predecessors, blocked_by)
CASES = [
    ("merge then blocked successor, whole-cycle decode", 1.0, 14.0,
     [(1, 0, (), None), (2, 0, (1,), 1)]),
    ("decode not a whole number of cycles", 1.0, 4.3,
     [(1, 0, (), None), (2, 0, (1,), 1)]),
    ("Willow 1.1 us cycle, 14 us decode", 1.1, 14.0,
     [(1, 0, (), None), (2, 0, (1,), 1)]),
    ("two patches, one waits on a decode, the other does not", 1.0, 6.5,
     [(1, 0, (), None), (2, 1, (), None), (3, 0, (1,), 1), (4, 1, (2,), None)]),
    ("chain of three with two waits", 1.0, 2.2,
     [(1, 0, (), None), (2, 0, (1,), 1), (3, 0, (2,), 2)]),
]
D = 3                                          # rounds per operation = one window


def simpy_model(cycle_us: float, decode_us: float, ops: list) -> dict:
    """The reference rule on SimPy: returns start/done/release per op and idle rounds."""
    env = simpy.Environment()
    cycle, decode = round(cycle_us * NS), round(decode_us * NS)
    start, done, release, idle = {}, {}, {}, {"count": 0}
    running, patch_idle = {}, {}                  # op -> rounds emitted; patch -> op id

    def ready(op_id, patch, preds, blocked_by):
        return (op_id not in start and all(p in done for p in preds)
                and (blocked_by is None or op_id in release))

    def decoder(op_id):
        yield env.timeout(decode)
        for blocked_id, _p, _preds, blocked_by in ops:      # keyed by the released op, as decsim records it
            if blocked_by == op_id:
                release[blocked_id] = env.now

    def qpu():
        while True:
            now = env.now
            if now > 0:
                for patch in list(patch_idle):    # patches idle during the cycle just ended
                    idle["count"] += 1
                for op_id in list(running):       # rounds of the cycle just ended
                    running[op_id] += 1
                    if running[op_id] == D:
                        del running[op_id]
                        done[op_id] = now
                        env.process(decoder(op_id))
                        patch_idle[next(p for i, p, _, _ in ops if i == op_id)] = op_id
            for op_id, patch, preds, blocked_by in ops:      # starts on this boundary
                if ready(op_id, patch, preds, blocked_by):
                    start[op_id] = now
                    running[op_id] = 0
                    patch_idle.pop(patch, None)
            if len(done) == len(ops):
                patch_idle.clear()
                return
            yield env.timeout(cycle)

    env.process(qpu())
    env.run()
    us = lambda t: t / NS
    return dict(start={k: us(v) for k, v in start.items()}, done={k: us(v) for k, v in done.items()},
                release={k: us(v) for k, v in release.items()}, idle_rounds=idle["count"])


def decsim_model(cycle_us: float, decode_us: float, ops: list) -> dict:
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=D, rounds=D,
                                     after_clifford_depolarization=0.001)
    operations = [Operation(id=i, name=f"op{i}", qubits=(patch,), patches=(patch,), circuit=circuit,
                            predecessors=preds, blocked_by=blocked_by)
                  for i, patch, preds, blocked_by in ops]
    engine = DecoderEngine(PyMatchingDecoder(PresetLatencyDecoder(decode_us)),
                           DecoderTiming(before=(), after=(), frequency_mhz=1000.0))
    completed = RunSpec(ops=operations, d=D, rounds_policy=FixedRounds(D), decoder=engine, num_units=1,
                        timing=TimingConfig(round_us=cycle_us), links=zero_latency_links(),
                        pauli_frame=PauliFrameConfig(commit_us=0.0, zero_commit_cost_justification="cadence check"),
                        seed=0).build()
    runtime = completed.execution_runtime
    us = lambda t: t / TICKS_PER_US
    return dict(start={k: us(v) for k, v in runtime.op_start_time.items()},
                done={k: us(v) for k, v in runtime.body_done_time.items()},
                release={k: us(v) for k, v in runtime.decode_release_time.items()},
                idle_rounds=completed.controller.idle_rounds_emitted)


def close(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(math.isclose(a[k], b[k], abs_tol=1e-6) for k in a)


def main() -> None:
    lines = ["# Gate 4: decsim QPU cycle clock vs an independent SimPy model of the reference rule", "",
             "Rule (Google readout every cycle; Skoric App. D; SWIPER device_manager): one global QEC cycle, "
             "every live patch one round per cycle idle or not, operations start on the next cycle boundary, "
             "a blocked operation is ready when its decode completes. SimPy pinned at "
             "tmp/references/code/simpy (f438164); zero links; one window per operation.", "",
             "| case | cycle us | decode us | starts SimPy / decsim | done | release | idle rounds SimPy / decsim | match |",
             "|---|---|---|---|---|---|---|---|"]
    all_match = True
    for name, cycle_us, decode_us, ops in CASES:
        s, d = simpy_model(cycle_us, decode_us, ops), decsim_model(cycle_us, decode_us, ops)
        match = (close(s["start"], d["start"]) and close(s["done"], d["done"])
                 and close(s["release"], d["release"]) and s["idle_rounds"] == d["idle_rounds"])
        all_match &= match
        fmt = lambda m: " ".join(f"{k}:{v:g}" for k, v in sorted(m.items()))
        lines.append(f"| {name} | {cycle_us:g} | {decode_us:g} | {fmt(s['start'])} / {fmt(d['start'])} | "
                     f"{fmt(s['done'])} / {fmt(d['done'])} | {fmt(s['release'])} / {fmt(d['release'])} | "
                     f"{s['idle_rounds']} / {d['idle_rounds']} | {'yes' if match else 'NO'} |")
    lines += ["", f"Verdict: {'PASS' if all_match else 'FAIL'}. Times in us from the run start; idle rounds are "
              "the cycles a patch spent between operations, each one a transmitted syndrome round in both models."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Gate 8: the whole weak-decoder loop against SWIPER (Vittal et al., ASPLOS
2025, tmp/references/code/swiper), an independent published simulator of
windowed decoding over lattice-surgery schedules.

SWIPER and decsim are given the same program, the same distance, the same
window rule (sliding: commit d rounds, buffer the next d rounds, windows of
one patch decode in order), the same constant decoder latency in rounds and
the same number of decoder units. decsim's links are set to zero latency and
unbounded capacity (SWIPER has no links). Then, window for window:

- memory: the round each window is fully constructed (its buffer complete),
  the round its decode starts, the round its result is available, and the
  round the whole program is decoded. SWIPER counts rounds from 0 and records
  a window's last busy round; decsim counts rounds from 1 and records the
  tick the result exists. So SWIPER start == decsim dispatch in rounds, and
  SWIPER completion + 1 == decsim done. Swept over distance, program length,
  decoder latency and units.
- conditional: one patch runs two d-round instructions and then an S gate
  conditioned on the decode of the second; the rounds the conditional waits
  (SWIPER conditioned_decode_wait_times) against decsim's reaction time with
  the same stream-bound feedback program.

SWIPER needs Python 3.10+; it is run in a subprocess with the interpreter
named by SWIPER_PYTHON (default: the reference venv used for the Toffoli
timing study). decsim runs in the current interpreter.

    python -m experiments.validation.validate_loop_swiper
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SWIPER = REPO / "tmp" / "references" / "code" / "swiper"
SWIPER_PYTHON = os.environ.get(
    "SWIPER_PYTHON",
    "/scratch/gpfs/MARTONOSI/sk2415/qlx-qec-sandbox/tmp/swiper_timing_v1/reference/venv/bin/python")
REPORT = REPO / "experiments" / "results" / "validation" / "loop_swiper.md"

SWIPER_SCRIPT = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from swiper.lattice_surgery_schedule import LatticeSurgerySchedule
from swiper.simulator import DecodingSimulator

def memory(d, rounds, latency, units, method):
    s = LatticeSurgerySchedule()
    s.idle([(0, 0)], num_rounds=rounds)
    ok, _, dev, win, dec = DecodingSimulator().run(
        schedule=s, distance=d, scheduling_method=method,
        decoding_latency_fn=lambda v: latency, max_parallel_processes=units,
        lightweight_setting=0, rng=0)
    windows = []
    for w in win.all_windows:
        if w is None:          # merged away (parallel mode)
            continue
        cr = min(w.commit_region, key=lambda r: r.round_start)
        last = max(w.commit_region, key=lambda r: r.round_start + r.duration)
        windows.append({"commit_lo": cr.round_start, "commit_hi": last.round_start + last.duration - 1,
                        "buffer": sorted((b.round_start, b.round_start + b.duration - 1) for b in w.buffer_regions),
                        "constructed": win.window_construction_times.get(w.window_idx),
                        "start": dec.window_decoding_start_times.get(w.window_idx),
                        "completion": dec.window_decoding_completion_times.get(w.window_idx)})
    return {"ok": ok, "device_rounds": dev.num_rounds, "decoder_rounds": dec.num_rounds,
            "windows": windows}

def conditional(d, latency, units):
    s = LatticeSurgerySchedule()
    s.idle([(0, 0)], num_rounds=d)          # idx 0
    s.idle([(0, 0)], num_rounds=d)          # idx 1, the decode the S waits for
    s.S((0, 0), (1, 0), conditioned_on_idx=1)   # idx 2.. merge, Y_meas, idle, discard
    ok, _, dev, win, dec = DecodingSimulator().run(
        schedule=s, distance=d, scheduling_method="sliding",
        decoding_latency_fn=lambda v: latency, max_parallel_processes=units,
        lightweight_setting=0, rng=0)
    windows = []
    for w in win.all_windows:
        cr = w.commit_region[0]
        windows.append({"patch": list(cr.patch), "commit_lo": cr.round_start,
                        "commit_hi": cr.round_start + cr.duration - 1,
                        "start": dec.window_decoding_start_times.get(w.window_idx),
                        "completion": dec.window_decoding_completion_times.get(w.window_idx)})
    return {"ok": ok, "device_rounds": dev.num_rounds, "decoder_rounds": dec.num_rounds,
            "waits": {str(k): v for k, v in dev.conditioned_decode_wait_times.items()},
            "windows": windows}

jobs = json.loads(sys.stdin.read())
out = []
for job in jobs:
    kind = job.pop("kind")
    out.append(memory(**job) if kind == "memory" else conditional(**job))
print(json.dumps(out))
'''


def run_swiper(jobs):
    completed = subprocess.run([SWIPER_PYTHON, "-c", SWIPER_SCRIPT, str(SWIPER)],
                               input=json.dumps(jobs), capture_output=True, text=True, check=True)
    return json.loads(completed.stdout.strip().splitlines()[-1])   # SWIPER prints a notice first


# ---- decsim side -----------------------------------------------------------

def zero_links():
    from experiments.refactor_lock import _links
    return _links()


def decsim_memory(d, rounds, latency, units, method):
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.message import Operation
    from decsim.qpu.code_geometry import SurfaceCodeModel
    from decsim.qpu.round_policies import FixedRounds
    from decsim.run_spec import RunSpec
    from decsim.windows.windowing_schemes import (ParallelWindowScheme, SlidingTerminalPolicy,
                                                   SlidingWindowScheme)
    scheme = (SlidingWindowScheme(terminal_policy=SlidingTerminalPolicy.REGULAR_STRIDE_LOOKAHEAD)
              if method == "sliding" else ParallelWindowScheme())
    op = Operation(0, "memory", (0,), clifford=True, patches=(0,))
    done = RunSpec(ops=[op], rounds_policy=FixedRounds(rounds), round_us=1.0,
                   code=SurfaceCodeModel(d=d, commit_rounds_override=d, buffer_rounds_override=d),
                   scheme=scheme, decoder=PresetLatencyDecoder(float(latency)), num_units=units,
                   links=zero_links(), seed=1).build()
    us = 1_000_000
    windows = []
    for key in sorted(done.window_manager.windows):
        w = done.window_manager.windows[key]
        windows.append({"commit_lo": w.commit_lo, "commit_hi": w.commit_hi, "buffer_hi": w.buffer_hi,
                        "data": w.t_data_complete / us, "dispatch": w.t_dispatch / us,
                        "done": w.t_done / us})
    return {"fully_done_rounds": done.result.fully_done_ticks / us, "windows": windows}


def decsim_conditional(d, latency, units):
    """Two d-round operations on one dynamic stream, then an S-like operation
    blocked by the second's decode; idle rounds extend the live stream (the
    patch keeps measuring syndromes while it waits, as SWIPER's DECODE_IDLE)."""
    from decsim.controller.policies import ExtendStream
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.message import Operation
    from decsim.observe.metrics import ConditionalReactionTime
    from decsim.qpu.code_geometry import SurfaceCodeModel
    from decsim.qpu.round_policies import PerOpRounds
    from decsim.qpu.syndrome_devices import TimingOnlyDevice
    from decsim.run_spec import RunSpec
    from decsim.windows.windowing_schemes import SlidingTerminalPolicy, SlidingWindowScheme
    stream = Operation(100, "patch-stream", (0,), clifford=True, patches=(0,))
    first = Operation(1, "idle-A", (0,), clifford=True, patches=(0,), stream_id=stream.id)
    second = Operation(2, "idle-B", (0,), clifford=True, patches=(0,), predecessors=(1,),
                       stream_id=stream.id)
    gate = Operation(3, "S", (0,), clifford=True, patches=(0,), predecessors=(2,),
                     decoder_boundary_predecessors=(2,), stream_id=stream.id, blocked_by=2)
    rounds = {1: d, 2: d, 3: d}
    done = RunSpec(ops=[first, second, gate], dynamic_streams=[stream],
                   idle_policy=ExtendStream(), device=TimingOnlyDevice(),
                   code=SurfaceCodeModel(d=d, commit_rounds_override=d, buffer_rounds_override=d),
                   rounds_policy=PerOpRounds(rounds),
                   scheme=SlidingWindowScheme(terminal_policy=SlidingTerminalPolicy.REGULAR_STRIDE_LOOKAHEAD),
                   decoder=PresetLatencyDecoder(float(latency)), num_units=units, round_us=1.0,
                   links=zero_links(), seed=7,
                   make_metrics=lambda e, wm, dm, runtime, f: [ConditionalReactionTime(runtime)]).build()
    us = 1_000_000
    metric = {m.name: m.value for m in done.result.metric_results}
    return {"fully_done_rounds": done.result.fully_done_ticks / us,
            "metrics": metric,
            "windows": [(k, w.commit_lo, w.commit_hi, w.t_dispatch / us, w.t_done / us)
                        for k, w in sorted(done.window_manager.windows.items())]}


# ---- comparison ------------------------------------------------------------

def compare_memory(theirs, ours):
    """Window for window: same commit span (offset by one), SWIPER start ==
    decsim dispatch, SWIPER completion + 1 == decsim done, program end."""
    problems = []
    their_windows = [w for w in theirs["windows"] if w["start"] is not None]
    if len(their_windows) != len(ours["windows"]):
        problems.append(("window count", len(their_windows), len(ours["windows"])))
    for theirs_w, ours_w in zip(their_windows, ours["windows"]):
        if (theirs_w["commit_lo"] + 1, theirs_w["commit_hi"] + 1) != (ours_w["commit_lo"], ours_w["commit_hi"]):
            problems.append(("commit span", theirs_w, ours_w))
        if theirs_w["start"] != ours_w["dispatch"]:
            problems.append(("decode start", theirs_w["commit_lo"], theirs_w["start"], ours_w["dispatch"]))
        if theirs_w["completion"] + 1 != ours_w["done"]:
            problems.append(("decode done", theirs_w["commit_lo"], theirs_w["completion"] + 1, ours_w["done"]))
    if theirs["decoder_rounds"] != ours["fully_done_rounds"]:
        problems.append(("program decoded at", theirs["decoder_rounds"], ours["fully_done_rounds"]))
    checks = 3 * len(their_windows) + 2
    return checks, problems


def main(argv=None) -> None:
    lines = ["# Gate 8: decsim vs SWIPER on the same program", ""]
    total_checks, total_problems = 0, []

    def report(title, checks, problems):
        nonlocal total_checks
        total_checks += checks
        total_problems.extend((title, p) for p in problems)
        state = "agree" if not problems else f"{len(problems)} disagreements"
        lines.append(f"- {title}: {checks} checks, {state}")

    lines.append("## 1. Memory, sliding windows (commit d, buffer d), one patch")
    memory_jobs = [dict(kind="memory", d=d, rounds=rounds, latency=latency, units=units, method="sliding")
                   for d in (3, 5) for rounds in (30, 47) for latency in (2, 5, 11) for units in (1, 2)]
    theirs_all = run_swiper([dict(job) for job in memory_jobs])
    for job, theirs in zip(memory_jobs, theirs_all):
        ours = decsim_memory(job["d"], job["rounds"], job["latency"], job["units"], "sliding")
        title = f"d={job['d']} rounds={job['rounds']} latency={job['latency']} units={job['units']}"
        report(title, *compare_memory(theirs, ours))

    lines.append("")
    lines.append("## 2. Memory, parallel windows")
    lines.append("SWIPER's parallel construction (sources of one commit region, sinks of "
                 "three) and decsim's Skoric block A/B differ as windows, so only the round "
                 "the program is decoded is compared; the window lists are printed.")
    parallel_jobs = [dict(kind="memory", d=3, rounds=30, latency=4, units=3, method="parallel")]
    theirs_all = run_swiper([dict(job) for job in parallel_jobs])
    for job, theirs in zip(parallel_jobs, theirs_all):
        ours = decsim_memory(job["d"], job["rounds"], job["latency"], job["units"], "parallel")
        lines.append(f"- SWIPER windows (lo, hi, buffers, start, completion): "
                     f"{[(w['commit_lo'], w['commit_hi'], w['buffer'], w['start'], w['completion']) for w in theirs['windows']]}")
        lines.append(f"- decsim windows (lo, hi, buffer_hi, dispatch, done), rounds from 0: "
                     f"{[(w['commit_lo'] - 1, w['commit_hi'] - 1, w['buffer_hi'] - 1, w['dispatch'], w['done']) for w in ours['windows']]}")
        problems = []
        if theirs["decoder_rounds"] != ours["fully_done_rounds"]:
            problems.append(("program decoded at", theirs["decoder_rounds"], ours["fully_done_rounds"]))
        report(f"d={job['d']} rounds={job['rounds']} latency={job['latency']} units={job['units']}: program decoded at",
               1, problems)

    lines.append("")
    lines.append("## 3. Conditional S on a patch stream: reaction wait")
    lines.append("SWIPER: idle d, idle d, then S conditioned on the second idle's decode; the "
                 "patch keeps emitting DECODE_IDLE rounds while it waits. decsim: two d-round "
                 "operations on one dynamic stream, then an operation blocked by the second, "
                 "ExtendStream idle policy. Compared: the rounds the conditional waits, and the "
                 "stream's windows up to the release.")
    cond_jobs = [dict(kind="conditional", d=d, latency=latency, units=1)
                 for d in (3, 5) for latency in (2, 5, 9)]
    theirs_all = run_swiper([dict(job) for job in cond_jobs])
    for job, theirs in zip(cond_jobs, theirs_all):
        ours = decsim_conditional(job["d"], job["latency"], job["units"])
        their_wait = theirs["waits"]["2"]
        our_wait = ours["metrics"]["conditional_reaction_time"]["conditioned_decode_wait_times"]["3"]
        problems = []
        if their_wait != our_wait:
            problems.append(("wait rounds", their_wait, our_wait))
        # the waited-for stream's windows whose buffer completes before the
        # release: same spans and decode ticks. After the release SWIPER cuts
        # the next window at the instruction boundary (a short dangling
        # window) while decsim's stream keeps full buffers, so those differ
        # by construction and are not compared.
        their_stream = [w for w in theirs["windows"] if w["patch"] == [0, 0]]
        released_at = 2 * job["d"] + their_wait
        their_before = [w for w in their_stream if w["commit_hi"] + job["d"] < released_at]
        our_before = [w for w in ours["windows"] if w[2] + job["d"] <= released_at]
        pairs = list(zip(their_before, our_before))
        for theirs_w, (key, lo, hi, dispatch, done) in pairs:
            if (theirs_w["commit_lo"] + 1, theirs_w["commit_hi"] + 1, theirs_w["start"], theirs_w["completion"] + 1) != (lo, hi, dispatch, done):
                problems.append(("stream window", theirs_w, (lo, hi, dispatch, done)))
        report(f"d={job['d']} latency={job['latency']}: wait {their_wait} rounds, {len(pairs)} stream windows",
               1 + len(pairs), problems)

    verdict = "PASS" if not total_problems else "FAIL"
    lines.append("")
    lines.append(f"Verdict: {verdict}. {total_checks} checks, {len(total_problems)} disagreements.")
    for title, problem in total_problems[:40]:
        lines.append(f"  - {title}: {problem}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])

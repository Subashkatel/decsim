"""decsim alone: sliding-window memory, window times to a JSON file.

    python -m experiments.validation.decsim_sliding_memory [d] [rounds] [latency_rounds] [units]

Same program as swiper_sliding_memory.py: one patch, memory for `rounds`
QEC rounds, sliding windows (commit d, buffer d), a constant decoder latency,
`units` decoder units, links at zero latency and unbounded capacity. Writes
experiments/results/validation/side_by_side/decsim_sliding_memory.json and
prints the windows; when the SWIPER file is present it prints both side by
side. decsim counts rounds from 1 and records the tick a result exists; SWIPER
counts from 0 and records the last busy round, so SWIPER start == decsim
dispatch and SWIPER last busy + 1 == decsim done.
"""

import json
import sys
from pathlib import Path

from decsim.config import microseconds
from decsim.decoders.decoders import PresetLatencyDecoder
from decsim.message import Operation
from decsim.qpu.code_geometry import SurfaceCodeModel
from decsim.qpu.round_policies import FixedRounds
from decsim.run_spec import RunSpec
from decsim.windows.windowing_schemes import SlidingTerminalPolicy, SlidingWindowScheme
from experiments.refactor_lock import _links as zero_links

RESULTS = Path(__file__).resolve().parents[1] / "results" / "validation" / "side_by_side"
OUTPUT = RESULTS / "decsim_sliding_memory.json"
SWIPER_OUTPUT = RESULTS / "swiper_sliding_memory.json"


def run(distance: int, rounds: int, latency_rounds: int, units: int) -> dict:
    operation = Operation(0, "memory", (0,), clifford=True, patches=(0,))
    code = SurfaceCodeModel(d=distance, commit_rounds_override=distance, buffer_rounds_override=distance)
    scheme = SlidingWindowScheme(terminal_policy=SlidingTerminalPolicy.REGULAR_STRIDE_LOOKAHEAD)
    done = RunSpec(ops=[operation], rounds_policy=FixedRounds(rounds), round_us=1.0, code=code,
                   scheme=scheme, decoder=PresetLatencyDecoder(float(latency_rounds)),
                   num_units=units, links=zero_links(), seed=1).build()
    windows = []
    for key in sorted(done.window_manager.windows):
        window = done.window_manager.windows[key]
        windows.append({
            "commit_first_round": window.commit_lo,
            "commit_last_round": window.commit_hi,
            "data_ready_us": microseconds(window.t_data_complete),
            "dispatch_us": microseconds(window.t_dispatch),
            "done_us": microseconds(window.t_done),
        })
    return {"simulator": "decsim", "distance": distance, "rounds": rounds, "latency_rounds": latency_rounds,
            "units": units, "rounds_count_from": 1,
            "program_decoded_us": microseconds(done.result.fully_done_ticks), "windows": windows}


def print_side_by_side(ours: dict, theirs: dict) -> None:
    print()
    print(f"{'window':>6} | {'SWIPER commit':>13} {'start':>6} {'last busy':>9} | {'decsim commit':>13} {'dispatch':>8} {'done':>6} | agree")
    their_windows = [window for window in theirs["windows"] if window["decode_start_round"] is not None]
    all_agree = len(their_windows) == len(ours["windows"])
    for index, (theirs_w, ours_w) in enumerate(zip(their_windows, ours["windows"])):
        same_commit = (theirs_w["commit_first_round"] + 1 == ours_w["commit_first_round"]
                       and theirs_w["commit_last_round"] + 1 == ours_w["commit_last_round"])
        same_start = theirs_w["decode_start_round"] == ours_w["dispatch_us"]
        same_end = theirs_w["decode_last_busy_round"] + 1 == ours_w["done_us"]
        agree = same_commit and same_start and same_end
        all_agree = all_agree and agree
        their_commit = f"{theirs_w['commit_first_round']}-{theirs_w['commit_last_round']}"
        our_commit = f"{ours_w['commit_first_round']}-{ours_w['commit_last_round']}"
        print(f"{index:>6} | {their_commit:>13} {theirs_w['decode_start_round']:>6} {theirs_w['decode_last_busy_round']:>9} | "
              f"{our_commit:>13} {ours_w['dispatch_us']:>8} {ours_w['done_us']:>6} | {'yes' if agree else 'NO'}")
    same_end = theirs["program_decoded_round"] == ours["program_decoded_us"]
    print(f"program decoded: SWIPER round {theirs['program_decoded_round']} | decsim {ours['program_decoded_us']} us | {'yes' if same_end else 'NO'}")
    print("ALL AGREE" if all_agree and same_end else "DISAGREEMENT")


def main(argv) -> None:
    distance = int(argv[1]) if len(argv) > 1 else 3
    rounds = int(argv[2]) if len(argv) > 2 else 30
    latency_rounds = int(argv[3]) if len(argv) > 3 else 5
    units = int(argv[4]) if len(argv) > 4 else 1
    result = run(distance, rounds, latency_rounds, units)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"decsim: memory d={distance}, {rounds} rounds, latency {latency_rounds} rounds, {units} unit(s); rounds count from 1")
    print(f"{'window':>6} {'commit':>8} {'data ready':>10} {'dispatch':>8} {'done':>6}")
    for index, window in enumerate(result["windows"]):
        commit = f"{window['commit_first_round']}-{window['commit_last_round']}"
        print(f"{index:>6} {commit:>8} {window['data_ready_us']:>10} {window['dispatch_us']:>8} {window['done_us']:>6}")
    print("program decoded at", result["program_decoded_us"], "us")
    print("written:", OUTPUT)
    if SWIPER_OUTPUT.exists():
        theirs = json.loads(SWIPER_OUTPUT.read_text())
        same_inputs = all(theirs[key] == result[key] for key in ("distance", "rounds", "latency_rounds", "units"))
        if same_inputs:
            print_side_by_side(result, theirs)
        else:
            print("SWIPER file has different inputs; rerun swiper_sliding_memory.py with the same arguments")


if __name__ == "__main__":
    main(sys.argv)

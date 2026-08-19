"""SWIPER alone: sliding-window memory, window times to a JSON file.

Run with a Python 3.10+ that has SWIPER's dependencies (networkx, numpy, tqdm,
matplotlib), e.g. the reference venv:

    /scratch/gpfs/MARTONOSI/sk2415/qlx-qec-sandbox/tmp/swiper_timing_v1/reference/venv/bin/python \
        -m experiments.validation.swiper_sliding_memory [d] [rounds] [latency_rounds] [units]

Writes experiments/results/validation/side_by_side/swiper_sliding_memory.json and
prints the windows. Compare with decsim_sliding_memory.py (same arguments).
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tmp" / "references" / "code" / "swiper"))

from swiper.lattice_surgery_schedule import LatticeSurgerySchedule  # noqa: E402
from swiper.simulator import DecodingSimulator  # noqa: E402

OUTPUT = REPO / "experiments" / "results" / "validation" / "side_by_side" / "swiper_sliding_memory.json"


def run(distance: int, rounds: int, latency_rounds: int, units: int) -> dict:
    schedule = LatticeSurgerySchedule()
    schedule.idle([(0, 0)], num_rounds=rounds)
    simulator = DecodingSimulator()
    ok, _, device, window_data, decoder_data = simulator.run(
        schedule=schedule, distance=distance, scheduling_method="sliding",
        decoding_latency_fn=lambda volume: latency_rounds, max_parallel_processes=units,
        lightweight_setting=0, rng=0)
    windows = []
    for window in window_data.all_windows:
        region = window.commit_region[0]
        windows.append({
            "commit_first_round": region.round_start,
            "commit_last_round": region.round_start + region.duration - 1,
            "constructed_round": window_data.window_construction_times.get(window.window_idx),
            "decode_start_round": decoder_data.window_decoding_start_times.get(window.window_idx),
            "decode_last_busy_round": decoder_data.window_decoding_completion_times.get(window.window_idx),
        })
    return {"simulator": "SWIPER", "distance": distance, "rounds": rounds, "latency_rounds": latency_rounds,
            "units": units, "rounds_count_from": 0, "program_decoded_round": decoder_data.num_rounds,
            "windows": windows}


def main(argv) -> None:
    distance = int(argv[1]) if len(argv) > 1 else 3
    rounds = int(argv[2]) if len(argv) > 2 else 30
    latency_rounds = int(argv[3]) if len(argv) > 3 else 5
    units = int(argv[4]) if len(argv) > 4 else 1
    result = run(distance, rounds, latency_rounds, units)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"SWIPER: memory d={distance}, {rounds} rounds, latency {latency_rounds} rounds, {units} unit(s); rounds count from 0")
    print(f"{'window':>6} {'commit':>8} {'constructed':>11} {'start':>6} {'last busy':>9}")
    for index, window in enumerate(result["windows"]):
        commit = f"{window['commit_first_round']}-{window['commit_last_round']}"
        print(f"{index:>6} {commit:>8} {window['constructed_round']:>11} {window['decode_start_round']:>6} {window['decode_last_busy_round']:>9}")
    print("program decoded at round", result["program_decoded_round"])
    print("written:", OUTPUT)


if __name__ == "__main__":
    main(sys.argv)

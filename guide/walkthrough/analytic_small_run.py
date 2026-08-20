"""The small example by hand: 12 rounds, 3 sliding windows, zero noise.

experiments/validation/analytic_oracle_d3_r12.yaml freezes every cost, so
every timestamp of the run follows from arithmetic:

    publication(r) = r x round_period + qc + binary + pack + c2b
    ready(w)       = publication(last round the window reads)
    queued(w)      = max(ready(w), previous window's decode done + dd)
    done(w)        = queued(w) + cwd + fetch + algorithm + release
    commit(w)      = done(w) + wdo + frame commit

This script is that arithmetic written out once, so the simulator has an
exact answer key. guide/walkthrough/simulated_small_run.py replays the same
yaml in the simulator and fails loudly on any difference.

    PYTHONPATH=. python guide/walkthrough/analytic_small_run.py
"""

from pathlib import Path

from experiments.baseline.baseline_closed_loop import load_config

CONFIG = Path("experiments/validation/analytic_oracle_d3_r12.yaml")

COLUMNS = ("window_id", "read_lo", "read_hi", "commit_lo", "commit_hi",
           "buffer0_ready_us", "queued_us", "dispatch_us", "decode_done_us",
           "dd_delivery_us", "frame_commit_us", "buffer0_ready_to_frame_us")


def window_layout(rounds: int, commit_rounds: int, buffer_rounds: int) -> list:
    """Sliding windows: window k commits rounds k*c+1 .. k*c+c and reads
    buffer_rounds beyond them for context; the last window that still fits
    commits and reads everything left."""
    windows = []
    window_id = 0
    while window_id * commit_rounds + commit_rounds + buffer_rounds <= rounds:
        read_lo = window_id * commit_rounds + 1
        windows.append({"window_id": window_id,
                        "read_lo": read_lo,
                        "read_hi": read_lo + commit_rounds + buffer_rounds - 1,
                        "commit_lo": read_lo,
                        "commit_hi": read_lo + commit_rounds - 1})
        window_id += 1
    last_window = windows[-1]
    last_window["read_hi"] = rounds
    last_window["commit_hi"] = rounds
    return windows


def analytic_timeline(config: dict) -> list:
    """One row per window, every timestamp in microseconds, from the yaml's
    numbers alone."""
    block = config["sweep"][0]
    round_period_us = block["round_period_us"][0]
    algorithm_us = block["algorithm_latency_us"][0]
    links = config["links"]
    controller = config["controller"]
    engine = config["decoder"]["engine"]
    engine_cycle_us = 1.0 / engine["frequency_mhz"]

    # Round r leaves the QPU when it finishes, at r x round_period. It is
    # published in Buffer 0 one constant offset later: the QC link, the
    # controller's pulses-to-binary step, packing, and the C2B link. Every
    # link of this yaml has unbounded bandwidth, so there is no serialization.
    publication_offset_us = (links["qc"]["latency_us"]
                             + controller["t_binary_availability_us"]
                             + controller["t_pack_us"]
                             + links["c2b"]["latency_us"])

    windows = window_layout(config["rounds_per_shot"],
                            config["windowing"]["commit_rounds"],
                            config["windowing"]["buffer_rounds"])
    rows = []
    previous_done_us = None
    for window in windows:
        ready_us = window["read_hi"] * round_period_us + publication_offset_us
        if previous_done_us is None:
            queued_us = ready_us          # the first window has no dependency
        else:
            dependency_arrival_us = previous_done_us + links["dd"]["latency_us"]
            queued_us = max(ready_us, dependency_arrival_us)
        dispatch_us = queued_us           # one decoder unit, idle again by then

        rounds_fetched = window["read_hi"] - window["read_lo"] + 1
        service_us = (links["cwd"]["latency_us"]
                      + rounds_fetched * engine["fetch_cycles_per_round"] * engine_cycle_us
                      + algorithm_us
                      + engine["release_cycles_per_job"] * engine_cycle_us)
        done_us = dispatch_us + service_us
        commit_us = done_us + links["wdo"]["latency_us"] + config["pauli_frame"]["commit_us"]

        last_window = window["read_hi"] == config["rounds_per_shot"]
        rows.append({**window,
                     "buffer0_ready_us": ready_us,
                     "queued_us": queued_us,
                     "dispatch_us": dispatch_us,
                     "decode_done_us": done_us,
                     # DD hands the boundary to the NEXT window's decode; the
                     # last window has no successor and sends nothing.
                     "dd_delivery_us": None if last_window else done_us + links["dd"]["latency_us"],
                     "frame_commit_us": commit_us,
                     "buffer0_ready_to_frame_us": commit_us - ready_us})
        previous_done_us = done_us
    return rows


def print_table(rows: list, columns: tuple = COLUMNS) -> None:
    def cell(value):
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for row in rows:
        print("| " + " | ".join(cell(row[column]) for column in columns) + " |")


if __name__ == "__main__":
    print_table(analytic_timeline(load_config(CONFIG)))

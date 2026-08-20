"""The same small run through the simulator, timestamp for timestamp.

Builds the run from the same yaml the answer key uses, replays it, reads
each window's timestamps out of the simulator's own records, prints both
tables and fails loudly on any difference.

    PYTHONPATH=. python guide/walkthrough/simulated_small_run.py
"""

import sys

from analytic_small_run import COLUMNS, CONFIG, analytic_timeline, print_table
from decsim.config import TICKS_PER_US
from experiments.baseline.baseline_closed_loop import build_run, load_config


def us(ticks: int) -> float:
    return ticks / TICKS_PER_US


def simulated_timeline(config: dict) -> list:
    """One row per window, the same columns as the answer key, but every
    number read out of the completed simulation."""
    block = config["sweep"][0]
    spec, _ = build_run(config,
                        physical_error_probability=block["physical_error_probability"][0],
                        round_period_us=block["round_period_us"][0],
                        algorithm_latency_us=block["algorithm_latency_us"][0],
                        seed=0)
    completed = spec.build()

    transfers = completed.result.link_traffic["transfers"]
    delivery = {(row["path"], row["attribution"]["window_id"]): row["delivery_ticks"]
                for row in transfers if row["path"] in ("dd", "wdo")}
    frame_by_window = {record.window_key[1]: record
                       for record in completed.pauli_frame.snapshot().records}

    rows = []
    last_round = config["rounds_per_shot"]
    for (operation_id, window_id), window in sorted(completed.window_manager.windows.items()):
        frame_record = frame_by_window[window_id]
        dd_delivery_ticks = delivery.get(("dd", window_id))
        rows.append({
            "window_id": window_id,
            "read_lo": window.start_round,
            "read_hi": min(window.buffer_hi, last_round),
            "commit_lo": window.commit_lo,
            "commit_hi": window.commit_hi,
            "buffer0_ready_us": us(window.t_data_complete),
            "queued_us": us(window.t_queued),
            "dispatch_us": us(window.t_dispatch),
            "decode_done_us": us(window.t_done),
            "dd_delivery_us": None if dd_delivery_ticks is None else us(dd_delivery_ticks),
            "frame_commit_us": us(frame_record.committed_ticks),
            "buffer0_ready_to_frame_us": us(frame_record.committed_ticks - window.t_data_complete),
        })
    return rows


def normalized(value):
    """Ticks are whole microseconds x 1e6, so six decimals compare exactly."""
    if isinstance(value, float):
        return round(value, 6)
    return value


def differences(analytic_rows: list, simulated_rows: list) -> list:
    """Every (window, column, analytic, simulated) cell that disagrees."""
    disagreements = []
    if len(analytic_rows) != len(simulated_rows):
        disagreements.append(("window count", "", len(analytic_rows), len(simulated_rows)))
    for analytic_row, simulated_row in zip(analytic_rows, simulated_rows):
        for column in COLUMNS:
            analytic_value = normalized(analytic_row[column])
            simulated_value = normalized(simulated_row[column])
            if analytic_value != simulated_value:
                disagreements.append((analytic_row["window_id"], column,
                                      analytic_value, simulated_value))
    return disagreements


def main() -> None:
    config = load_config(CONFIG)
    analytic_rows = analytic_timeline(config)
    simulated_rows = simulated_timeline(config)

    print("Analytic answer key:")
    print_table(analytic_rows)
    print()
    print("Simulator:")
    print_table(simulated_rows)
    print()

    disagreements = differences(analytic_rows, simulated_rows)
    if disagreements:
        for window_id, column, analytic_value, simulated_value in disagreements:
            print(f"MISMATCH window {window_id} {column}: "
                  f"analytic {analytic_value} vs simulated {simulated_value}")
        sys.exit(1)
    cells = len(analytic_rows) * len(COLUMNS)
    print(f"MATCH: all {cells} cells agree to the tick.")


if __name__ == "__main__":
    main()

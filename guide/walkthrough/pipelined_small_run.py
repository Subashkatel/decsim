"""The pipelined small run: sliding windows with input faster than decode.

Same whole-number card as the small run, rounds every 1 us instead of 3.
Prints the per-window table and draws the stage timeline; the overlap of
the colored bars is the pipelining.

    PYTHONPATH=. python guide/walkthrough/pipelined_small_run.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from decsim.config import TICKS_PER_US
from experiments.baseline.baseline_closed_loop import build_run

CONFIG_PATH = Path(__file__).parent / "pipelined_run.yaml"
FIGURE_PATH = Path(__file__).parent / "figures" / "pipelined_timeline.png"


def us(ticks):
    value = ticks / TICKS_PER_US
    if value == int(value):
        return int(value)
    return round(value, 6)


config = yaml.safe_load(CONFIG_PATH.read_text())
sweep = config["sweep"][0]
spec, decoder_engine = build_run(config,
                                 physical_error_probability=sweep["physical_error_probability"][0],
                                 round_period_us=sweep["round_period_us"][0],
                                 algorithm_latency_us=sweep["algorithm_latency_us"][0],
                                 seed=8)
completed = spec.build()

transfers = completed.result.link_traffic["transfers"]
windows = {window_id: window
           for (_, window_id), window in sorted(completed.window_manager.windows.items())}
frame_records = {record.window_key[1]: record
                 for record in completed.pauli_frame.snapshot().records}
operation = spec.ops[0]


def transfers_on(path):
    """round -> transfer for qc and c2b, window -> transfer for the rest."""
    selected = {}
    for transfer in transfers:
        if transfer["path"] != path:
            continue
        if path in ("qc", "c2b"):
            selected[transfer["attribution"]["round_lo"]] = transfer
        else:
            selected[transfer["attribution"]["window_id"]] = transfer
    return selected


qc = transfers_on("qc")
c2b = transfers_on("c2b")
cwd = transfers_on("cwd")
dd = transfers_on("dd")
wdo = transfers_on("wdo")

print("window | ready | dependency wait | dispatch | done | commit | reaction")
for window_id, window in windows.items():
    record = frame_records[window_id]
    dependency_wait = us(window.t_dispatch - window.t_data_complete)
    reaction = us(record.committed_ticks - window.t_data_complete)
    print(f"{window_id:6} | {us(window.t_data_complete):5} | {dependency_wait:15} "
          f"| {us(window.t_dispatch):8} | {us(window.t_done):4} "
          f"| {us(record.committed_ticks):6} | {reaction}")

ROWS = ("qpu round", "qc link", "c2b link", "buffer fill", "wait",
        "transfer (cwd)", "fetch", "algorithm", "release", "dd handoff",
        "wdo link", "frame commit")
WINDOW_COLORS = ("tab:blue", "tab:orange", "tab:green")

figure, axis = plt.subplots(figsize=(9, 5))

round_period = int(sweep["round_period_us"][0])
for round_index in sorted(qc):
    sent = us(qc[round_index]["send_ticks"])
    # rounds are back to back at this period, so alternating shades mark
    # where one ends and the next begins, on every per-round row
    shade = "0.55" if round_index % 2 else "0.75"
    axis.barh(0, round_period, left=sent - round_period, color=shade, height=0.55)
    axis.barh(1, 1, left=sent, color=shade, height=0.55)
    axis.barh(2, 1, left=us(c2b[round_index]["send_ticks"]), color=shade, height=0.55)

for window_id, window in windows.items():
    color = WINDOW_COLORS[window_id]

    def bar(row, start, end):
        lane = row + (window_id - 1) * 0.19
        axis.barh(lane, end - start, left=start, color=color, height=0.17)

    stages = {record.stage: record
              for record in decoder_engine.stage_records_for(operation.id, window_id)}
    bar(3, us(window.t_first_round), us(window.t_data_complete))
    bar(4, us(window.t_data_complete), us(window.t_dispatch))
    bar(5, us(cwd[window_id]["send_ticks"]), us(cwd[window_id]["delivery_ticks"]))
    for row, stage in ((6, "fetch"), (7, "algorithm"), (8, "release")):
        bar(row, us(stages[stage].start_ticks), us(stages[stage].end_ticks))
    if window_id in dd:
        bar(9, us(dd[window_id]["send_ticks"]), us(dd[window_id]["delivery_ticks"]))
    bar(10, us(wdo[window_id]["send_ticks"]), us(wdo[window_id]["delivery_ticks"]))
    bar(11, us(frame_records[window_id].accepted_ticks),
        us(frame_records[window_id].committed_ticks))

axis.set_yticks(range(len(ROWS)))
axis.set_yticklabels(ROWS, fontsize=9)
axis.invert_yaxis()
axis.set_xticks(range(0, 32))
axis.set_xlim(0, 31)
axis.set_xlabel("time from shot start (µs)")
axis.set_title("The pipelined small run: rounds every 1 µs, chain 7 µs per window")
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in ("0.6",) + WINDOW_COLORS]
axis.legend(handles, ("rounds", "window 0", "window 1", "window 2"),
            loc="lower left", fontsize=8)
axis.grid(alpha=0.5, linewidth=0.8)
axis.set_axisbelow(True)
figure.tight_layout()
figure.savefig(FIGURE_PATH, dpi=200)
print("figure:", FIGURE_PATH)

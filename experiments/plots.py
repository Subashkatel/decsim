"""The two experiment figures.

timeline.png   one shot of the first sweep point, seed 0: every stage of
               every window on its own row, in real time, the weak
               baseline's figure style (commit reads solid, buffer reads
               lighter, geometry in the subtitle)
ler.png        logical error rate vs physical error rate, Wilson 95%
               bars, drawn when more than one p was swept

Every time is in microseconds.
"""

from pathlib import Path

from decsim.config import TICKS_PER_US

from experiments.build_run import build_run
from experiments.experiment_config import ExperimentConfig
from experiments.measure_shot import INPUT_LINK, OUTPUT_LINK

WINDOW_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red",
                 "tab:purple", "tab:brown", "tab:pink", "tab:olive")
MAX_LEGEND_WINDOWS = 8


def us(ticks) -> float:
    return ticks / TICKS_PER_US


def card_label(algorithm) -> str:
    if isinstance(algorithm, str):
        return algorithm.capitalize()
    return f"{algorithm:g} µs"


# ---- the window timeline ---------------------------------------------------

def timeline_plot(config: ExperimentConfig, path: Path) -> None:
    """One shot at the first sweep point, seed 0, every stage in time."""
    import matplotlib.pyplot as plt

    block = config.sweep[0]
    physical_error_probability = block.physical_error_probabilities[0]
    distance = block.distances[0]
    round_period_us = block.round_periods_us[0]
    algorithm = config.active_decoder.algorithm
    spec, engine = build_run(
        config, physical_error_probability=physical_error_probability,
        distance=distance, round_period_us=round_period_us, seed=0)
    completed = spec.build()

    input_path = INPUT_LINK[config.mode]
    output_path = OUTPUT_LINK[config.mode]
    store_path = "csb" if config.mode == "strong_only" else "cwb"
    store_name = ("syndrome buffer 1" if config.mode == "strong_only"
                  else "buffer 0")

    transfers = completed.result.link_traffic["transfers"]

    def on(path_name, by_round):
        selected = {}
        for transfer in transfers:
            if transfer["path"] != path_name:
                continue
            key = (transfer["attribution"]["round_lo"] if by_round
                   else transfer["attribution"]["window_id"])
            selected[key] = transfer
        return selected

    qc = on("qc", by_round=True)
    store = on(store_path, by_round=True)
    input_link = on(input_path, by_round=False)
    dd = on("dd", by_round=False)
    output_link = on(output_path, by_round=False)
    windows = {window_id: window for (_, window_id), window
               in sorted(completed.window_manager.windows.items())}
    frame_records = {record.window_key[1]: record
                     for record in completed.pauli_frame.snapshot().records}
    rounds = config.rounds_per_shot.rounds_for(distance)

    rows = ["qpu round", "qc link"]
    if store:
        rows.append(f"{store_name} link")
    rows += [f"{store_name} fill", "wait", f"transfer ({input_path})",
             "fetch", "algorithm", "release", "dd handoff",
             f"{output_path} link", "frame commit"]
    row_index = {name: index for index, name in enumerate(rows)}

    figure, axis = plt.subplots(figsize=(11, 0.45 * len(rows) + 1.6))
    for round_number in sorted(qc):
        sent = us(qc[round_number]["send_ticks"])
        shade = "0.55" if round_number % 2 else "0.75"
        axis.barh(row_index["qpu round"], round_period_us,
                  left=sent - round_period_us, color=shade, height=0.55)
        axis.barh(row_index["qc link"],
                  us(qc[round_number]["delivery_ticks"]) - sent,
                  left=sent, color=shade, height=0.55)
        if round_number in store:
            axis.barh(row_index[f"{store_name} link"],
                      us(store[round_number]["delivery_ticks"])
                      - us(store[round_number]["send_ticks"]),
                      left=us(store[round_number]["send_ticks"]),
                      color=shade, height=0.55)

    # the round's landing tick in the store the windows read
    stored_row = store if store else qc
    window_ranges = []
    lane_count = max(len(windows), 3)
    for window_id, window in windows.items():
        if window.t_done is None:
            continue
        color = WINDOW_COLORS[window_id % len(WINDOW_COLORS)]

        def bar(row, start, end, window_id=window_id, color=color, alpha=1.0):
            lane = (row_index[row]
                    + (window_id - (lane_count - 1) / 2) * (0.5 / lane_count))
            axis.barh(lane, end - start, left=start, color=color,
                      height=0.17, alpha=alpha)

        stages = {record.stage: record
                  for record in engine.stage_records_for(1, window_id)}
        read_hi = min(window.buffer_hi, rounds)
        stored_tick = us(stored_row[read_hi]["delivery_ticks"])
        window_ranges.append(
            f"window {window_id}: "
            f"commits {window.commit_lo}-{window.commit_hi}, "
            f"reads {window.start_round}-{read_hi}")
        # commit rounds land solid; the trailing buffer reads land lighter
        commit_stored = us(
            stored_row[min(window.commit_hi, rounds)]["delivery_ticks"])
        bar(f"{store_name} fill", us(window.t_first_round), commit_stored)
        if read_hi > window.commit_hi:
            bar(f"{store_name} fill", commit_stored, stored_tick, alpha=0.45)
        bar("wait", stored_tick, us(window.t_dispatch))
        if window_id in input_link:
            bar(f"transfer ({input_path})",
                us(input_link[window_id]["send_ticks"]),
                us(input_link[window_id]["delivery_ticks"]))
        for stage in ("fetch", "algorithm", "release"):
            if stage in stages:
                bar(stage, us(stages[stage].start_ticks),
                    us(stages[stage].end_ticks))
        if window_id in dd:
            bar("dd handoff", us(dd[window_id]["send_ticks"]),
                us(dd[window_id]["delivery_ticks"]))
        if window_id in output_link:
            bar(f"{output_path} link",
                us(output_link[window_id]["send_ticks"]),
                us(output_link[window_id]["delivery_ticks"]))
        if window_id in frame_records:
            bar("frame commit",
                us(frame_records[window_id].accepted_ticks),
                us(frame_records[window_id].committed_ticks))

    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels(rows, fontsize=9)
    axis.invert_yaxis()
    axis.set_xlabel("time from shot start (µs)")
    axis.set_title(f"{config.name}: one shot, every stage at its real time",
                   pad=22)
    commit_rounds = config.windowing.commit_rounds or distance
    buffer_rounds = config.windowing.buffer_rounds or distance
    axis.text(0.5, 1.005,
              f"d={distance} {config.code_task.split(':')[0]}"
              f" · {config.windowing.scheme} windows: commit {commit_rounds},"
              f" buffer {buffer_rounds} rounds"
              f" · rounds every {round_period_us:g} µs"
              f" · algorithm {card_label(algorithm)}"
              f" · p={physical_error_probability:g}",
              transform=axis.transAxes, ha="center", va="bottom",
              fontsize=8.5, color="0.35")
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.6")]
    labels = ["rounds"]
    for window_id in list(windows)[:MAX_LEGEND_WINDOWS]:
        handles.append(plt.Rectangle(
            (0, 0), 1, 1, color=WINDOW_COLORS[window_id % len(WINDOW_COLORS)]))
        labels.append(window_ranges[window_id]
                      if window_id < len(window_ranges)
                      else f"window {window_id}")
    if len(windows) > MAX_LEGEND_WINDOWS:
        labels[-1] += "  (…)"
    handles.append(plt.Rectangle((0, 0), 1, 1, color="0.4", alpha=0.45))
    labels.append("lighter fill = buffer reads (not committed)")
    axis.legend(handles, labels, loc="lower left", fontsize=7.5)
    axis.grid(alpha=0.5, linewidth=0.8)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


# ---- the logical error rate ------------------------------------------------

def ler_groups(rows: list) -> list:
    """(distance, round time, rows sorted by p) for every distance and
    round time that swept more than one physical error rate: the papers'
    convention, one LER curve per code distance."""
    distances = sorted({row["distance"] for row in rows})
    periods = sorted({row["round_period_us"] for row in rows})
    groups = []
    for distance in distances:
        for period in periods:
            group = [row for row in rows
                     if row["distance"] == distance
                     and row["round_period_us"] == period]
            probabilities = {row["physical_error_probability"] for row in group}
            if len(probabilities) < 2:
                continue
            group.sort(key=lambda row: row["physical_error_probability"])
            groups.append((distance, period, group))
    return groups


def ler_plot(rows: list, path: Path) -> None:
    """Logical error rate against physical error rate, Wilson 95% bars, one
    line per card and round time that swept more than one p."""
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(4.8, 3.6))
    groups = ler_groups(rows)
    plotted_periods = {period for _, period, _ in groups}
    for distance, period, group in groups:
        probabilities = [row["physical_error_probability"] for row in group]
        rates = [row["logical_error_rate"] for row in group]
        lower = [row["logical_error_rate"] - row["ler_wilson_low"]
                 for row in group]
        upper = [row["ler_wilson_high"] - row["logical_error_rate"]
                 for row in group]
        label = (f"d={distance}" if len(plotted_periods) == 1
                 else f"d={distance}, {period:g} µs")
        axis.errorbar(probabilities, rates, yerr=[lower, upper], fmt="o-",
                      capsize=3, label=label)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Physical error rate")
    axis.set_ylabel("Logical error rate")
    axis.set_title("Logical error rate")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plots(config: ExperimentConfig, rows: list, report_dir: Path) -> None:
    """timeline.png always; ler.png when more than one p was swept."""
    import matplotlib
    matplotlib.use("Agg")
    timeline_plot(config, report_dir / "timeline.png")
    if len({row["physical_error_probability"] for row in rows}) > 1:
        ler_plot(rows, report_dir / "ler.png")

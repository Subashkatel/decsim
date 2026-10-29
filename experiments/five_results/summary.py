r"""The pilot's numbers, one table per run, written as a Markdown page.

    python summary.py RESULTS DATE

RESULTS is the five results' results folder and DATE the runs' folder
name, so RESULTS/r1_shots/DATE is R1's folded shots. Each run that has
finished adds its table; one still running is named and skipped. The
page is RESULTS/summary_DATE.md.

- R1 on shots: each decoder's failures and logical error rate per round
  with its 95% interval (sweep.csv), its reaction time, and the two
  paired on the same shots (shots.csv; McNemar's exact test on the shots
  only one of them failed, failure_statistics.exact_mcnemar).
- R1 on streams: each decoder's waiting line and reaction time over one
  long stream (the stream's json record).
- The calibration: the threshold that escalates each share
  (calibrate.py's calibration.csv).
- The switching pilot (R3 and R5): each design's failures, escalated
  share, reaction time and windows past one logical cycle, and what it
  did to union-find's shots, paired with both decoders alone
  (paired_shots.paired_repairs).

Reaction is buffer0_ready_to_frame: a window's data complete to its
correction at the frame. The share past one logical cycle is reported
beside it, not as a pass mark (experiments.md, decision D1).
"""

import csv
import json
import pathlib
import sys

import decsim.experiments.paired_shots as paired_shots

R1_SHOTS = "r1_shots"
R1_STREAMS = "r1_streams"
CALIBRATION = "gap_calibration"
SWITCHING_PILOT = "switching_pilot"
WEAK_ALONE = {"decoding": "weak_alone"}
STRONG_ALONE = {"decoding": "strong_alone"}
DESIGN_COLUMNS = ("scheme", "strong_unit_count", "share")
DECODER_HEADER = (
    "| Decoder | Shots | Failures | Error per round (95%) | "
    "Reaction mean / p99 (us) | Past one cycle |\n"
    "|---|---|---|---|---|---|"
)
STREAM_HEADER = (
    "| Stream | Rounds | Windows | Reaction mean / max (us) | "
    "Growth (us per us) | Most waiting | Past one cycle |\n"
    "|---|---|---|---|---|---|---|"
)
CALIBRATION_HEADER = (
    "| d | p | Target share | Threshold (dB) | Share it escalates |\n"
    "|---|---|---|---|---|"
)
DESIGN_HEADER = (
    "| Design | Failures | Error per round (95%) | Escalated | "
    "Reaction mean / p99 (us) | Past one cycle | "
    "Repaired / broken / missed |\n"
    "|---|---|---|---|---|---|---|"
)


def main(results: pathlib.Path, date: str) -> None:
    """Write the page into RESULTS and print it."""
    r1_folder = results / R1_SHOTS / date
    streams_folder = results / R1_STREAMS / date
    calibration_folder = results / CALIBRATION / date
    switching_folder = results / SWITCHING_PILOT / date
    sections = [
        r1_shots_section(r1_folder),
        r1_streams_section(streams_folder),
        calibration_section(calibration_folder),
        switching_section(switching_folder, r1_folder),
    ]
    joined = "\n\n".join(sections)
    text = f"{joined}\n"
    summary_path = results / f"summary_{date}.md"
    summary_path.write_text(text)
    print(text)


def r1_shots_section(folder: pathlib.Path) -> str:
    """Each decoder alone on the same shots, then the pair of them."""
    title = "## R1 on shots"
    if not _is_folded(folder):
        return _not_yet(title, folder)
    sweep = _read_named(folder, "sweep.csv")
    lines = [title, "", DECODER_HEADER]
    for row in sweep:
        line = _decoder_line(row)
        lines.append(line)
    shots = _read_named(folder, "shots.csv")
    weak = paired_shots.failures_by_seed(shots, WEAK_ALONE)
    strong = paired_shots.failures_by_seed(shots, STRONG_ALONE)
    # the strong decoder as the pair: repaired counts the shots only
    # union-find failed, broken those only correlated matching failed
    pair = paired_shots.paired_repairs(weak, strong, strong)
    sentence = _pair_sentence(pair)
    lines.extend(("", sentence))
    return "\n".join(lines)


def r1_streams_section(folder: pathlib.Path) -> str:
    """Each decoder alone over one long stream."""
    title = "## R1 on streams"
    paths = folder.glob("*.json")
    records = sorted(paths)
    if not records:
        return _not_yet(title, folder)
    lines = [title, "", STREAM_HEADER]
    for path in records:
        text = path.read_text()
        record = json.loads(text)
        line = _stream_line(path.stem, record["measurement"])
        lines.append(line)
    return "\n".join(lines)


def calibration_section(folder: pathlib.Path) -> str:
    """The threshold that escalates each share, on the calibration seeds."""
    title = "## Gap calibration"
    path = folder / "calibration.csv"
    if not path.exists():
        return _not_yet(title, folder)
    lines = [title, "", CALIBRATION_HEADER]
    for row in _read(path):
        line = _calibration_line(row)
        lines.append(line)
    return "\n".join(lines)


def switching_section(folder: pathlib.Path, r1_folder: pathlib.Path) -> str:
    """Each design's numbers and its effect on union-find's shots."""
    title = "## Switching pilot (R3, R5)"
    if not _is_folded(folder) or not _is_folded(r1_folder):
        return _not_yet(title, folder)
    sweep = _read_named(folder, "sweep.csv")
    shots = _read_named(folder, "shots.csv")
    r1_shots = _read_named(r1_folder, "shots.csv")
    weak = paired_shots.failures_by_seed(r1_shots, WEAK_ALONE)
    strong = paired_shots.failures_by_seed(r1_shots, STRONG_ALONE)
    lines = [title, "", DESIGN_HEADER]
    for row in sweep:
        selection = {column: row[column] for column in DESIGN_COLUMNS}
        design = paired_shots.failures_by_seed(shots, selection)
        pair = paired_shots.paired_repairs(weak, strong, design)
        line = _design_line(selection, row, pair)
        lines.append(line)
    return "\n".join(lines)


def _decoder_line(row: dict) -> str:
    rate = _rate_text(row)
    reaction = _reaction_text(row)
    past = _past_cycle_text(row)
    return (
        f"| {row['decoding']} | {row['scored_shots']} | "
        f"{row['logical_failures']} | {rate} | {reaction} | {past} |"
    )


def _calibration_line(row: dict) -> str:
    return (
        f"| {row['distance']} | {row['physical_error_probability']} | "
        f"{row['target_share']} | {row['threshold_decibels']} | "
        f"{row['escalated_share']} |"
    )


def _design_line(selection: dict, row: dict, pair) -> str:
    name = (
        f"{selection['scheme']}, {selection['strong_unit_count']} core, "
        f"{selection['share']}"
    )
    rate = _rate_text(row)
    escalated = _fixed(row["escalated_fraction"], 3)
    reaction = _reaction_text(row)
    past = _past_cycle_text(row)
    effect = f"{pair.repaired} / {pair.broken} / {pair.missed}"
    return (
        f"| {name} | {row['logical_failures']} | {rate} | {escalated} | "
        f"{reaction} | {past} | {effect} |"
    )


def _stream_line(name: str, measurement: dict) -> str:
    reaction = measurement["values"]["buffer0_ready_to_frame"]
    growth = _growth_text(measurement)
    past = _stream_past_cycle_text(measurement)
    return (
        f"| {name} | {measurement['rounds']} | {measurement['decoded_windows']}"
        f" | {reaction['mean']:.1f} / {reaction['maximum']:.1f} | {growth} | "
        f"{measurement['max_queued_windows']} | {past} |"
    )


def _pair_sentence(pair) -> str:
    return (
        f"Paired on {pair.shots} shots: union-find alone failed "
        f"{pair.fast_failures}, correlated matching alone "
        f"{pair.strong_failures}; {pair.repaired} shots only union-find "
        f"failed, {pair.broken} only correlated matching failed "
        f"(McNemar's exact p = {pair.p_value:.2g})."
    )


def _rate_text(row: dict) -> str:
    """The rate and its limits; with no failure the fold writes no rate."""
    rate = _scientific(row["logical_error_rate_per_round"])
    low = _scientific(row["logical_error_rate_per_round_low"])
    high = _scientific(row["logical_error_rate_per_round_high"])
    return f"{rate} ({low} to {high})"


def _reaction_text(row: dict) -> str:
    mean = _fixed(row["buffer0_ready_to_frame_mean_us"], 1)
    tail = _fixed(row["buffer0_ready_to_frame_p99_us"], 1)
    return f"{mean} / {tail}"


def _past_cycle_text(row: dict) -> str:
    """The fold's missed fraction over the recorded windows."""
    fraction = row.get("deadline_missed_fraction", "")
    if fraction == "":
        return "not set"
    return _fixed(fraction, 3)


def _stream_past_cycle_text(measurement: dict) -> str:
    """The recorded windows over the deadline, as a share of them."""
    missed = measurement["deadline_missed_windows"]
    recorded = measurement["recorded_windows"]
    if missed is None or recorded == 0:
        return "not set"
    share = missed / recorded
    return f"{share:.3f}"


def _growth_text(measurement: dict) -> str:
    rates = [
        measurement["weak_reaction_growth_rate"],
        measurement["strong_reaction_growth_rate"],
    ]
    written = [f"{rate:.3f}" for rate in rates if rate is not None]
    if not written:
        return "none"
    return " / ".join(written)


def _scientific(written: str) -> str:
    """A csv number in scientific form, or "none" when the cell is blank."""
    if written == "":
        return "none"
    number = float(written)
    return f"{number:.2e}"


def _fixed(written: str, digits: int) -> str:
    """A csv number with digits after the point, or "none" when blank."""
    if written == "":
        return "none"
    number = float(written)
    return f"{number:.{digits}f}"


def _is_folded(folder: pathlib.Path) -> bool:
    sweep_path = folder / "sweep.csv"
    return sweep_path.exists()


def _not_yet(title: str, folder: pathlib.Path) -> str:
    return f"{title}\n\nNot finished yet: {folder}."


def _read_named(folder: pathlib.Path, name: str) -> list:
    path = folder / name
    return _read(path)


def _read(path: pathlib.Path) -> list:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        return list(reader)


if __name__ == "__main__":
    results_folder = pathlib.Path(sys.argv[1])
    main(results_folder, sys.argv[2])

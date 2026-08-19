"""Baseline logical error rate through the closed loop.

Many shots of the same rotated memory circuit through the whole loop (the
same run as baseline_closed_loop.py), at one timing point, over physical
error probability; the logical failure is read from the Pauli frame's
prediction against the sampled truth. Writes ler.csv and ler.png.

Usage: python -m experiments.baseline.baseline_logical_error [config.yaml]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from experiments.baseline.baseline_closed_loop import (DEFAULT_CONFIG, load_config, measure_shot,
                                                       write_csv)


# ---- the logical error rate sweep -------------------------------------------

def wilson_interval(failures: int, shots: int, z: float = 1.96) -> tuple:
    """Wilson 95% confidence interval for a failure fraction."""
    if shots == 0:
        return (0.0, 0.0)
    fraction = failures / shots
    denominator = 1 + z * z / shots
    center = (fraction + z * z / (2 * shots)) / denominator
    half_width = z * math.sqrt(fraction * (1 - fraction) / shots + z * z / (4 * shots * shots)) / denominator
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def run_ler_sweep(config: dict) -> list:
    """One row per physical error probability: shots, failures, LER and its interval."""
    sweep = config["ler_sweep"]
    rows = []
    for probability in sweep["physical_error_probabilities"]:
        point_config = dict(config)
        point_config["physical_error_probability"] = probability
        failures = 0
        for seed in range(sweep["shots"]):
            shot = measure_shot(point_config, round_period_us=sweep["round_period_us"],
                                algorithm_latency_us=sweep["algorithm_latency_us"], seed=seed)
            failures += int(shot.logical_failure)
        low, high = wilson_interval(failures, sweep["shots"])
        rows.append({"physical_error_probability": probability, "shots": sweep["shots"],
                     "failures": failures, "logical_error_rate": failures / sweep["shots"],
                     "wilson_low": low, "wilson_high": high})
        print(f"LER p={probability}: {failures}/{sweep['shots']}", file=sys.stderr)
    return rows


def ler_plot(rows: list, config: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    probabilities = [row["physical_error_probability"] for row in rows]
    rates = [row["logical_error_rate"] for row in rows]
    lower_errors = [row["logical_error_rate"] - row["wilson_low"] for row in rows]
    upper_errors = [row["wilson_high"] - row["logical_error_rate"] for row in rows]
    figure, axis = plt.subplots(figsize=(5, 3.6))
    axis.errorbar(probabilities, rates, yerr=[lower_errors, upper_errors], fmt="o-", capsize=3,
                  label=f"d={config['distance']}, {config['rounds_per_shot']} rounds, {config['windowing']['scheme']} windows")
    axis.set_xlabel("physical error probability")
    axis.set_ylabel(f"logical error rate per shot")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)



def main(argv) -> None:
    config_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_CONFIG
    config = load_config(config_path)
    report_dir = Path(config["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = run_ler_sweep(config)
    write_csv(rows, report_dir / "ler.csv")
    import matplotlib
    matplotlib.use("Agg")
    ler_plot(rows, config, report_dir / "ler.png")
    for row in rows:
        print(f"p={row['physical_error_probability']:g}: LER {row['logical_error_rate']:.4f} "
              f"[{row['wilson_low']:.4f}, {row['wilson_high']:.4f}] ({row['failures']}/{row['shots']})")


if __name__ == "__main__":
    main(sys.argv)

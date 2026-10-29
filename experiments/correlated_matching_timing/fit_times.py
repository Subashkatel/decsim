r"""Fit correlated matching's window time to its detection events.

Reads time_windows.py's times.csv. For each distance and scheme it takes
the full windows (2 d rounds for sliding, 3 d for the double window) at
every noise factor and fits the window's median time t to its detection
events n in three laws, each a DetectionEventTime
(minimum_weight_perfect_matching/detection_event_time.py):

- line: t = a + b n, least squares with a and b at zero or more;
- power: t = c n^k, least squares on log t, so each window counts by
  its relative error;
- intercept and power: t = a + c n^k, least squares on t.

Each law is scored by five-fold held-out error: the mean of |predicted
- measured| / measured over the windows left out. The law with the
least is chosen. Its scale makes the law's mean over the calm windows
(factor 1, p = 0.003) equal 2604.12841 Table IX's time per round times
the window's rounds, five_results/config.py's average; the law keeps
the shape measured here. Table IX also times p = 0.006 (p.22), so the
ratio of the measured mean at factor 2 to the calm mean is printed
beside the paper's: a check of the shape against a referent.

It writes fit.json and prints, per cell, the windows, the raw median and
99th percentile, each law with its held-out error, and the scale.

Usage
-----

With this checkout first on the path (STYLE.md):

```
python fit_times.py --times DIR/times.csv --out DIR
```
"""

import argparse
import csv
import json
import pathlib

import numpy
import scipy.optimize

import decsim.experiments.sweeps as sweeps

five_results = sweeps.load_beside(__file__, "../five_results/config.py")
# 2604.12841 Table IX (p.22) at p = 0.003, five_results' average
TABLE_ROUND_MICROSECONDS = (
    five_results.CORRELATED_MATCHING_ROUND_MICROSECONDS_BY_DISTANCE
)
# 2604.12841 Table IX (p.22): correlated matching on a Grace CPU, batch
# 1, microseconds per round at p = 0.006; p = 0.003 is five_results'
TABLE_ROUND_MICROSECONDS_AT_DOUBLE_NOISE = {5: 1.78, 9: 7.51, 13: 21.51}
FOLDS = 5
FOLD_SEED = 20261029
LAWS = ("line", "power", "intercept_and_power")


def fit_line(events, microseconds) -> dict:
    """The line t = a + b n by least squares, a and b held at zero or more.

    DetectionEventTime refuses a negative intercept, a time below zero
    for a quiet window.
    """
    ones = numpy.ones_like(events)
    design = numpy.column_stack([ones, events])
    solution, _residual = scipy.optimize.nnls(design, microseconds)
    intercept, slope = solution
    return _law(intercept, slope, 1.0)


def fit_power(events, microseconds) -> dict:
    """The power t = c n^k on log t; a window with no event is left out."""
    has_events = events > 0
    log_events = numpy.log(events[has_events])
    log_microseconds = numpy.log(microseconds[has_events])
    exponent, log_coefficient = numpy.polyfit(log_events, log_microseconds, 1)
    coefficient = numpy.exp(log_coefficient)
    return _law(0.0, coefficient, exponent)


def fit_intercept_and_power(events, microseconds) -> dict:
    """The law t = a + c n^k on t, with a and c held at zero or more."""
    start = fit_power(events, microseconds)
    guess = (0.0, start["coefficient_microseconds"], start["exponent"])
    bounds = ([0.0, 0.0, 0.1], [numpy.inf, numpy.inf, 4.0])
    solution, _ = scipy.optimize.curve_fit(
        _intercept_and_power, events, microseconds, p0=guess, bounds=bounds
    )
    intercept, coefficient, exponent = solution
    return _law(intercept, coefficient, exponent)


FITTERS = {
    "line": fit_line,
    "power": fit_power,
    "intercept_and_power": fit_intercept_and_power,
}


def law_microseconds(law: dict, events):
    """The unscaled law at these event counts."""
    growth = numpy.power(events, law["exponent"])
    event_microseconds = law["coefficient_microseconds"] * growth
    return law["intercept_microseconds"] + event_microseconds


def held_out_error(fitter, events, microseconds) -> float:
    """The mean relative error over the windows each fold leaves out."""
    count = len(events)
    order = numpy.arange(count)
    generator = numpy.random.default_rng(FOLD_SEED)
    generator.shuffle(order)
    folds = numpy.array_split(order, FOLDS)
    errors = []
    for left_out in folds:
        kept = numpy.setdiff1d(order, left_out)
        law = fitter(events[kept], microseconds[kept])
        predicted = law_microseconds(law, events[left_out])
        measured = microseconds[left_out]
        difference = predicted - measured
        absolute = numpy.abs(difference)
        relative = absolute / measured
        errors.append(relative)
    every_error = numpy.concatenate(errors)
    mean_error = every_error.mean()
    return float(mean_error)


def fit_cell(windows: list, distance: int, scheme: str) -> dict:
    """Every law of one cell, its held-out error, the chosen law's scale."""
    events = _column(windows, "events")
    microseconds = _column(windows, "microseconds")
    laws = {}
    for name in LAWS:
        fitter = FITTERS[name]
        law = fitter(events, microseconds)
        law["held_out_relative_error"] = held_out_error(
            fitter, events, microseconds
        )
        laws[name] = law
    chosen = min(LAWS, key=lambda name: laws[name]["held_out_relative_error"])
    calm = _at_factor(windows, 1.0)
    calm_events = _column(calm, "events")
    calm_law = law_microseconds(laws[chosen], calm_events)
    calm_mean = calm_law.mean()
    full_rounds = _full_rounds(distance, scheme)
    table_microseconds = TABLE_ROUND_MICROSECONDS[distance] * full_rounds
    scale = table_microseconds / calm_mean
    return {
        "distance": distance,
        "scheme": scheme,
        "rounds": full_rounds,
        "laws": laws,
        "chosen": chosen,
        "table_microseconds": table_microseconds,
        "calm_law_mean_microseconds": float(calm_mean),
        "scale": float(scale),
        "factors": _factor_summaries(windows),
        "double_noise_check": _double_noise_check(windows, distance),
    }


def main() -> None:
    """Fit every cell of the times file; write fit.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--times", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    rows = _full_windows(arguments.times)
    cells = sorted({(row["distance"], row["scheme"]) for row in rows})
    fits = []
    for distance, scheme in cells:
        windows = _in_cell(rows, distance, scheme)
        fit = fit_cell(windows, distance, scheme)
        fits.append(fit)
        _print_fit(fit)
    out = pathlib.Path(arguments.out)
    path = out / "fit.json"
    text = json.dumps(fits, indent=2)
    path.write_text(text)
    print(f"wrote {path}")


def _law(intercept, coefficient, exponent) -> dict:
    return {
        "intercept_microseconds": float(intercept),
        "coefficient_microseconds": float(coefficient),
        "exponent": float(exponent),
    }


def _intercept_and_power(events, intercept, coefficient, exponent):
    growth = numpy.power(events, exponent)
    return intercept + coefficient * growth


def _full_rounds(distance: int, scheme: str) -> int:
    distances = five_results.STRONG_DECODE_DISTANCES_BY_SCHEME[scheme]
    return distances * distance


def _full_windows(path: str) -> list:
    """The windows that read their scheme's full rounds, typed."""
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [_typed(row) for row in reader]
    return [row for row in rows if row["is_full"]]


def _typed(row: dict) -> dict:
    distance = int(row["distance"])
    rounds = int(row["rounds"])
    full_rounds = _full_rounds(distance, row["scheme"])
    nanoseconds = int(row["median_nanoseconds"])
    return {
        "distance": distance,
        "scheme": row["scheme"],
        "factor": float(row["factor"]),
        "rounds": rounds,
        "is_full": rounds == full_rounds,
        "events": float(row["events"]),
        "microseconds": nanoseconds / 1000.0,
    }


def _in_cell(rows: list, distance: int, scheme: str) -> list:
    return [
        row
        for row in rows
        if row["distance"] == distance and row["scheme"] == scheme
    ]


def _at_factor(windows: list, factor: float) -> list:
    return [window for window in windows if window["factor"] == factor]


def _column(windows: list, name: str):
    values = [window[name] for window in windows]
    return numpy.asarray(values, dtype=float)


def _factor_summaries(windows: list) -> list:
    """Each factor's windows, events and raw times, as measured."""
    factors = sorted({window["factor"] for window in windows})
    summaries = []
    for factor in factors:
        at_factor = _at_factor(windows, factor)
        summary = _factor_summary(at_factor, factor)
        summaries.append(summary)
    return summaries


def _factor_summary(at_factor: list, factor: float) -> dict:
    events = _column(at_factor, "events")
    microseconds = _column(at_factor, "microseconds")
    mean_events = events.mean()
    mean_microseconds = microseconds.mean()
    median = numpy.median(microseconds)
    percentile_99 = numpy.percentile(microseconds, 99)
    return {
        "factor": factor,
        "windows": len(at_factor),
        "mean_events": float(mean_events),
        "mean_microseconds": float(mean_microseconds),
        "median_microseconds": float(median),
        "percentile_99_microseconds": float(percentile_99),
    }


def _double_noise_check(windows: list, distance: int) -> dict:
    """The mean at twice the noise over the calm mean: measured, Table IX's."""
    calm = _at_factor(windows, 1.0)
    doubled = _at_factor(windows, 2.0)
    calm_table = TABLE_ROUND_MICROSECONDS[distance]
    doubled_table = TABLE_ROUND_MICROSECONDS_AT_DOUBLE_NOISE[distance]
    table_ratio = doubled_table / calm_table
    if not calm or not doubled:
        return {"table_ratio": table_ratio, "measured_ratio": None}
    calm_times = _column(calm, "microseconds")
    doubled_times = _column(doubled, "microseconds")
    calm_mean = calm_times.mean()
    doubled_mean = doubled_times.mean()
    measured_ratio = doubled_mean / calm_mean
    return {"table_ratio": table_ratio, "measured_ratio": float(measured_ratio)}


def _print_fit(fit: dict) -> None:
    print(
        f"d={fit['distance']} {fit['scheme']} ({fit['rounds']} rounds): "
        f"chosen {fit['chosen']}, scale {fit['scale']:.4f} to "
        f"{fit['table_microseconds']:.2f} us"
    )
    for summary in fit["factors"]:
        print(
            f"  factor {summary['factor']:g}: {summary['windows']} windows, "
            f"mean events {summary['mean_events']:.1f}, median "
            f"{summary['median_microseconds']:.2f} us, 99th percentile "
            f"{summary['percentile_99_microseconds']:.2f} us"
        )
    for name, law in fit["laws"].items():
        print(
            f"  {name}: a={law['intercept_microseconds']:.4f} "
            f"c={law['coefficient_microseconds']:.5f} "
            f"k={law['exponent']:.4f} held-out error "
            f"{law['held_out_relative_error']:.4f}"
        )
    check = fit["double_noise_check"]
    print(f"  twice the noise over calm: {check}")


if __name__ == "__main__":
    main()

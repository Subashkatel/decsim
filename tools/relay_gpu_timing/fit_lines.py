"""Fit each region's decode times to a line in Relay-BP iterations.

Reads the CSV files time_decodes.py wrote and prints, for each region
size, the least-squares line through every decode at every noise
strength, its r2, the fastest decode, the iterations the decodes span,
and how far decodes sit off the line (the ratio of measured to line
time at the 1st, 50th and 99th percentiles). It also reads the line
decsim's table holds for the same device and region size at the same
keys, so a reader sees how the table's line, fit on fewer iterations,
reads over the new span.

Run with decsim's interpreter, PYTHONPATH=<tree>:<pydeps>:

    python tools/relay_gpu_timing/fit_lines.py --region-dir DIR CSV...
"""

import argparse
import csv
import os

import numpy

import decsim.decoders.measured_table.measurements as measurements
import decsim.decoders.relay_belief_propagation.decoder as relay_decoder


def read_decodes(paths):
    """Every timed decode, grouped by its region's detector count."""
    by_detectors = {}
    for path in paths:
        rows = _read_rows(path)
        for row in rows:
            detectors = int(row["detectors"])
            group = by_detectors.setdefault(detectors, [])
            group.append(row)
    return by_detectors


def least_squares_line(iterations, microseconds):
    """The line's intercept, slope and r2."""
    ones = numpy.ones_like(iterations)
    design = numpy.column_stack([ones, iterations])
    solution, _, _, _ = numpy.linalg.lstsq(design, microseconds, rcond=None)
    intercept, slope = solution
    fitted = design @ solution
    residuals = microseconds - fitted
    residual_squares = residuals**2
    residual_total = residual_squares.sum()
    residual_sum = float(residual_total)
    mean = microseconds.mean()
    deviations = microseconds - mean
    spread_squares = deviations**2
    spread_total = spread_squares.sum()
    spread_sum = float(spread_total)
    r_squared = 1.0 - residual_sum / spread_sum
    return float(intercept), float(slope), r_squared


def table_line(device, detectors, settings):
    """The table's line for this device, region and keys, if it has one."""
    wanted = (device, "whole", "together", detectors, settings)
    for cell in measurements.RELAY_BP_TIMES:
        cell_keys = (
            cell.device,
            cell.partition,
            cell.bases,
            cell.detectors,
            cell.relay_settings,
        )
        if cell_keys == wanted:
            return cell
    return None


def region_keys(region_dir, distance):
    """The Relay-BP keys the region files of this distance carry."""
    names = os.listdir(region_dir)
    for name in sorted(names):
        if not name.startswith(f"region_d{distance}_"):
            continue
        path = os.path.join(region_dir, name)
        region = numpy.load(path)
        return relay_decoder.RelayBeliefPropagationDecoder.Settings(
            gamma0=float(region["gamma0"]),
            gamma_interval=tuple(
                float(bound) for bound in region["gamma_interval"]
            ),
            pre_iterations=int(region["pre_iterations"]),
            relay_set_count=int(region["relay_set_count"]),
            iterations_per_set=int(region["iterations_per_set"]),
            converged_solution_count=int(region["converged_solution_count"]),
        )
    raise FileNotFoundError(f"no region file for d={distance} in {region_dir}")


def describe(detectors, rows, region_dir):
    """One region size's line, spread and the table's line beside it."""
    iterations = numpy.array([float(row["iterations"]) for row in rows])
    microseconds = numpy.array([float(row["time_ns"]) for row in rows]) / 1e3
    intercept, slope, r_squared = least_squares_line(iterations, microseconds)
    line_microseconds = intercept + slope * iterations
    ratios = microseconds / line_microseconds
    low, middle, high = numpy.percentile(ratios, [1, 50, 99])
    distance = int(rows[0]["distance"])
    settings = region_keys(region_dir, distance)
    cell = table_line(rows[0]["device"], detectors, settings)
    probabilities = sorted({row["probability"] for row in rows})
    fastest = microseconds.min()
    median_iterations = numpy.median(iterations)
    p99_iterations = numpy.percentile(iterations, 99)
    most_iterations = iterations.max()
    print(f"d={distance} detectors={detectors} p={probabilities} n={len(rows)}")
    print(
        f"  line: {intercept:.3f} + {slope:.4f} x iterations us, "
        f"r2 {r_squared:.4f}, fastest {fastest:.3f} us"
    )
    print(
        f"  iterations: median {median_iterations:.0f}, "
        f"p99 {p99_iterations:.0f}, max {most_iterations:.0f}"
    )
    print(
        f"  measured / line: p1 {low:.3f}, median {middle:.3f}, p99 {high:.3f}"
    )
    _print_table_line(cell, iterations, microseconds)


def main():
    """Print each region size's line from the CSV files named."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-dir", required=True)
    parser.add_argument("csv", nargs="+")
    args = parser.parse_args()
    by_detectors = read_decodes(args.csv)
    for detectors in sorted(by_detectors):
        describe(detectors, by_detectors[detectors], args.region_dir)


def _read_rows(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _print_table_line(cell, iterations, microseconds):
    if cell is None:
        print("  table: no line at these keys")
        return
    per_iteration = cell.microseconds_per_iteration * iterations
    table_microseconds = cell.intercept_microseconds + per_iteration
    table_ratios = table_microseconds / microseconds
    table_error = numpy.median(table_ratios) - 1.0
    print(
        f"  table: {cell.intercept_microseconds} + "
        f"{cell.microseconds_per_iteration} x iterations us, "
        f"median error over these decodes {table_error:+.1%}"
    )


if __name__ == "__main__":
    main()

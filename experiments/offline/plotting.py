"""Small paper-facing plots from reduced experiment counts."""

from collections import defaultdict
from pathlib import Path


def _binomial_interval(failures, shots, confidence=0.95):
    from scipy.stats import beta

    tail = (1 - confidence) / 2
    lower = 0 if failures == 0 else beta.ppf(
        tail, failures, shots - failures + 1
    )
    upper = 1 if failures == shots else beta.ppf(
        1 - tail, failures + 1, shots - failures
    )
    return float(lower), float(upper)


def plot_logical_error_rate(rows, output, *, title="Logical error rate"):
    """Plot exact failure counts by physical error rate and code distance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    rows = tuple(rows)
    for row in rows:
        grouped[row["distance"]].append(row)

    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    for distance, points in sorted(grouped.items()):
        points.sort(key=lambda point: point["physical_error_rate"])
        x_values = []
        y_values = []
        lower_errors = []
        upper_errors = []
        for point in points:
            failures = point["failures"]
            shots = point["shots"]
            rate = failures / shots
            lower, upper = _binomial_interval(failures, shots)
            if failures == 0:
                rate = upper
                lower = upper
            x_values.append(point["physical_error_rate"])
            y_values.append(rate)
            lower_errors.append(rate - lower)
            upper_errors.append(upper - rate)
        axis.errorbar(
            x_values,
            y_values,
            yerr=(lower_errors, upper_errors),
            marker="o",
            capsize=3,
            label=f"d={distance}",
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Physical error rate")
    axis.set_ylabel("Logical error rate per shot")
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_logical_error_rate_card(rows, output, *, title="Logical error rate"):
    """Describe an exact-count logical-error-rate plot in plain language."""
    rows = tuple(rows)
    grouped = {row["distance"] for row in rows}
    zero_failure_points = sum(row["failures"] == 0 for row in rows)

    distances = ", ".join(str(distance) for distance in sorted(grouped))
    shot_counts = ", ".join(
        f"{shots:,}" for shots in sorted({row["shots"] for row in rows})
    )
    zero_note = (
        f" {zero_failure_points} zero-failure point(s) are shown at their "
        "95% confidence upper limit."
        if zero_failure_points else ""
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"# {title}\n\n"
        "This offline figure asks how the weak-decoder logical error rate "
        "changes with physical error rate. It does not include event-engine "
        "latency, links, queues, or switching.\n\n"
        f"- Distances: {distances}\n"
        f"- Shots per point: {shot_counts}\n"
        "- Horizontal axis: physical error rate (log scale)\n"
        "- Vertical axis: logical error rate per attempted shot (log scale)\n"
        "- Error bars: exact 95% Clopper-Pearson binomial intervals."
        f"{zero_note}\n",
        encoding="utf-8",
    )

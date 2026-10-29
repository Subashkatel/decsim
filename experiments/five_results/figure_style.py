"""The five results' figure style, after the closest papers.

An ISCA column is 3.35 in wide. SWIPER (Viszlai et al. 2412.05115,
Figs. 3, 11 and 13) draws one plot per column at about 2:1, in the
body's serif, with ticks inward on all four sides, a colour-blind
palette with one colour per method, and its own method hatched in bar
charts. Toshio et al. (2510.25222, Figs. 15 and 16) draw each decoder
alone as a dotted or dashed line and switching as solid, with the
uncertainty as a shaded band. No figure carries a title: the caption
says the one thing it shows.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLUMN_WIDTH_INCHES = 3.35
HEIGHT_INCHES = 1.9
# seaborn's colour-blind palette, as SWIPER uses; the three pass the
# all-pairs colour-blind check, and each also has its own marker
RULE_COLORS = {
    "sliding": "#0173B2",
    "speculation": "#DE8F05",
    "double": "#029E73",
}
RULE_MARKERS = {"sliding": "o", "speculation": "^", "double": "s"}
RULE_NAMES = {
    "sliding": "Sliding window",
    "speculation": "Speculation",
    "double": "Double window",
}
# each decoder alone in neutral ink, told apart by its line and marker
DECODER_STYLES = {
    "weak_alone": {"color": "#7a7a7a", "linestyle": ":", "marker": "s"},
    "strong_alone": {"color": "#000000", "linestyle": "--", "marker": "D"},
}
DECODER_NAMES = {
    "weak_alone": "Union-find alone",
    "strong_alone": "Correlated matching alone",
}
DEADLINE_COLOR = "#b2182b"
BAND_ALPHA = 0.18
RC_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "errorbar.capsize": 0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "hatch.linewidth": 0.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
}


def use() -> None:
    """Set the style for every figure drawn after."""
    plt.rcParams.update(RC_PARAMS)


def figure(height: float = HEIGHT_INCHES) -> tuple:
    """One column-wide figure and its axes."""
    size = (COLUMN_WIDTH_INCHES, height)
    return plt.subplots(figsize=size)


def deadline(axes, microseconds: float, vertical: bool) -> None:
    """The deadline as a dashed line, named beside it."""
    style = {"color": DEADLINE_COLOR, "linestyle": "--", "linewidth": 0.8}
    label = f"Deadline ({microseconds:g} µs)"
    if vertical:
        axes.axvline(microseconds, **style)
        axes.annotate(
            label,
            (microseconds, 1),
            xycoords=("data", "axes fraction"),
            xytext=(2, -2),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=6.5,
        )
    else:
        axes.axhline(microseconds, **style)
        axes.annotate(
            label,
            (0, microseconds),
            xycoords=("axes fraction", "data"),
            xytext=(2, 2),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=6.5,
        )


def save(figure_handle, folder: pathlib.Path, name: str) -> None:
    """Write the figure as PDF for the paper and PNG for a page."""
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = folder / f"{name}.{suffix}"
        figure_handle.savefig(path)
    plt.close(figure_handle)

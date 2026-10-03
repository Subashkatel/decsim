"""Relay-BP decode times measured on NVIDIA GPUs, as a line in iterations.

Each row is one measured cell: a device, how it was shared (whole: one
process on the whole GPU; mps: one client of an MPS daemon; 3g.40gb and
1g.10gb: one process on one Multi-Instance GPU slice of the NVIDIA
profile of that name), how many decodes ran on it at once, and the
region decoded, named by its detector count. The time of one decode is
intercept_microseconds plus microseconds_per_iteration times the
Relay-BP iterations it ran, the least-squares line through the cell's
2,000 timed decodes, and never less than fastest_decode_microseconds,
the fastest of those decodes (the minimum of the time_ns column of the
cell's measurement file, in microseconds). Below its fastest decode a
line is extrapolating to times the device never showed.

How the times were made. The decoder is NVIDIA's Relay-BP,
nv-qldpc-decoder from cudaq-qec-cu12 0.8.0 (CUDA-Q QEC 0.8.0, cudaqx
commit 6ac00e5c), with decsim's relay_bp profile as its keys: bp_method
3, composition 1, gamma0 0.1, 80 pre-iterations, 300 relay sets of 60
iterations, stop at the first converged solution, fp32, and decsim's own
gamma table. Each time is one decode() call from Python on the host
clock: the binding, the syndrome's copy to the device, the kernels and
the answer's copy back, with no dispatcher and no link. The GPUs are an
NVIDIA A100-SXM4-80GB (x86 host) and a GH200 144G HBM3e (Grace host,
NVLink-C2C), driver 610.57.04, CUDA user-mode driver 13.3, toolkit 12.9.
The two slice profiles were cut on NVIDIA A100 80GB PCIe boards, as
nvidia-smi names them in the measurement's logs, so a slice row is not
the whole-card row's own board cut in parts.
The regions are decsim's memory circuit (Stim
surface_code:rotated_memory_z, p = 0.001 on all four noise channels) of
3d rounds, r_strong = r_com + 2 r_buf at r_com = r_buf = d (Toshio et
al. arXiv:2510.25222 lines 1246-1250), 2,000 Stim shots per distance.
Rows with bases apart time the X part and the Z part of the same
regions on the same shots, each one decode at a time on the whole card:
decsim's split (detector_error_model/basis_split.py) cuts the region's
model, each part is given decsim's gamma table for its own column
count, and each part is a row of its own, named by its detector count.

Every whole-card and MPS line explains its cell's times with r2 at
least 0.996 except the GH200 at d = 13, 0.683, where a few slow outliers
sit off the line; the part lines explain r2 0.990 to 1.000 except the
A100's d = 5 X part, 0.864, where one decode of one iteration took
1,704 us, an outlier the line cannot follow. The slice
lines explain r2 0.951 to 0.9999, and a line leans on the long decodes:
at a cell's median decode a slice's line reads 1 percent high to 13
percent low, the whole card's 1 percent high to 6 percent low. The
1g.10gb line at d = 13 leans the hardest: it reads 522 us at the three
iterations its fastest region ran and crosses zero below two, while
that region took 1,406 us, which is why a line is floored at its cell's
fastest decode.
Cells with two or more decodes at once are not here: without MPS the
processes time-slice the GPU and a decode's time follows the others'
work, not its own iterations (their lines explain r2 0.11 to 0.96), so
no line prices them.

The rows that name SURFACE_CODE_RELAY_BP_5 were measured at the Relay-BP
paper's surface code Relay-BP-5 (Mueller et al. 2506.01779 lines 307,
332 and 343), the decoder baseline's strong decoder: gamma0 0.35, the
interval [-0.254, 0.985], 600 relay sets of 60 after 80 pre-iterations,
stop after 5 converged solutions (nv-qldpc-decoder's NConv criterion), and
the rest as above, with decsim's gamma table drawn at these keys. They
are one decode at a time on the whole A100-SXM4-80GB, X and Z
together, the same regions and shots, Slurm job 14676845 on 2026-09-29
(driver 610.57.04, CUDA user-mode driver 13.3). Each slope is the
Relay-BP-1 line's to within 7 percent and each intercept 32 to 108 us
higher, so at a cell's median decode the Relay-BP-1 line at the same
iterations reads 1.5 to 13 percent low, the cell's own line 2.5
percent low to 0.6 percent high.
"""

import dataclasses

from decsim.decoders.relay_belief_propagation import (
    decoder as relay_belief_propagation,
)

RelaySettings = relay_belief_propagation.RelayBeliefPropagationDecoder.Settings


@dataclasses.dataclass(frozen=True)
class MeasuredTime:
    """One cell's decode time as a line in its Relay-BP iterations.

    relay_settings are the relay_bp row's keys every decode of the cell
    ran at, its defaults unless the row names others, each decode X and
    Z together; bases says whether the region was cut into parts first.
    """

    device: str
    partition: str
    bases: str
    decodes_running: int
    detectors: int
    intercept_microseconds: float
    microseconds_per_iteration: float
    fastest_decode_microseconds: float
    relay_settings: RelaySettings = RelaySettings()

    def decode_microseconds(self, iterations: int) -> float:
        """The line at these iterations, floored at the fastest decode."""
        return floored_line_microseconds(
            self.intercept_microseconds,
            self.microseconds_per_iteration,
            self.fastest_decode_microseconds,
            iterations,
        )


def floored_line_microseconds(
    intercept_microseconds: float,
    microseconds_per_iteration: float,
    fastest_microseconds: float,
    iterations: int,
) -> float:
    """A time line in Relay-BP iterations, floored at the fastest measured.

    A line fit to the long decodes can read below the fastest decode
    measured at few iterations, or cross zero, so it is floored there.
    """
    per_iteration = microseconds_per_iteration * iterations
    line_microseconds = intercept_microseconds + per_iteration
    return max(line_microseconds, fastest_microseconds)


# Rows in device, partition, bases, settings, then region order; the
# comment above a row names the code distance of its region, its part
# when it is one, its settings when they are not the relay_bp row's
# defaults, and its line's r2.
RELAY_BP_TIMES = (
    # d 5, 0.9962
    MeasuredTime("a100", "whole", "together", 1, 360, 78.957, 9.762, 85.872),
    # d 7, 0.9971
    MeasuredTime(
        "a100", "whole", "together", 1, 1008, 106.435, 20.797, 122.491
    ),
    # d 9, 0.9997
    MeasuredTime(
        "a100", "whole", "together", 1, 2160, 147.553, 26.852, 170.963
    ),
    # d 11, 0.9999
    MeasuredTime(
        "a100", "whole", "together", 1, 3960, 189.257, 36.131, 277.895
    ),
    # d 13, 0.9997
    MeasuredTime(
        "a100", "whole", "together", 1, 6552, 310.036, 48.329, 464.768
    ),
    # d 5, Relay-BP-5, 0.9963
    MeasuredTime(
        "a100",
        "whole",
        "together",
        1,
        360,
        110.852,
        9.827,
        147.187,
        relay_settings=relay_belief_propagation.SURFACE_CODE_RELAY_BP_5,
    ),
    # d 7, Relay-BP-5, 0.9897
    MeasuredTime(
        "a100",
        "whole",
        "together",
        1,
        1008,
        178.551,
        19.533,
        259.108,
        relay_settings=relay_belief_propagation.SURFACE_CODE_RELAY_BP_5,
    ),
    # d 9, Relay-BP-5, 0.9940
    MeasuredTime(
        "a100",
        "whole",
        "together",
        1,
        2160,
        218.469,
        26.372,
        640.656,
        relay_settings=relay_belief_propagation.SURFACE_CODE_RELAY_BP_5,
    ),
    # d 11, Relay-BP-5, 0.9993
    MeasuredTime(
        "a100",
        "whole",
        "together",
        1,
        3960,
        276.795,
        36.079,
        1092.257,
        relay_settings=relay_belief_propagation.SURFACE_CODE_RELAY_BP_5,
    ),
    # d 13, Relay-BP-5, 0.9985
    MeasuredTime(
        "a100",
        "whole",
        "together",
        1,
        6552,
        417.132,
        48.252,
        1748.302,
        relay_settings=relay_belief_propagation.SURFACE_CODE_RELAY_BP_5,
    ),
    # d 5, X part, 0.8641
    MeasuredTime("a100", "whole", "apart", 1, 168, 59.437, 7.752, 64.058),
    # d 5, Z part, 0.9991
    MeasuredTime("a100", "whole", "apart", 1, 192, 59.673, 7.346, 64.831),
    # d 7, X part, 0.9996
    MeasuredTime("a100", "whole", "apart", 1, 480, 67.169, 10.024, 74.726),
    # d 7, Z part, 0.9999
    MeasuredTime("a100", "whole", "apart", 1, 528, 69.844, 11.465, 77.773),
    # d 9, X part, 0.9997
    MeasuredTime("a100", "whole", "apart", 1, 1040, 80.712, 20.042, 98.627),
    # d 9, Z part, 0.9994
    MeasuredTime("a100", "whole", "apart", 1, 1120, 85.380, 20.228, 99.995),
    # d 11, X part, 1.0000
    MeasuredTime("a100", "whole", "apart", 1, 1920, 92.328, 22.866, 118.381),
    # d 11, Z part, 0.9989
    MeasuredTime("a100", "whole", "apart", 1, 2040, 96.495, 25.718, 124.116),
    # d 13, X part, 0.9999
    MeasuredTime("a100", "whole", "apart", 1, 3192, 125.165, 28.965, 163.399),
    # d 13, Z part, 0.9985
    MeasuredTime("a100", "whole", "apart", 1, 3360, 123.099, 32.976, 162.063),
    # d 5, 0.9971
    MeasuredTime("a100", "mps", "together", 1, 360, 80.911, 9.839, 86.273),
    # d 9, 0.9985
    MeasuredTime("a100", "mps", "together", 1, 2160, 155.414, 26.961, 171.383),
    # d 13, 0.9988
    MeasuredTime("a100", "mps", "together", 1, 6552, 326.723, 48.319, 464.064),
    # d 5, 0.9514
    MeasuredTime("a100", "3g.40gb", "together", 1, 360, 72.377, 9.527, 77.529),
    # d 7, 0.9810
    MeasuredTime(
        "a100", "3g.40gb", "together", 1, 1008, 101.534, 14.972, 112.184
    ),
    # d 9, 0.9967
    MeasuredTime(
        "a100", "3g.40gb", "together", 1, 2160, 139.312, 27.765, 175.893
    ),
    # d 11, 0.9999
    MeasuredTime(
        "a100", "3g.40gb", "together", 1, 3960, 200.997, 47.165, 328.342
    ),
    # d 13, 0.9999
    MeasuredTime(
        "a100", "3g.40gb", "together", 1, 6552, 209.000, 79.085, 582.489
    ),
    # d 5, 0.9997
    MeasuredTime("a100", "1g.10gb", "together", 1, 360, 68.534, 15.951, 81.145),
    # d 7, 0.9981
    MeasuredTime(
        "a100", "1g.10gb", "together", 1, 1008, 98.295, 36.404, 136.857
    ),
    # d 9, 0.9998
    MeasuredTime(
        "a100", "1g.10gb", "together", 1, 2160, 96.910, 76.640, 229.207
    ),
    # d 11, 0.9999
    MeasuredTime(
        "a100", "1g.10gb", "together", 1, 3960, 96.049, 136.788, 502.685
    ),
    # d 13, 0.9993
    MeasuredTime(
        "a100", "1g.10gb", "together", 1, 6552, -585.136, 369.200, 1405.683
    ),
    # d 5, 0.9996
    MeasuredTime("gh200", "whole", "together", 1, 360, 70.463, 7.219, 71.808),
    # d 7, 0.9969
    MeasuredTime("gh200", "whole", "together", 1, 1008, 95.243, 11.044, 99.552),
    # d 9, 0.9999
    MeasuredTime(
        "gh200", "whole", "together", 1, 2160, 125.843, 16.437, 137.280
    ),
    # d 11, 0.9999
    MeasuredTime(
        "gh200", "whole", "together", 1, 3960, 171.485, 23.870, 211.552
    ),
    # d 13, 0.6833
    MeasuredTime(
        "gh200", "whole", "together", 1, 6552, 312.388, 34.671, 328.545
    ),
    # d 5, X part, 0.9985
    MeasuredTime("gh200", "whole", "apart", 1, 168, 54.440, 6.678, 56.640),
    # d 5, Z part, 0.9932
    MeasuredTime("gh200", "whole", "apart", 1, 192, 55.712, 6.664, 56.896),
    # d 7, X part, 0.9995
    MeasuredTime("gh200", "whole", "apart", 1, 480, 63.201, 7.745, 65.953),
    # d 7, Z part, 0.9997
    MeasuredTime("gh200", "whole", "apart", 1, 528, 63.061, 8.220, 66.433),
    # d 9, X part, 0.9981
    MeasuredTime("gh200", "whole", "apart", 1, 1040, 76.305, 10.804, 80.352),
    # d 9, Z part, 0.9901
    MeasuredTime("gh200", "whole", "apart", 1, 1120, 76.863, 12.163, 81.440),
    # d 11, X part, 1.0000
    MeasuredTime("gh200", "whole", "apart", 1, 1920, 88.026, 14.421, 95.968),
    # d 11, Z part, 0.9985
    MeasuredTime("gh200", "whole", "apart", 1, 2040, 97.686, 14.612, 96.736),
    # d 13, X part, 1.0000
    MeasuredTime("gh200", "whole", "apart", 1, 3192, 128.126, 19.358, 140.032),
    # d 13, Z part, 0.9990
    MeasuredTime("gh200", "whole", "apart", 1, 3360, 128.366, 20.625, 129.952),
)

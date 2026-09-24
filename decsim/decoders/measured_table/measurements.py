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

Every whole-card and MPS line explains its cell's times with r2 at
least 0.996 except the GH200 at d = 13, 0.683, where a few slow outliers
sit off the line. The slice lines explain r2 0.951 to 0.9999, and a
line leans on the long decodes: at a cell's median decode a slice's line
reads 1 percent high to 13 percent low, the whole card's 1 percent high
to 6 percent low. The 1g.10gb
line at d = 13 leans the hardest: it reads 522 us at the three
iterations its fastest region ran and crosses zero below two, while
that region took 1,406 us, which is why a line is floored at its cell's
fastest decode.
Cells with two or more decodes at once are not here: without MPS the
processes time-slice the GPU and a decode's time follows the others'
work, not its own iterations (their lines explain r2 0.11 to 0.96), so
no line prices them.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class MeasuredTime:
    """One cell's decode time as a line in its Relay-BP iterations."""

    device: str
    partition: str
    decodes_running: int
    detectors: int
    intercept_microseconds: float
    microseconds_per_iteration: float
    fastest_decode_microseconds: float

    def decode_microseconds(self, iterations: int) -> float:
        """The line at these iterations, floored at the fastest decode."""
        per_iteration = self.microseconds_per_iteration * iterations
        line_microseconds = self.intercept_microseconds + per_iteration
        return max(line_microseconds, self.fastest_decode_microseconds)


# Rows in device, partition, then region order; the comment above a row
# names the code distance of its region and its line's r2.
RELAY_BP_TIMES = (
    # d 5, 0.9962
    MeasuredTime("a100", "whole", 1, 360, 78.957, 9.762, 85.872),
    # d 7, 0.9971
    MeasuredTime("a100", "whole", 1, 1008, 106.435, 20.797, 122.491),
    # d 9, 0.9997
    MeasuredTime("a100", "whole", 1, 2160, 147.553, 26.852, 170.963),
    # d 11, 0.9999
    MeasuredTime("a100", "whole", 1, 3960, 189.257, 36.131, 277.895),
    # d 13, 0.9997
    MeasuredTime("a100", "whole", 1, 6552, 310.036, 48.329, 464.768),
    # d 5, 0.9971
    MeasuredTime("a100", "mps", 1, 360, 80.911, 9.839, 86.273),
    # d 9, 0.9985
    MeasuredTime("a100", "mps", 1, 2160, 155.414, 26.961, 171.383),
    # d 13, 0.9988
    MeasuredTime("a100", "mps", 1, 6552, 326.723, 48.319, 464.064),
    # d 5, 0.9514
    MeasuredTime("a100", "3g.40gb", 1, 360, 72.377, 9.527, 77.529),
    # d 7, 0.9810
    MeasuredTime("a100", "3g.40gb", 1, 1008, 101.534, 14.972, 112.184),
    # d 9, 0.9967
    MeasuredTime("a100", "3g.40gb", 1, 2160, 139.312, 27.765, 175.893),
    # d 11, 0.9999
    MeasuredTime("a100", "3g.40gb", 1, 3960, 200.997, 47.165, 328.342),
    # d 13, 0.9999
    MeasuredTime("a100", "3g.40gb", 1, 6552, 209.000, 79.085, 582.489),
    # d 5, 0.9997
    MeasuredTime("a100", "1g.10gb", 1, 360, 68.534, 15.951, 81.145),
    # d 7, 0.9981
    MeasuredTime("a100", "1g.10gb", 1, 1008, 98.295, 36.404, 136.857),
    # d 9, 0.9998
    MeasuredTime("a100", "1g.10gb", 1, 2160, 96.910, 76.640, 229.207),
    # d 11, 0.9999
    MeasuredTime("a100", "1g.10gb", 1, 3960, 96.049, 136.788, 502.685),
    # d 13, 0.9993
    MeasuredTime("a100", "1g.10gb", 1, 6552, -585.136, 369.200, 1405.683),
    # d 5, 0.9996
    MeasuredTime("gh200", "whole", 1, 360, 70.463, 7.219, 71.808),
    # d 7, 0.9969
    MeasuredTime("gh200", "whole", 1, 1008, 95.243, 11.044, 99.552),
    # d 9, 0.9999
    MeasuredTime("gh200", "whole", 1, 2160, 125.843, 16.437, 137.280),
    # d 11, 0.9999
    MeasuredTime("gh200", "whole", 1, 3960, 171.485, 23.870, 211.552),
    # d 13, 0.6833
    MeasuredTime("gh200", "whole", 1, 6552, 312.388, 34.671, 328.545),
)

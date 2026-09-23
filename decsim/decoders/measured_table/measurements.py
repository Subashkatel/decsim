"""Relay-BP decode times measured on NVIDIA GPUs, as a line in iterations.

Each row is one measured cell: a device, how it was shared (whole: one
process on the whole GPU; mps: one client of an MPS daemon), how many
decodes ran on it at once, and the region decoded, named by its
detector count. The time of one decode is intercept_microseconds plus
microseconds_per_iteration times the Relay-BP iterations it ran, the
least-squares line through the cell's 2,000 timed decodes.

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
The regions are decsim's memory circuit (Stim
surface_code:rotated_memory_z, p = 0.001 on all four noise channels) of
3d rounds, r_strong = r_com + 2 r_buf at r_com = r_buf = d (Toshio et
al. arXiv:2510.25222 lines 1246-1250), 2,000 Stim shots per distance.

Every line explains its cell's times with r2 at least 0.996 except the
GH200 at d = 13, 0.683, where a few slow outliers sit off the line.
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


# Rows in device, partition, then region order; the comment names the
# code distance of the region and the line's r2.
RELAY_BP_TIMES = (
    MeasuredTime("a100", "whole", 1, 360, 78.957, 9.762),  # d 5, 0.9962
    MeasuredTime("a100", "whole", 1, 1008, 106.435, 20.797),  # d 7, 0.9971
    MeasuredTime("a100", "whole", 1, 2160, 147.553, 26.852),  # d 9, 0.9997
    MeasuredTime("a100", "whole", 1, 3960, 189.257, 36.131),  # d 11, 0.9999
    MeasuredTime("a100", "whole", 1, 6552, 310.036, 48.329),  # d 13, 0.9997
    MeasuredTime("a100", "mps", 1, 360, 80.911, 9.839),  # d 5, 0.9971
    MeasuredTime("a100", "mps", 1, 2160, 155.414, 26.961),  # d 9, 0.9985
    MeasuredTime("a100", "mps", 1, 6552, 326.723, 48.319),  # d 13, 0.9988
    MeasuredTime("gh200", "whole", 1, 360, 70.463, 7.219),  # d 5, 0.9996
    MeasuredTime("gh200", "whole", 1, 1008, 95.243, 11.044),  # d 7, 0.9969
    MeasuredTime("gh200", "whole", 1, 2160, 125.843, 16.437),  # d 9, 0.9999
    MeasuredTime("gh200", "whole", 1, 3960, 171.485, 23.870),  # d 11, 0.9999
    MeasuredTime("gh200", "whole", 1, 6552, 312.388, 34.671),  # d 13, 0.6833
)

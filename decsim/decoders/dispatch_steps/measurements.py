"""A CUDA-Q dispatcher's steps on NVIDIA GPUs, each timed on its own.

KERNEL_TIMES are the Relay-BP kernels alone. The decodes measured_table's
lines were timed on (the same decoder, keys, regions and shots) were
traced under Nsight Systems 2025.3, each decode an NVTX range; a decode
runs one cooperative kernel, _bp_decoder_gpu_decode_csr_relay, and its
kernel time is a line in the Relay-BP iterations it ran with r2 0.9995
to 1.0 on every region, floored at the fastest traced kernel as
measured_table floors its lines. The line's slope is measured_table's
per-iteration slope to within 3 percent on the GH200 and 11 percent on
the A100; what the kernels and copies leave out of one decode() call,
52 to 278 microseconds at the median, is the Python binding's host
work between the calls, which a dispatcher does not run.

CARDS are the rest of a path, from a CUDA microbenchmark that times
each piece alone with a stream sync after it (the medians of 2,000
timings). On the host path, launch is an empty graph's launch and sync,
the one sync a worker pays per decode, and each copy is the GPU's own
time for the smallest copy the trace saw in that direction: inside a
graph a copy is a node with no sync of its own. A request or an answer
through d = 13 is under 1 kB, and in the microbenchmark a copy's time
is flat to within 0.1 microseconds from 16 bytes to 1 kB, so a copy is
one number. On the device path the dispatcher reads the ring in place,
so nothing is copied. Fire is two round trips from a mapped 16-byte
slot on the GH200 apart, both through cuda-quantum's dispatch_kernel.cu
with a handler that answers at once: the dispatcher firing a child
graph whose one kernel answers, 10.112 microseconds at the median, less
the dispatcher calling the same answer in place, the echo, 4.576
microseconds, which the link card prices (the StrongBackend rule).

Devices: an NVIDIA A100-SXM4-80GB (x86 host) and a GH200 144G HBM3e
(Grace host, NVLink-C2C), as measured_table names them.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class KernelTime:
    """One region's Relay-BP kernel time as a line in its iterations."""

    device: str
    detectors: int
    intercept_microseconds: float
    microseconds_per_iteration: float
    fastest_kernel_microseconds: float

    def decode_microseconds(self, iterations: int) -> float:
        """The line at these iterations, floored at the fastest kernel."""
        per_iteration = self.microseconds_per_iteration * iterations
        line_microseconds = self.intercept_microseconds + per_iteration
        return max(line_microseconds, self.fastest_kernel_microseconds)


@dataclasses.dataclass(frozen=True)
class PathCard:
    """A device's launch and copies on one dispatcher path."""

    launch_microseconds: float
    copy_in_microseconds: float
    copy_out_microseconds: float


# Rows in device, then region order; the comment names the region's code
# distance.
KERNEL_TIMES = (
    # d 5
    KernelTime("a100", 360, 18.120, 8.700, 26.080),
    # d 7
    KernelTime("a100", 1008, 28.932, 19.787, 41.696),
    # d 9
    KernelTime("a100", 2160, 25.396, 26.634, 61.664),
    # d 11
    KernelTime("a100", 3960, 9.560, 35.730, 138.529),
    # d 13
    KernelTime("a100", 6552, 34.665, 48.418, 214.177),
    # d 5
    KernelTime("gh200", 360, 11.840, 7.243, 19.008),
    # d 7
    KernelTime("gh200", 1008, 14.416, 10.866, 24.992),
    # d 9
    KernelTime("gh200", 2160, 15.102, 15.933, 31.456),
    # d 11
    KernelTime("gh200", 3960, 15.338, 23.590, 88.000),
    # d 13
    KernelTime("gh200", 6552, 13.354, 35.019, 159.104),
)

CARDS = {
    ("a100", "host"): PathCard(5.691, 1.184, 1.472),
    ("gh200", "host"): PathCard(5.664, 1.056, 1.344),
    ("gh200", "device"): PathCard(5.536, 0.0, 0.0),
}

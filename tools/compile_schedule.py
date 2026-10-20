"""Compile an OpenQASM circuit into a SWIPER lattice-surgery schedule.

    python tools/compile_schedule.py SWIPER_CHECKOUT TOOLS_FOLDER QASM OUT

SWIPER's pipeline (2412.05115 Sec. 5.1, Fig. 9), run by SWIPER's own code
(github.com/jviszlai/swiper benchmarks/benchmark_circuits.py
QASMBenchmark): Qiskit transpiles the circuit, Cirq and pyLIQTR decompose
it to Clifford+Rz, Gridsynth approximates each Rz to 1e-10, the Lattice
Surgery Compiler maps and routes it on the EDPC layout (lsqecc_slicer -L
edpc --disttime 1 --nostagger -P wave) and cirq_to_ls reads its slices
into instructions. OUT is the .lss that frontends/lattice_surgery.py
reads.

Runs in SWIPER's environment as its lock stood when it regenerated its
cached schedules (commit 482bfd28, 2024-11-26): Python 3.11, cirq 1.4.0,
qualtran 0.4.0, pyLIQTR 1.3.1 and qiskit 1.2.4; a later qiskit merges
rotations SWIPER's schedules keep, and so compiles fewer T gates.
TOOLS_FOLDER holds the gridsynth (newsynth 0.3.0.4) and lsqecc_slicer
(liblsqecc d76c318, 2024-10-25, the last before that date) executables;
cirq_to_ls runs them as benchmarks/gridsynth and benchmarks/lsqecc_slicer
from its working folder, so a scratch folder that links them there is
the working folder. So pinned, adder_n18 and qpeexact_5 compile to
SWIPER's T counts and qft_10 to within 0.4%; gridsynth's random seed
and the router's choices move the rest.
"""

import os
import pathlib
import sys
import tempfile

EXECUTABLES = ("gridsynth", "lsqecc_slicer")


def main(
    checkout: pathlib.Path,
    tools_folder: pathlib.Path,
    qasm_path: pathlib.Path,
    out: pathlib.Path,
) -> None:
    """The circuit's schedule, written as SWIPER writes one."""
    folder = str(checkout)
    sys.path.insert(0, folder)
    import benchmarks.benchmark_circuits as benchmark_circuits

    qasm = qasm_path.resolve()
    written = out.resolve()
    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = pathlib.Path(scratch)
        _link_executables(scratch_path, tools_folder)
        os.chdir(scratch_path)
        benchmark = benchmark_circuits.QASMBenchmark(str(qasm))
    schedule = benchmark.get_schedule()
    text = str(schedule)
    written.write_text(text)


def _link_executables(scratch: pathlib.Path, tools_folder: pathlib.Path):
    """The executables where cirq_to_ls runs them, under benchmarks/."""
    linked = scratch / "benchmarks"
    linked.mkdir()
    for name in EXECUTABLES:
        target = tools_folder / name
        resolved = target.resolve()
        link = linked / name
        link.symlink_to(resolved)


if __name__ == "__main__":
    swiper_checkout = pathlib.Path(sys.argv[1])
    executables_folder = pathlib.Path(sys.argv[2])
    circuit_file = pathlib.Path(sys.argv[3])
    out_file = pathlib.Path(sys.argv[4])
    main(swiper_checkout, executables_folder, circuit_file, out_file)

r"""The windows correlated matching decodes on decsim's machines, kept to time.

B1 stage 2 (specs/adaptive-switching/experiments.md, decision D4) times
correlated matching on the windows decsim hands it. This streams the
five results' correlated matching alone (five_results/config.py) as a
long memory (decsim stream's machine, experiments/stream_command.py),
and a recording row keeps every window the strong pool decodes: the
window's model as the row reads it (minimum_weight_perfect_matching
correlated_model), one Stim file for each distinct model, named by a
digest of the arrays the model is written from, and the rows its
syndrome fires. A stream moves a model in time for each window, so the
row writes a model the first time it meets it and keeps no model. A
strong decode reads 2 d rounds in a sliding window (d committed, d
looked ahead), which a sliding redo and speculation also decode, or 3 d
rounds in a double window's region
(r_com + 2 r_buf, Toshio et al. 2510.25222 line 1250). A stream does
not build the double window, so the 3 d cell is a sliding window of d
committed and 2 d looked ahead: the same rounds and detectors, its far
face open where the region's is pinned. check_machine.py prices real
double-window regions by the fitted law.

The chip's noise is SD6 at p = 0.003 times a factor from the second
round on (FlipStreamDevice's drift; SD6 is linear in p), while the
decoders keep p = 0.003's window models, as in a burst they are not told
of (specs/adaptive-switching/bursts.md). Factor 1 is the calm chip; 2
is Table IX's other point, p = 0.006 (2604.12841 p.22); 4 and 10 are
the burst peaks bursts.md sweeps, here over the whole patch, so a window
holds more events than a burst's window of the same peak does. Factor
35 saturates the detectors and is left out.

time_windows.py times the kept windows on one pinned core.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md),
through Slurm:

```
export PYTHONPATH=$PWD/../..
python capture_windows.py --out DIR --distances 5 9 13 \
    --factors 1 2 4 10 --windows 400
```
"""

import argparse
import csv
import dataclasses
import hashlib
import pathlib

import numpy

import decsim
import decsim.decoders.decoders as decoders
import decsim.decoders.minimum_weight_perfect_matching.decoder as matching
import decsim.experiments.stream_command as stream_command
import decsim.experiments.sweeps as sweeps
import decsim.presets.machines as machines
import decsim.qpu.flip_stream_device as flip_stream_device
import decsim.records.fault_model_contracts as fault_models
import decsim.windows.schemes.sliding as sliding

five_results = sweeps.load_beside(__file__, "../five_results/config.py")
CALM_PROBABILITY = 0.003
SCHEMES = ("sliding", "double")
# the noise rises from the round after this one (FlipStreamDevice)
DRIFT_ROUND = 1
# the stream's first and last windows read fewer rounds, so it holds
# this many windows more than the full ones asked for
EDGE_WINDOWS = 3
STREAM_SEED = 1
COLUMNS = (
    "distance",
    "scheme",
    "factor",
    "physical_error_probability",
    "first_commit_round",
    "rounds",
    "detectors",
    "events",
    "model",
    "fired_rows",
)
# what the recording row kept of each window decoded in the stream that
# is running: (model file, first commit round, rounds, syndrome); the
# capture loop empties it after each stream
_DECODED_WINDOWS: list = []


class RecordingCorrelatedMatching(matching.CorrelatedPyMatchingDecoder):
    """Correlated matching at its preset time, keeping each window decoded."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The row the machine's strong pool held, at its preset time.

        models_folder is where the row writes each distinct model.
        """

        preset_latency_microseconds: float
        models_folder: str
        # the word the reports name this row by
        name = "recording_correlated_pymatching"

        def build(self) -> "RecordingCorrelatedMatching":
            """A fresh decoder at this preset time."""
            latency_model = decoders.PresetLatencyDecoder(
                self.preset_latency_microseconds
            )
            models_folder = pathlib.Path(self.models_folder)
            return RecordingCorrelatedMatching(latency_model, models_folder)

    def __init__(self, latency_model, models_folder: pathlib.Path) -> None:
        matching.CorrelatedPyMatchingDecoder.__init__(self, latency_model)
        self.models_folder = models_folder

    def decode_window(self, backend, model, faults, syndrome):
        """The row's own answer; the window is kept to time later."""
        model_file = _written_model(self.models_folder, faults, model)
        rounds = _round_count(model)
        kept_syndrome = syndrome.copy()
        window = (model_file, model.first_commit_round, rounds, kept_syndrome)
        _DECODED_WINDOWS.append(window)
        return matching.CorrelatedPyMatchingDecoder.decode_window(
            self, backend, model, faults, syndrome
        )


def capture_machine(
    out: pathlib.Path, scheme: str, distance: int, factor: float
):
    """Correlated matching alone on windows of the scheme's rounds."""
    machine = five_results.alone("strong_alone", distance, CALM_PROBABILITY)
    strong_pool = machine.part(machines.STRONG_POOL)
    preset = strong_pool.settings.algorithm.preset_latency_microseconds
    models_folder = out / "models"
    recording = RecordingCorrelatedMatching.Settings(preset, str(models_folder))
    machine = machine.with_settings(machines.STRONG_POOL, algorithm=recording)
    read_distances = five_results.STRONG_DECODE_DISTANCES_BY_SCHEME[scheme]
    buffer_distances = read_distances - 1
    buffer_rounds = buffer_distances * distance
    shape = sliding.SlidingWindowScheme.Settings(
        commit_rounds=distance, buffer_rounds=buffer_rounds
    )
    windows = machine.held(machines.WINDOWS)
    shaped = dataclasses.replace(windows, scheme=shape)
    machine = machine.with_held(machines.WINDOWS, shaped)
    source = _drifted_source(factor)
    qpu = machine.held(machines.QPU)
    flipped = dataclasses.replace(qpu, source=source)
    return machine.with_held(machines.QPU, flipped)


def capture_cell(
    out: pathlib.Path,
    writer: csv.DictWriter,
    scheme: str,
    distance: int,
    factors: list,
    windows: int,
) -> None:
    """One stream at each factor, long enough for the windows asked for.

    A window commits d rounds, so a stream of that many windows more
    EDGE_WINDOWS holds them; its first and last windows read fewer
    rounds and are kept too.
    """
    stream_windows = windows + EDGE_WINDOWS
    rounds = stream_windows * distance
    for factor in factors:
        machine = capture_machine(out, scheme, distance, factor)
        name = f"capture_{scheme}_d{distance}_f{factor:g}"
        task = decsim.Task(name, machine)
        stream_command.stream_record(task, rounds, 0, STREAM_SEED)
        cell = (distance, scheme, factor)
        rows = _drained_rows(cell)
        writer.writerows(rows)
        print(f"d={distance} {scheme} factor {factor:g}: {len(rows)} windows")


def main() -> None:
    """Keep each distance's and scheme's windows at every factor."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--distances", type=int, nargs="+", default=[5, 9])
    parser.add_argument(
        "--factors", type=float, nargs="+", default=[1.0, 2.0, 4.0, 10.0]
    )
    parser.add_argument("--schemes", nargs="+", default=list(SCHEMES))
    parser.add_argument("--windows", type=int, default=400)
    arguments = parser.parse_args()
    out = pathlib.Path(arguments.out)
    models = out / "models"
    models.mkdir(parents=True, exist_ok=True)
    file_name = _windows_file_name(arguments)
    path = out / file_name
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, COLUMNS)
        writer.writeheader()
        _capture_every_cell(out, handle, writer, arguments)
    print(f"wrote {path}")


def _windows_file_name(arguments) -> str:
    """The cells the run keeps, so runs of other cells write other files."""
    distance_names = [str(distance) for distance in arguments.distances]
    factor_names = [f"{factor:g}" for factor in arguments.factors]
    distance_text = "_".join(distance_names)
    scheme_text = "_".join(arguments.schemes)
    factor_text = "_".join(factor_names)
    return f"windows_d{distance_text}_{scheme_text}_f{factor_text}.csv"


def _capture_every_cell(
    out: pathlib.Path, handle, writer: csv.DictWriter, arguments
) -> None:
    """Every distance and scheme, each written out as it ends."""
    for distance in arguments.distances:
        for scheme in arguments.schemes:
            capture_cell(
                out,
                writer,
                scheme,
                distance,
                arguments.factors,
                arguments.windows,
            )
            handle.flush()


def _drifted_source(factor: float):
    """The flip stream source, its noise times the factor after round one."""
    if factor == 1:
        return flip_stream_device.FlipStreamDevice.Settings()
    return flip_stream_device.FlipStreamDevice.Settings(
        drift_round=DRIFT_ROUND, drift_factor=factor
    )


def _drained_rows(cell: tuple) -> list:
    """One row for each window decoded in the stream; the list is emptied."""
    rows = []
    for window in _DECODED_WINDOWS:
        row = _window_row(cell, window)
        rows.append(row)
    _DECODED_WINDOWS.clear()
    return rows


def _written_model(folder: pathlib.Path, faults, model) -> str:
    """The window's model file, written the first time it is met.

    correlated_model writes the model from the graphlike check and
    priors, the physical priors and the link between them, so a digest
    of those arrays names it.
    """
    physical = model.require_faults(fault_models.FaultRepresentation.PHYSICAL)
    link = model.physical_to_graphlike_detector_projection
    digest = hashlib.sha256()
    arrays = (
        faults.check.indices,
        faults.check.indptr,
        faults.priors,
        physical.priors,
        link.indices,
        link.indptr,
    )
    for array in arrays:
        contiguous = numpy.ascontiguousarray(array)
        digest.update(contiguous)
    name = digest.hexdigest()
    file_name = f"{name[:16]}.dem"
    path = folder / file_name
    if path.exists():
        return file_name
    detector_error_model = matching.correlated_model(faults, model)
    text = str(detector_error_model)
    path.write_text(text)
    return file_name


def _window_row(cell: tuple, window: tuple) -> dict:
    distance, scheme, factor = cell
    model_file, first_commit_round, rounds, syndrome = window
    fired = numpy.flatnonzero(syndrome)
    fired_names = [str(row) for row in fired]
    probability = CALM_PROBABILITY * factor
    return {
        "distance": distance,
        "scheme": scheme,
        "factor": factor,
        "physical_error_probability": f"{probability:g}",
        "first_commit_round": first_commit_round,
        "rounds": rounds,
        "detectors": len(syndrome),
        "events": len(fired),
        "model": model_file,
        "fired_rows": " ".join(fired_names),
    }


def _round_count(model) -> int:
    """The rounds the window's rows lie in."""
    rounds = set()
    for detector_id in model.detector_ids:
        round_index, _position = model.defect_positions[detector_id]
        rounds.add(round_index)
    return len(rounds)


if __name__ == "__main__":
    main()

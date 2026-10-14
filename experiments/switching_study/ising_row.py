"""NVIDIA's Ising pre-decoder, then matching, on one window: a plugin row.

Chamberland et al. arXiv 2604.12841: a 3D convolutional network reads a
window's detection events laid on a D x D grid per round (Sec. IV.2.1),
predicts data-qubit flips and measurement flips (Sec. IV.2.6), and
PyMatching matches the detection events those flips leave (Eqs. 44-53).
The released Fast model (Hugging Face
nvidia/Ising-Decoder-SurfaceCode-1-Fast) is loaded with NVIDIA's own
loader, and the stabilizer layout is NVIDIA's (Ising-Decoding
code/evaluation/logical_error_rate.py, _build_stab_maps). The model was
trained at d = T = 9 on NVIDIA's circuit and noise, not on SD6; this row
runs it as released.

Each predicted flip becomes the window's own graphlike column with the
same detectors, the residual is matched on the window's graph, and the
row returns both sets of columns. The answer therefore explains the
syndrome whatever the network predicts; the network only moves which
explanation is chosen. A flip whose detectors no column of the window
has is not applied.

decsim imports nothing of this file; a run passes
IsingMatchingDecoder.Settings as a pool's algorithm. NVIDIA's code and
torch are imported when the first window compiles.
"""

import dataclasses
import functools
import itertools
import sys
from typing import Optional

import numpy
import pymatching

import decsim.config as config
import decsim.decoders.decoder as decoder_module
import decsim.decoders.minimum_weight_perfect_matching.weights as weights
import decsim.detector_error_model.basis_split as basis_split
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models

# The network's four output channels (model card "Outputs"): Z flips on
# data qubits, X flips on data qubits, then X- and Z-stabilizer
# measurement flips.
Z_DATA_CHANNEL = 0
X_DATA_CHANNEL = 1
X_MEASUREMENT_CHANNEL = 2
Z_MEASUREMENT_CHANNEL = 3
ORIENTATIONS = ("XV", "XH", "ZV", "ZH")
WINDOW_ROUNDS = "window_round_count"
PREDICTED_FLIPS = "predicted_flip_count"
APPLIED_FLIPS = "applied_flip_count"


@dataclasses.dataclass(frozen=True)
class IsingMatchingDecoderSettings:
    """Where NVIDIA's code and model are, and how a decode is timed.

    repository_path is the Ising-Decoding checkout, checkpoint_path the
    .safetensors file. microseconds_per_round prices a decode at that
    time per window round (2604.12841 Table 7 gives 3.72 us a round at
    d = 13, p = 0.003, Fast model plus PyMatching); None times the call
    on the host.

    layout names the placement of the window on NVIDIA's grid, an
    orientation and a symmetry of the square (_square_symmetries). The
    eight placements that fit Stim's rotated memory agree on which data
    qubits each stabilizer checks but differ in direction, and the
    network was trained on one gate order, so the direction matters.
    ("XV", 6) is the one with fewest failures on 2,000 calibration shots
    (d = 9, SD6 at p = 0.006, seed 5, ising_layouts.py): 146 against
    PyMatching's 184, the other seven 217 to 423.
    applied_channels are the output channels whose flips are applied,
    all four as released; fewer is a diagnostic.
    """

    repository_path: str
    checkpoint_path: str
    device: str = "cuda"
    microseconds_per_round: Optional[float] = None
    layout: tuple = ("XV", 6)
    applied_channels: tuple = (0, 1, 2, 3)
    # the word the reports name this row by
    name = "ising_pymatching"

    def build(self) -> "IsingMatchingDecoder":
        """A fresh decoder of these settings."""
        return IsingMatchingDecoder(self)


class IsingMatchingDecoder(decoder_module.WindowDecoderBase):
    """Decode one window with the Ising pre-decoder, then PyMatching."""

    fault_model_requirement = (
        fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED.joined(
            fault_models.DETECTOR_BASES_REQUIRED
        )
    )
    fault_representation = fault_models.FaultRepresentation.GRAPHLIKE
    Settings = IsingMatchingDecoderSettings

    def __init__(self, settings: IsingMatchingDecoderSettings):
        decoder_module.WindowDecoderBase.__init__(self)
        self.settings = settings
        self.compile_key = (IsingMatchingDecoder, settings)

    def compile(
        self,
        faults: fault_models.PlacedFaultModel,
        model: fault_models.WindowErrorModel,
    ) -> "_WindowBackend":
        """The window's grid, its column lookup and its matching graph."""
        network = _network(
            self.settings.repository_path,
            self.settings.checkpoint_path,
            self.settings.device,
        )
        grid = _window_grid(
            self.settings.repository_path, model, self.settings.layout
        )
        columns_by_rows = _columns_by_rows(faults)
        check = faults.check.copy()
        edge_weights = weights.matching_weights(faults.priors)
        error_probabilities = weights.finite_priors(faults.priors)
        matching = pymatching.Matching.from_check_matrix(
            check,
            weights=edge_weights,
            error_probabilities=error_probabilities,
            merge_strategy="independent",
        )
        return _WindowBackend(network, grid, columns_by_rows, matching)

    def decode_window(
        self,
        backend: "_WindowBackend",
        model: fault_models.WindowErrorModel,
        faults: fault_models.PlacedFaultModel,
        syndrome: numpy.ndarray,
    ) -> decoding_records.WindowDecode:
        """The applied flips' columns plus the matching of what they leave."""
        del model
        predicted = _predicted_flips(backend, syndrome)
        channels = self.settings.applied_channels
        flips = [flip for flip in predicted if flip[0] in channels]
        column_count = faults.check.shape[1]
        applied = _applied_columns(backend, flips, column_count)
        lit = faults.check @ applied
        explained = numpy.remainder(lit, 2)
        explained = explained.astype(numpy.uint8)
        residual = numpy.bitwise_xor(syndrome, explained)
        try:
            matched = backend.matching.decode(residual)
        except ValueError as error:
            if "perfect matching" not in str(error):
                raise
            matched = numpy.zeros(column_count, dtype=numpy.uint8)
        selected = numpy.bitwise_xor(applied, matched)
        evidence = {
            WINDOW_ROUNDS: float(backend.grid.round_count),
            PREDICTED_FLIPS: float(len(predicted)),
            APPLIED_FLIPS: float(applied.sum()),
        }
        return decoding_records.WindowDecode(selected, evidence=evidence)

    def ticks_after_decode(
        self,
        result: Optional[decoding_records.DecodeResult],
        elapsed_nanoseconds: int,
        now: int,
    ) -> int:
        """The stated time per round times the window's rounds, else the host's."""
        per_round = self.settings.microseconds_per_round
        if per_round is None or result is None:
            return decoder_module.WindowDecoderBase.ticks_after_decode(
                self, result, elapsed_nanoseconds, now
            )
        round_count = result.evidence[WINDOW_ROUNDS]
        microseconds = per_round * round_count
        return config.microseconds_to_ticks(microseconds)


@dataclasses.dataclass(frozen=True)
class _WindowGrid:
    """Where each window row sits on NVIDIA's grid, and back.

    stabilizer_by_row and time_by_row place row r at stabilizer index
    stabilizer_by_row[r] of its type (0 for X, 1 for Z) at local round
    time_by_row[r]; row_by_place inverts that. cell_by_stabilizer[type]
    is NVIDIA's stab_x or stab_z (stabilizer to grid cell),
    stabilizers_by_cell[type] lists the stabilizers of that type each
    data cell is in (NVIDIA's Hx or Hz by column), present[type] is
    NVIDIA's normalized weight map.
    """

    distance: int
    round_count: int
    type_by_row: numpy.ndarray
    stabilizer_by_row: numpy.ndarray
    time_by_row: numpy.ndarray
    row_by_place: dict
    cell_by_stabilizer: tuple
    stabilizers_by_cell: tuple
    present: tuple


@dataclasses.dataclass(frozen=True)
class _WindowBackend:
    network: object
    grid: _WindowGrid
    columns_by_rows: dict
    matching: pymatching.Matching


def _predicted_flips(backend: _WindowBackend, syndrome) -> list:
    """(channel, round, cell) of every flip the network predicts."""
    import torch

    grid = backend.grid
    inputs = _network_input(grid, syndrome)
    parameter = next(backend.network.parameters())
    tensor = torch.from_numpy(inputs)
    tensor = tensor.to(device=parameter.device, dtype=parameter.dtype)
    with torch.no_grad():
        logits = backend.network(tensor)
    # the released threshold: a flip where the logit is at least 0
    # (logical_error_rate.py sample_predictions, test.th_data = th_syn = 0)
    predicted = logits[0] >= 0
    positions = predicted.nonzero()
    positions = positions.cpu()
    positions = positions.numpy()
    flips = []
    for channel, time, row, column in positions:
        cell = int(row) * grid.distance + int(column)
        flips.append((int(channel), int(time), cell))
    return flips


def _network_input(grid: _WindowGrid, syndrome) -> numpy.ndarray:
    """(1, 4, T, D, D): X and Z detection events, then the present maps.

    A round with no detector of a type, the first round of a memory and
    the round of the final data readout, has that type's present map
    zeroed, as NVIDIA zeroes the non-basis maps there
    (predecoder_transform.py, _predecoder_transform_core).
    """
    distance = grid.distance
    cell_count = distance * distance
    shape = (4, grid.round_count, cell_count)
    inputs = numpy.zeros(shape, dtype=numpy.float32)
    has_type = numpy.zeros((2, grid.round_count), dtype=bool)
    for row, bit in enumerate(syndrome):
        stabilizer_type = grid.type_by_row[row]
        time = grid.time_by_row[row]
        has_type[stabilizer_type, time] = True
        if not bit:
            continue
        stabilizer = grid.stabilizer_by_row[row]
        cell = grid.cell_by_stabilizer[stabilizer_type][stabilizer]
        inputs[stabilizer_type, time, cell] = 1.0
    for stabilizer_type in (0, 1):
        present_channel = stabilizer_type + 2
        rounds_with_type = has_type[stabilizer_type]
        weight_map = grid.present[stabilizer_type]
        inputs[present_channel, rounds_with_type] = weight_map
    grid_shape = (1, 4, grid.round_count, distance, distance)
    return inputs.reshape(grid_shape)


def _applied_columns(
    backend: _WindowBackend, flips: list, column_count: int
) -> numpy.ndarray:
    """The window columns that carry the predicted flips, as a 0/1 vector."""
    applied = numpy.zeros(column_count, dtype=numpy.uint8)
    for flip in flips:
        rows = _flip_rows(backend.grid, flip)
        column = backend.columns_by_rows.get(rows)
        if column is None:
            continue
        applied[column] ^= 1
    return applied


def _flip_rows(grid: _WindowGrid, flip: tuple) -> frozenset:
    """The window rows one predicted flip lights (2604.12841 Eqs. 44-53).

    A data flip at round t lights its neighbouring stabilizers of the
    other type at t: a Z flip the X stabilizers (Hx), an X flip the Z
    ones (Hz). A measurement flip of a stabilizer at t lights it at t
    and at t + 1. A place outside the window lights nothing.
    """
    channel, time, cell = flip
    places = []
    if channel == Z_DATA_CHANNEL:
        for stabilizer in grid.stabilizers_by_cell[0][cell]:
            places.append((0, stabilizer, time))
    if channel == X_DATA_CHANNEL:
        for stabilizer in grid.stabilizers_by_cell[1][cell]:
            places.append((1, stabilizer, time))
    if channel in (X_MEASUREMENT_CHANNEL, Z_MEASUREMENT_CHANNEL):
        places = _measurement_places(grid, channel, time, cell)
    rows = []
    for place in places:
        row = grid.row_by_place.get(place)
        if row is not None:
            rows.append(row)
    return frozenset(rows)


def _measurement_places(grid: _WindowGrid, channel: int, time, cell) -> list:
    """The stabilizer at that cell, at t and t + 1, or none at all."""
    stabilizer_type = channel - X_MEASUREMENT_CHANNEL
    cells = grid.cell_by_stabilizer[stabilizer_type]
    stabilizers = numpy.flatnonzero(cells == cell)
    if not len(stabilizers):
        return []
    stabilizer = int(stabilizers[0])
    next_time = time + 1
    return [
        (stabilizer_type, stabilizer, time),
        (stabilizer_type, stabilizer, next_time),
    ]


def _columns_by_rows(faults: fault_models.PlacedFaultModel) -> dict:
    """Each in-window detector set's most likely column."""
    best_by_rows = {}
    for column in range(faults.check.shape[1]):
        rows = basis_split.column_rows(faults.check, column)
        key = frozenset(rows)
        prior = float(faults.priors[column])
        best = best_by_rows.get(key)
        if best is not None and best[1] >= prior:
            continue
        best_by_rows[key] = (column, prior)
    columns = {}
    for key, best in best_by_rows.items():
        columns[key] = best[0]
    return columns


def layouts(repository_path: str, model: fault_models.WindowErrorModel):
    """Every (orientation, symmetry index) that fits the window's rows."""
    coordinates = numpy.asarray(model.detector_coordinates, dtype=float)
    distance = int(coordinates[:, 0].max()) // 2
    type_by_row = _types_by_row(model)
    supports = _supports_by_row(coordinates, distance)
    fitting = _fitting_layouts(repository_path, distance, type_by_row, supports)
    return [(orientation, index) for orientation, index, _, _ in fitting]


def _window_grid(
    repository_path: str, model: fault_models.WindowErrorModel, layout: int
) -> _WindowGrid:
    """Place the window's rows on NVIDIA's grid by Stim's coordinates.

    Stim's rotated surface code puts data qubits at odd (x, y) and each
    stabilizer's measure qubit at even (x, y) between the data qubits it
    checks; a detector's coordinates are its stabilizer's (x, y) and its
    round (Stim's surface_code generator). The orientation is found by
    matching every stabilizer's data cells to a row of NVIDIA's Hx or Hz.
    """
    coordinates = numpy.asarray(model.detector_coordinates, dtype=float)
    distance = int(coordinates[:, 0].max()) // 2
    type_by_row = _types_by_row(model)
    supports = _supports_by_row(coordinates, distance)
    fitting = _fitting_layouts(repository_path, distance, type_by_row, supports)
    if not fitting:
        raise ValueError(
            f"no orientation of NVIDIA's d = {distance} layout matches the "
            "window's stabilizers"
        )
    chosen = None
    for orientation, index, stabilizers, maps in fitting:
        if (orientation, index) == layout:
            chosen = (stabilizers, maps)
    if chosen is None:
        raise ValueError(
            f"the layout {layout} does not fit the window's stabilizers; "
            f"these do: {[fit[:2] for fit in fitting]}"
        )
    stabilizer_by_row, maps = chosen
    times = coordinates[:, 2].astype(int)
    first_time = times.min()
    time_by_row = times - first_time
    round_count = int(time_by_row.max()) + 1
    row_by_place = {}
    for row, stabilizer in enumerate(stabilizer_by_row):
        place = (int(type_by_row[row]), int(stabilizer), int(time_by_row[row]))
        row_by_place[place] = row
    return _WindowGrid(
        distance=distance,
        round_count=round_count,
        type_by_row=type_by_row,
        stabilizer_by_row=stabilizer_by_row,
        time_by_row=time_by_row,
        row_by_place=row_by_place,
        cell_by_stabilizer=maps.cell_by_stabilizer,
        stabilizers_by_cell=maps.stabilizers_by_cell,
        present=maps.present,
    )


def _types_by_row(model: fault_models.WindowErrorModel) -> numpy.ndarray:
    types = []
    for detector_id in model.detector_ids:
        basis = model.detector_bases[detector_id]
        type_index = 0
        if basis == "Z":
            type_index = 1
        types.append(type_index)
    return numpy.asarray(types, dtype=int)


def _supports_by_row(coordinates: numpy.ndarray, distance: int) -> list:
    """The data cells (row, column) each detector's stabilizer checks."""
    limit = 2 * distance - 1
    supports = []
    for x, y, _time in coordinates[:, :3]:
        cells = []
        for step_x, step_y in itertools.product((-1, 1), (-1, 1)):
            data_x = int(x) + step_x
            data_y = int(y) + step_y
            if 1 <= data_x <= limit and 1 <= data_y <= limit:
                cells.append(((data_y - 1) // 2, (data_x - 1) // 2))
        supports.append(tuple(cells))
    return supports


@dataclasses.dataclass(frozen=True)
class _NvidiaMaps:
    cell_by_stabilizer: tuple
    stabilizers_by_cell: tuple
    present: tuple
    stabilizer_by_support: tuple


def _fitting_layouts(repository_path, distance, type_by_row, supports):
    """(orientation, symmetry index, stabilizers, maps) of every fit."""
    symmetries = _square_symmetries(distance)
    fitting = []
    for orientation in ORIENTATIONS:
        maps = _nvidia_maps(repository_path, distance, orientation)
        for index, symmetry in enumerate(symmetries):
            stabilizers = _stabilizers_under(
                maps, symmetry, type_by_row, supports
            )
            if stabilizers is not None:
                fitting.append((orientation, index, stabilizers, maps))
    return fitting


def _stabilizers_under(maps, symmetry, type_by_row, supports):
    """NVIDIA's index of every row's stabilizer, or None if one has none."""
    stabilizers = []
    for row, cells in enumerate(supports):
        moved = frozenset(symmetry(cell) for cell in cells)
        stabilizer_type = int(type_by_row[row])
        stabilizer = maps.stabilizer_by_support[stabilizer_type].get(moved)
        if stabilizer is None:
            return None
        stabilizers.append(stabilizer)
    return numpy.asarray(stabilizers, dtype=int)


def _square_symmetries(distance: int) -> list:
    """The eight maps of the D x D square onto itself, as cell indices."""
    last = distance - 1

    def symmetry(transpose, flip_row, flip_column):
        def moved(cell):
            row, column = cell
            if transpose:
                row, column = column, row
            if flip_row:
                row = last - row
            if flip_column:
                column = last - column
            return row * distance + column

        return moved

    flags = itertools.product((False, True), repeat=3)
    return [symmetry(*flag) for flag in flags]


@functools.cache
def _nvidia_maps(repository_path: str, distance: int, orientation: str):
    """NVIDIA's stabilizer layout for one distance and orientation."""
    _import_path(repository_path)
    import evaluation.logical_error_rate as nvidia_evaluation
    import qec.surface_code.data_mapping as nvidia_mapping

    maps = nvidia_evaluation._build_stab_maps(distance, orientation)
    checks = (maps["Hx_i32"].numpy(), maps["Hz_i32"].numpy())
    cell_by_stabilizer = (
        maps["stab_x"].numpy(),
        maps["stab_z"].numpy(),
    )
    stabilizers_by_cell = tuple(_stabilizers_by_cell(check) for check in checks)
    stabilizer_by_support = tuple(_by_support(check) for check in checks)
    present_x = nvidia_mapping.normalized_weight_mapping_Xstab_memory(
        distance, orientation
    )
    present_z = nvidia_mapping.normalized_weight_mapping_Zstab_memory(
        distance, orientation
    )
    present = (numpy.asarray(present_x), numpy.asarray(present_z))
    return _NvidiaMaps(
        cell_by_stabilizer, stabilizers_by_cell, present, stabilizer_by_support
    )


def _stabilizers_by_cell(check: numpy.ndarray) -> list:
    stabilizers = []
    for cell in range(check.shape[1]):
        rows = numpy.flatnonzero(check[:, cell])
        stabilizers.append(tuple(int(row) for row in rows))
    return stabilizers


def _by_support(check: numpy.ndarray) -> dict:
    by_support = {}
    for stabilizer in range(check.shape[0]):
        cells = numpy.flatnonzero(check[stabilizer])
        support = frozenset(int(cell) for cell in cells)
        by_support[support] = stabilizer
    return by_support


@functools.cache
def _network(repository_path: str, checkpoint_path: str, device: str):
    """The released model, once a process, in evaluation mode."""
    _import_path(repository_path)
    import export.safetensors_utils as nvidia_safetensors

    network, _metadata = nvidia_safetensors.load_safetensors(
        checkpoint_path, device=device
    )
    network.eval()
    if device != "cpu":
        return network
    # the weights are fp16; a CPU runs the same weights in fp32, since
    # the card validates GPUs only and CPU half-precision 3D convolution
    # is slow where PyTorch has it at all
    return network.float()


def _import_path(repository_path: str) -> None:
    code_path = f"{repository_path}/code"
    if code_path not in sys.path:
        sys.path.insert(0, code_path)

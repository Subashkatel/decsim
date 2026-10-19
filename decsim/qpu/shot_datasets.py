"""Saved shots: a circuit's draws written once and read back by every run.

A dataset holds, for each root seed of a range, the shot the error-model
source would draw for that seed: its detection events, observable flips
and fired errors (error_model_sampler.ErrorModelSampler.draw). Runs that
read it decode the same data whatever the code, noise or Stim version do
later, and a run on it matches a run on the live error-model source shot
for shot. The draws are written in chunks of root seeds, each its own
folder with a chunk.json of file hashes, so chunks can be written in
parallel; seal then writes the manifest beside the circuit.
"""

import hashlib
import json
import pathlib
from typing import Any, Optional

import numpy
import stim

import decsim.detector_error_model.circuit_builds as circuit_builds
import decsim.detector_error_model.error_model_sampler as error_model_sampler
import decsim.records.seeds as seed_records
import decsim.seeding as seeding

DATASET_FORMAT = "decsim.shots/1"
MANIFEST_FILE = "manifest.json"
CIRCUIT_FILE = "circuit.stim"
CHUNK_FILE = "chunk.json"
# the arrays of one chunk, row k holding root seed first_root_seed + k
ARRAY_NAMES = (
    "sample_seeds",
    "detection_events",
    "observable_flips",
    "error_offsets",
    "error_indices",
)
# the machine binds the syndrome source under this field name
# (machine._seed_roots), so its draws are keyed by it
SOURCE_SEED_FIELD = "device"
# packed bits are read in this order, as stim's b8 format stores them
BIT_ORDER = "little"


def sample_seed_for(
    root_seed: int,
    decode_identity: Any,  # an opaque identity
) -> int:
    """The seed the source draws a stream's shot with, for a root seed.

    The machine gives the source derive_component_seed(root, its field)
    and the source draws a stream under substream_seed(that, (identity,))
    (stim_device.StimDevice._sample_seed).
    """
    segment = seed_records.RunSeedPathSegment("field", SOURCE_SEED_FIELD)
    path = (segment,)
    source_seed = seeding.derive_component_seed(root_seed, path)
    return seeding.substream_seed(source_seed, (decode_identity,))


def write_chunk(
    folder: pathlib.Path,
    circuit: stim.Circuit,
    decode_identity: Any,  # an opaque identity
    first_root_seed: int,
    shot_count: int,
) -> None:
    """Draw shot_count shots from first_root_seed on and save them."""
    sampler = error_model_sampler.sampler_of(circuit)
    rows = _ChunkRows()
    for offset in range(shot_count):
        root_seed = first_root_seed + offset
        sample_seed = sample_seed_for(root_seed, decode_identity)
        drawn = sampler.draw(sample_seed)
        rows.add(sample_seed, drawn)
    chunk_folder = folder / _chunk_name(first_root_seed)
    chunk_folder.mkdir(parents=True, exist_ok=True)
    file_hashes = rows.save(chunk_folder)
    chunk = {
        "first_root_seed": first_root_seed,
        "shot_count": shot_count,
        "file_sha256": file_hashes,
    }
    chunk_path = chunk_folder / CHUNK_FILE
    _write_json(chunk_path, chunk)


def seal(
    folder: pathlib.Path,
    circuit: stim.Circuit,
    decode_identity: Any,  # an opaque identity
    description: dict,
) -> None:
    """Write the circuit and the manifest over every chunk in the folder.

    The chunks must cover one unbroken range of root seeds.
    """
    chunks = _chunks_in(folder)
    _refuse_a_broken_range(chunks)
    circuit_text = str(circuit)
    circuit_path = folder / CIRCUIT_FILE
    circuit_path.write_text(circuit_text)
    model = circuit.detector_error_model()
    manifest = {
        "format": DATASET_FORMAT,
        "circuit_sha256": _text_sha256(circuit_text),
        "decode_identity": decode_identity,
        "detector_count": circuit.num_detectors,
        "observable_count": circuit.num_observables,
        "error_count": model.num_errors,
        "stim_version": stim.__version__,
        "description": description,
        "chunks": chunks,
    }
    manifest_path = folder / MANIFEST_FILE
    _write_json(manifest_path, manifest)


def opened(folder: str) -> "ShotDataset":
    """The dataset at folder, read once per process."""
    key = (folder,)
    dataset = _DATASETS.held(key)
    if dataset is not None:
        return dataset
    folder_path = pathlib.Path(folder)
    dataset = ShotDataset(folder_path)
    _DATASETS.keep(key, dataset)
    return dataset


class ShotDataset:
    """One saved dataset, its shots found by the seed that drew them."""

    def __init__(self, folder: pathlib.Path) -> None:
        manifest = _read_manifest(folder)
        self.folder = folder
        self.manifest = manifest
        self.locations = _ShotLocations(folder, manifest)

    @property
    def root_seeds(self) -> range:
        """The root seeds the dataset holds, first to last."""
        chunks = self.manifest["chunks"]
        first = chunks[0]["first_root_seed"]
        shot_count = sum(chunk["shot_count"] for chunk in chunks)
        end = first + shot_count
        return range(first, end)

    def shot(self, circuit: stim.Circuit, sample_seed: int) -> tuple:
        """(detection events, observable flips, fired error indices).

        The circuit must be the one the dataset was drawn from, and the
        seed one it holds.
        """
        self._refuse_another_circuit(circuit)
        return self.locations.shot(sample_seed)

    def verify(self) -> None:
        """Check every saved file against the hash its chunk recorded."""
        for chunk in self.manifest["chunks"]:
            first_root_seed = chunk["first_root_seed"]
            chunk_folder = self.folder / _chunk_name(first_root_seed)
            recorded = chunk["file_sha256"]
            for name, expected in recorded.items():
                path = chunk_folder / name
                actual = _file_sha256(path)
                _refuse_a_changed_file(path, actual, expected)

    def _refuse_another_circuit(self, circuit: stim.Circuit) -> None:
        circuit_text = str(circuit)
        circuit_hash = _text_sha256(circuit_text)
        if circuit_hash == self.manifest["circuit_sha256"]:
            return
        raise ValueError(
            f"the dataset at {self.folder} was drawn from another circuit "
            f"(sha256 {self.manifest['circuit_sha256']}); this run's "
            f"circuit has sha256 {circuit_hash}"
        )


# (folder,) -> dataset, in this process
_DATASETS = circuit_builds.KeptBuilds(kept_count=4)


class _ChunkRows:
    """The rows of one chunk as they are drawn, then saved as arrays."""

    def __init__(self) -> None:
        self.sample_seeds = []
        self.detection_events = []
        self.observable_flips = []
        self.error_indices = []

    def add(self, sample_seed: int, drawn: tuple) -> None:
        """One drawn shot: (events, flips, fired error indices)."""
        detection_events, observable_flips, error_indices = drawn
        self.sample_seeds.append(sample_seed)
        self.detection_events.append(detection_events)
        self.observable_flips.append(observable_flips)
        self.error_indices.append(error_indices)

    def save(self, chunk_folder: pathlib.Path) -> dict:
        """Write the arrays; their sha256 by file name."""
        arrays = self._arrays()
        file_hashes = {}
        for name in ARRAY_NAMES:
            file_name = f"{name}.npy"
            path = chunk_folder / file_name
            numpy.save(path, arrays[name])
            file_hashes[file_name] = _file_sha256(path)
        return file_hashes

    def _arrays(self) -> dict:
        sample_seeds = numpy.array(self.sample_seeds, dtype=numpy.uint64)
        events = numpy.array(self.detection_events, dtype=bool)
        flips = numpy.array(self.observable_flips, dtype=bool)
        lengths = [len(indices) for indices in self.error_indices]
        offset_count = len(lengths) + 1
        offsets = numpy.zeros(offset_count, dtype=numpy.int64)
        numpy.cumsum(lengths, out=offsets[1:])
        indices = numpy.concatenate(self.error_indices)
        return {
            "sample_seeds": sample_seeds,
            "detection_events": _packed(events),
            "observable_flips": _packed(flips),
            "error_offsets": offsets,
            "error_indices": indices.astype(numpy.uint32),
        }


class _ShotLocations:
    """Each saved shot's chunk and row, looked up by its sample seed."""

    def __init__(self, folder: pathlib.Path, manifest: dict) -> None:
        self.detector_count = manifest["detector_count"]
        self.observable_count = manifest["observable_count"]
        self.chunks = []
        seeds = []
        for chunk in manifest["chunks"]:
            first_root_seed = chunk["first_root_seed"]
            chunk_folder = folder / _chunk_name(first_root_seed)
            arrays = _mapped_arrays(chunk_folder)
            self.chunks.append(arrays)
            seeds.append(arrays["sample_seeds"])
        self.seed_order, self.sorted_seeds = _seed_index(seeds)

    def shot(self, sample_seed: int) -> tuple:
        """The saved shot drawn with this sample seed."""
        chunk_index, row = self._location(sample_seed)
        arrays = self.chunks[chunk_index]
        packed_events = arrays["detection_events"][row]
        packed_flips = arrays["observable_flips"][row]
        first = arrays["error_offsets"][row]
        last = arrays["error_offsets"][row + 1]
        events = _unpacked(packed_events, self.detector_count)
        flips = _unpacked(packed_flips, self.observable_count)
        error_indices = arrays["error_indices"][first:last]
        return events, flips, numpy.asarray(error_indices, dtype=numpy.int64)

    def _location(self, sample_seed: int) -> tuple[int, int]:
        seed = numpy.uint64(sample_seed)
        position = numpy.searchsorted(self.sorted_seeds, seed)
        is_held = position < len(self.sorted_seeds)
        if is_held:
            is_held = self.sorted_seeds[position] == seed
        if not is_held:
            raise ValueError(
                f"the dataset holds no shot for sample seed {sample_seed}; "
                "run a root seed inside its range"
            )
        chunk_index, row = self.seed_order[position]
        return int(chunk_index), int(row)


def _seed_index(seeds: list) -> tuple:
    """Every seed's (chunk, row), in seed order, beside the sorted seeds."""
    locations = []
    for chunk_index, chunk_seeds in enumerate(seeds):
        for row in range(len(chunk_seeds)):
            locations.append((chunk_index, row))
    all_seeds = numpy.concatenate(seeds)
    order = numpy.argsort(all_seeds, kind="stable")
    location_array = numpy.array(locations, dtype=numpy.int64)
    return location_array[order], all_seeds[order]


def _mapped_arrays(chunk_folder: pathlib.Path) -> dict:
    arrays = {}
    for name in ARRAY_NAMES:
        path = chunk_folder / f"{name}.npy"
        arrays[name] = numpy.load(path, mmap_mode="r")
    return arrays


def _chunks_in(folder: pathlib.Path) -> list:
    """Every chunk's record, in root seed order."""
    chunks = []
    for chunk_path in folder.glob(f"*/{CHUNK_FILE}"):
        text = chunk_path.read_text()
        chunk = json.loads(text)
        chunks.append(chunk)
    return sorted(chunks, key=_first_root_seed)


def _first_root_seed(chunk: dict) -> int:
    return chunk["first_root_seed"]


def _refuse_a_broken_range(chunks: list) -> None:
    if not chunks:
        raise ValueError("the dataset folder holds no chunk to seal")
    expected = chunks[0]["first_root_seed"]
    for chunk in chunks:
        first = chunk["first_root_seed"]
        if first != expected:
            raise ValueError(
                f"the chunks leave out the root seeds from {expected} up "
                f"to, not including, {first}"
            )
        expected = first + chunk["shot_count"]


def _read_manifest(folder: pathlib.Path) -> dict:
    path = folder / MANIFEST_FILE
    text = path.read_text()
    manifest = json.loads(text)
    found_format = manifest.get("format")
    if found_format != DATASET_FORMAT:
        raise ValueError(
            f"{path} is not a {DATASET_FORMAT} dataset; its format is "
            f"{found_format!r}"
        )
    return manifest


def _refuse_a_changed_file(
    path: pathlib.Path, actual: str, expected: Optional[str]
) -> None:
    if actual == expected:
        return
    raise ValueError(
        f"{path} has sha256 {actual}, not the {expected} its chunk recorded"
    )


def _packed(bits: numpy.ndarray) -> numpy.ndarray:
    return numpy.packbits(bits, axis=1, bitorder=BIT_ORDER)


def _unpacked(packed: numpy.ndarray, bit_count: int) -> numpy.ndarray:
    bits = numpy.unpackbits(packed, count=bit_count, bitorder=BIT_ORDER)
    return bits.astype(bool)


def _chunk_name(first_root_seed: int) -> str:
    return f"chunk_{first_root_seed:012d}"


def _write_json(path: pathlib.Path, value: dict) -> None:
    text = json.dumps(value, indent=2, sort_keys=True)
    path.write_text(text)


def _text_sha256(text: str) -> str:
    encoded = text.encode()
    digest = hashlib.sha256(encoded)
    return digest.hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256")
    return digest.hexdigest()

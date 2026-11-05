"""A recorded run's per-shot rows, read one shot at a time.

A replay (escalation/verdicts.py Replay, decoders/decode_time_replay.py)
reads what a recorded run wrote for each seed it replays. A million
shots write gigabytes of rows, more than every job of a replay should
hold, so a recording may be given as its task's pieces folder
(pieces/TASK_ID, one FIRST-LAST folder per piece of seeds FIRST to LAST)
and only the piece holding the seed is read, a few kept at a time. A
single file is read whole, for small recordings and tests. The piece
folder's name is made and read here alone (piece_name, piece_range),
for the writer (experiments/pieces.py) and this reader.
"""

import csv
import functools
import pathlib
from typing import Optional


def rows_of_seed(path: str, file_name: str, seed: int) -> Optional[list]:
    """The rows the recording wrote for one seed; None when it ran no such seed.

    path is the recorded file itself, or the task's pieces folder, whose
    piece covering the seed holds file_name. A piece that ran the seed
    and wrote no row for it gives an empty list.
    """
    recorded = pathlib.Path(path)
    if recorded.is_file():
        file_rows = _rows_by_seed(str(recorded))
        return file_rows.get(seed)
    piece = _piece_of(str(recorded), seed)
    if piece is None:
        return None
    piece_file = piece / file_name
    piece_rows = _rows_by_seed(str(piece_file))
    return piece_rows.get(seed, [])


def piece_name(first_seed: int, count: int) -> str:
    """The folder name of a piece of count seeds from first_seed."""
    last_seed = first_seed + count - 1
    return f"{first_seed}-{last_seed}"


def piece_range(name: str) -> Optional[tuple]:
    """(first seed, count) a piece folder's name holds; None for another."""
    first, _separator, last = name.partition("-")
    if not (first.isdigit() and last.isdigit()):
        return None
    first_seed = int(first)
    count = int(last) - first_seed + 1
    return first_seed, count


@functools.lru_cache(maxsize=1)
def _pieces(folder: str) -> tuple:
    """(first seed, count, folder) of every piece, by first seed."""
    pieces = []
    folder_path = pathlib.Path(folder)
    for child in folder_path.iterdir():
        seed_range = piece_range(child.name)
        if child.is_dir() and seed_range is not None:
            first_seed, count = seed_range
            pieces.append((first_seed, count, child))
    return tuple(sorted(pieces))


def _piece_of(folder: str, seed: int) -> Optional[pathlib.Path]:
    for first_seed, count, piece in _pieces(folder):
        if first_seed <= seed < first_seed + count:
            return piece
    return None


@functools.lru_cache(maxsize=4)
def _rows_by_seed(file_path: str) -> dict:
    """Each seed's rows, in file order; a missing file is a piece of no rows."""
    by_seed: dict = {}
    recorded = pathlib.Path(file_path)
    if not recorded.exists():
        return by_seed
    with open(file_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seed = int(row["seed"])
            seed_rows = by_seed.setdefault(seed, [])
            seed_rows.append(row)
    return by_seed

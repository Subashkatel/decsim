"""A recorded run's per-shot rows, read one shot at a time.

A replay (escalation/verdicts.py Replay, decoders/decode_time_replay.py)
reads what a recorded run wrote for each seed it replays. A million
shots write gigabytes of rows, more than every job of a replay should
hold, so a recording may be given as its task's pieces folder
(pieces/TASK_ID, one FIRST-LAST folder per piece of seeds FIRST to LAST)
and only the piece holding the seed is read, a few kept at a time. A
single file is read whole, for small recordings and tests.
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


@functools.lru_cache(maxsize=1)
def _pieces(folder: str) -> tuple:
    """(first seed, last seed, folder) of every piece, by first seed."""
    pieces = []
    folder_path = pathlib.Path(folder)
    for child in folder_path.iterdir():
        first, separator, last = child.name.partition("-")
        is_seed_range = first.isdigit() and last.isdigit()
        is_piece_name = bool(separator) and is_seed_range
        if child.is_dir() and is_piece_name:
            pieces.append((int(first), int(last), child))
    return tuple(sorted(pieces))


def _piece_of(folder: str, seed: int) -> Optional[pathlib.Path]:
    for first, last, piece in _pieces(folder):
        if first <= seed <= last:
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

"""A piece: seeds [first, first + count) of one task, kept as one folder.

pieces/<task id>/<first>-<last>/ holds the piece's additive files and
piece.json. They are written into a hidden staging folder beside it,
piece.json last, and the folder is renamed into place last, so a piece
folder exists only whole: a rename within one directory is atomic under
POSIX, and GPFS keeps POSIX semantics under its distributed locking
(Schmuck and Haskin, FAST 2002). A run skips a piece whose folder
exists, which is how a killed collect resumes, as sinter resumes from
the counts its save file already holds
(sinter/_collection/_collection.py:387-397). A staging folder a killed
writer left is no piece and is passed over; no writer can tell from
another host whether its owner is still writing, so none removes it.
"""

import hashlib
import json
import os
import pathlib
import pickle
import shutil
import uuid
from typing import Optional

import decsim.experiments.refusal as refusal
import decsim.experiments.report as report
import decsim.experiments.run_folder as run_folder
import decsim.ports as ports

PIECES_FOLDER = "pieces"
PIECE_FILE = "piece.json"
# an adaptive task's calibrator as its piece left it
STATE_FILE = "state.pickle"


def piece_dir(
    experiment_dir: pathlib.Path, task_id: str, first_seed: int, count: int
) -> pathlib.Path:
    """Where the piece's folder is: its task, then its first and last seed."""
    last_seed = first_seed + count - 1
    task_dir = experiment_dir / PIECES_FOLDER / task_id
    return task_dir / f"{first_seed}-{last_seed}"


def write(
    experiment_dir: pathlib.Path,
    task_id: str,
    first_seed: int,
    measurements: list,
    facts: dict,
    state: Optional[ports.ThresholdSource] = None,
) -> pathlib.Path:
    """One piece's files, whole or not at all, and where they went.

    state is an adaptive task's calibrator after the piece's last shot,
    which its next piece starts from; it is pickled beside the files with
    its sha256 in piece.json.
    """
    count = len(measurements)
    folder = piece_dir(experiment_dir, task_id, first_seed, count)
    staging = _staging_dir(folder)
    staging.mkdir(parents=True)
    record = report.record_of(measurements)
    report.write_record(record, staging, None)
    identity = run_folder.piece_identity()
    confidence_shot_count = report.confidence_shot_count_of(measurements)
    piece = {
        "task_id": task_id,
        "first_seed": first_seed,
        "count": count,
        "confidence_shot_count": confidence_shot_count,
        **facts,
        **identity,
    }
    if state is not None:
        piece["state_sha256"] = _write_state(staging, state)
    piece_path = staging / PIECE_FILE
    run_folder.write_json(piece_path, piece)
    _publish(staging, folder)
    return folder


def folders_of(experiment_dir: pathlib.Path, task_ids: list) -> list:
    """Every whole piece of these tasks, task by task, in seed order.

    A staging folder is no piece and is passed over.
    """
    folders = []
    for task_id in task_ids:
        task_dir = experiment_dir / PIECES_FOLDER / task_id
        if not task_dir.is_dir():
            continue
        written = _whole_pieces(task_dir)
        folders.extend(written)
    return folders


def every_folder(experiment_dir: pathlib.Path) -> list:
    """Every whole piece the folder holds, of whichever task."""
    pieces_dir = experiment_dir / PIECES_FOLDER
    every_task_dir = pieces_dir.glob("*")
    task_dirs = sorted(every_task_dir)
    task_ids = [task_dir.name for task_dir in task_dirs]
    return folders_of(experiment_dir, task_ids)


def refuse_pieces_of_another_tree(run_dir: pathlib.Path, folders: list) -> None:
    """Every piece ran the tree the folder's run.json names.

    A piece another tree saved would be skipped as done, folded and reported
    as this tree's. run.json's tree is read first, so a fold with no piece
    yet still stops on a run.json an older tree wrote.
    """
    run_path = run_dir / run_folder.RUN_FILE
    run_record = run_folder.read_json(run_path)
    folder_tree = run_record["git"]
    folder_text = run_folder.tree_text(folder_tree)
    for folder in folders:
        piece = read_piece(folder)
        if run_folder.is_one_tree(folder_tree, piece):
            continue
        piece_text = run_folder.tree_text(piece)
        message = (
            f"{folder} ran {piece_text}, and {run_path} "
            f"names {folder_text}; a folder holds "
            "one tree's results, so collect into a new folder, or move "
            "that tree's pieces out of this one"
        )
        raise refusal.RefusalError(message)


def task_folders(folders: list, task_id: str) -> list:
    """The pieces of one task among folders, in their order."""
    return [folder for folder in folders if folder.parent.name == task_id]


def saved_counts(folders: list) -> dict:
    """One task's whole pieces, each first seed mapped to its count.

    folders are the task's pieces as folders_of gave them, so a caller
    that read them once counts what it folds.
    """
    counts = {}
    for folder in folders:
        first_seed, count = _range_of(folder)
        counts[first_seed] = count
    return counts


def contiguous_ranges(saved: dict, first_seed: int) -> list:
    """The saved (first seed, count) pieces from first_seed on, up to a gap.

    saved maps each saved piece's first seed to its count, as
    saved_counts gives it.
    """
    ranges = []
    next_seed = first_seed
    while next_seed in saved:
        count = saved[next_seed]
        ranges.append((next_seed, count))
        next_seed += count
    return ranges


def unsaved_ranges(saved: dict, shot_count: int) -> list:
    """The (first seed, count) gaps of [0, shot_count) no saved piece holds.

    saved maps each saved piece's first seed to its count, as
    saved_counts gives it, so a resubmitted plan cuts new pieces only
    between the saved ones and runs no seed twice.
    """
    ranges = []
    next_seed = 0
    for first_seed in sorted(saved):
        gap_end = min(first_seed, shot_count)
        if gap_end > next_seed:
            gap_count = gap_end - next_seed
            ranges.append((next_seed, gap_count))
        piece_end = first_seed + saved[first_seed]
        next_seed = max(next_seed, piece_end)
    if shot_count > next_seed:
        last_count = shot_count - next_seed
        ranges.append((next_seed, last_count))
    return ranges


def seed_ranges_of(folders: list) -> dict:
    """Each task's seed ranges, [first, count], from its pieces' names."""
    ranges = {}
    for folder in folders:
        task_id = folder.parent.name
        first_seed, count = _range_of(folder)
        task_ranges = ranges.setdefault(task_id, [])
        task_ranges.append((first_seed, count))
    joined = {}
    for task_id, task_ranges in ranges.items():
        joined[task_id] = run_folder.seed_ranges(task_ranges)
    return joined


def read_piece(folder: pathlib.Path) -> dict:
    """One piece's piece.json."""
    piece_path = folder / PIECE_FILE
    text = piece_path.read_text()
    return json.loads(text)


def read_state(folder: pathlib.Path) -> ports.ThresholdSource:
    """The calibrator a piece saved, once its bytes match their sha256.

    The file comes off a shared disk and may have been written on
    another host, so bytes that do not hash to what piece.json records
    are refused rather than unpickled.
    """
    piece = read_piece(folder)
    state_path = folder / STATE_FILE
    state_bytes = state_path.read_bytes()
    digest = hashlib.sha256(state_bytes)
    if digest.hexdigest() != piece["state_sha256"]:
        raise refusal.RefusalError(
            f"{state_path} does not hash to the state_sha256 its "
            "piece.json records; the piece is damaged, so delete its "
            "folder and collect it again"
        )
    return pickle.loads(state_bytes)


def _staging_dir(folder: pathlib.Path) -> pathlib.Path:
    """A hidden folder beside the piece that only this writer uses.

    Two tasks handed the same piece may write it at once, so each
    stages its own copy under a random name and neither touches the
    other's files.
    """
    identifier = uuid.uuid4()
    token = identifier.hex
    return folder.with_name(f".{folder.name}.{token}.partial")


def _publish(staging: pathlib.Path, folder: pathlib.Path) -> None:
    """The staged piece renamed into place, or dropped when one is there.

    A rename onto a folder that holds files fails (rename(2), ENOTEMPTY),
    so of two writers of one piece the first to rename wins. The other's
    copy holds the same seeds of the same task, the same piece, and is
    dropped.
    """
    try:
        os.replace(staging, folder)
    except OSError:
        if not folder.is_dir():
            raise
        shutil.rmtree(staging)


def _write_state(staging: pathlib.Path, state: ports.ThresholdSource) -> str:
    """The state pickled into the staging folder; its sha256."""
    state_bytes = pickle.dumps(state)
    state_path = staging / STATE_FILE
    state_path.write_bytes(state_bytes)
    digest = hashlib.sha256(state_bytes)
    return digest.hexdigest()


def _whole_pieces(task_dir: pathlib.Path) -> list:
    """One task's piece folders in first-seed order, staging ones left out."""
    pieces = []
    for folder in task_dir.iterdir():
        if folder.name.startswith("."):
            continue
        pieces.append(folder)
    return sorted(pieces, key=_range_of)


def _range_of(folder: pathlib.Path) -> tuple:
    """(first seed, count) from a piece folder's <first>-<last> name."""
    first_text, last_text = folder.name.split("-")
    first_seed = int(first_text)
    last_seed = int(last_text)
    count = last_seed - first_seed + 1
    return (first_seed, count)

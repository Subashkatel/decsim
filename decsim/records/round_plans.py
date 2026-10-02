"""A batch plan's records: one piece a batch deals to a task.

decsim run --slurm writes them to batches/<k>/plan.csv, and each task
of the batch reads its share back (decsim/experiments/pieces.py).
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class PlannedPiece:
    """One piece of the plan: seeds [first_seed, first_seed + count)."""

    point_id: str
    first_seed: int
    count: int

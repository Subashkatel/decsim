"""A round plan's records: one piece a round deals to a task.

decsim plan writes them to round<k>/plan.csv, and decsim collect --plan
reads its task's share back (decsim/experiments/pieces.py).
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class PlannedPiece:
    """One piece of the plan: seeds [first_seed, first_seed + count)."""

    configuration_id: str
    point_id: str
    first_seed: int
    count: int

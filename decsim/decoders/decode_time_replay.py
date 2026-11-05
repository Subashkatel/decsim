"""A recorded run's decode times, replayed on the same windows.

A study that changes one part (a decoder that is always right) keeps
every other part's timing: each decode holds its unit for the time the
same decode took in the recorded run, so the schedule, the queues and
every latency stay the recorded run's. The decoder still answers on this
run's own input, which can differ from the recorded run's once an
earlier answer changed; only its time is the recorded one.

The times are the recorded run's decode_times.csv
(observation.record_decode_times), or its task's pieces folder: one row
a decode, its seed, pool, stream, window and the ticks its algorithm
stage held the unit. A decode is found by the shot's root seed
(seeding.RootSeedReader) and its pool, stream and window; one the
recording does not hold is refused, never priced some other way.
"""

from typing import Optional

import decsim.records.decoding as decoding_records
import decsim.records.recorded_rows as recorded_rows


class DecodeTimeReplay:
    """The recorded algorithm ticks of one pool's decodes, by shot."""

    def __init__(self, path: str, pool: str) -> None:
        # the recorded decode_times.csv, or the recorded task's pieces
        # folder, read a piece at a time (records/recorded_rows.py)
        self.path = path
        self.pool = pool
        self.seed: Optional[int] = None
        self.shot_ticks: Optional[dict] = None

    def read_root_seed(self, root_seed: Optional[int]) -> None:
        """The shot's seed picks its recorded times.

        A window decoded twice on one pool would make the key ambiguous,
        so the recording is refused then.
        """
        rows = recorded_rows.rows_of_seed(
            self.path, "decode_times.csv", root_seed
        )
        if rows is None:
            raise ValueError(
                f"{self.pool}.replayed_decode_times: the recorded run holds "
                f"no decode of seed {root_seed}"
            )
        ticks = {}
        for row in rows:
            if row["pool"] != self.pool:
                continue
            key = (row["stream_id"], int(row["window_id"]))
            if key in ticks:
                raise ValueError(
                    f"{self.path}: seed {root_seed} decodes window {key} "
                    f"twice on {self.pool}; a replay cannot tell them apart"
                )
            ticks[key] = int(row["ticks"])
        self.seed = root_seed
        self.shot_ticks = ticks

    def ticks(self, job: decoding_records.DecodeJob) -> int:
        """The ticks the same decode held the unit in the recorded run."""
        key = (str(job.stream_id), job.window_id)
        if self.shot_ticks is None or key not in self.shot_ticks:
            raise ValueError(
                f"{self.pool}.replayed_decode_times: seed {self.seed} has no "
                f"recorded decode of window {key}"
            )
        return self.shot_ticks[key]

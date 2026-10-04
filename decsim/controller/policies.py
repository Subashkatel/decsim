"""How an idle round of a waiting patch travels: the idle policies.

Idle rounds are real decoder workload: every generated syndrome bit
counts against the decoder's rate (Terhal 1302.3428: "the decoding should
never lead to a increasing backlog of syndrome data"; Battistel et al.
2303.00054 line 144). Deferring them is legitimate and deleting them is
a modeling choice, since only data feeding the next non-Clifford
decision is latency-critical (Skoric 2209.08552), so each policy suits a
different claim.
"""

import dataclasses
from typing import Any

import decsim.controller.idle_rounds as idle_rounds_module
import decsim.records.program as program_records


@dataclasses.dataclass(frozen=True)
class IgnoreSettings:
    """The optimistic card: idle rounds cost no decode work (Ignore)."""

    def build(self) -> "Ignore":
        """A fresh policy."""
        return Ignore()


@dataclasses.dataclass(frozen=True)
class SeparateDecodeJobsSettings:
    """The default card: idle rounds are decode jobs (SeparateDecodeJobs)."""

    def build(self) -> "SeparateDecodeJobs":
        """A fresh policy."""
        return SeparateDecodeJobs()


class Ignore:
    """Idle rounds travel as feedback-memory rounds and cost no decode work.

    Valid for latency studies of the active path; it undercounts decoder
    throughput, utilization and unit counts on multi-operation workloads.
    """

    def relay(
        self,
        idle_rounds: idle_rounds_module.IdleRoundAccounting,
        operation: program_records.Operation,
        patch: Any,  # an opaque identity
        round_index: int,
    ) -> None:
        """Send the round as a memory round."""
        idle_rounds.emit_memory_round(operation, patch, round_index)

    def end_idle_period(
        self,
        idle_rounds: idle_rounds_module.IdleRoundAccounting,
        operation: program_records.Operation,
        patch: Any,  # an opaque identity
    ) -> None:
        """Nothing was charged, so nothing settles."""
        del idle_rounds
        del operation
        del patch


class SeparateDecodeJobs:
    """Idle rounds travel as memory rounds and are charged as decode jobs.

    Every commit region costs one load-only decode job sized to the region
    plus the buffer rounds. The rounds left when an operation claims the
    patch, or the workload completes, cost one shorter job: a final window
    may be smaller (Tan et al. 2209.09219; Skoric et al. 2209.08552), and no
    validated system leaves the end of a stream undecoded. The honest
    default for throughput, utilization, backlog or unit-count claims.
    """

    def relay(
        self,
        idle_rounds: idle_rounds_module.IdleRoundAccounting,
        operation: program_records.Operation,
        patch: Any,  # an opaque identity
        round_index: int,
    ) -> None:
        """Send the memory round and count it toward the next job."""
        idle_rounds.emit_memory_round(operation, patch, round_index)
        idle_rounds.submit_idle_decode_if_due(operation, patch, round_index)

    def end_idle_period(
        self,
        idle_rounds: idle_rounds_module.IdleRoundAccounting,
        operation: program_records.Operation,
        patch: Any,  # an opaque identity
    ) -> None:
        """Charge the rounds left after the last full commit region."""
        idle_rounds.submit_idle_decode_for_remaining_rounds(operation, patch)

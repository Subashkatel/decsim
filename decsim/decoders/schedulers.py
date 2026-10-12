"""The ready-queue discipline of a decoder pool: which waiting job is next.

A scheduler is a class with pop(queue), built by its frozen Settings
record, which the decoder manager's scheduler setting holds. It reads
the planned operations once before the run (read_program) and hears of
each job its pool dispatches (served). It stays a research knob: Triage
(2605.04459) makes the M-for-N scheduler the thing to vary.

    FifoScheduler      admission order (the default)
    longest_undecoded  VQD's MLS
    round_robin        VQD's RR
    most_critical      VQD's MFD

VQD (Maurya and Tannu 2406.17995v2) shares M decoders among N logical
qubits. Every policy has two components (Sec. 4.2): decoders go first
to the critical decodes, those a non-Clifford gate waits on, which take
"all syndromes generated up to that point for that logical qubit" (Sec.
2.4); the rest go by the policy. The authors' simulator
(github.com/satvikmaurya/decoder-resources, scripts/sim.py,
Sim.schedule) also puts any other multi-patch measurement ahead of the
policy, and an older task ahead of a newer one of the same priority.
Each VQD row here orders the waiting jobs in those three tiers:

    1. every job on a patch with a waiting critical job, a job of an
       operation another operation is blocked_by;
    2. every job on a patch with a waiting job over several patches;
    3. the rest, by the row's policy.

VQD schedules once per slice of d rounds at compile time; a row here
orders the queue at each dispatch, Triage's online form (Sec. III-B1).
A stream's windows are refused: its critical point is a segment's end.
Triage's EDF and MDF rank slices of a constraint graph decsim does not
model, and are not built.
"""

import dataclasses
from typing import Any, Protocol

import decsim.records.decoding as decoding_records
import decsim.refusals as refusals

_CRITICAL_TIER = 0
_MULTI_PATCH_TIER = 1
_POLICY_TIER = 2


class Scheduler(Protocol):
    """A ready-queue rule: the waiting job a free unit serves next."""

    def read_program(self, operations: tuple) -> None:
        """Learn the planned operations, once, before the run."""

    def pop(
        self, queue: list[decoding_records.DecodeJob]
    ) -> decoding_records.DecodeJob:
        """Remove and return the next job of the queue."""

    def served(self, job: decoding_records.DecodeJob) -> None:
        """The pool dispatched this job."""


class FifoScheduler:
    """Dispatch ready jobs in admission order within each decoder pool.

    FIFO is the minimal work-conserving baseline over identical
    non-preemptive units: it claims no deadline, cost or
    microarchitecture.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """FIFO's settings: the rule has no knob."""

        def build(self) -> "FifoScheduler":
            """A fresh scheduler: each manager orders its own queue."""
            return FifoScheduler()

    def read_program(self, operations: tuple) -> None:
        """FIFO reads nothing of the program."""
        del operations

    def pop(
        self, queue: list[decoding_records.DecodeJob]
    ) -> decoding_records.DecodeJob:
        """Remove the ready job admitted earliest."""
        return queue.pop(0)

    def served(self, job: decoding_records.DecodeJob) -> None:
        """FIFO keeps no history."""
        del job


class LongestUndecoded:
    """VQD's MLS: the patch with the most rounds waiting goes first.

    VQD sorts the qubits by their current undecoded sequence length
    (Sec. 4.2.3, Fig. 6(d)); sim.py by a backlog that grows with each
    undecoded slice and shrinks by the work a decode does. Here a
    patch's undecoded sequence is the rounds of its jobs in the queue.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """MLS's settings: the rule has no knob."""

        name = "longest_undecoded"

        def build(self) -> "LongestUndecoded":
            """A fresh scheduler over no program yet."""
            return LongestUndecoded()

    def __init__(self) -> None:
        self.program = ProgramTable()

    def read_program(self, operations: tuple) -> None:
        """The patches and critical operations of the program."""
        self.program = program_table(operations)

    def pop(
        self, queue: list[decoding_records.DecodeJob]
    ) -> decoding_records.DecodeJob:
        """Remove the next job by tier, then by the most waiting rounds."""
        rounds_by_patch = _waiting_rounds_by_patch(self.program, queue)
        ranks = []
        for job in queue:
            patch = self.program.patch_of(job)
            ranks.append(-rounds_by_patch[patch])
        return _pop_by_tier(self.program, queue, ranks)

    def served(self, job: decoding_records.DecodeJob) -> None:
        """MLS reads the queue alone."""
        del job


class RoundRobin:
    """VQD's RR: the patches take turns.

    VQD chooses the qubits "in a round-robin manner" (Sec. 4.2.2,
    Fig. 6(c)); sim.py keeps them in a deque and moves every decoded
    qubit to its back, a critical one too. Here the turn order starts
    as the program first names the patches.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """RR's settings: the rule has no knob."""

        name = "round_robin"

        def build(self) -> "RoundRobin":
            """A fresh scheduler over no program yet."""
            return RoundRobin()

    def __init__(self) -> None:
        self.program = ProgramTable()
        self.turns: list = []

    def read_program(self, operations: tuple) -> None:
        """The patches, in the order of their first turn."""
        self.program = program_table(operations)
        self.turns = list(self.program.patch_order)

    def pop(
        self, queue: list[decoding_records.DecodeJob]
    ) -> decoding_records.DecodeJob:
        """Remove the next job by tier, then by its patch's turn."""
        ranks = []
        for job in queue:
            patch = self.program.patch_of(job)
            turn = _turn_of(patch, self.turns)
            ranks.append(turn)
        return _pop_by_tier(self.program, queue, ranks)

    def served(self, job: decoding_records.DecodeJob) -> None:
        """The served job's patches go to the back of the turns."""
        patches = self.program.patches_of(job)
        for patch in patches:
            _to_the_back(patch, self.turns)


class MostCritical:
    """VQD's MFD: the patch with the most critical decodes ahead first.

    VQD prioritises the qubits "that have numerous critical decodes in
    the future" (Sec. 4.2.1, Fig. 6(b)); sim.py sums each qubit's
    critical decodes from the current slice on. Here a critical
    operation is ahead until a job of it is served.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """MFD's settings: the rule has no knob."""

        name = "most_critical"

        def build(self) -> "MostCritical":
            """A fresh scheduler over no program yet."""
            return MostCritical()

    def __init__(self) -> None:
        self.program = ProgramTable()
        self.served_operation_ids: set = set()

    def read_program(self, operations: tuple) -> None:
        """The critical operations on each patch."""
        self.program = program_table(operations)

    def pop(
        self, queue: list[decoding_records.DecodeJob]
    ) -> decoding_records.DecodeJob:
        """Remove the next job by tier, then by critical decodes ahead."""
        ranks = []
        for job in queue:
            patch = self.program.patch_of(job)
            ahead = self._critical_decodes_ahead(patch)
            ranks.append(-ahead)
        return _pop_by_tier(self.program, queue, ranks)

    def served(self, job: decoding_records.DecodeJob) -> None:
        """A served job's critical operation is no longer ahead."""
        self.served_operation_ids.add(job.operation_id)

    def _critical_decodes_ahead(self, patch: Any) -> int:
        critical_ids = self.program.critical_ids_by_patch.get(patch, ())
        ahead = [
            operation_id
            for operation_id in critical_ids
            if operation_id not in self.served_operation_ids
        ]
        return len(ahead)


@dataclasses.dataclass(frozen=True)
class ProgramTable:
    """What a VQD row reads of the program: patches and critical decodes.

    patches_by_operation maps each operation id to its patches;
    critical_ids holds every operation another is blocked_by;
    patch_order names each patch once, in the program's order.
    """

    patches_by_operation: dict = dataclasses.field(default_factory=dict)
    critical_ids: frozenset = frozenset()
    critical_ids_by_patch: dict = dataclasses.field(default_factory=dict)
    patch_order: tuple = ()

    def patches_of(self, job: decoding_records.DecodeJob) -> tuple:
        """The job's patches; a job the program does not list has none."""
        return self.patches_by_operation.get(job.operation_id, ())

    def patch_of(self, job: decoding_records.DecodeJob) -> Any:
        """The patch a policy ranks the job by: its operation's first.

        A job the program does not list (a merged strong batch, a
        factory's correction) is a group of its own.
        """
        patches = self.patches_of(job)
        if patches:
            return patches[0]
        return ("operation", job.operation_id)

    def is_critical(self, job: decoding_records.DecodeJob) -> bool:
        """Whether another operation waits on this job's decision."""
        return job.operation_id in self.critical_ids

    def spans_patches(self, job: decoding_records.DecodeJob) -> bool:
        """Whether the job's operation measures over several patches."""
        patches = self.patches_of(job)
        return len(patches) > 1


def program_table(operations: tuple) -> ProgramTable:
    """The table of the planned operations; a stream is refused."""
    patches_by_operation = {}
    critical_ids = set()
    patch_order = {}
    for operation in operations:
        _refuse_a_stream(operation)
        patches_by_operation[operation.id] = tuple(operation.patches)
        patches = dict.fromkeys(operation.patches)
        patch_order.update(patches)
        if operation.blocked_by is not None:
            critical_ids.add(operation.blocked_by)
    critical_ids_by_patch = _critical_ids_by_patch(
        patches_by_operation, critical_ids
    )
    frozen_ids = frozenset(critical_ids)
    ordered_patches = tuple(patch_order)
    return ProgramTable(
        patches_by_operation,
        frozen_ids,
        critical_ids_by_patch,
        ordered_patches,
    )


def _critical_ids_by_patch(
    patches_by_operation: dict, critical_ids: set
) -> dict:
    critical_ids_by_patch: dict = {}
    for operation_id in sorted(critical_ids):
        patches = patches_by_operation.get(operation_id, ())
        for patch in patches:
            listed = critical_ids_by_patch.setdefault(patch, [])
            listed.append(operation_id)
    return critical_ids_by_patch


def _refuse_a_stream(operation: Any) -> None:
    if operation.stream_id is None:
        return
    raise NotImplementedError(
        f"operation {operation.name!r} is a segment of stream "
        f"{operation.stream_id!r}: a VQD scheduler over a stream's windows "
        "is not implemented yet; its critical point is a segment's end, "
        f"which the scheduler does not read; {refusals.REFUSALS}"
    )


def _pop_by_tier(
    program: ProgramTable,
    queue: list,
    ranks: list,
) -> decoding_records.DecodeJob:
    """Remove the job first by tier, then rank, then age."""
    tier_by_patch = _tier_by_patch(program, queue)
    keys = []
    for index, job in enumerate(queue):
        key = _order_key(program, job, tier_by_patch, ranks[index], index)
        keys.append(key)
    first = keys.index(min(keys))
    return queue.pop(first)


def _order_key(
    program: ProgramTable,
    job: decoding_records.DecodeJob,
    tier_by_patch: dict,
    rank: int,
    index: int,
) -> tuple:
    """(tier, rank, admission tick, place): the policy ranks its tier only."""
    tier = _POLICY_TIER
    for patch in program.patches_of(job):
        known = tier_by_patch.get(patch, _POLICY_TIER)
        tier = min(tier, known)
    if tier != _POLICY_TIER:
        rank = 0
    return (tier, rank, job.ready_time, index)


def _tier_by_patch(program: ProgramTable, queue: list) -> dict:
    """Each patch's tier: critical, multi-patch, or absent for the policy."""
    tier_by_patch: dict = {}
    for job in queue:
        tier = _own_tier(program, job)
        for patch in program.patches_of(job):
            known = tier_by_patch.get(patch, _POLICY_TIER)
            tier_by_patch[patch] = min(known, tier)
    return tier_by_patch


def _own_tier(program: ProgramTable, job: decoding_records.DecodeJob) -> int:
    if program.is_critical(job):
        return _CRITICAL_TIER
    if program.spans_patches(job):
        return _MULTI_PATCH_TIER
    return _POLICY_TIER


def _waiting_rounds_by_patch(program: ProgramTable, queue: list) -> dict:
    rounds_by_patch: dict = {}
    for job in queue:
        patch = program.patch_of(job)
        waiting = rounds_by_patch.get(patch, 0)
        rounds_by_patch[patch] = waiting + job.round_count
    return rounds_by_patch


def _turn_of(patch: Any, turns: list) -> int:
    if patch in turns:
        return turns.index(patch)
    return len(turns)


def _to_the_back(patch: Any, turns: list) -> None:
    if patch in turns:
        turns.remove(patch)
        turns.append(patch)

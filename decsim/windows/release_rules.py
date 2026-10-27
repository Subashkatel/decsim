"""When an outcome is usable: which decodes it waits for.

An outcome is needed when an operation is blocked on it or the QPU
needs it returned (requires_result_return_to_qpu); either waits on the
same decodes, so a "blocker" below is either.

The windows package's own seam: OperationResults asks its row which
operations' segments must be final before a blocker releases its
waiters, and which blockers a segment's commit may release.

A logical outcome is usable once every decoding task that can flip it
has committed: Bombín et al. 2303.04846 Eq. 4 sums the tasks T(M) whose
commit region holds an error flipping the outcome's membrane, and the
membrane "continues towards the past and depends on the history of the
two qubits" (Sec. I.A), branching through earlier merges into other
patches (Fig. 1). The rows bound that set from either side.

OwnStreams waits for each of the blocker's streams committed from its
first round through the blocker's rounds: the membrane's history along
its own patches, not the branches through earlier merges with others.
DecodedAncestors is SWIPER's (2412.05115 Sec. 2.3, a Clifford correction
"based on up-to-date decoding results"; in its code,
swiper/decoder_manager.py get_incomplete_instruction_indices), as is
Triage's causal cone (2605.04459 Sec. II-B): an instruction is
incomplete while any instruction it descends from, along a path of
decoded instructions, has a window not yet decoded. It holds every
branch, and some instructions that cannot flip the outcome.
"""

import dataclasses
from typing import Protocol


class ReleaseRule(Protocol):
    """Which segments a blocker's release waits for, both ways."""

    def awaited_by(self, operation_id: int) -> tuple:
        """The operations whose segments the blocker's release waits for."""

    def awaiting(self, operation_id: int) -> tuple:
        """The blockers a commit of the operation's segments may release."""


class OwnStreams:
    """A blocker waits for its own segments, each a committed prefix."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The own-streams row, which takes no setting."""

        name = "own_streams"
        awaits_ancestors = False

        def build(self, operations: tuple) -> "OwnStreams":
            """The rule; it reads no operation but the blocker."""
            del operations
            return OwnStreams()

    def awaited_by(self, operation_id: int) -> tuple:
        """The operations whose segments the blocker's release waits for."""
        return (operation_id,)

    def awaiting(self, operation_id: int) -> tuple:
        """The blockers a commit of the operation's segments may release."""
        return (operation_id,)


class DecodedAncestors:
    """A needed outcome waits for itself and every decoded operation before it.

    An ancestor is reached through predecessors that each decode rounds;
    one that decodes none ends the walk, as SWIPER's walk passes only
    instructions whose windows it has seen.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The decoded-ancestors row, which takes no setting."""

        name = "decoded_ancestors"
        # an ancestor's decodes are read from its stream segments, which
        # only patch streams bind for every operation
        awaits_ancestors = True

        def build(self, operations: tuple) -> "DecodedAncestors":
            """The rule over the workload's operations."""
            return DecodedAncestors(operations)

    def __init__(self, operations: tuple) -> None:
        operation_by_id = {operation.id: operation for operation in operations}
        # blocker id -> the operations its release waits for, itself first
        self.awaited_by_blocker: dict = {}
        # operation id -> the blockers whose release waits for it
        self.blockers_by_awaited: dict = {}
        for operation in operations:
            if operation.blocked_by is not None:
                self._await(operation.blocked_by, operation_by_id)
            if operation.requires_result_return_to_qpu:
                self._await(operation.id, operation_by_id)

    def awaited_by(self, operation_id: int) -> tuple:
        """The operations whose segments the blocker's release waits for."""
        return self.awaited_by_blocker[operation_id]

    def awaiting(self, operation_id: int) -> tuple:
        """The blockers a commit of the operation's segments may release."""
        blockers = self.blockers_by_awaited.get(operation_id, ())
        return tuple(blockers)

    def _await(self, blocker_id: int, operation_by_id: dict) -> None:
        if blocker_id in self.awaited_by_blocker:
            return
        awaited = _decoded_ancestry(blocker_id, operation_by_id)
        self.awaited_by_blocker[blocker_id] = awaited
        for operation_id in awaited:
            blockers = self.blockers_by_awaited.setdefault(operation_id, [])
            blockers.append(blocker_id)


def _decoded_ancestry(blocker_id: int, operation_by_id: dict) -> tuple:
    """The blocker, then each ancestor reached through decoding operations."""
    awaited = [blocker_id]
    blocker = operation_by_id[blocker_id]
    frontier = list(blocker.predecessors)
    visited = set(frontier)
    while frontier:
        predecessor = operation_by_id[frontier.pop()]
        if not predecessor.emits_detector_data:
            continue
        awaited.append(predecessor.id)
        earlier = _unvisited(predecessor.predecessors, visited)
        frontier.extend(earlier)
    return tuple(awaited)


def _unvisited(operation_ids: tuple, visited: set) -> list:
    """The ids not visited yet, marked visited now."""
    unvisited = [each for each in operation_ids if each not in visited]
    visited.update(unvisited)
    return unvisited

"""Which consumer keeps which rounds alive in a round store.

A hold is a token (a window key, a request key) that names the rounds it
reads. A round stays in its store while any live hold names it; the store
frees a round when its last hold releases. This object only counts: it
knows nothing of packets, slots or ticks.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class HoldRecord:
    """The rounds one holder keeps and the operations they belong to."""

    round_keys: tuple
    referenced_operation_ids: frozenset


class RoundHolds:
    """The live holds, the holders of every round, and the released tokens."""

    def __init__(self) -> None:
        self.record_by_holder: dict = {}
        self.holders_by_round: dict = {}
        # a released or transferred token, with the operations it named;
        # it never registers again, and it is forgotten once they close
        self.released_holders: dict = {}

    def register(self, holder, record: HoldRecord) -> None:
        """Keep the record's rounds alive for a new token."""
        self._check_new_holder(holder)
        self.record_by_holder[holder] = record
        for round_key in record.round_keys:
            self._attach(round_key, holder)

    def replace(self, holder, record: HoldRecord) -> list:
        """Re-point a live hold; the rounds that lost their last holder."""
        old = self.record_by_holder.get(holder)
        if old is None:
            raise RuntimeError(f"hold {holder!r} is not live")
        if record == old:
            return []
        self.record_by_holder[holder] = record
        for round_key in record.round_keys:
            if round_key not in old.round_keys:
                self._attach(round_key, holder)
        orphaned = []
        for round_key in old.round_keys:
            if round_key in record.round_keys:
                continue
            if self._detach(round_key, holder):
                orphaned.append(round_key)
        return orphaned

    def transfer(self, old_holder, new_holder) -> None:
        """Move a live hold to a new token without freeing its rounds."""
        self._check_new_holder(new_holder)
        record = self.record_by_holder.pop(old_holder, None)
        if record is None:
            raise RuntimeError(f"hold {old_holder!r} is not live")
        self.record_by_holder[new_holder] = record
        for round_key in record.round_keys:
            holders = self.holders_by_round[round_key]
            holders.discard(old_holder)
            holders.add(new_holder)
        self.released_holders[old_holder] = record.referenced_operation_ids

    def release(self, holder) -> list:
        """Drop a hold; the rounds that lost their last holder."""
        assert holder not in self.released_holders, (
            f"hold {holder!r} was released twice"
        )
        record = self.record_by_holder.pop(holder, None)
        if record is None:
            raise RuntimeError(f"hold {holder!r} was never registered")
        orphaned = []
        for round_key in record.round_keys:
            if self._detach(round_key, holder):
                orphaned.append(round_key)
        self.released_holders[holder] = record.referenced_operation_ids
        return orphaned

    def is_live(self, holder) -> bool:
        """True while the token holds rounds."""
        return holder in self.record_by_holder

    def round_keys_of(self, holder) -> tuple:
        """The rounds a live holder keeps."""
        return self.record_by_holder[holder].round_keys

    def is_held(self, round_key) -> bool:
        """True while any live hold names the round."""
        holders = self.holders_by_round.get(round_key)
        return bool(holders)

    def held_round_keys(self) -> list:
        """Every round some live hold names."""
        return list(self.holders_by_round)

    def references(self, operation_id) -> bool:
        """True while any live hold names the operation."""
        for record in self.record_by_holder.values():
            if operation_id in record.referenced_operation_ids:
                return True
        return False

    def forget_released_outside(self, open_operation_ids) -> None:
        """Forget released tokens whose every operation has closed."""
        stale = []
        for holder, referenced in self.released_holders.items():
            if referenced.isdisjoint(open_operation_ids):
                stale.append(holder)
        for holder in stale:
            del self.released_holders[holder]

    def _check_new_holder(self, holder) -> None:
        is_live = holder in self.record_by_holder
        was_released = holder in self.released_holders
        if is_live or was_released:
            raise RuntimeError(f"hold {holder!r} was registered before")

    def _attach(self, round_key, holder) -> None:
        holders = self.holders_by_round.setdefault(round_key, set())
        holders.add(holder)

    def _detach(self, round_key, holder) -> bool:
        """Drop one holder; True when the round has no holder left."""
        holders = self.holders_by_round.get(round_key)
        if holders is None:
            return False
        holders.discard(holder)
        if holders:
            return False
        del self.holders_by_round[round_key]
        return True

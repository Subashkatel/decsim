"""The logical ledger: which owner committed which rounds of a stream.

A contribution is one owner (an ordinary window or a strong window)
over an exact inclusive round extent; the contributions of one stream
tile it without gap or overlap, and the observables of an interval are
the XOR of the contributions that cover it. A strong result may replace
the prediction of an owner; a strong window may replace ordinary
windows. Row W5 checks the ledger against a per-round oracle.
"""

from typing import Optional

import decsim.records.decoding as decoding_records

OWNERSHIP_KINDS = ("ordinary_window", "strong_window")
BOUNDARY_POLICIES = ("strict", "stream_segment")


class LogicalLedger:
    """The contributions by owner key, and one observable arity per stream."""

    def __init__(self):
        self.contributions: dict[
            tuple, decoding_records.LogicalContribution
        ] = {}
        self._arity_by_stream: dict[object, int] = {}

    def get(
        self, owner_key: tuple
    ) -> Optional[decoding_records.LogicalContribution]:
        """The owner's contribution, or None."""
        return self.contributions.get(owner_key)

    def drop(self, owner_key: tuple) -> None:
        """A window is decoded again: its contribution is gone until commit."""
        self.contributions.pop(owner_key, None)

    def replace_contributions(
        self,
        owner_key: tuple,
        commit_lo: int,
        commit_hi: int,
        replaced_keys: tuple,
    ) -> None:
        """A strong window takes the extent of the windows it replaces.

        The windows it absorbs, and the escalated window's own entry,
        leave the ledger; every contribution that stays must sit outside
        the strong window's extent, since two owners never claim one
        round (Toshio et al. 2510.25222 Theorem 1's single strong
        decoder). Nothing live moves: the strong window's own
        contribution carries no observables until it commits.
        """
        replaced = {owner_key, *replaced_keys}
        kept = {}
        for other_key, contribution in self.contributions.items():
            if other_key not in replaced:
                kept[other_key] = contribution
        for other_key, contribution in kept.items():
            _refuse_overlapping_extent(
                owner_key, commit_lo, commit_hi, other_key, contribution
            )
        kept[owner_key] = decoding_records.LogicalContribution(
            owner_key=owner_key,
            commit_lo=commit_lo,
            commit_hi=commit_hi,
            ownership_kind="strong_window",
            logical_observables=None,
        )
        self.contributions = kept

    def owns_strong_window(self, owner_key: tuple) -> bool:
        """Whether a strong window already claims that owner's extent."""
        contribution = self.contributions.get(owner_key)
        if contribution is None:
            return False
        return contribution.ownership_kind == "strong_window"

    def install(
        self, contribution: decoding_records.LogicalContribution
    ) -> None:
        """Record who owns an extent.

        An owner never changes kind or extent, extents never overlap,
        and one stream has one observable arity.
        """
        if contribution.ownership_kind not in OWNERSHIP_KINDS:
            raise ValueError(
                "logical contribution ownership_kind must be "
                "'ordinary_window' or 'strong_window'"
            )
        if contribution.commit_lo < 1:
            _refuse_extent(contribution)
        if contribution.commit_hi < contribution.commit_lo:
            _refuse_extent(contribution)
        previous = self.contributions.get(contribution.owner_key)
        if previous is not None:
            _check_ownership_unchanged(previous, contribution)
        self._check_no_overlap(contribution)
        self._check_arity(contribution)
        self.contributions[contribution.owner_key] = contribution

    def observables_for_interval(
        self,
        stream_id,
        commit_lo: int,
        commit_hi: int,
        *,
        boundary_policy: str,
    ) -> Optional[tuple]:
        """The XOR of the contributions covering [commit_lo, commit_hi].

        The interval must be tiled without gap or overlap. Under strict,
        no contribution may cross the interval's edge; under
        stream_segment, only a functional (observable-bearing) one may
        not. None when any covering contribution is timing-only.
        """
        if boundary_policy not in BOUNDARY_POLICIES:
            raise ValueError(
                f"unknown logical contribution boundary policy "
                f"{boundary_policy!r}"
            )
        if commit_lo < 1 or commit_hi < commit_lo:
            raise ValueError(
                f"invalid logical prediction interval {commit_lo}-{commit_hi}"
            )
        contributions = self._covering(stream_id, commit_lo, commit_hi)
        if not contributions:
            raise RuntimeError(
                f"logical prediction interval {stream_id!r} "
                f"{commit_lo}-{commit_hi} has no contribution coverage"
            )
        _check_tiling(contributions, stream_id, commit_lo, commit_hi)
        for contribution in contributions:
            _check_inside_interval(
                contribution, commit_lo, commit_hi, boundary_policy
            )
        for contribution in contributions:
            if contribution.logical_observables is None:
                return None
        return _xor_of(contributions, stream_id)

    def replace_prediction(
        self, owner_key: tuple, logical_observables: tuple
    ) -> None:
        """A strong result replaces the owner's prediction, extent unchanged."""
        contribution = self.contributions.get(owner_key)
        if contribution is None:
            raise RuntimeError(
                f"result for {owner_key} has no logical contribution owner"
            )
        replaced = decoding_records.LogicalContribution(
            owner_key=contribution.owner_key,
            commit_lo=contribution.commit_lo,
            commit_hi=contribution.commit_hi,
            ownership_kind=contribution.ownership_kind,
            logical_observables=logical_observables,
        )
        self.install(replaced)

    def _check_no_overlap(
        self, contribution: decoding_records.LogicalContribution
    ) -> None:
        stream_id = contribution.owner_key[0]
        for other_key, other in self.contributions.items():
            if other_key == contribution.owner_key:
                continue
            if other_key[0] != stream_id:
                continue
            if _overlaps(contribution, other):
                raise RuntimeError(
                    f"logical contribution {contribution.owner_key} extent "
                    f"{contribution.commit_lo}-{contribution.commit_hi} "
                    f"overlaps {other_key} extent "
                    f"{other.commit_lo}-{other.commit_hi}"
                )

    def _check_arity(
        self, contribution: decoding_records.LogicalContribution
    ) -> None:
        """One stream has one arity, set by its first functional owner."""
        logical_observables = contribution.logical_observables
        if logical_observables is None:
            return
        stream_id = contribution.owner_key[0]
        observed_arity = len(logical_observables)
        expected_arity = self._arity_by_stream.get(stream_id)
        if expected_arity is None:
            self._arity_by_stream[stream_id] = observed_arity
            return
        if observed_arity != expected_arity:
            raise ValueError(
                f"logical contribution {contribution.owner_key} has "
                f"observable length {observed_arity}; expected "
                f"{expected_arity} for stream {stream_id!r}"
            )

    def _covering(self, stream_id, commit_lo: int, commit_hi: int) -> list:
        """The stream's contributions touching the interval, in extent order."""
        covering = []
        for key, contribution in self.contributions.items():
            if key[0] != stream_id:
                continue
            if contribution.commit_lo > commit_hi:
                continue
            if contribution.commit_hi < commit_lo:
                continue
            covering.append(contribution)
        covering.sort(key=_extent_order)
        return covering


def _refuse_overlapping_extent(
    owner_key: tuple,
    commit_lo: int,
    commit_hi: int,
    other_key: tuple,
    contribution: decoding_records.LogicalContribution,
) -> None:
    """A kept contribution may not touch the strong window's extent."""
    if other_key[0] != owner_key[0]:
        return
    is_begun = contribution.commit_lo <= commit_hi
    is_unfinished = commit_lo <= contribution.commit_hi
    if is_begun and is_unfinished:
        raise RuntimeError(
            f"strong window {owner_key} extent {commit_lo}-"
            f"{commit_hi} overlaps unabsorbed logical "
            f"contribution {other_key} extent "
            f"{contribution.commit_lo}-{contribution.commit_hi}"
        )


def _refuse_extent(contribution: decoding_records.LogicalContribution) -> None:
    raise ValueError(
        f"logical contribution {contribution.owner_key} has invalid "
        f"extent {contribution.commit_lo}-{contribution.commit_hi}"
    )


def _check_ownership_unchanged(
    previous: decoding_records.LogicalContribution,
    contribution: decoding_records.LogicalContribution,
) -> None:
    same_extent = previous.commit_lo == contribution.commit_lo
    if previous.commit_hi != contribution.commit_hi:
        same_extent = False
    same_kind = previous.ownership_kind == contribution.ownership_kind
    if same_extent and same_kind:
        return
    raise RuntimeError(
        f"logical contribution {contribution.owner_key} cannot "
        f"change ownership from {previous.ownership_kind} "
        f"{previous.commit_lo}-{previous.commit_hi} to "
        f"{contribution.ownership_kind} "
        f"{contribution.commit_lo}-{contribution.commit_hi}"
    )


def _overlaps(
    left: decoding_records.LogicalContribution,
    right: decoding_records.LogicalContribution,
) -> bool:
    if left.commit_lo > right.commit_hi:
        return False
    return right.commit_lo <= left.commit_hi


def _extent_order(contribution: decoding_records.LogicalContribution) -> tuple:
    owner_text = repr(contribution.owner_key)
    return (contribution.commit_lo, contribution.commit_hi, owner_text)


def _check_tiling(
    contributions: list, stream_id, commit_lo: int, commit_hi: int
) -> None:
    """The covering contributions tile the interval without gap or overlap."""
    cursor = commit_lo
    for contribution in contributions:
        covered_lo = max(contribution.commit_lo, commit_lo)
        covered_hi = min(contribution.commit_hi, commit_hi)
        if covered_lo != cursor:
            relation = _relation_of(covered_lo, cursor)
            raise RuntimeError(
                f"logical prediction interval {stream_id!r} "
                f"{commit_lo}-{commit_hi} has a contribution "
                f"{relation} at round {cursor}"
            )
        cursor = covered_hi + 1
    if cursor != commit_hi + 1:
        raise RuntimeError(
            f"logical prediction interval {stream_id!r} "
            f"{commit_lo}-{commit_hi} has a contribution gap at "
            f"round {cursor}"
        )


def _relation_of(covered_lo: int, cursor: int) -> str:
    if covered_lo < cursor:
        return "overlap"
    return "gap"


def _check_inside_interval(
    contribution: decoding_records.LogicalContribution,
    commit_lo: int,
    commit_hi: int,
    boundary_policy: str,
) -> None:
    """A contribution crossing the interval's edge is refused by policy."""
    crosses_boundary = contribution.commit_lo < commit_lo
    if contribution.commit_hi > commit_hi:
        crosses_boundary = True
    if not crosses_boundary:
        return
    if boundary_policy == "stream_segment":
        if contribution.logical_observables is None:
            return
        raise RuntimeError(
            f"functional logical contribution "
            f"{contribution.owner_key} crosses stream-segment "
            f"boundary {commit_lo}-{commit_hi}"
        )
    raise RuntimeError(
        f"logical contribution {contribution.owner_key} crosses "
        f"strict interval boundary {commit_lo}-{commit_hi}"
    )


def _xor_of(contributions: list, stream_id) -> tuple:
    """The XOR of every contribution's observables, one arity throughout."""
    first = contributions[0]
    arity = len(first.logical_observables)
    aggregate = [0] * arity
    for contribution in contributions:
        logical_observables = contribution.logical_observables
        if len(logical_observables) != arity:
            raise RuntimeError(
                f"logical prediction interval {stream_id!r} changed "
                "observable arity during aggregation"
            )
        for observable_index, bit in enumerate(logical_observables):
            aggregate[observable_index] ^= bit
    return tuple(aggregate)

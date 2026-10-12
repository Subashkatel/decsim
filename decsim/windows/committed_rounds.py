"""The logical ledger: which owner committed which rounds of a stream.

A contribution is one owner (an ordinary window, a strong window, or the
folded windows a stream let go) over an exact inclusive round extent.
The contributions of one stream tile it without gap or overlap, which
every read checks, and the observables of an interval are the XOR of
the contributions that cover it. A strong
result may replace an owner's prediction; a strong window may replace
ordinary windows. tests/windows/test_committed_rounds.py checks the
ledger against a per-round oracle.
"""

import dataclasses
from typing import Any, Optional

import decsim.records.decoding as decoding_records


class LogicalLedger:
    """Which owner committed which rounds of each stream.

    Contributions sit by owner key, and all of a stream's carry one
    observable arity.
    """

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

    def replace_contributions(
        self,
        owner_key: tuple,
        commit_lo: int,
        commit_hi: int,
        replaced_keys: tuple,
    ) -> None:
        """A strong window takes the extent of the windows it replaces.

        The windows it absorbs and the escalated window's own entry
        leave the ledger. Its contribution carries no observables until
        it commits, and a read refuses any owner left inside its extent,
        since two owners never claim one round (Toshio et al. 2510.25222
        Theorem 1).
        """
        replaced = {owner_key, *replaced_keys}
        kept = {}
        for other_key, contribution in self.contributions.items():
            if other_key not in replaced:
                kept[other_key] = contribution
        kept[owner_key] = decoding_records.LogicalContribution(
            owner_key=owner_key,
            commit_lo=commit_lo,
            commit_hi=commit_hi,
            ownership_kind="strong_window",
            logical_observables=None,
        )
        self.contributions = kept

    def install(
        self, contribution: decoding_records.LogicalContribution
    ) -> None:
        """Record who owns an extent, one observable arity per stream."""
        if contribution.commit_lo < 1:
            _refuse_extent(contribution)
        if contribution.commit_hi < contribution.commit_lo:
            _refuse_extent(contribution)
        copied = _with_copied_observables(contribution)
        self._check_arity(copied)
        self.contributions[copied.owner_key] = copied

    def observables_for_interval(
        self,
        stream_id: Any,  # an opaque identity
        commit_lo: int,
        commit_hi: int,
        *,
        boundary_policy: str,
    ) -> Optional[tuple]:
        """The XOR of the contributions covering [commit_lo, commit_hi].

        None when any covering contribution is timing-only.
        """
        contributions = self.contributions_for_interval(
            stream_id, commit_lo, commit_hi, boundary_policy=boundary_policy
        )
        for contribution in contributions:
            if contribution.logical_observables is None:
                return None
        return _xor_of(contributions)

    def contributions_for_interval(
        self,
        stream_id: Any,  # an opaque identity
        commit_lo: int,
        commit_hi: int,
        *,
        boundary_policy: str,
    ) -> list:
        """The contributions covering [commit_lo, commit_hi], in extent order.

        The interval must be tiled without gap or overlap. Under strict,
        no contribution may cross the interval's edge; under
        stream_segment, only a functional (observable-bearing) one may
        not.
        """
        contributions = self._covering(stream_id, commit_lo, commit_hi)
        _check_tiling(contributions, stream_id, commit_lo, commit_hi)
        for contribution in contributions:
            _check_inside_interval(
                contribution, commit_lo, commit_hi, boundary_policy
            )
        return contributions

    def replace_prediction(
        self, owner_key: tuple, logical_observables: tuple
    ) -> None:
        """A strong result replaces the owner's prediction, extent unchanged."""
        contribution = self.contributions[owner_key]
        replaced = decoding_records.LogicalContribution(
            owner_key=contribution.owner_key,
            commit_lo=contribution.commit_lo,
            commit_hi=contribution.commit_hi,
            ownership_kind=contribution.ownership_kind,
            logical_observables=logical_observables,
        )
        self.install(replaced)

    def fold_before(
        self,
        stream_id: Any,  # an opaque identity
        first_kept_round: int,
        cut_rounds: tuple,
    ) -> None:
        """The stream's contributions before the round, folded by stretch.

        A stream lets its final windows go (window_release.py), so the
        contributions that end before the round become one per stretch
        between the segment cuts: a "folded" owner, keyed (stream,
        "folded", its first round), over the stretch's rounds folded so
        far, with the XOR of their observables, or None when any is
        timing-only. A read XORs what covers its interval and is None
        when any of it is, so a read over folded rounds gives what it
        gave before; a segment's first round starts a stretch, so a read
        from it crosses no fold.
        """
        folding = self._ending_before(stream_id, first_kept_round)
        stretches = []
        for contribution in folding:
            del self.contributions[contribution.owner_key]
            _fold_into(stretches, contribution, cut_rounds)
        for stretch in stretches:
            self.contributions[stretch.owner_key] = stretch

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

    def _ending_before(self, stream_id, round_index: int) -> list:
        """The stream's contributions that end before the round, in order."""
        ending = []
        for key, contribution in self.contributions.items():
            if key[0] != stream_id:
                continue
            if contribution.commit_hi >= round_index:
                continue
            ending.append(contribution)
        ending.sort(key=_extent_order)
        return ending

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


def _refuse_extent(contribution: decoding_records.LogicalContribution) -> None:
    raise ValueError(
        f"logical contribution {contribution.owner_key} has invalid "
        f"extent {contribution.commit_lo}-{contribution.commit_hi}"
    )


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
            f"functional logical contribution {contribution.owner_key} "
            f"spans rounds {contribution.commit_lo}-"
            f"{contribution.commit_hi} and crosses the stream segment "
            f"{commit_lo}-{commit_hi}: the window plan starts and ends "
            f"a window on each segment's rounds"
        )
    raise RuntimeError(
        f"logical contribution {contribution.owner_key} crosses "
        f"strict interval boundary {commit_lo}-{commit_hi}"
    )


def _fold_into(
    stretches: list,
    contribution: decoding_records.LogicalContribution,
    cut_rounds: tuple,
) -> None:
    """The contribution joined to the last stretch, or one of its own.

    A stretch ends on a cut, where a segment's rounds end; folded
    contributions tile, so a gap is refused as a read refuses it.
    """
    if not stretches:
        stretch = _stretch_of(contribution)
        stretches.append(stretch)
        return
    previous = stretches[-1]
    following_round = previous.commit_hi + 1
    if contribution.commit_lo != following_round:
        raise RuntimeError(
            f"folding the logical ledger of stream {previous.owner_key[0]!r} "
            f"found no contribution at round {following_round}"
        )
    if previous.commit_hi in cut_rounds:
        stretch = _stretch_of(contribution)
        stretches.append(stretch)
        return
    stretches[-1] = _joined(previous, contribution)


def _stretch_of(
    contribution: decoding_records.LogicalContribution,
) -> decoding_records.LogicalContribution:
    """The contribution as a folded stretch of its own."""
    stream_id = contribution.owner_key[0]
    owner_key = (stream_id, "folded", contribution.commit_lo)
    return dataclasses.replace(
        contribution, owner_key=owner_key, ownership_kind="folded"
    )


def _joined(
    stretch: decoding_records.LogicalContribution,
    contribution: decoding_records.LogicalContribution,
) -> decoding_records.LogicalContribution:
    """The stretch through the contribution's rounds, observables XORed."""
    observables = None
    if _is_functional(stretch) and _is_functional(contribution):
        observables = _xor_of([stretch, contribution])
    return dataclasses.replace(
        stretch,
        commit_hi=contribution.commit_hi,
        logical_observables=observables,
    )


def _is_functional(contribution: decoding_records.LogicalContribution) -> bool:
    return contribution.logical_observables is not None


def _with_copied_observables(
    contribution: decoding_records.LogicalContribution,
) -> decoding_records.LogicalContribution:
    """The contribution with its observables as ints, as they are now.

    A decoder may keep and later change the sequence it returned, so the
    ledger holds the values of the install, not the decoder's object.
    """
    observables = contribution.logical_observables
    if observables is None:
        return contribution
    copied = tuple(int(bit) for bit in observables)
    return dataclasses.replace(contribution, logical_observables=copied)


def _xor_of(contributions: list) -> tuple:
    """The XOR of every contribution's observables, one arity throughout."""
    first = contributions[0]
    arity = len(first.logical_observables)
    aggregate = [0] * arity
    for contribution in contributions:
        observables = contribution.logical_observables
        for observable_index, bit in enumerate(observables):
            aggregate[observable_index] ^= bit
    return tuple(aggregate)

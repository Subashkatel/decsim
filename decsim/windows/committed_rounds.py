"""Which window owns which committed rounds of a stream, and the logical
observables that result. A contribution is one owner (an ordinary window or
a strong slab) over an exact inclusive round extent; contributions of one
stream tile it without gap or overlap, and the observables of an interval
are the XOR of the contributions that cover it. A strong result may replace
the prediction of an owner; a strong slab may replace ordinary windows.
"""

from __future__ import annotations

from typing import Optional

from ..message import LogicalContribution


class LogicalLedger:
    def __init__(self):
        self.contributions: dict[tuple, LogicalContribution] = {}
        self._arity_by_stream: dict[object, int] = {}

    def get(self, owner_key: tuple) -> Optional[LogicalContribution]:
        return self.contributions.get(owner_key)

    def drop(self, owner_key: tuple) -> None:
        """A window is decoded again: its contribution is gone until it commits."""
        self.contributions.pop(owner_key, None)

    def install(self, contribution: LogicalContribution) -> None:
        """Record who owns an extent; an owner never changes kind or extent,
        extents never overlap, and one stream has one observable arity."""
        if contribution.ownership_kind not in (
            "ordinary_window",
            "strong_slab",
        ):
            raise ValueError(
                "logical contribution ownership_kind must be "
                "'ordinary_window' or 'strong_slab'")
        if (
            contribution.commit_lo < 1
            or contribution.commit_hi < contribution.commit_lo
        ):
            raise ValueError(
                f"logical contribution {contribution.owner_key} has invalid "
                f"extent {contribution.commit_lo}-{contribution.commit_hi}")

        logical_observables = contribution.logical_observables
        stream_id = contribution.owner_key[0]
        previous = self.contributions.get(contribution.owner_key)
        if previous is not None and (
            previous.commit_lo != contribution.commit_lo
            or previous.commit_hi != contribution.commit_hi
            or previous.ownership_kind != contribution.ownership_kind
        ):
            raise RuntimeError(
                f"logical contribution {contribution.owner_key} cannot "
                f"change ownership from {previous.ownership_kind} "
                f"{previous.commit_lo}-{previous.commit_hi} to "
                f"{contribution.ownership_kind} "
                f"{contribution.commit_lo}-{contribution.commit_hi}")

        for other_key, other in self.contributions.items():
            if other_key == contribution.owner_key \
                    or other_key[0] != stream_id:
                continue
            if (
                contribution.commit_lo <= other.commit_hi
                and other.commit_lo <= contribution.commit_hi
            ):
                raise RuntimeError(
                    f"logical contribution {contribution.owner_key} extent "
                    f"{contribution.commit_lo}-{contribution.commit_hi} "
                    f"overlaps {other_key} extent "
                    f"{other.commit_lo}-{other.commit_hi}")

        if logical_observables is not None:
            observed_arity = len(logical_observables)
            expected_arity = self._arity_by_stream.get(stream_id)
            if (
                expected_arity is not None
                and observed_arity != expected_arity
            ):
                raise ValueError(
                    f"logical contribution {contribution.owner_key} has "
                    f"observable length {observed_arity}; expected "
                    f"{expected_arity} for stream {stream_id!r}")
            if expected_arity is None:
                self._arity_by_stream[stream_id] = observed_arity

        self.contributions[contribution.owner_key] = contribution
    def observables_for_interval(
        self,
        stream_id,
        commit_lo: int,
        commit_hi: int,
        *,
        boundary_policy: str,
    ) -> Optional[tuple[int, ...]]:
        if boundary_policy not in ("strict", "stream_segment"):
            raise ValueError(
                f"unknown logical contribution boundary policy "
                f"{boundary_policy!r}")
        if commit_lo < 1 or commit_hi < commit_lo:
            raise ValueError(
                f"invalid logical prediction interval "
                f"{commit_lo}-{commit_hi}")

        contributions = sorted(
            (
                contribution
                for key, contribution in self.contributions.items()
                if key[0] == stream_id
                and contribution.commit_lo <= commit_hi
                and contribution.commit_hi >= commit_lo
            ),
            key=lambda contribution: (
                contribution.commit_lo,
                contribution.commit_hi,
                repr(contribution.owner_key),
            ),
        )
        if not contributions:
            raise RuntimeError(
                f"logical prediction interval {stream_id!r} "
                f"{commit_lo}-{commit_hi} has no contribution coverage")

        cursor = commit_lo
        for contribution in contributions:
            covered_lo = max(contribution.commit_lo, commit_lo)
            covered_hi = min(contribution.commit_hi, commit_hi)
            if covered_lo != cursor:
                relation = "overlap" \
                    if covered_lo < cursor else "gap"
                raise RuntimeError(
                    f"logical prediction interval {stream_id!r} "
                    f"{commit_lo}-{commit_hi} has a contribution "
                    f"{relation} at round {cursor}")
            cursor = covered_hi + 1
        if cursor != commit_hi + 1:
            raise RuntimeError(
                f"logical prediction interval {stream_id!r} "
                f"{commit_lo}-{commit_hi} has a contribution gap at "
                f"round {cursor}")

        for contribution in contributions:
            crosses_boundary = (
                contribution.commit_lo < commit_lo
                or contribution.commit_hi > commit_hi
            )
            if not crosses_boundary:
                continue
            if (
                boundary_policy == "stream_segment"
                and contribution.logical_observables is None
            ):
                continue
            if boundary_policy == "stream_segment":
                raise RuntimeError(
                    f"functional logical contribution "
                    f"{contribution.owner_key} crosses stream-segment "
                    f"boundary {commit_lo}-{commit_hi}")
            raise RuntimeError(
                f"logical contribution {contribution.owner_key} crosses "
                f"strict interval boundary {commit_lo}-{commit_hi}")

        if any(
            contribution.logical_observables is None
            for contribution in contributions
        ):
            return None

        arity = len(contributions[0].logical_observables)
        aggregate = [0] * arity
        for contribution in contributions:
            logical_observables = contribution.logical_observables
            if len(logical_observables) != arity:
                raise RuntimeError(
                    f"logical prediction interval {stream_id!r} changed "
                    "observable arity during aggregation")
            for observable_index, bit in enumerate(logical_observables):
                aggregate[observable_index] ^= bit
        return tuple(aggregate)
    def replace_prediction(
        self,
        owner_key: tuple,
        logical_observables: tuple[int, ...],
    ) -> None:
        contribution = self.contributions.get(owner_key)
        if contribution is None:
            raise RuntimeError(
                f"result for {owner_key} has no logical contribution owner")
        self.install(
            LogicalContribution(
                owner_key=contribution.owner_key,
                commit_lo=contribution.commit_lo,
                commit_hi=contribution.commit_hi,
                ownership_kind=contribution.ownership_kind,
                logical_observables=logical_observables,
            )
        )

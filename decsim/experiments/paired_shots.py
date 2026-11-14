"""Three runs of the same shots, paired by seed: what the pair changed.

A hybrid decoder pairs a fast decoder with a strong one. Run on the
same seeds as the fast decoder alone and the strong one alone, each
shot says what the pair did to it (CoDecoder counts per shot against
the observable truth, tune.py lines 76-99, studies/adaptive-threshold):

- repaired: the fast decoder alone failed it and the pair did not;
- broken: the fast decoder alone did not fail it and the pair did;
- missed: the pair failed it and the strong decoder alone did not.

The best-of-two line is the shots both decoders alone failed: a pair
that chose the right decoder for every shot fails only those, so no
choice per shot between the two fails fewer. It is the ensemble
decoder that is right when at least one of its decoders is (Sheth et
al. 1905.02345 lines 186-196, equation 1).

Over the same shots the pair fails broken - repaired more times than
the fast decoder alone, and McNemar's exact test reads those two
counts (failure_statistics.exact_mcnemar). The runs are read from
shots.csv, a run's shots chosen by the values of its columns.

Runs that differ only in their decoder draw the same shot at the same
seed, and shots.csv holds each shot's sample_digest so that this is
checked (experiments/measure.py). A seed whose three shots are not one
draw is refused: its outcomes are not of one shot, so they do not pair.
"""

import dataclasses
from collections.abc import Iterable, Mapping

import decsim.experiments.failure_statistics as failure_statistics
import decsim.experiments.fold as fold


@dataclasses.dataclass(frozen=True)
class ShotOutcome:
    """One scored shot of one run: the draw it decoded and its outcome.

    sample_digest is the shot's cell of shots.csv, the sha256 of its
    sampled detection events and truth.
    """

    sample_digest: str
    failed: bool


@dataclasses.dataclass(frozen=True)
class PairedRepairs:
    """The pair's effect on the shots all three runs scored.

    best_of_two_failures is the best-of-two line: the shots the fast
    decoder alone and the strong one alone both failed. p_value is
    McNemar's exact two-sided p that the pair and the fast decoder alone
    fail as often, read off repaired and broken.
    """

    shots: int
    fast_failures: int
    strong_failures: int
    pair_failures: int
    best_of_two_failures: int
    repaired: int
    broken: int
    missed: int
    p_value: float


def outcomes_by_seed(rows: Iterable[Mapping], selection: Mapping) -> dict:
    """{seed: ShotOutcome} of the scored shots whose columns hold the selection.

    rows are shots.csv's, as csv.DictReader reads them; selection maps a
    column to the text its cell holds, such as {"decoding": "switching"}.
    An unscored shot has no outcome to pair, so it is left out.
    """
    outcome_by_seed = {}
    for row in rows:
        if not _holds(row, selection):
            continue
        if not fold.number_of(row["is_scored"]):
            continue
        seed = fold.number_of(row["seed"])
        failed = fold.number_of(row["logical_failure"])
        outcome_by_seed[seed] = ShotOutcome(row["sample_digest"], failed)
    return outcome_by_seed


def paired_repairs(
    fast: Mapping, strong: Mapping, pair: Mapping
) -> PairedRepairs:
    """Each run's {seed: ShotOutcome}, compared on the seeds all three hold.

    Raises:
        ValueError: a seed's three shots are not one draw.
    """
    seeds = set(fast) & set(strong) & set(pair)
    _refuse_seeds_of_several_draws(fast, strong, pair, seeds)
    fast_failed = _failed_seeds(fast, seeds)
    strong_failed = _failed_seeds(strong, seeds)
    pair_failed = _failed_seeds(pair, seeds)
    repaired = fast_failed - pair_failed
    broken = pair_failed - fast_failed
    missed = pair_failed - strong_failed
    both_failed = fast_failed & strong_failed
    p_value = failure_statistics.exact_mcnemar(len(repaired), len(broken))
    return PairedRepairs(
        shots=len(seeds),
        fast_failures=len(fast_failed),
        strong_failures=len(strong_failed),
        pair_failures=len(pair_failed),
        best_of_two_failures=len(both_failed),
        repaired=len(repaired),
        broken=len(broken),
        missed=len(missed),
        p_value=p_value,
    )


def _failed_seeds(outcome_by_seed: Mapping, seeds: set) -> set:
    return {seed for seed in seeds if outcome_by_seed[seed].failed}


def _refuse_seeds_of_several_draws(
    fast: Mapping, strong: Mapping, pair: Mapping, seeds: set
) -> None:
    """Runs that differ only in their decoder hold one draw a seed."""
    for seed in sorted(seeds):
        digests = {
            fast[seed].sample_digest,
            strong[seed].sample_digest,
            pair[seed].sample_digest,
        }
        if len(digests) == 1:
            continue
        raise ValueError(
            f"seed {seed} holds {len(digests)} different draws in the "
            "three runs (their sample_digest cells differ), so its outcomes "
            "cannot be paired; pair runs that differ only in their decoder"
        )


def _holds(row: Mapping, selection: Mapping) -> bool:
    """Whether every selected column's cell is the selected text."""
    return all(row[column] == text for column, text in selection.items())

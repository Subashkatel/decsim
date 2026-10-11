"""The ensemble: several decoders on one window, pooled into one answer.

Every ensemble the sources build runs its members inside one decoder,
in parallel, and pools what they return. Harmony (Shutty, Newman and
Villalonga 2401.12434) pools correlated-matching members three ways
(Sec. III.2); VibeLSD (2508.15743 Sec. III.4) stops once M of its BP
members converge and ranks their corrections by likelihood; the Coset
Ensemble (2606.11076 Eq. 12) votes over logical outcomes. So an
ensemble here is a decoder row: its members decode the window at once
on the pool's unit, and its pooling row picks the answer.

    vote             the prediction most members give (Harmony)
    sum_likelihood   the prediction whose members' likelihoods sum
                     largest (Harmony)
    most_likely      the most likely member's answer (Harmony)
    first_converged  the most likely of the first count members to
                     converge; the rest are stopped (VibeLSD)

A member's likelihood is its whole selected error set's, priced on the
shared window model as the sum of log((1 - p) / p) over its faults, the
weight a matching minimises; Harmony asks each member for "the set of
errors it used". Each answer carries the ensemble's agreement, the
share of members that vote with the majority prediction, which Harmony
uses as a confidence score (Sec. V) and the consensus row reads.

Members differ only as their rows do: Harmony's perturbed priors (Sec.
III.1) are a matching row's knob, not built. A vote tie goes to the
earliest listed member's prediction, and the vote answers with the
earliest listed member that gives it (the Coset Ensemble takes any).
"""

import dataclasses
import math
from typing import Optional, Protocol

import numpy

import decsim.decoders.decoder as decoder_module
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.seeds as seed_records

_SUCCEEDED = decoding_records.BackendDecodeStatus.SUCCEEDED


class PoolingSettings(Protocol):
    """A pooling row's settings record (this module), which builds it."""

    name: str
    reads_likelihood: bool

    def build(self) -> object:
        """The pooling, which answers pooled(runs, model)."""


@dataclasses.dataclass(frozen=True)
class MemberRun:
    """One member's decode of the window: its result and its time."""

    index: int
    result: Optional[decoding_records.DecodeResult]
    ticks: int

    @property
    def prediction(self) -> Optional[tuple]:
        """The member's logical observables; None without a prediction."""
        if self.result is None:
            return None
        return self.result.logical_observables

    @property
    def has_converged(self) -> bool:
        """Whether the member answered with a succeeded correction."""
        if self.prediction is None:
            return False
        status = self.result.decode_status
        return status is None or status is _SUCCEEDED


@dataclasses.dataclass(frozen=True)
class Pooled:
    """The member whose result answers, and when the ensemble answers."""

    answer: MemberRun
    ticks: int


class Vote:
    """Harmony's vote pooling: the prediction most members give."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The vote row, which takes no setting."""

        name = "vote"
        reads_likelihood = False

        def build(self) -> "Vote":
            """The vote."""
            return Vote()

    def pooled(self, runs: tuple, model: object) -> Pooled:
        """The earliest listed member of the winning prediction, at the last."""
        del model
        winning = majority_prediction(runs)
        answer = _first_giving(runs, winning)
        ticks = _last_finish(runs)
        return Pooled(answer, ticks)


class SumLikelihood:
    """Harmony's sum-likelihood pooling over the logical predictions.

    The likelihoods of the members giving each prediction are summed,
    and the largest sum wins; its most likely member answers.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The sum-likelihood row, which takes no setting."""

        name = "sum_likelihood"
        reads_likelihood = True

        def build(self) -> "SumLikelihood":
            """The sum-likelihood pooling."""
            return SumLikelihood()

    def pooled(self, runs: tuple, model: object) -> Pooled:
        """The winning prediction's most likely member, at the last finish."""
        weights = _weights(runs, model)
        winning = _largest_summed_likelihood(runs, weights)
        giving = _runs_giving(runs, winning)
        answer = _most_likely(giving, weights)
        ticks = _last_finish(runs)
        return Pooled(answer, ticks)


class MostLikely:
    """Harmony's most-likely-error pooling: the most likely member answers."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The most-likely row, which takes no setting."""

        name = "most_likely"
        reads_likelihood = True

        def build(self) -> "MostLikely":
            """The most-likely pooling."""
            return MostLikely()

    def pooled(self, runs: tuple, model: object) -> Pooled:
        """The member of least weight, once every member has finished."""
        weights = _weights(runs, model)
        with_prediction = _runs_with_prediction(runs)
        answer = _most_likely(with_prediction, weights)
        ticks = _last_finish(runs)
        return Pooled(answer, ticks)


class FirstConverged:
    """VibeLSD's stop rule: the first count converged members, ranked.

    Once count members have converged the rest are terminated, and the
    most likely of the count answers then (2508.15743 Sec. III.4.2,
    steps 3 and 4). With fewer converged, every member runs out and the
    converged ones are ranked; with none, the most likely of all
    answers with its own status, a best-effort correction (VibeLSD
    hands its averaged beliefs to LSD, which only BP members have).
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """count is VibeLSD's M, the converged members it waits for."""

        count: int = 1
        name = "first_converged"
        reads_likelihood = True

        def __post_init__(self) -> None:
            if self.count < 1:
                raise ValueError(
                    "ensemble.pooling.count must be at least 1, "
                    f"got {self.count}"
                )

        def build(self) -> "FirstConverged":
            """The stop rule over count converged members."""
            return FirstConverged(self.count)

    def __init__(self, count: int) -> None:
        self.count = count

    def pooled(self, runs: tuple, model: object) -> Pooled:
        """The most likely of the first count converged, when the last lands."""
        weights = _weights(runs, model)
        finished = sorted(runs, key=_finish_order)
        converged = [run for run in finished if run.has_converged]
        if len(converged) >= self.count:
            first = converged[: self.count]
            answer = _most_likely(first, weights)
            return Pooled(answer, first[-1].ticks)
        ranked = _best_effort(converged, runs)
        answer = _most_likely(ranked, weights)
        ticks = _last_finish(runs)
        return Pooled(answer, ticks)


class EnsembleDecoder(decoder_module.DecoderBase):
    """Members decoding one window at once, pooled into one answer.

    The members share the pool's unit, as an ensemble's instances share
    its hardware; the unit is held until the pooling row answers. The
    answer is one member's whole result, with the ensemble's agreement.
    """

    decoder_evidence = decoding_records.CONSENSUS_EVIDENCE

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """members are decoder rows; pooling is a row of this module."""

        members: tuple = ()
        pooling: PoolingSettings = MostLikely.Settings()
        name = "ensemble"

        def __post_init__(self) -> None:
            if len(self.members) < 2:
                raise ValueError(
                    "ensemble.members must list at least two decoder "
                    f"rows, got {len(self.members)}"
                )

        def build(self) -> "EnsembleDecoder":
            """Each member built, under its pooling row."""
            members = []
            for member_settings in self.members:
                member = member_settings.build()
                members.append(member)
            pooling = self.pooling.build()
            reads_likelihood = self.pooling.reads_likelihood
            return EnsembleDecoder(tuple(members), pooling, reads_likelihood)

    def __init__(
        self, members: tuple, pooling: object, reads_likelihood: bool
    ) -> None:
        _refuse_a_member_without_windows(members)
        if reads_likelihood:
            _refuse_mixed_representations(members)
        self.members = members
        self.pooling = pooling
        self.fault_model_requirement = _joined_requirement(members)

    def run_seed_children(self) -> tuple:
        """Each member under members, by its place in the list."""
        children = []
        for index, member in enumerate(self.members):
            path = (
                seed_records.RunSeedPathSegment("field", "members"),
                seed_records.RunSeedPathSegment("integer_key", index),
            )
            child = seed_records.RunSeedChild(path, member)
            children.append(child)
        return tuple(children)

    def latency(self, job: decoding_records.DecodeJob) -> int:
        """The slowest member's latency."""
        latencies = []
        for member in self.members:
            member_latency = member.latency(job)
            latencies.append(member_latency)
        return max(latencies)

    def occupancy(self, job: decoding_records.DecodeJob) -> Optional[int]:
        """None: the unit frees when the pooling answers, known at the end."""
        del job
        return None

    def start(
        self,
        job: decoding_records.DecodeJob,
        engine: engine_module.Engine,
        on_result: decoder_module.OnResult,
    ) -> None:
        """Decode on every member now; answer when the pooling says."""
        if job.cancelled or job.on_done is not None:
            ticks = self.latency(job)
            engine.schedule(
                ticks,
                lambda: on_result(None),
                label=f"decode_done({job.label})",
            )
            return
        runs = self._member_runs(job, engine.now)
        answer, ticks = self._answer(job, runs)
        engine.schedule(
            ticks,
            lambda: _deliver(job, answer, on_result),
            label=f"decode_done({job.label})",
        )

    def decode(
        self, job: decoding_records.DecodeJob
    ) -> decoding_records.DecodeResult:
        """The pooled answer, its timing dropped."""
        runs = self._member_runs(job, 0)
        answer, _ticks = self._answer(job, runs)
        return answer

    def _member_runs(self, job: decoding_records.DecodeJob, now: int) -> tuple:
        runs = []
        for index, member in enumerate(self.members):
            run = _member_run(index, member, job, now)
            runs.append(run)
        return tuple(runs)

    def _answer(self, job: decoding_records.DecodeJob, runs: tuple) -> tuple:
        """(The answer with the agreement, the ticks it answers at)."""
        pooled = self.pooling.pooled(runs, job.detector_error_model)
        agreement = agreement_of(runs)
        answer = dataclasses.replace(pooled.answer.result, agreement=agreement)
        return answer, pooled.ticks


def majority_prediction(runs: tuple) -> Optional[tuple]:
    """The prediction most members give; a tie to the earliest listed."""
    counts = _prediction_counts(runs)
    if not counts:
        return None
    member_counts = counts.values()
    largest = max(member_counts)
    for prediction, count in counts.items():
        if count == largest:
            return prediction
    return None


def agreement_of(runs: tuple) -> float:
    """The share of members voting with the majority (Harmony Sec. V).

    A member with no prediction votes with nobody.
    """
    counts = _prediction_counts(runs)
    if not counts:
        return 0.0
    member_counts = counts.values()
    largest = max(member_counts)
    return largest / len(runs)


def selection_weight(
    result: decoding_records.DecodeResult, model: object
) -> float:
    """The sum of log((1 - p) / p) over the selected faults; +inf without."""
    if result.selected_faults is None or model is None:
        return math.inf
    faults = model.require_faults(result.fault_representation)
    priors = numpy.asarray(faults.priors, dtype=float)
    selected = numpy.asarray(result.selected_faults, dtype=bool)
    chosen = priors[selected]
    with numpy.errstate(divide="ignore"):
        weights = numpy.log1p(-chosen) - numpy.log(chosen)
    total = numpy.sum(weights)
    return float(total)


def _deliver(
    job: decoding_records.DecodeJob,
    answer: decoding_records.DecodeResult,
    on_result: decoder_module.OnResult,
) -> None:
    if job.cancelled:
        on_result(None)
        return
    on_result(answer)


def _member_run(
    index: int, member: object, job: decoding_records.DecodeJob, now: int
) -> MemberRun:
    """The member's result and time: measured on the host, or declared."""
    if member.occupancy(job) is None:
        result, elapsed_nanoseconds = member.decode_timed(job)
        ticks = member.ticks_after_decode(result, elapsed_nanoseconds, now)
        return MemberRun(index, result, ticks)
    result = member.decode(job)
    ticks = member.latency(job)
    return MemberRun(index, result, ticks)


def _prediction_counts(runs: tuple) -> dict:
    """Each prediction's member count, in the order members first give it."""
    counts: dict = {}
    for run in runs:
        prediction = run.prediction
        if prediction is None:
            continue
        counts[prediction] = counts.get(prediction, 0) + 1
    return counts


def _weights(runs: tuple, model: object) -> dict:
    weights = {}
    for run in runs:
        weight = math.inf
        if run.result is not None:
            weight = selection_weight(run.result, model)
        weights[run.index] = weight
    return weights


def _largest_summed_likelihood(runs: tuple, weights: dict) -> Optional[tuple]:
    """The prediction whose members' likelihoods, exp(-weight), sum largest.

    Summed as a log-sum-exp; a tie goes to the earliest listed.
    """
    summed: dict = {}
    for run in _runs_with_prediction(runs):
        terms = summed.setdefault(run.prediction, [])
        terms.append(-weights[run.index])
    best = None
    best_score = -math.inf
    for prediction, terms in summed.items():
        score = numpy.logaddexp.reduce(terms)
        if score > best_score:
            best, best_score = prediction, score
    return best


def _most_likely(runs: list, weights: dict) -> MemberRun:
    """The run of least weight; a tie to the earliest listed."""
    keys = []
    for run in runs:
        key = (weights[run.index], run.index)
        keys.append(key)
    first = keys.index(min(keys))
    return runs[first]


def _best_effort(converged: list, runs: tuple) -> list:
    """The converged runs; else those with a prediction; else every run."""
    if converged:
        return converged
    with_prediction = _runs_with_prediction(runs)
    if with_prediction:
        return with_prediction
    return list(runs)


def _runs_with_prediction(runs: tuple) -> list:
    return [run for run in runs if run.prediction is not None]


def _runs_giving(runs: tuple, prediction: Optional[tuple]) -> list:
    return [run for run in runs if run.prediction == prediction]


def _first_giving(runs: tuple, prediction: Optional[tuple]) -> MemberRun:
    giving = _runs_giving(runs, prediction)
    if giving:
        return giving[0]
    return runs[0]


def _last_finish(runs: tuple) -> int:
    ticks = [run.ticks for run in runs]
    return max(ticks)


def _finish_order(run: MemberRun) -> tuple:
    return (run.ticks, run.index)


def _joined_requirement(members: tuple) -> object:
    requirement = members[0].fault_model_requirement
    for member in members[1:]:
        requirement = requirement.joined(member.fault_model_requirement)
    return requirement


def _refuse_a_member_without_windows(members: tuple) -> None:
    for member in members:
        if getattr(member, "fault_representation", None) is not None:
            continue
        row = type(member)
        message = (
            f"ensemble member {row.__name__} decodes no window model: an "
            "ensemble pools corrections, so every member must return one"
        )
        raise ValueError(message)


def _refuse_mixed_representations(members: tuple) -> None:
    representations = set()
    for member in members:
        representations.add(member.fault_representation)
    if len(representations) <= 1:
        return
    names = sorted(representation.value for representation in representations)
    raise NotImplementedError(
        f"ensemble members select faults in {names}: pricing one against "
        "another is not implemented yet (stage 5); Harmony recovers a "
        "matching's hyperedge errors by an edge-decomposition matching "
        "(2401.12434 Sec. III.2). Pool by vote, or list members of one "
        "representation"
    )

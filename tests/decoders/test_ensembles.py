"""The ensemble: Harmony's three poolings and VibeLSD's stop rule.

Harmony (Shutty, Newman and Villalonga 2401.12434 Sec. III.2) pools by
vote, by summed likelihood within each logical prediction, or by the
most likely member; VibeLSD (2508.15743 Sec. III.4.2) answers once
count members have converged, with the most likely of them. A member's
weight here is sum log((1 - p) / p) over its selected faults: on four
faults of prior 0.1, 0.1, 0.01 and 0.01, a fault weighs log 9 = 2.197
or log 99 = 4.595.

The timelines run one memory, mem1, on the declared fabric
(tests/declared_run.py): W0's input lands at 20 us, its answer rides
weak_decoder_to_frame (2 us) and commits 1 us later; W1 waits for W0's
boundary (decoder_to_decoder, 0.5 us), then its own input (5 us).
"""

import dataclasses
import math
import types

import numpy
import pytest

import decsim.confidence.consensus as consensus
import decsim.config as config
import decsim.decoders.decoder as decoder_module
import decsim.decoders.ensembles as ensembles
import decsim.engine as engine_module
import decsim.escalation.verdicts as verdicts
import decsim.machine as machine_module
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models
import tests.declared_run as declared_run
import tests.escalation.declared_fabric as declared_fabric

GRAPHLIKE = fault_models.FaultRepresentation.GRAPHLIKE
PHYSICAL = fault_models.FaultRepresentation.PHYSICAL
PRIORS = numpy.array([0.1, 0.1, 0.01, 0.01])
NONCONVERGED = decoding_records.BackendDecodeStatus.NONCONVERGED


class _Model:
    """A window model of four faults, the same in either representation."""

    def require_faults(self, representation):
        del representation
        return types.SimpleNamespace(priors=PRIORS)


MODEL = _Model()


def test_a_fault_weighs_the_log_of_its_odds_against():
    result = _result((1,), faults=(0, 2))

    weight = ensembles.selection_weight(result, MODEL)

    expected = math.log(9) + math.log(99)
    assert weight == pytest.approx(expected)


def test_vote_answers_the_prediction_most_members_give():
    """Two of three say (1,): the earliest listed of them answers."""
    runs = (
        _run(0, (0,), faults=(0,)),
        _run(1, (1,), faults=(2, 3)),
        _run(2, (1,), faults=(1, 2)),
    )

    pooling = ensembles.Vote()
    pooled = pooling.pooled(runs, MODEL)

    assert pooled.answer.index == 1


def test_a_vote_tie_goes_to_the_earliest_listed_prediction():
    runs = (_run(0, (1,)), _run(1, (0,)))

    pooling = ensembles.Vote()
    pooled = pooling.pooled(runs, MODEL)

    assert pooled.answer.index == 0


def test_most_likely_answers_the_member_of_least_weight():
    """Weights 2.2, 9.2 and 6.8: member 0 answers, though outvoted."""
    runs = (
        _run(0, (0,), faults=(0,)),
        _run(1, (1,), faults=(2, 3)),
        _run(2, (1,), faults=(1, 2)),
    )

    pooling = ensembles.MostLikely()
    pooled = pooling.pooled(runs, MODEL)

    assert pooled.answer.index == 0


def test_sum_likelihood_lets_two_likely_members_outweigh_one():
    """(1,) sums exp(-2.20) twice, (0,) exp(-2.20) once: (1,) wins.

    Its most likely member, member 1 (weight 2.20 against 6.79),
    answers.
    """
    runs = (
        _run(0, (0,), faults=(0,)),
        _run(1, (1,), faults=(1,)),
        _run(2, (1,), faults=(1, 2)),
    )

    pooling = ensembles.SumLikelihood()
    pooled = pooling.pooled(runs, MODEL)

    assert pooled.answer.index == 1


def test_a_pooling_that_waits_for_all_answers_at_the_slowest_member():
    runs = (_run(0, (0,), ticks=4), _run(1, (0,), ticks=9))

    pooling = ensembles.MostLikely()
    pooled = pooling.pooled(runs, MODEL)

    assert pooled.ticks == 9


def test_first_converged_answers_when_the_first_member_converges():
    """Member 1 converges at 3 ticks; member 0 at 8 is terminated."""
    runs = (_run(0, (0,), ticks=8), _run(1, (1,), ticks=3))

    pooling = ensembles.FirstConverged(1)
    pooled = pooling.pooled(runs, MODEL)

    assert (pooled.answer.index, pooled.ticks) == (1, 3)


def test_first_converged_skips_a_member_that_did_not_converge():
    runs = (
        _run(0, (0,), ticks=2, status=NONCONVERGED),
        _run(1, (1,), ticks=5),
        _run(2, (1,), ticks=7),
    )

    pooling = ensembles.FirstConverged(1)
    pooled = pooling.pooled(runs, MODEL)

    assert (pooled.answer.index, pooled.ticks) == (1, 5)


def test_first_converged_of_two_ranks_the_two_by_likelihood():
    """Members 1 and 2 converge by 6 ticks; member 2 is the more likely."""
    runs = (
        _run(0, (0,), ticks=9, faults=()),
        _run(1, (1,), ticks=4, faults=(2, 3)),
        _run(2, (1,), ticks=6, faults=(0,)),
    )

    pooling = ensembles.FirstConverged(2)
    pooled = pooling.pooled(runs, MODEL)

    assert (pooled.answer.index, pooled.ticks) == (2, 6)


def test_first_converged_with_none_converged_answers_best_effort():
    runs = (
        _run(0, (0,), ticks=3, faults=(2,), status=NONCONVERGED),
        _run(1, (1,), ticks=5, faults=(0,), status=NONCONVERGED),
    )

    pooling = ensembles.FirstConverged(1)
    pooled = pooling.pooled(runs, MODEL)

    assert (pooled.answer.index, pooled.ticks) == (1, 5)
    assert pooled.answer.result.decode_status is NONCONVERGED


def test_agreement_is_the_share_voting_with_the_majority():
    runs = (_run(0, (0,)), _run(1, (1,)), _run(2, (1,)), _run(3, None))

    agreement = ensembles.agreement_of(runs)

    assert agreement == 0.5


def test_the_ensemble_answers_after_its_pooling_with_the_agreement():
    """Members of 4 and 9 ticks disagree: the answer lands at 9 ticks."""
    vote = ensembles.Vote.Settings()
    fast = _FixedMember.Settings((0,), 4)
    slow = _FixedMember.Settings((1,), 9)
    ensemble = _ensemble(vote, fast, slow)
    engine = engine_module.Engine()
    delivered = []
    job = _job()

    ensemble.start(job, engine, delivered.append)
    engine.run()

    assert engine.now == 9
    assert delivered[0].logical_observables == (0,)
    assert delivered[0].agreement == 0.5


def test_a_race_answers_at_the_first_member_to_converge():
    """The member of 5 ticks answers then; the one of 12 is terminated."""
    race = ensembles.FirstConverged.Settings(count=1)
    slow = _FixedMember.Settings((0,), 12)
    fast = _FixedMember.Settings((1,), 5)
    ensemble = _ensemble(race, slow, fast)
    engine = engine_module.Engine()
    delivered = []
    job = _job()

    ensemble.start(job, engine, delivered.append)
    engine.run()

    assert delivered[0].logical_observables == (1,)
    assert engine.now == 5


def test_an_ensemble_needs_two_members():
    alone = (_FixedMember.Settings((0,), 1),)

    with pytest.raises(ValueError, match="at least two"):
        ensembles.EnsembleDecoder.Settings(members=alone)


def test_likelihood_pooling_over_two_representations_is_refused_for_now():
    graphlike = _FixedMember.Settings((0,), 1)
    physical = _FixedMember.Settings((0,), 1, representation=PHYSICAL)
    most_likely = ensembles.MostLikely.Settings()
    members = (graphlike, physical)
    settings = ensembles.EnsembleDecoder.Settings(
        members=members, pooling=most_likely
    )

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        settings.build()


def test_the_consensus_row_reports_the_agreement_as_its_soft_output():
    result = _result((1,))
    result.agreement = 0.75

    signal = consensus.Consensus()
    computation = signal.compute((result,))

    assert computation.soft_output.gap == 0.75
    assert computation.soft_output.source == consensus.CONSENSUS_SOURCE


def test_the_agreement_verdict_escalates_on_any_dissent_by_default():
    """Harmony's second pass runs whenever the first is not unanimous."""
    settings = verdicts.Agreement.Settings()
    verdict = settings.build()
    unanimous = _scored(1.0)
    dissent = _scored(0.75)
    job = _job()

    kept = verdict.decide(job, unanimous, None)
    escalated = verdict.decide(job, dissent, None)

    assert kept is decoding_records.Verdict.KEEP
    assert escalated is decoding_records.Verdict.ESCALATE


def test_a_lower_minimum_keeps_a_majority():
    settings = verdicts.Agreement.Settings(minimum=0.6)
    verdict = settings.build()
    majority = _scored(0.75)
    job = _job()

    kept = verdict.decide(job, majority, None)

    assert kept is decoding_records.Verdict.KEEP


def test_a_race_of_two_members_answers_at_the_faster():
    """Members of 10 and 4 us: W0 answers at 24 us, lands 26, commits 27.

    Its boundary reaches W1 at 24.5 us and W1's input lands at 29.5 us;
    W1 races too and lands at 35.5 us.
    """
    race = ensembles.FirstConverged.Settings(count=1)
    members = (_WindowMember.Settings(10.0), _WindowMember.Settings(4.0))
    machine = _weak_only(race, members)

    records = _records_by_window(machine)

    assert records[0] == ("weak", _ticks(26.0), _ticks(27.0), (0,))
    assert records[1] == ("weak", _ticks(35.5), _ticks(36.5), (0,))


def test_a_vote_of_three_answers_two_against_one_at_the_slowest():
    """Members of 10, 6 and 8 us say (0,), (1,), (1,) on W0.

    The vote waits for all three: (1,) lands at 32 us and commits 33.
    """
    members = (
        _WindowMember.Settings(10.0, ((0, (0,)),)),
        _WindowMember.Settings(6.0, ((0, (1,)),)),
        _WindowMember.Settings(8.0, ((0, (1,)),)),
    )
    vote = ensembles.Vote.Settings()
    machine = _weak_only(vote, members)

    records = _records_by_window(machine)

    assert records[0] == ("weak", _ticks(32.0), _ticks(33.0), (1,))


def test_a_disagreement_escalates_the_window_to_the_strong_pool():
    """W0's two members disagree: agreement 0.5 escalates it.

    The serial escalation of the declared switching machine follows
    (tests/escalation/test_deadlines.py): the strong decode runs 39-69
    us and its answer lands at 73 us. W1's members agree, so it is kept;
    its boundary crosses strong_decoder_to_weak_decoder (4 us) and its
    input (5 us), and it decodes 78-88 us, landing at 90.
    """
    members = (
        _WindowMember.Settings(10.0, ((0, (0,)),)),
        _WindowMember.Settings(6.0, ((0, (1,)),)),
    )
    machine = _layered(members)

    records = _records_by_window(machine)

    assert records[0][:3] == ("strong", _ticks(73.0), _ticks(74.0))
    assert records[1] == ("weak", _ticks(90.0), _ticks(91.0), (0,))


class _WindowMember(decoder_module.DecoderBase):
    """A member of declared latency and a declared prediction per window."""

    fault_model_requirement = fault_models.NO_FAULT_MODEL_REQUIRED
    fault_representation = GRAPHLIKE

    @dataclasses.dataclass(frozen=True)
    class Settings:
        microseconds: float
        # (window id, prediction) pairs; (0,) for a window not named
        prediction_by_window: tuple = ()
        name = "declared_member"

        def build(self):
            return _WindowMember(self)

    def __init__(self, settings) -> None:
        self.settings = settings

    def latency(self, job) -> int:
        del job
        return config.microseconds_to_ticks(self.settings.microseconds)

    def decode(self, job):
        table = dict(self.settings.prediction_by_window)
        prediction = table.get(job.window_id, (0,))
        return decoding_records.DecodeResult(
            job.operation_id, job.window_id, logical_observables=prediction
        )


def _weak_only(pooling, members) -> machine_module.Machine:
    """The declared weak-only machine over mem1, its pool an ensemble."""
    memory = declared_run.memory_operation(1)
    settings = declared_run.weak_only_settings(rounds=9, operations=[memory])
    ensemble = ensembles.EnsembleDecoder.Settings(
        members=members, pooling=pooling
    )
    weak = dataclasses.replace(settings.weak_decoder, algorithm=ensemble)
    settings = dataclasses.replace(settings, weak_decoder=weak)
    return declared_run.run_machine(settings)


def _layered(members) -> machine_module.Machine:
    """The declared switching machine: an ensemble, then the strong pool."""
    confidence = consensus.Consensus.Settings()
    agreement = verdicts.Agreement.Settings()
    switching = declared_run.declared_switching(
        confidence=confidence, verdict=agreement
    )
    settings = declared_fabric.switching_settings(
        rounds=9, escalated_windows=set(), switching=switching
    )
    vote = ensembles.Vote.Settings()
    ensemble = ensembles.EnsembleDecoder.Settings(members=members, pooling=vote)
    weak = dataclasses.replace(settings.weak_decoder, algorithm=ensemble)
    settings = dataclasses.replace(settings, weak_decoder=weak)
    return declared_run.run_machine(settings)


def _records_by_window(machine: machine_module.Machine) -> dict:
    """Each window's tier, landing, commit and prediction, by window id."""
    snapshot = machine.control.pauli_frame.snapshot()
    records = {}
    for record in snapshot.records:
        records[record.window_key[1]] = (
            record.tier,
            record.accepted_ticks,
            record.committed_ticks,
            record.logical_observables,
        )
    return records


def _ticks(microseconds: float) -> int:
    return config.microseconds_to_ticks(microseconds)


class _FixedMember(decoder_module.DecoderBase):
    """A member that always answers one prediction after fixed ticks."""

    fault_model_requirement = fault_models.NO_FAULT_MODEL_REQUIRED

    @dataclasses.dataclass(frozen=True)
    class Settings:
        prediction: tuple
        ticks: int
        representation: object = GRAPHLIKE
        name = "fixed"

        def build(self):
            return _FixedMember(self)

    def __init__(self, settings) -> None:
        self.settings = settings
        self.fault_representation = settings.representation

    def decode(self, job):
        del job
        prediction = self.settings.prediction
        return _result(prediction, representation=self.fault_representation)

    def latency(self, job) -> int:
        del job
        return self.settings.ticks


def _ensemble(pooling, *members) -> ensembles.EnsembleDecoder:
    settings = ensembles.EnsembleDecoder.Settings(
        members=members, pooling=pooling
    )
    return settings.build()


def _job() -> decoding_records.DecodeJob:
    job = decoding_records.DecodeJob(operation_id=1, window_id=0, round_count=3)
    job.detector_error_model = MODEL
    return job


def _scored(agreement: float) -> decoding_records.DecodeResult:
    result = _result((0,))
    result.soft_output = decoding_records.SoftOutput(
        gap=agreement, source=consensus.CONSENSUS_SOURCE
    )
    return result


def _run(index, prediction, ticks=1, faults=(0,), status=None):
    result = _result(prediction, faults=faults, status=status)
    return ensembles.MemberRun(index, result, ticks)


def _result(prediction, faults=(0,), status=None, representation=GRAPHLIKE):
    selected = numpy.zeros(len(PRIORS), dtype=bool)
    for fault in faults:
        selected[fault] = True
    return decoding_records.DecodeResult(
        1,
        0,
        logical_observables=prediction,
        decode_status=status,
        selected_faults=selected,
        fault_representation=representation,
    )

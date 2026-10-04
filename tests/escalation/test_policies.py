"""The escalation policies: one law per port method, and the paper's identity.

Toshio et al. 2510.25222 Sec. IV C: the switching rate is the mass of the
window-gap distribution below the threshold, so on one run the windows
escalated equal the recorded weak gaps below g_th equal the strong
corrections in the frame (one identity, checked here on its threshold and
its complementary-gap signal, which the weak decoder computes from its
own two forced-class solves). The port is gem5's conditional predictor
(src/cpu/pred/conditional.hh): a row answers and is told; the root acts.
"""

import dataclasses
import math
from typing import Optional

import pytest
import stim

import decsim.build.escalation as escalation_build
import decsim.confidence.complementary as complementary
import decsim.config as config
import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.decoders.settings as decoder_settings
import decsim.engine as engine_module
import decsim.escalation.policies as policies
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.observe.settings as observe_settings
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.windows.boundary_policies as boundary_policies
import decsim.windows.schemes.sliding as sliding_scheme
import decsim.windows.settings as window_settings
import tests.declared_run as declared_run
import tests.escalation.declared_fabric as fabric
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

# the decoder engines run at 100 MHz, a 10000-tick period
ENGINE_CLOCK = config.Clock(10_000)
ENGINE_CARD = decoder_settings.EngineSettings(clock=ENGINE_CLOCK)
# the frame writes one cycle of a 250 MHz clock, 4 ns
FRAME_CLOCK = config.Clock(4000)
SOURCE = decoding_records.SoftOutputSource(method="matching-gap")
WINDOW = window_records.Window(
    operation_id=1,
    window_index=1,
    commit_lo=4,
    commit_hi=6,
    buffer_hi=9,
    round_count=6,
)
JOB = decoding_records.DecodeJob(operation_id=1, window_id=1, round_count=6)
BOTH_TIERS = (
    window_records.DecoderTier.WEAK,
    window_records.DecoderTier.STRONG,
)


def _result(gap) -> decoding_records.DecodeResult:
    soft_output = None
    if gap is not None:
        soft_output = decoding_records.SoftOutput(gap=gap, source=SOURCE)
    return decoding_records.DecodeResult(
        1, 1, logical_observables=(0,), soft_output=soft_output
    )


def _switching(**arguments) -> policies.Switching:
    fixed = threshold_sources.FixedThreshold(2.0)
    return policies.Switching(threshold=fixed, **arguments)


def _always_auditing_online_threshold(
    *, threshold: float
) -> threshold_sources.OnlineThreshold:
    """An online source that audits every kept window (audit rate 1.0)."""
    tracker = threshold_sources.EscalationRateTracker(
        target_escalation_rate=0.0, threshold=threshold, step=0.0
    )
    audit = threshold_sources.AuditLane(audit_rate=1.0)
    adjustment = threshold_sources.TargetAdjustment(
        kept_bad_budget=0.5,
        adjust_factor=2.0,
        min_escalation_rate=1e-5,
        max_escalation_rate=0.9,
    )
    controller = threshold_sources.OnlineThresholdController(
        tracker, audit, adjustment
    )
    draws = _Draws()
    return threshold_sources.OnlineThreshold(controller, draws)


# ---- the paper's identity, Sec. IV C


def _weak_gaps(machine) -> list[float]:
    """The gap of every weak request that carries one, in request order.

    One gap per window: the companion forced-class request carries none.
    """
    gaps = []
    for record in machine.observation.decode_records.requests:
        if record.request_key.tier is not window_records.DecoderTier.WEAK:
            continue
        if record.soft_output is None:
            continue
        gaps.append(record.soft_output.gap)
    return gaps


def _strong_frame_writes(machine) -> int:
    """The frame's commits that the strong tier wrote."""
    frame = machine.control.pauli_frame.snapshot()
    return sum(1 for record in frame.records if record.tier == "strong")


def test_escalations_equal_gaps_below_the_threshold_equal_strong_frame_writes():
    """One d=3, 30-round memory shot at p=0.005 and g_th = 15 dB (seed 1)."""
    threshold_nats = 15.0 * math.log(10.0) / 10.0
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=30,
        distance=3,
        after_clifford_depolarization=0.005,
        before_round_data_depolarization=0.005,
        before_measure_flip_probability=0.005,
        after_reset_flip_probability=0.005,
    )
    operation = program_records.Operation(
        id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit
    )
    rounds_policy = round_policies.FixedRounds(30)
    workload = workload_settings.WorkloadSettings(
        operations=[operation], rounds_policy=rounds_policy
    )
    device = stim_device.StimDevice()
    source = declared_run.GivenSource(device)
    qpu = qpu_settings.QpuSettings(distance=3, source=source)
    held = boundary_policies.Held.Settings()
    windows = window_settings.WindowSettings(
        terminal_policy="lookahead", boundary_policy=held
    )
    decoder_manager = decoder_settings.DecoderManagerSettings()
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings()
    weak_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=matching,
        engine=ENGINE_CARD,
    )
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings()
    strong_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=matching,
        engine=ENGINE_CARD,
    )
    confidence = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=15.0
    )
    switching = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold
    )
    pauli_frame = pauli_frame_module.PauliFrameConfig(
        write_cycles=1, clock=FRAME_CLOCK
    )
    observation = observe_settings.ObservationSettings(
        record_switching_windows=True
    )
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        windows=windows,
        decoder_manager=decoder_manager,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        switching=switching,
        pauli_frame=pauli_frame,
        observation=observation,
    )
    machine = machine_module.Machine.build(settings, 1)
    machine.run()
    weak_gaps = _weak_gaps(machine)
    below = sum(1 for gap in weak_gaps if gap < threshold_nats)
    strong_frame_writes = _strong_frame_writes(machine)
    assert len(weak_gaps) == 10
    assert below == 3
    assert machine.decoders.decoder_manager.strong_requests.counts.needed == 3
    assert strong_frame_writes == 3


# ---- one law per port method


def test_switching_decodes_both_tiers_at_once_when_asked():
    parallel = _switching(run_both_at_once=True)
    assert parallel.tiers_for_ready_window(WINDOW) == BOTH_TIERS


def test_a_strong_result_teaches_the_online_source():
    online = _always_auditing_online_threshold(threshold=0.0)
    switching = policies.Switching(threshold=online)
    confident = _result(5.0)
    # the kept window is audited, so its verdict escalates
    verdict = switching.verdict_for_weak_result(JOB, confident)
    assert verdict is decoding_records.Verdict.ESCALATE
    revised = decoding_records.DecodeResult(1, 1, logical_observables=(1,))
    switching.learn_from_strong_result((1, 1), revised)
    assert online.controller.raise_count == 1


def test_a_plan_that_contradicts_itself_is_refused_at_build():
    with pytest.raises(ValueError):
        fabric.switching_machine(
            rounds=9,
            escalated_windows=set(),
            strong_window=declared_run.DOUBLE_WINDOW,
            run_both_at_once=True,
        )


def _double_window_settings(
    commit_rounds: int, buffer_rounds: int
) -> machine_settings.MachineSettings:
    """The gate's switching card with the double window and the sizes.

    Every part is written out as its record.
    """
    scheme = sliding_scheme.SlidingWindowScheme.Settings(
        commit_rounds=commit_rounds, buffer_rounds=buffer_rounds
    )
    windows = window_settings.WindowSettings(
        scheme=scheme, terminal_policy="lookahead"
    )
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings()
    weak_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=matching,
        engine=ENGINE_CARD,
    )
    belief_matching_settings = belief_matching.BeliefMatchingDecoder.Settings()
    strong_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=belief_matching_settings,
        engine=ENGINE_CARD,
    )
    confidence = complementary.ComplementaryGap.Settings()
    one_nat = threshold_sources.nats_to_decibels(1.0)
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=one_nat
    )
    double_window = strong_window_shapes.DoubleWindow.Settings()
    switching = escalation_settings.SwitchingSettings(
        confidence=confidence,
        threshold=threshold,
        strong_window=double_window,
    )
    return machine_settings.MachineSettings(
        windows=windows,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        switching=switching,
    )


def test_an_online_source_under_a_double_window_is_refused_as_serial():
    """check_plan's serial-only law.

    An audit label compares one window's weak and strong committed
    observables, and a double-window strong result owns a larger extent
    than the audited window.
    """
    online = _always_auditing_online_threshold(threshold=2.0)
    online_settings = threshold_sources.OnlineThreshold.Settings(
        threshold_decibels=20.0
    )
    double_window = strong_window_shapes.DoubleWindow.Settings()
    switching = declared_run.declared_switching(
        threshold=online_settings, strong_window=double_window
    )
    with pytest.raises(ValueError):
        fabric.switching_machine(
            rounds=9,
            escalated_windows=set(),
            strong_window=declared_run.DOUBLE_WINDOW,
            switching=switching,
            online_threshold=online,
        )


def test_an_online_source_beside_run_both_at_once_is_refused():
    online = _always_auditing_online_threshold(threshold=2.0)
    with pytest.raises(ValueError):
        policies.Switching(threshold=online, run_both_at_once=True)


class _KeepEverything:
    """A threshold source written outside decsim: the port, and no more.

    Its one constructor argument is the sweep point's threshold in nats,
    and its Settings record is the fixed row's, which builds this row.
    """

    audits_by_escalating = False

    @dataclasses.dataclass(frozen=True)
    class Settings(threshold_sources.FixedThreshold.Settings):
        def build(self) -> "_KeepEverything":
            return _KeepEverything(self.threshold_nats)

    def __init__(self, threshold_nats: float) -> None:
        self.threshold_nats = threshold_nats

    def decide_keep(self, job, result) -> bool:
        """Every weak result is confident enough."""
        del job
        del result
        return True

    def learn_from_strong_result(self, window_key, result) -> None:
        """This source learns nothing."""
        del window_key
        del result


def test_a_threshold_source_written_outside_decsim_runs_from_its_record():
    """A threshold source is one class and its Settings record, nothing else.

    Every window's declared gap asks to escalate, and the source keeps
    every weak result, so no strong request is made.
    """
    threshold = _KeepEverything.Settings(threshold_decibels=20.0)
    switching = declared_run.declared_switching(threshold=threshold)
    machine = fabric.switching_machine(
        rounds=9, escalated_windows={0, 1, 2}, switching=switching
    )
    assert isinstance(machine.switching.policy.threshold, _KeepEverything)
    result = machine.run()
    assert result.terminal_status == "complete"
    assert machine.decoders.decoder_manager.strong_requests.counts.needed == 0


class _Draws:
    """A uniform draw that always audits (0.0 is below every audit rate)."""

    def random(self) -> float:
        return 0.0


def _serial_switching_settings(
    boundary_policy,
) -> machine_settings.MachineSettings:
    """Serial switching (no double window) with the boundary policy given."""
    windows = window_settings.WindowSettings(
        terminal_policy="lookahead", boundary_policy=boundary_policy
    )
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings()
    weak_decoder = decoder_settings.DecoderPoolSettings(algorithm=matching)
    belief_matching_settings = belief_matching.BeliefMatchingDecoder.Settings()
    strong_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=belief_matching_settings
    )
    confidence = complementary.ComplementaryGap.Settings()
    one_nat = threshold_sources.nats_to_decibels(1.0)
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=one_nat
    )
    switching = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold
    )
    return machine_settings.MachineSettings(
        windows=windows,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        switching=switching,
    )


def test_serial_switching_refuses_eager_boundaries_at_build():
    """A provisional boundary shipped eagerly is never corrected.

    Under serial switching the weak result may be revised by the strong
    decoder, so the boundary waits for the final result; Eager would
    hand a successor a correction the strong tier later replaces.
    """
    eager = boundary_policies.Eager.Settings()
    settings = _serial_switching_settings(eager)
    with pytest.raises(ValueError):
        machine_module.Machine.build(settings, 0)


def test_the_double_window_refuses_held_boundaries_at_build():
    """The far boundary IS the restart window's weak commit.

    Toshio 2510.25222 Sec. III C: under the double window the weak chain
    keeps committing while the strong region decodes, so holding the
    weak boundaries until the strong result arrives would deadlock the
    strong window on itself.
    """
    settings = _double_window_settings(3, 3)
    held = boundary_policies.Held.Settings()
    scheme = sliding_scheme.SlidingWindowScheme.Settings(
        commit_rounds=3, buffer_rounds=3
    )
    windows = window_settings.WindowSettings(
        scheme=scheme, terminal_policy="lookahead", boundary_policy=held
    )
    settings = dataclasses.replace(settings, windows=windows)
    with pytest.raises(ValueError):
        machine_module.Machine.build(settings, 0)


class DelegatingWindowScheme:
    """A windowing scheme row a study adds: sliding windows, wrapped.

    It fills the WindowingScheme port by holding a sliding scheme and
    passing every call to it, and it declares the three facts the port
    asks a row to declare. That is what a new row is: one class, the
    declarations, and nothing else changes (gem5's port API, arXiv
    2007.03152 lines 489-491).
    """

    has_trailing_tail_context = True
    commits_in_one_serial_chain = True
    supports_dynamic_streams = True

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The delegating row's record: the sizes, and its build."""

        commit_rounds: Optional[int] = None
        buffer_rounds: Optional[int] = None
        name = "delegating"

        def build(self, terminal_policy) -> "DelegatingWindowScheme":
            del terminal_policy
            return DelegatingWindowScheme()

    def __init__(self) -> None:
        self.inner = sliding_scheme.SlidingWindowScheme(
            terminal_policy="lookahead"
        )

    def plan_operation(
        self,
        operation_id: int,
        round_count: int,
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ):
        """The windows the sliding scheme lays out."""
        return self.inner.plan_operation(
            operation_id,
            round_count,
            commit_round_count=commit_round_count,
            buffer_round_count=buffer_round_count,
        )

    def data_complete(self, window, *, readiness) -> bool:
        """Whether the window has every round it reads."""
        return self.inner.data_complete(window, readiness=readiness)


class UndeclaredWindowScheme:
    """The same kind of row with the port's declarations left off.

    It carries no port method either: the build refuses the row before
    it plans a window.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The undeclared row's record: the sizes, and its build."""

        commit_rounds: Optional[int] = None
        buffer_rounds: Optional[int] = None
        name = "undeclared"

        def build(self, terminal_policy) -> "UndeclaredWindowScheme":
            del terminal_policy
            return UndeclaredWindowScheme()


def test_a_windowing_scheme_added_from_outside_runs_under_switching():
    """The double window reads the row's declaration, not its class.

    A row that commits in one serial chain and keeps trailing tail
    context serves the double window, whatever class it is.
    """
    scheme = DelegatingWindowScheme.Settings()
    machine = fabric.switching_machine(
        rounds=15,
        escalated_windows={1},
        strong_window=declared_run.DOUBLE_WINDOW,
        round_microseconds=4.0,
        scheme=scheme,
    )
    machine.run()
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 4), "weak"),
        ((1, 1), "strong"),
    ]


def test_a_flush_tail_is_refused_under_switching():
    """A strong recovery reads past the last commit, which flush drops."""
    held = boundary_policies.Held.Settings()
    settings = _serial_switching_settings(held)
    windows = dataclasses.replace(settings.windows, terminal_policy="flush")
    settings = dataclasses.replace(settings, windows=windows)
    with pytest.raises(ValueError, match="no trailing tail context"):
        machine_module.Machine.build(settings, 0)


def test_a_switching_run_on_a_scheme_that_declares_nothing_still_stops():
    """The policy reads the facts the row does not declare."""
    scheme = UndeclaredWindowScheme.Settings()
    with pytest.raises(AttributeError):
        fabric.switching_machine(rounds=9, escalated_windows={1}, scheme=scheme)


def _switching_policy(threshold_decibels: float):
    """A switching row over a fixed threshold and the complementary gap."""
    confidence = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=threshold_decibels
    )
    settings = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold
    )
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings()
    weak = decoder_settings.DecoderPoolSettings(algorithm=matching)
    engine = engine_module.Engine()
    switching = escalation_build.Switching.build(settings, weak, engine)
    return switching.policy


def test_a_weak_result_with_no_soft_output_escalates_its_window():
    """The fail-safe: no confidence is not a confident answer.

    A job with no window model runs no growth and a signal that reports
    nothing leaves the result bare, so the window goes to the strong
    tier rather than being kept on a confidence nobody computed.
    """
    policy = _switching_policy(20.0)
    job = decoding_records.DecodeJob(operation_id=1, window_id=0, round_count=3)
    bare = decoding_records.DecodeResult(1, 0)

    verdict = policy.verdict_for_weak_result(job, bare)

    assert bare.soft_output is None
    assert verdict is decoding_records.Verdict.ESCALATE


def test_the_papers_twenty_decibels_is_the_threshold_in_nats():
    """Toshio 2510.25222 line 1623: "fix the gap threshold to be gth = 20 dB".

    The threshold record is in decibels because that is the paper's
    unit; a gap is compared in nats, so the record converts once.
    """
    twenty_decibels = threshold_sources.decibels_to_nats(20.0)
    natural_log_of_ten = math.log(10.0)
    by_hand = 20.0 * natural_log_of_ten / 10.0

    assert twenty_decibels == pytest.approx(4.605170185988092)
    assert twenty_decibels == pytest.approx(by_hand)


def test_a_gap_at_the_papers_threshold_is_kept_and_one_below_escalates():
    """The equality case is a keep, which is Toshio's Fig. 12 caption."""
    threshold = threshold_sources.decibels_to_nats(20.0)
    policy = _switching_policy(20.0)
    job = decoding_records.DecodeJob(operation_id=1, window_id=0, round_count=3)

    a_hair_under = threshold - 1e-9
    well_over = threshold + 1.0
    at_the_threshold = _result(threshold)
    just_below = _result(a_hair_under)
    well_above = _result(well_over)

    assert (
        policy.verdict_for_weak_result(job, at_the_threshold)
        is decoding_records.Verdict.KEEP
    )
    assert (
        policy.verdict_for_weak_result(job, just_below)
        is decoding_records.Verdict.ESCALATE
    )
    assert (
        policy.verdict_for_weak_result(job, well_above)
        is decoding_records.Verdict.KEEP
    )

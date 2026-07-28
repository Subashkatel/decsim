#==================================================================
# TESTS FOR POLICY SEAMS (deadlines, routing, switching, round time, idle decode)
#==================================================================
import pytest

from decsim.codes import SurfaceCodeModel
from decsim.config import us
from decsim.decoders import (CodeRouter, FunctionLatencyDecoder,
                             PerRoundDecoder, PresetLatencyDecoder,
                             SAMPLED_CONFIDENCE_SOURCE,
                             SampledConfidenceDecoder, SwitchingDecoder,
                             SwitchingRouter)
from decsim.frontends.circuit import CircuitFrontend, cnot_plus_two_t_circuit
from decsim.message import DecodeJob, DecodeResult, Operation
from decsim.schedulers import (EarliestDeadlineScheduler, EnqueueTimeDeadline,
                               ReactionPathDeadline)
from decsim.policies import from_mode
from decsim.planner import FixedRounds
from decsim.run_spec import RunSpec, simulate
from decsim.schemes import SlidingWindowScheme
from decsim.switching import Switching


class _RecordingTimingChild:
    def __init__(self, latency_ticks, logical_bit):
        self.latency_ticks = latency_ticks
        self.logical_bit = logical_bit
        self.decode_calls = []

    def run_manifest_config(self):
        return {
            "kind": "recording_timing_child",
            "latency_ticks": self.latency_ticks,
            "logical_bit": self.logical_bit,
        }

    def latency(self, job):
        return self.latency_ticks

    def decode(self, job):
        self.decode_calls.append((job.op_id, job.window_id, job.hint))
        return DecodeResult(
            job.op_id,
            job.window_id,
            correction=(self.logical_bit,),
            logical_observables=(self.logical_bit,),
        )


# ---- deadline policies ----------------------------------------------------------------

def test_deadline_policies():
    assert EnqueueTimeDeadline().deadline(None, None, 42, on_reaction_path=True) == 42
    pol = ReactionPathDeadline(slack_ticks=100)
    assert pol.deadline(None, None, 42, on_reaction_path=True) == 42
    assert pol.deadline(None, None, 42, on_reaction_path=False) == 142


def _contended_circuit():
    """Four background CNOTs registered before a feedback-blocked T chain, so under FIFO the
    reaction-path windows queue behind the Clifford windows."""
    ops = [Operation(i, f"CNOT(q{2*i+2},q{2*i+3})", (2*i + 2, 2*i + 3), clifford=True)
           for i in range(4)]
    ops.append(Operation(4, "T(q0)", (0,), clifford=False))
    ops.append(Operation(5, "T2(q0)", (0,), clifford=False, blocked_by=4))
    return CircuitFrontend(ops).build()


def test_reaction_path_deadline_beats_fifo_under_contention():
    def run(scheduler=None, deadline_policy=None):
        r = simulate(RunSpec(ops=_contended_circuit(), num_units=1, d=3,
                             rounds_policy=FixedRounds(11),
                             decoder=PresetLatencyDecoder(5.0), scheduler=scheduler,
                             deadline_policy=deadline_policy), verbose=False)
        return r.result.chip_done_ticks                  # ends with the blocked T's last round

    fifo = run()
    edf = run(scheduler=EarliestDeadlineScheduler(),
              deadline_policy=ReactionPathDeadline(slack_ticks=us(100.0)))
    assert edf < fifo                          # reaction-path-first shrinks the stall


@pytest.mark.parametrize(
    "switching",
    [
        Switching(
            confidence_threshold=0.5,
            expected_source=SAMPLED_CONFIDENCE_SOURCE,
        ),
        Switching(
            confidence_threshold=0.5,
            expected_source=SAMPLED_CONFIDENCE_SOURCE,
            run_both_at_once=True,
        ),
        Switching(
            confidence_threshold=0.5,
            expected_source=SAMPLED_CONFIDENCE_SOURCE,
            double_window=True,
        ),
    ],
    ids=["serial", "parallel", "double-window"],
)
def test_switching_strong_jobs_keep_their_policy_deadline(switching):
    slack_ticks = 123_456_789

    class RecordingPolicy(ReactionPathDeadline):
        def __init__(self):
            super().__init__(slack_ticks)
            self.deadline_by_window = {}

        def deadline(self, op, window, now, on_reaction_path):
            value = super().deadline(
                op,
                window,
                now,
                on_reaction_path=on_reaction_path,
            )
            self.deadline_by_window[id(window)] = value
            return value

    class RecordingScheduler(EarliestDeadlineScheduler):
        def __init__(self):
            self.insertions = []

        def insert(self, queue, job):
            self.insertions.append(job)
            super().insert(queue, job)

    policy = RecordingPolicy()
    scheduler = RecordingScheduler()
    execution = simulate(
        RunSpec(
            ops=[Operation(0, "memory", (0,), clifford=True)],
            d=3,
            rounds_policy=FixedRounds(12),
            round_us=1.0,
            scheme=SlidingWindowScheme(),
            strategy=switching,
            router=SwitchingRouter(
                SampledConfidenceDecoder(
                    PerRoundDecoder(0.2),
                    escalation_probability=1.0,
                ),
                PerRoundDecoder(5.0),
            ),
            unit_pools={"default": 1, "strong": 1},
            scheduler=scheduler,
            deadline_policy=policy,
            seed=7,
        ),
        verbose=False,
    )

    strong_jobs = [
        job for job in scheduler.insertions
        if job.strong_decode_for is not None
    ]
    assert len(strong_jobs) == execution.cluster.strong_needed
    assert strong_jobs
    for job in strong_jobs:
        assert job.window.t_first_round is not None
        assert job.deadline == policy.deadline_by_window[id(job.window)]


# ---- decoder routing --------------------------------------------------------------------

def test_code_router_routes_by_code_with_default():
    surface, bb = PresetLatencyDecoder(1.0), PresetLatencyDecoder(2.0)
    router = CodeRouter(default=surface, by_code={"bb": bb})
    assert router.route(DecodeJob(0, 0, 6, code="bb")) is bb
    assert router.route(DecodeJob(0, 0, 6, code="surface")) is surface
    assert router.route(DecodeJob(0, 0, 6)) is surface


def test_custom_router_by_hint():
    class HintRouter:
        def __init__(self, normal, strong):
            self.normal, self.strong = normal, strong
            self.needs_hyperedges = False

        def route(self, job):
            return self.strong if job.hint == "strong" else self.normal

    weak, strong = PresetLatencyDecoder(1.0), PresetLatencyDecoder(10.0)
    router = HintRouter(weak, strong)
    assert router.route(DecodeJob(0, 0, 6)) is weak
    assert router.route(DecodeJob(0, 0, 6, hint="strong")) is strong
    # The router seam accepts it end to end and dispatches every job to
    # weak/strong.
    r = simulate(RunSpec(ops=cnot_plus_two_t_circuit(), num_units=2, d=3,
                         rounds_policy=FixedRounds(11), router=router),
                 verbose=False)
    assert r.result.fully_done_ticks > 0


# ---- timing-level decoder switching (arXiv:2510.25222) ---------------------------------

def test_switching_decoder_latency_mix():
    weak = _RecordingTimingChild(latency_ticks=3, logical_bit=0)
    strong = _RecordingTimingChild(latency_ticks=11, logical_bit=1)
    job = DecodeJob(0, 0, 6)
    never = SwitchingDecoder(weak, strong, gamma_switch=0.0, handoff_us=0.0)
    assert never.latency(job) == 3
    assert job.hint is None
    never_result = never.decode(job)
    assert (
        never_result.correction,
        never_result.logical_observables,
        never_result.soft_output,
        never_result.boundary_defects,
        never_result.boundary_data,
    ) == (None, None, None, None, None)

    job2 = DecodeJob(0, 0, 6)
    always = SwitchingDecoder(weak, strong, gamma_switch=1.0, handoff_us=0.5)
    assert always.latency(job2) == 3 + 2 * us(0.5) + 11
    assert job2.hint is None
    assert always.switches == 1
    always_result = always.decode(job2)
    assert (
        always_result.correction,
        always_result.logical_observables,
        always_result.soft_output,
        always_result.boundary_defects,
        always_result.boundary_data,
    ) == (None, None, None, None, None)
    assert weak.decode_calls == []
    assert strong.decode_calls == []


def test_switching_decoder_manifest_declares_timing_only_sampling():
    decoder = SwitchingDecoder(
        PresetLatencyDecoder(1.0),
        PresetLatencyDecoder(10.0),
        gamma_switch=0.25,
    )

    expected = {
        "kind": "sampled_inline_switching_timing",
        "switch_probability": 0.25,
        "handoff_ticks": us(0.5),
        "weak_communication_ticks": 0,
        "timing_path_source": "bernoulli_per_job",
    }
    assert decoder.run_manifest_config() == expected

    completed = RunSpec(ops=[], decoder=decoder).build(verbose=False)
    published = [
        component["configuration"]
        for component in completed.manifest.to_json_value()["components"]
        if component["implementation"].endswith(".SwitchingDecoder")
    ]
    assert published == [expected]


def test_switching_decoder_cannot_reroute_completion_from_latency():
    weak_timing = _RecordingTimingChild(latency_ticks=3, logical_bit=0)
    strong_timing = _RecordingTimingChild(latency_ticks=11, logical_bit=1)
    bypass = _RecordingTimingChild(latency_ticks=5, logical_bit=1)
    inline = SwitchingDecoder(
        weak_timing,
        strong_timing,
        gamma_switch=1.0,
        handoff_us=0.0,
        seed=7,
    )
    execution = simulate(
        RunSpec(
            ops=[Operation(0, "memory", (0,), clifford=True)],
            d=3,
            rounds_policy=FixedRounds(3),
            router=SwitchingRouter(inline, bypass),
            unit_pools={"default": 1, "strong": 1},
            seed=None,
        ),
        verbose=False,
    )

    contribution = execution.window_manager.logical_contributions[(0, 0)]
    assert contribution.logical_observables is None
    assert bypass.decode_calls == []
    assert weak_timing.decode_calls == []
    assert strong_timing.decode_calls == []


def test_switching_decoder_changes_run_timing_without_changing_behavior():
    def run(gamma_switch):
        weak = _RecordingTimingChild(latency_ticks=6, logical_bit=0)
        strong = _RecordingTimingChild(latency_ticks=12, logical_bit=1)
        execution = simulate(
            RunSpec(
                ops=[Operation(0, "memory", (0,), clifford=True)],
                d=3,
                rounds_policy=FixedRounds(3),
                decoder=SwitchingDecoder(
                    weak,
                    strong,
                    gamma_switch=gamma_switch,
                    handoff_us=0.0,
                    seed=7,
                ),
                seed=None,
            ),
            verbose=False,
        )
        logical = execution.window_manager.logical_contributions[
            (0, 0)
        ].logical_observables
        return execution.engine.now, logical, weak.decode_calls, strong.decode_calls

    weak_only = run(0.0)
    weak_and_strong = run(1.0)

    assert weak_and_strong[0] - weak_only[0] == 12
    assert weak_only[1:] == (None, [], [])
    assert weak_and_strong[1:] == (None, [], [])


def test_switching_decoder_charges_t_comm_weak_on_every_path():
    """The paper's T_comm^weak is paid on EVERY decode (weak path included, its backlog
    recursion has both T_comm terms); default 0 keeps the old latencies exactly."""
    weak, strong = PresetLatencyDecoder(1.0), PresetLatencyDecoder(10.0)
    never = SwitchingDecoder(weak, strong, gamma_switch=0.0, t_comm_weak_us=1.1)
    assert never.latency(DecodeJob(0, 0, 6)) == us(1.1) + us(1.0)
    always = SwitchingDecoder(weak, strong, gamma_switch=1.0, handoff_us=0.5,
                              t_comm_weak_us=1.1)
    assert always.latency(DecodeJob(0, 0, 6)) == us(1.1) + us(1.0) + 2 * us(0.5) + us(10.0)


def test_switching_decoder_end_to_end():
    sw = SwitchingDecoder(PresetLatencyDecoder(1.0), PresetLatencyDecoder(10.0),
                          gamma_switch=0.5)
    r = simulate(RunSpec(ops=cnot_plus_two_t_circuit(), num_units=2, d=3,
                         rounds_policy=FixedRounds(11), decoder=sw, seed=7),
                 verbose=False)
    assert r.result.fully_done_ticks > 0                 # runs to completion with mixed latencies


def test_switching_decoder_run_seed_binding_replays_and_rejects_direct_use():
    def build():
        decoder = SwitchingDecoder(
            PresetLatencyDecoder(1.0),
            PresetLatencyDecoder(10.0),
            gamma_switch=0.5,
        )
        reservation = decoder.reserve_run_seed(37)
        decoder.commit_run_seed(reservation)
        return decoder

    first = build()
    second = build()
    first_latencies = []
    second_latencies = []
    for window_id in range(64):
        first_job = DecodeJob(0, window_id, 3)
        second_job = DecodeJob(0, window_id, 3)
        first_latencies.append(first.latency(first_job))
        second_latencies.append(second.latency(second_job))
    assert first_latencies == second_latencies
    assert len(set(first_latencies)) == 2
    assert first.switches == second.switches
    assert first.switches == sum(
        latency != us(1.0) for latency in first_latencies
    )

    with pytest.raises(ValueError, match=r"SwitchingDecoder.*already used"):
        first.reserve_run_seed(37)


def test_sampled_confidence_run_seed_binding_and_explicit_conflict():
    explicit = SampledConfidenceDecoder(
        PerRoundDecoder(1.0),
        escalation_probability=0.5,
        seed=0,
    )
    with pytest.raises(
        ValueError,
        match=r"SampledConfidenceDecoder.*explicit seed",
    ):
        explicit.reserve_run_seed(37)

    def build():
        decoder = SampledConfidenceDecoder(
            PerRoundDecoder(1.0),
            escalation_probability=0.5,
        )
        reservation = decoder.reserve_run_seed(37)
        decoder.commit_run_seed(reservation)
        return decoder

    first = build()
    second = build()
    assert [
        first.decode(DecodeJob(0, window_id, 3)).soft_output.gap
        for window_id in range(12)
    ] == [
        second.decode(DecodeJob(0, window_id, 3)).soft_output.gap
        for window_id in range(12)
    ]


def test_sampled_decoders_do_not_expose_their_rng_state():
    switching = SwitchingDecoder(
        PresetLatencyDecoder(1.0),
        PresetLatencyDecoder(10.0),
        gamma_switch=0.5,
    )
    confidence = SampledConfidenceDecoder(
        PerRoundDecoder(1.0),
        escalation_probability=0.5,
    )

    assert not hasattr(switching, "rng")
    assert not hasattr(confidence, "rng")


# ---- per-code round time (heterogeneous cadence infrastructure) ------------------------

def _memory_op():
    op = Operation(0, "M(q0)", (0,), clifford=True)
    return CircuitFrontend([op]).build()


def test_code_round_time_overrides_global_cadence():
    slow = SurfaceCodeModel(d=3, round_us=2.0)
    r = simulate(RunSpec(ops=_memory_op(), num_units=1, code=slow,
                         rounds_policy=FixedRounds(5), round_us=1.1,
                         decoder=PresetLatencyDecoder(0.5)), verbose=False)
    assert r.result.chip_done_ticks == 5 * us(2.0)       # the CODE's cadence, not the global


def test_global_cadence_is_default():
    r = simulate(RunSpec(ops=_memory_op(), num_units=1, d=3,
                         rounds_policy=FixedRounds(5), round_us=1.1,
                         decoder=PresetLatencyDecoder(0.5)), verbose=False)
    assert r.result.chip_done_ticks == 5 * us(1.1)


# ---- idle-round decoding mode (arXiv:2511.10633: memory rounds need decoding) -----------

def test_idle_rounds_decoded_only_when_enabled():
    def run(mode):
        r = simulate(RunSpec(ops=cnot_plus_two_t_circuit(), num_units=2, d=3,
                             rounds_policy=FixedRounds(11),
                             decoder=PresetLatencyDecoder(3.0),
                             idle_policy=from_mode(mode)), verbose=False)
        return [l for l in r.engine.log_lines if "mem(" in l]

    assert run("ignore") == []                 # ignored idle rounds do not load the decoder
    assert len(run("separate_decode_jobs")) > 0


# ---- caller-supplied latency functions ------------------------------------------------

def test_function_latency_decoder_prices_jobs_from_the_supplied_function():
    """The service time is whatever the caller's job->microseconds function
    says (e.g. a GPU model: fixed overhead + throughput terms); decode()
    stays a timing-only empty result."""
    decoder = FunctionLatencyDecoder(
        lambda job: 2.0 + 0.5 * job.n_rounds + 0.01 * job.spatial_nodes)
    short_wide = DecodeJob(op_id=0, window_id=0, n_rounds=4, spatial_nodes=100)
    long_narrow = DecodeJob(op_id=0, window_id=1, n_rounds=12, spatial_nodes=25)
    assert decoder.latency(short_wide) == us(2.0 + 2.0 + 1.0)
    assert decoder.latency(long_narrow) == us(2.0 + 6.0 + 0.25)
    result = decoder.decode(short_wide)
    assert (result.op_id, result.window_id) == (0, 0)
    assert result.correction is None and result.logical_observables is None

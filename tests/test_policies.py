#==================================================================
# TESTS FOR POLICY SEAMS (deadlines, routing, switching, round time, idle decode)
#==================================================================
import pytest

from decsim.codes import SurfaceCodeModel
from decsim.config import us
from decsim.decoders import (CodeRouter, FunctionLatencyDecoder,
                             PerRoundDecoder, PresetLatencyDecoder,
                             SampledConfidenceDecoder, SwitchingDecoder)
from decsim.frontends.circuit import CircuitFrontend, cnot_plus_two_t_circuit
from decsim.message import DecodeJob, Operation
from decsim.schedulers import (EarliestDeadlineScheduler, EnqueueTimeDeadline,
                               ReactionPathDeadline)
from decsim.policies import from_mode
from decsim.planner import FixedRounds
from decsim.run_spec import RunSpec, simulate


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
        def route(self, job):
            return self.strong if job.hint == "strong" else self.normal

    weak, strong = PresetLatencyDecoder(1.0), PresetLatencyDecoder(10.0)
    router = HintRouter(weak, strong)
    assert router.route(DecodeJob(0, 0, 6)) is weak
    assert router.route(DecodeJob(0, 0, 6, hint="strong")) is strong
    # the router seam accepts it end to end (decoder= is the placeholder the
    # router overrides; the custom router dispatches every job to weak/strong)
    r = simulate(RunSpec(ops=cnot_plus_two_t_circuit(), num_units=2, d=3,
                         rounds_policy=FixedRounds(11), router=router,
                         decoder=weak), verbose=False)
    assert r.result.fully_done_ticks > 0


# ---- timing-level decoder switching (arXiv:2510.25222) ---------------------------------

def test_switching_decoder_latency_mix():
    weak, strong = PresetLatencyDecoder(1.0), PresetLatencyDecoder(10.0)
    job = DecodeJob(0, 0, 6)
    never = SwitchingDecoder(weak, strong, gamma_switch=0.0, handoff_us=0.5)
    assert never.latency(job) == us(1.0) and job.hint is None
    assert never.decode(job).soft_output == 1.0

    job2 = DecodeJob(0, 0, 6)
    always = SwitchingDecoder(weak, strong, gamma_switch=1.0, handoff_us=0.5)
    # weak decode + two handoffs (decoder->decoder messaging) + strong decode
    assert always.latency(job2) == us(1.0) + 2 * us(0.5) + us(10.0)
    assert job2.hint == "strong" and always.switches == 1
    assert always.decode(job2).soft_output == 0.0


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
    first_paths = []
    second_paths = []
    for window_id in range(12):
        first_job = DecodeJob(0, window_id, 3)
        second_job = DecodeJob(0, window_id, 3)
        first.latency(first_job)
        second.latency(second_job)
        first_paths.append(first_job.hint)
        second_paths.append(second_job.hint)
    assert first_paths == second_paths

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
        first.decode(DecodeJob(0, window_id, 3)).soft_output
        for window_id in range(12)
    ] == [
        second.decode(DecodeJob(0, window_id, 3)).soft_output
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

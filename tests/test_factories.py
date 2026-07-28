#==================================================================
# TESTS FOR FACTORIES (continuous production + per-state provenance)
#==================================================================
import pytest

from decsim.config import us
from decsim.engine import Engine
from decsim.factories import (DistillationFactory, DistillLevel,
                              InfiniteFactory, MultiLevelDistillationFactory)
from decsim.metrics import MagicStateLatency


class ImmediateService:
    """DecodeService test helper: every correction decode completes instantly."""
    def submit_decode(self, round_count, on_done, label="", deadline=None,
                      code=None, spatial_nodes=None):
        on_done()


class DelayedService:
    """DecodeService test helper: fixed decode latency, records submit times."""
    def __init__(self, engine, latency_ticks):
        self.engine = engine
        self.latency_ticks = latency_ticks
        self.submit_times = []

    def submit_decode(self, round_count, on_done, label="", deadline=None,
                      code=None, spatial_nodes=None):
        self.submit_times.append(self.engine.now)
        self.engine.schedule(self.latency_ticks, on_done, label=label)


def _seeded_single_level_factory(explicit_seed=None):
    engine = Engine(verbose=False)
    factory = DistillationFactory(
        engine,
        num_units=1,
        cycle_ticks=1,
        decode_service=None,
        corr_rounds=0,
        n_corr=0,
        p_success=0.5,
        seed=explicit_seed,
    )
    return engine, factory


def _seeded_multi_level_factory(explicit_seed=None):
    engine = Engine(verbose=False)
    factory = MultiLevelDistillationFactory(
        engine,
        [DistillLevel(units=1, d=1, O=1, P=0.5)],
        W_ticks=1,
        M=1,
        N=1,
        prep_units=1,
        prep_O=0,
        prep_P=1.0,
        seed=explicit_seed,
    )
    return engine, factory


@pytest.mark.parametrize(
    "build_factory",
    [_seeded_single_level_factory, _seeded_multi_level_factory],
)
def test_factory_run_seed_binding_replays_and_explicit_zero_conflicts(
    build_factory,
):
    _, explicit = build_factory(0)
    with pytest.raises(ValueError, match=r"Factory.*explicit seed"):
        explicit.reserve_run_seed(53)

    runs = []
    for _ in range(2):
        engine, factory = build_factory()
        reservation = factory.reserve_run_seed(53)
        factory.commit_run_seed(reservation)
        delivered = []
        factory.request(0, lambda: delivered.append(engine.now))
        engine.run()
        runs.append((
            delivered,
            tuple(engine.log_lines),
            getattr(factory, "failures", None),
        ))
    assert runs[0] == runs[1]


@pytest.mark.parametrize(
    "build_factory",
    [_seeded_single_level_factory, _seeded_multi_level_factory],
)
def test_factory_direct_random_use_prevents_later_run_seed_binding(
    build_factory,
):
    engine, factory = build_factory()
    factory.request(0, lambda: None)
    engine.run()

    with pytest.raises(ValueError, match=r"Factory.*already used"):
        factory.reserve_run_seed(53)


@pytest.mark.parametrize(
    "build_factory",
    [_seeded_single_level_factory, _seeded_multi_level_factory],
)
def test_factory_rng_state_has_no_public_bypass(build_factory):
    _, factory = build_factory()

    assert not hasattr(factory, "rng")


@pytest.mark.parametrize(
    "factory_cls",
    [DistillationFactory, MultiLevelDistillationFactory],
)
@pytest.mark.parametrize("invalid_count", [True, -1, 0.0])
def test_factory_rejects_nonexact_or_negative_correction_count_before_events(
    factory_cls,
    invalid_count,
):
    engine = Engine(verbose=False)

    with pytest.raises(
        (TypeError, ValueError),
        match=r"n_corr.*nonnegative.*built-in int",
    ):
        if factory_cls is DistillationFactory:
            factory_cls(
                engine,
                num_units=1,
                cycle_ticks=1,
                decode_service=None,
                corr_rounds=0,
                n_corr=invalid_count,
                production="continuous",
                buffer_capacity=1,
            )
        else:
            factory_cls(
                engine,
                [DistillLevel(units=1, d=1)],
                W_ticks=1,
                decode_service=None,
                corr_rounds=0,
                n_corr=invalid_count,
                production="continuous",
                buffer_capacity=1,
            )

    assert engine._event_queue == []


@pytest.mark.parametrize(
    "factory_cls",
    [DistillationFactory, MultiLevelDistillationFactory],
)
def test_factory_correction_service_presence_matches_correction_count(
    factory_cls,
):
    engine = Engine(verbose=False)

    def construct(decode_service, n_corr):
        if factory_cls is DistillationFactory:
            return factory_cls(
                engine,
                num_units=1,
                cycle_ticks=1,
                decode_service=decode_service,
                corr_rounds=0,
                n_corr=n_corr,
            )
        return factory_cls(
            engine,
            [DistillLevel(units=1, d=1)],
            W_ticks=1,
            decode_service=decode_service,
            corr_rounds=0,
            n_corr=n_corr,
        )

    with pytest.raises(ValueError, match=r"decode_service.*None.*n_corr.*zero"):
        construct(ImmediateService(), 0)
    with pytest.raises(ValueError, match=r"decode_service.*required.*n_corr"):
        construct(None, 1)

    service = ImmediateService()
    active = construct(service, 1)
    assert active.run_seed_children()[0].child is service
    assert active.run_seed_children()[0].relative_path[0].value == (
        "decode_service"
    )
    assert construct(None, 0).run_seed_children() == ()


def test_continuous_requires_capacity():
    eng = Engine(verbose=False)
    with pytest.raises(ValueError):
        DistillationFactory(eng, 1, us(10), ImmediateService(), corr_rounds=1,
                            production="continuous")
    with pytest.raises(ValueError):
        DistillationFactory(eng, 1, us(10), ImmediateService(), corr_rounds=1,
                            production="freerun")


def test_continuous_fills_buffer_and_halts():
    eng = Engine(verbose=False)
    f = DistillationFactory(eng, num_units=2, cycle_ticks=us(10),
                            decode_service=ImmediateService(), corr_rounds=1, n_corr=2,
                            production="continuous", buffer_capacity=3)
    eng.run()                                  # free-runs from t=0, no request needed
    assert f.produced == 3 and f.store == 3    # filled to capacity, then HALTED
    # consuming a state re-opens a buffer slot and production resumes
    delivered = []
    f.request(0, lambda: delivered.append(True))
    eng.run()
    assert delivered == [True]
    assert f.produced == 4 and f.store == 3    # refilled the slot just taken


def test_demand_mode_produces_nothing_unasked():
    eng = Engine(verbose=False)
    f = DistillationFactory(eng, num_units=2, cycle_ticks=us(10),
                            decode_service=ImmediateService(), corr_rounds=1, n_corr=2)
    eng.run()
    assert f.produced == 0                     # demand-driven: idle without requests


def test_state_trace_provenance():
    eng = Engine(verbose=False)
    f = DistillationFactory(eng, num_units=1, cycle_ticks=us(10),
                            decode_service=ImmediateService(), corr_rounds=1, n_corr=2,
                            return_ticks=us(2.0))
    f.request(0, lambda: None)
    eng.run()
    assert len(f.traces) == 1
    tr = f.traces[0]
    assert tr.t_phys_done - tr.t_distill_start == us(10)   # the distillation cycle
    assert tr.t_corr_done == tr.t_phys_done                # immediate correction decodes
    assert tr.t_released - tr.t_corr_done == us(2.0)       # the return trip
    assert tr.t_delivered == tr.t_released                 # a consumer was waiting


def test_magic_state_latency_metric():
    eng = Engine(verbose=False)
    f = DistillationFactory(eng, num_units=1, cycle_ticks=us(10),
                            decode_service=ImmediateService(), corr_rounds=1, n_corr=2)
    f.request(0, lambda: None)
    eng.run()
    res = MagicStateLatency(f).result()
    assert res["distill"] == {"mean": us(10), "max": us(10), "n": 1}
    assert res["total"]["n"] == 1


def test_multilevel_continuous_fills_top_buffer():
    eng = Engine(verbose=False)
    f = MultiLevelDistillationFactory(
        eng, [DistillLevel(units=1, d=3)], W_ticks=us(1.0), M=2, N=1,
        prep_units=4, production="continuous", buffer_capacity=2)
    eng.run()
    assert f.buffer[1] == 2 and f.produced[1] == 2         # filled to capacity, halted


#==================================================================
# TICKET API (spec 5.21 port 19): cancel never delivers, FIFO holds
#==================================================================

def test_ticket_cancel_never_delivers_and_preserves_fifo():
    eng = Engine(verbose=False)
    f = DistillationFactory(eng, num_units=1, cycle_ticks=us(10),
                            decode_service=ImmediateService(),
                            corr_rounds=1, n_corr=1)
    delivered = []
    t1 = f.request(1, lambda: delivered.append(1))
    t2 = f.request(2, lambda: delivered.append(2))
    t3 = f.request(3, lambda: delivered.append(3))
    assert t2.cancel() is True                 # withdrawn before any delivery
    assert t2.cancel() is False                # idempotent: already gone
    eng.run()
    assert delivered == [1, 3]                 # op2 never delivered; FIFO kept
    assert 2 not in f._stall_start             # no stall accounting leak
    assert t1.cancel() is False and t3.cancel() is False   # already delivered


def test_multilevel_ticket_cancel_never_delivers_and_preserves_fifo():
    eng = Engine(verbose=False)
    f = MultiLevelDistillationFactory(
        eng, [DistillLevel(units=1, d=1, O=10)], W_ticks=us(1.0), M=1, N=1,
        prep_units=1, prep_O=0)
    delivered = []
    f.request(1, lambda: delivered.append(1))
    t2 = f.request(2, lambda: delivered.append(2))
    f.request(3, lambda: delivered.append(3))
    assert t2.cancel() is True
    eng.run()
    assert delivered == [1, 3]


def test_infinite_factory_ticket_is_dead():
    eng = Engine(verbose=False)
    delivered = []
    ticket = InfiniteFactory(eng).request(0, lambda: delivered.append(True))
    assert delivered == [True]                 # delivery was instant
    assert ticket.cancel() is False            # nothing pending to cancel


#==================================================================
# ONE OVERLAP RULE (spec 5.21): correction decodes are submitted at
# the END of the physical attempt in every factory, so single- and
# multi-level factories stall identically at equal parameters.
#==================================================================

def test_multilevel_corrections_submitted_at_end_of_physical_attempt():
    eng = Engine(verbose=False)
    svc = DelayedService(eng, us(2.0))
    f = MultiLevelDistillationFactory(
        eng, [DistillLevel(units=1, d=1, O=10)], W_ticks=us(1.0), M=1, N=1,
        prep_units=1, prep_O=0, decode_service=svc, corr_rounds=3, n_corr=2)
    f.request(0, lambda: None)
    eng.run()
    assert svc.submit_times == [us(10), us(10)]   # at phys end, not round start
    assert f.produced[1] == 1


def test_single_and_multi_level_stall_identically_at_equal_params():
    # single level: one unit, 10us physical cycle, 2 corrections at 2us each
    eng1 = Engine(verbose=False)
    svc1 = DelayedService(eng1, us(2.0))
    single = DistillationFactory(eng1, num_units=1, cycle_ticks=us(10),
                                 decode_service=svc1, corr_rounds=3, n_corr=2,
                                 return_ticks=0)
    single.request(0, lambda: None)
    eng1.run()

    # multi level, ONE level with the same physical time (O*d*W = 10us),
    # free input preparation (prep_O=0), same correction decode load
    eng2 = Engine(verbose=False)
    svc2 = DelayedService(eng2, us(2.0))
    multi = MultiLevelDistillationFactory(
        eng2, [DistillLevel(units=1, d=1, O=10)], W_ticks=us(1.0), M=1, N=1,
        prep_units=1, prep_O=0, decode_service=svc2, corr_rounds=3, n_corr=2)
    multi.request(0, lambda: None)
    eng2.run()

    assert single.total_stall == multi.total_stall == us(12)   # 10 phys + 2 corr

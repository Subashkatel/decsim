import hashlib
import random
import threading
from types import SimpleNamespace

import pytest

from decsim.message import RunSeedChild, RunSeedPathSegment, RunSeedReservation
from decsim.run_spec import RunSpec
from decsim.seeding import (
    _AtomicRunSeedConsumer,
    _RandomSeedConsumer,
    bind_run_seed,
    derive_component_seed,
)


MAX_SEED = (1 << 64) - 1


def field(name):
    return RunSeedPathSegment("field", name)


def join_threads(threads):
    for thread in threads:
        thread.join(timeout=3)
    assert all(not thread.is_alive() for thread in threads)


class IntegerSubclass(int):
    pass


class FatalReservationError(BaseException):
    pass


class BareAtomicConsumer(_AtomicRunSeedConsumer):
    def __init__(self, seed=None):
        self._initialize_run_seed_binding(seed)


class AtomicProbe(_AtomicRunSeedConsumer):
    def __init__(self, seed=None, *, explicit_result=None, entropy_result=None):
        self._initialize_run_seed_binding(seed)
        self.explicit_result = explicit_result
        self.entropy_result = entropy_result
        self.calls = []
        self.active_state = "original"

    def _explicit_run_seed(self):
        self.calls.append("explicit")
        if self.explicit_result is None:
            return self._explicit_seed
        return self.explicit_result

    def _entropy_seed(self):
        self.calls.append("entropy")
        return self.entropy_result

    def _prepare_run_seed_state(self, effective_seed):
        self.calls.append(("prepare", effective_seed, self._run_seed_lock.locked()))
        return ("prepared", effective_seed)

    def _install_run_seed_state(self, prepared_state):
        self.calls.append(("install", prepared_state, self._run_seed_lock.locked()))
        self.active_state = prepared_state


class RandomProbe(_RandomSeedConsumer):
    def __init__(self, seed=None):
        self._initialize_run_seed_state(seed)


class RecordingConsumer:
    def __init__(self, name, events=None):
        self.name = name
        self.events = [] if events is None else events
        self.reservations = []

    def reserve_run_seed(self, seed):
        self.events.append(("reserve", self.name, seed))
        source = "entropy" if seed is None else "derived"
        reservation = RunSeedReservation(source, seed, self.name)
        self.reservations.append(reservation)
        return reservation

    def commit_run_seed(self, reservation):
        self.events.append(("commit", self.name, reservation.proposed_seed))

    def cancel_run_seed(self, reservation):
        self.events.append(("cancel", self.name, reservation.proposed_seed))


class ChildrenComposite:
    def __init__(self, children):
        self.children = children

    def run_seed_children(self):
        return self.children


class DualComponent(RecordingConsumer):
    def __init__(self, name, children, events=None):
        super().__init__(name, events)
        self.children = children

    def run_seed_children(self):
        return self.children


class ChildRecordSubclass(RunSeedChild):
    pass


class ReservationSubclass(RunSeedReservation):
    pass


def test_atomic_consumer_initializes_default_hooks_and_live_label():
    """Atomic consumers start unclaimed, expose default hooks, and honor the live seed label."""
    consumer = BareAtomicConsumer(17)

    assert consumer._explicit_seed == 17
    assert type(consumer._run_seed_lock) is type(threading.Lock())
    assert not consumer._run_seed_lock.locked()
    assert consumer._pending_run_seed is None
    assert consumer._run_seed_claimed is False
    assert consumer._stochastic_use_started is False
    assert consumer._explicit_seed_label == "seed"
    assert consumer._explicit_run_seed() == 17
    assert consumer._entropy_seed() is None
    marker = object()
    assert consumer._prepare_run_seed_state(marker) is marker
    with pytest.raises(NotImplementedError):
        consumer._install_run_seed_state(marker)

    from decsim.decoders.relay_bp.window_decoder import RelayBpWindowDecoder

    relay = RelayBpWindowDecoder(gamma_table_seed=5)
    with pytest.raises(ValueError, match="explicit gamma-table seed"):
        relay.reserve_run_seed(9)


@pytest.mark.parametrize("seed", [None, 0, MAX_SEED])
def test_reservation_accepts_only_unsigned_builtin_seed_boundaries(seed):
    """Reservation accepts None and both exact unsigned seed boundaries."""
    consumer = AtomicProbe()

    reservation = consumer.reserve_run_seed(seed)

    assert consumer._pending_run_seed is reservation
    assert reservation.proposed_seed == seed
    consumer.cancel_run_seed(reservation)


@pytest.mark.parametrize(
    ("used", "claimed", "pending", "explicit", "message"),
    [
        (True, True, True, 2, "already used"),
        (False, True, True, 2, "already claimed"),
        (False, False, True, 2, "pending run-seed reservation"),
        (False, False, False, 2, "conflicts with numeric run root"),
    ],
)
def test_reservation_refusals_follow_transaction_precedence(
    used, claimed, pending, explicit, message
):
    """Overlapping refusal states report the first transaction condition in contract order."""
    consumer = AtomicProbe(explicit)
    consumer._stochastic_use_started = used
    consumer._run_seed_claimed = claimed
    if pending:
        consumer._pending_run_seed = RunSeedReservation("derived", 1, None)

    before = (
        consumer._stochastic_use_started,
        consumer._run_seed_claimed,
        consumer._pending_run_seed,
        consumer._explicit_seed,
    )
    with pytest.raises(ValueError, match=message):
        consumer.reserve_run_seed(1)
    after = (
        consumer._stochastic_use_started,
        consumer._run_seed_claimed,
        consumer._pending_run_seed,
        consumer._explicit_seed,
    )
    assert after == before


@pytest.mark.parametrize(
    ("consumer", "root", "source", "seed", "hook"),
    [
        (AtomicProbe(), 11, "derived", 11, None),
        (AtomicProbe(13), None, "explicit_local", 13, "explicit"),
        (AtomicProbe(entropy_result=19), None, "entropy", None, "entropy"),
    ],
)
def test_reservation_selects_effective_seed_source_and_prepares_under_lock(
    consumer, root, source, seed, hook
):
    """Reservation selects the required source and stages preparation while holding its lock."""
    reservation = consumer.reserve_run_seed(root)

    assert reservation.proposed_seed_source == source
    assert reservation.proposed_seed == seed
    effective_seed = 19 if source == "entropy" else seed
    expected_calls = [] if hook is None else [hook]
    expected_calls.append(("prepare", effective_seed, True))
    assert consumer.calls == expected_calls
    assert consumer.active_state == "original"


def test_preparation_failure_leaves_no_pending_or_active_change():
    """A preparation exception leaves no pending reservation and preserves active state."""
    class FailingPreparation(AtomicProbe):
        def _prepare_run_seed_state(self, effective_seed):
            assert self._run_seed_lock.locked()
            raise LookupError("preparation failed")

    consumer = FailingPreparation()

    with pytest.raises(LookupError, match="preparation failed"):
        consumer.reserve_run_seed(6)

    assert consumer._pending_run_seed is None
    assert consumer.active_state == "original"


def test_one_consumer_serializes_competing_reservations():
    """Two simultaneous reservations on one consumer yield one claim and one refusal."""
    consumer = AtomicProbe()
    start = threading.Barrier(3)
    outcomes = []
    outcome_lock = threading.Lock()

    def reserve():
        start.wait(timeout=3)
        try:
            result = ("reserved", consumer.reserve_run_seed(4))
        except ValueError as error:
            result = ("refused", str(error))
        with outcome_lock:
            outcomes.append(result)

    threads = [threading.Thread(target=reserve) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait(timeout=3)
    join_threads(threads)

    assert sorted(outcome[0] for outcome in outcomes) == ["refused", "reserved"]
    reservation = next(value for status, value in outcomes if status == "reserved")
    consumer.cancel_run_seed(reservation)


def test_distinct_consumers_can_prepare_concurrently():
    """Separate consumers can enter their own preparation sections at the same time."""
    preparation_barrier = threading.Barrier(2)

    class ConcurrentProbe(AtomicProbe):
        def _prepare_run_seed_state(self, effective_seed):
            preparation_barrier.wait(timeout=3)
            return super()._prepare_run_seed_state(effective_seed)

    consumers = [ConcurrentProbe(), ConcurrentProbe()]
    reservations = [None, None]

    def reserve(index):
        reservations[index] = consumers[index].reserve_run_seed(index)

    threads = [threading.Thread(target=reserve, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    join_threads(threads)

    assert all(reservation is not None for reservation in reservations)
    for consumer, reservation in zip(consumers, reservations):
        consumer.cancel_run_seed(reservation)


def test_random_consumer_stages_swap_and_reproduces_integer_seed():
    """Random consumers preserve the active generator until commit and reproduce integer-seeded draws."""
    initialized = RandomProbe(2)
    assert initialized._rng.getstate() == random.Random(2).getstate()

    consumer = RandomProbe(None)
    active_generator = consumer._rng
    active_state = active_generator.getstate()

    reservation = consumer.reserve_run_seed(29)

    assert consumer._rng is active_generator
    assert consumer._rng.getstate() == active_state
    assert reservation.prepared_state is not active_generator
    consumer.commit_run_seed(reservation)
    assert consumer._rng is reservation.prepared_state
    reference = random.Random(29)
    assert [consumer._rng.random() for _ in range(4)] == [
        reference.random() for _ in range(4)
    ]



def test_random_consumer_accepts_entropy_preparation_without_comparing_outputs():
    """Random consumers can stage and install None-seeded state without output inequality claims."""
    consumer = RandomProbe(None)
    active_generator = consumer._rng
    active_state = active_generator.getstate()

    reservation = consumer.reserve_run_seed(None)

    assert reservation.proposed_seed_source == "entropy"
    assert reservation.proposed_seed is None
    assert consumer._rng is active_generator
    assert consumer._rng.getstate() == active_state
    consumer.commit_run_seed(reservation)
    assert consumer._rng is reservation.prepared_state
    assert 0.0 <= consumer._rng.random() < 1.0


@pytest.mark.parametrize(
    ("root", "path", "expected"),
    [
        (0, (), 4535200295187781575),
        (1, (RunSeedPathSegment("field", "device"),), 14587833418448047332),
        (
            MAX_SEED,
            (
                RunSeedPathSegment("field", "metrics"),
                RunSeedPathSegment("string_key", "latency"),
            ),
            3551856422068386999,
        ),
        (
            23,
            (
                RunSeedPathSegment("none_key", None),
                RunSeedPathSegment("integer_key", -2),
            ),
            351941757342560240,
        ),
    ],
)
def test_component_seed_derivation_matches_fixed_vectors_and_framing(root, path, expected):
    """Component seeds match fixed vectors built from the versioned framed preimage."""
    preimage = (
        b"decsim.run-seed.v1"
        + root.to_bytes(8, "big")
        + b"".join(segment.canonical_bytes() for segment in path)
    )
    independent = int.from_bytes(
        hashlib.blake2b(preimage, digest_size=8).digest(), "big"
    )

    assert independent == expected
    assert derive_component_seed(root, path) == expected


def stable_binder_trace(reverse):
    events = []
    child_early = RecordingConsumer("child-early", events)
    child_late = RecordingConsumer("child-late", events)
    dual_children = [
        RunSeedChild((field("z_child"),), child_late),
        RunSeedChild((field("a_child"),), child_early),
    ]
    if reverse:
        dual_children.reverse()
    dual = DualComponent("dual", dual_children, events)
    middle_leaf = RecordingConsumer("middle-leaf", events)
    composite = ChildrenComposite(
        (RunSeedChild((field("leaf"),), middle_leaf),)
    )
    last = RecordingConsumer("last", events)
    roots = [
        ((field("z_root"),), last),
        ((field("n_root"),), object()),
        ((field("m_root"),), composite),
        ((field("a_root"),), dual),
    ]
    if reverse:
        roots.reverse()

    bind_run_seed(41, roots)
    return events


def test_binder_dispatches_duck_typed_roles_in_stable_preorder():
    """Structural roles traverse sorted roots and children in depth-first consumer preorder."""
    forward = stable_binder_trace(False)
    reverse = stable_binder_trace(True)
    expected_names = ["dual", "child-early", "child-late", "middle-leaf", "last"]

    assert forward == reverse
    assert [name for action, name, _ in forward if action == "reserve"] == expected_names
    assert [name for action, name, _ in forward if action == "commit"] == expected_names
    assert [action for action, _, _ in forward] == ["reserve"] * 5 + ["commit"] * 5


@pytest.mark.parametrize(
    ("path", "rendered"),
    [
        ((), ""),
        ((field("device"),), "device"),
        ((RunSeedPathSegment("string_key", "latency"),), "['latency']"),
        ((RunSeedPathSegment("none_key", None),), "[None]"),
        ((RunSeedPathSegment("integer_key", -2),), "[-2]"),
        (
            (
                field("metrics"),
                RunSeedPathSegment("string_key", "latency"),
                RunSeedPathSegment("integer_key", 3),
            ),
            "metrics.['latency'].[3]",
        ),
    ],
)
def test_binder_rejects_duplicate_paths_through_rendered_diagnostics(path, rendered):
    """Duplicate canonical paths are rejected with their field and key rendering."""
    with pytest.raises(ValueError) as caught:
        bind_run_seed(1, [(path, object()), (path, object())])

    assert str(caught.value) == f"duplicate seed path {rendered}"


@pytest.mark.parametrize("same_child", [False, True])
def test_binder_rejects_duplicate_sibling_paths_for_same_or_distinct_objects(same_child):
    """Duplicate sibling paths fail before shared-object identity can suppress traversal."""
    first = object()
    second = first if same_child else object()
    composite = ChildrenComposite(
        (
            RunSeedChild((field("leaf"),), first),
            RunSeedChild((field("leaf"),), second),
        )
    )

    with pytest.raises(ValueError, match=r"duplicate seed path root\.leaf"):
        bind_run_seed(1, [((field("root"),), composite)])


def test_binder_rejects_self_and_multinode_cycles_with_both_paths():
    """Self and multi-node cycles report both the current path and first identity path."""
    self_cycle = ChildrenComposite([])
    self_cycle.children = (RunSeedChild((field("again"),), self_cycle),)
    with pytest.raises(ValueError) as self_error:
        bind_run_seed(1, [((field("root"),), self_cycle)])
    assert str(self_error.value) == "seed cycle from root.again to root"

    first = ChildrenComposite([])
    second = ChildrenComposite([])
    first.children = (RunSeedChild((field("second"),), second),)
    second.children = (RunSeedChild((field("first"),), first),)
    with pytest.raises(ValueError) as multi_error:
        bind_run_seed(1, [((field("root"),), first)])
    assert str(multi_error.value) == "seed cycle from root.second.first to root"


def test_binder_reserves_shared_alias_at_first_sorted_path_once():
    """A shared leaf is reserved once at its lexicographically first encountered path."""
    events = []
    leaf = RecordingConsumer("leaf", events)
    composite = ChildrenComposite(
        (
            RunSeedChild((field("z_path"),), leaf),
            RunSeedChild((field("a_path"),), leaf),
        )
    )

    bind_run_seed(7, [((field("root"),), composite)])

    expected_seed = derive_component_seed(7, (field("root"), field("a_path")))
    assert events == [
        ("reserve", "leaf", expected_seed),
        ("commit", "leaf", expected_seed),
    ]


@pytest.mark.parametrize(
    "record_factory",
    [
        lambda child: RunSeedChild((field("leaf"),), child),
        lambda child: ChildRecordSubclass((field("leaf"),), child),
        lambda child: SimpleNamespace(relative_path=(field("leaf"),), child=child),
    ],
)
def test_binder_accepts_structural_child_records(record_factory):
    """Exact, subclass, and duck child records all reach their nested consumer."""
    events = []
    leaf = RecordingConsumer("leaf", events)
    composite = ChildrenComposite((record_factory(leaf),))

    bind_run_seed(12, [((field("root"),), composite)])

    expected_seed = derive_component_seed(12, (field("root"), field("leaf")))
    assert events == [
        ("reserve", "leaf", expected_seed),
        ("commit", "leaf", expected_seed),
    ]


@pytest.mark.parametrize(
    ("record", "error"),
    [
        (SimpleNamespace(child=object()), AttributeError),
        (SimpleNamespace(relative_path=(field("leaf"),)), AttributeError),
        (SimpleNamespace(relative_path=[field("leaf")], child=object()), TypeError),
        (SimpleNamespace(relative_path=(object(),), child=object()), AttributeError),
    ],
)
def test_binder_allows_child_record_shape_to_fail_naturally(record, error):
    """Malformed structural child records fail naturally at the attribute or path operation used."""
    composite = ChildrenComposite((record,))

    with pytest.raises(error):
        bind_run_seed(1, [((field("root"),), composite)])


def test_binder_reserves_every_consumer_before_committing_in_order():
    """The binder completes all reservations before committing in acquisition order."""
    events = []
    consumers = [RecordingConsumer(name, events) for name in ("a", "b", "c")]
    roots = [((field(consumer.name),), consumer) for consumer in consumers]

    bind_run_seed(5, roots)

    assert [action for action, _, _ in events] == ["reserve"] * 3 + ["commit"] * 3
    assert [name for action, name, _ in events if action == "reserve"] == ["a", "b", "c"]
    assert [name for action, name, _ in events if action == "commit"] == ["a", "b", "c"]


@pytest.mark.parametrize("error", [RuntimeError("late"), FatalReservationError("fatal")])
@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_binder_rolls_back_prior_acquisitions_in_reverse_on_reserve_failure(
    error, failure_index
):
    """A reservation failure cancels every earlier acquisition in reverse order."""
    events = []

    class FailingConsumer(RecordingConsumer):
        def reserve_run_seed(self, seed):
            self.events.append(("reserve", self.name, seed))
            raise error

    names = ("a", "b", "c")
    consumers = [
        FailingConsumer(name, events)
        if index == failure_index
        else RecordingConsumer(name, events)
        for index, name in enumerate(names)
    ]

    with pytest.raises(type(error)):
        bind_run_seed(
            8,
            [((field(name),), consumer) for name, consumer in zip(names, consumers)],
        )

    expected = [
        ("reserve", name) for name in names[: failure_index + 1]
    ] + [
        ("cancel", name) for name in reversed(names[:failure_index])
    ]
    assert [event[:2] for event in events] == expected


@pytest.mark.parametrize("source", ["derived", "other", ""])
def test_binder_rejects_other_unseeded_source_labels(source):
    """Unseeded binding rejects every source label outside explicit local and entropy."""
    class InvalidSourceConsumer(RecordingConsumer):
        def reserve_run_seed(self, seed):
            return SimpleNamespace(
                proposed_seed_source=source,
                proposed_seed=None,
                prepared_state=None,
            )

    with pytest.raises(ValueError, match="unseeded components"):
        bind_run_seed(None, [((field("leaf"),), InvalidSourceConsumer("leaf"))])


@pytest.mark.parametrize(
    ("source", "proposed"),
    [("entropy", 999), ("explicit_local", None), ("explicit_local", "local")],
)
def test_binder_accepts_both_unseeded_sources_without_checking_proposed_seed(source, proposed):
    """Unseeded binding accepts both source labels without validating the proposed seed field."""
    committed = []

    class UnseededConsumer(RecordingConsumer):
        def reserve_run_seed(self, seed):
            assert seed is None
            return SimpleNamespace(
                proposed_seed_source=source,
                proposed_seed=proposed,
                prepared_state=None,
            )

        def commit_run_seed(self, reservation):
            committed.append(reservation)

    bind_run_seed(None, [((field("leaf"),), UnseededConsumer("leaf"))])

    assert len(committed) == 1


@pytest.mark.parametrize(
    "reservation_factory",
    [
        lambda seed: RunSeedReservation("derived", seed, "state"),
        lambda seed: ReservationSubclass("derived", seed, "state"),
        lambda seed: SimpleNamespace(
            proposed_seed_source="derived", proposed_seed=seed, prepared_state="state"
        ),
    ],
)
def test_binder_accepts_structural_reservations(reservation_factory):
    """Exact, subclass, and duck reservations all pass binder provenance checks."""
    committed = []

    class StructuralConsumer(RecordingConsumer):
        def reserve_run_seed(self, seed):
            return reservation_factory(seed)

        def commit_run_seed(self, reservation):
            committed.append(reservation)

    bind_run_seed(14, [((field("leaf"),), StructuralConsumer("leaf"))])

    assert len(committed) == 1


def test_binder_allows_empty_paths_and_naturally_rejects_malformed_graph_shapes():
    """Empty and list paths work where operations allow them, while malformed shapes fail naturally."""
    events = []
    empty = RecordingConsumer("empty", events)
    list_path = RecordingConsumer("list", events)

    bind_run_seed(4, [((), empty)])
    bind_run_seed(4, [([], list_path)])
    assert [event[0] for event in events] == ["reserve", "commit", "reserve", "commit"]
    assert events[0][2] == derive_component_seed(4, ())

    with pytest.raises(TypeError):
        bind_run_seed(4, None)
    with pytest.raises(AttributeError):
        bind_run_seed(4, [((object(),), object())])


@pytest.mark.parametrize("seed", [None, 0, MAX_SEED, IntegerSubclass(31)])
def test_run_spec_normalizes_seed_boundaries_at_public_entry(seed):
    """RunSpec accepts public seed boundaries and converts integral subclasses before binding."""
    events = []
    policy = RecordingConsumer("boundary", events)

    RunSpec(ops=[], boundary_policy=policy, seed=seed).build()

    reserved_seed = next(value for action, _, value in events if action == "reserve")
    if seed is None:
        assert reserved_seed is None
    else:
        assert reserved_seed == derive_component_seed(
            int(seed), (field("boundary_policy"),)
        )
        assert type(reserved_seed) is int


@pytest.mark.parametrize(
    ("seed", "error"),
    [
        ("x", ValueError),
        (-1, ValueError),
        (1 << 64, ValueError),
    ],
)
def test_run_spec_refuses_invalid_public_seeds(seed, error):
    """RunSpec rejects seeds outside [0, 2**64) at build entry."""
    with pytest.raises(error):
        RunSpec(ops=[], seed=seed).build()


def test_run_spec_binding_failure_precedes_conditional_release_connection():
    """A seed reservation failure stops the build before runtime owners are connected."""
    events = []

    class FailingPolicy(RecordingConsumer):
        def reserve_run_seed(self, seed):
            events.append("reserve")
            raise RuntimeError("reservation failed")

    class OrchestratorProbe:
        def connect(self, controller, callback):
            events.append("connect")

    def make_conditional_release(engine):
        events.append("construct")
        return OrchestratorProbe()

    with pytest.raises(RuntimeError, match="reservation failed"):
        RunSpec(
            ops=[],
            boundary_policy=FailingPolicy("boundary"),
            make_conditional_release=make_conditional_release,
            seed=4,
        ).build()

    assert events == ["construct", "reserve"]


def test_run_spec_binds_outside_composite_at_stable_policy_path():
    """An outside composite injected through a policy axis binds its duck child at a stable path."""
    events = []
    leaf = RecordingConsumer("outside-leaf", events)
    duck_child = SimpleNamespace(relative_path=(field("leaf"),), child=leaf)
    outside_policy = ChildrenComposite((duck_child,))

    RunSpec(ops=[], boundary_policy=outside_policy, seed=23).build()

    expected = derive_component_seed(
        23, (field("boundary_policy"), field("leaf"))
    )
    assert events == [
        ("reserve", "outside-leaf", expected),
        ("commit", "outside-leaf", expected),
    ]

"""One run seed, one seed per component, bound in two phases.

A component's seed is blake2b-8 over a versioned namespace, the run's
root seed and the component's framed path in the machine, so a component
draws the same numbers whatever else the run holds and a rerun of one
point reproduces it. The framed path bytes are pinned in
tests/records/test_seeds.py; here the derivation over them, checked
against hashlib computed independently in the test.

Binding is a two-phase transaction, prepare then commit: every leaf
reserves before any commits, so a leaf whose preparation fails leaves no
other leaf half bound, and reserve stages the new state beside the live
one so a refused binding never disturbs a run.
"""

import hashlib

import pytest

import decsim.records.seeds as seed_records
import decsim.seeding as seeding

NAMESPACE = b"decsim.run-seed.v1"


def field(name):
    return seed_records.RunSeedPathSegment("field", name)


def framed(path):
    """The path's canonical bytes, as decsim/records/seeds.py frames them."""
    pieces = []
    for segment in path:
        piece = segment.canonical_bytes()
        pieces.append(piece)
    return b"".join(pieces)


def blake2b_seed(root_seed, path):
    """The seed an independent reader of the docstring would compute."""
    root_bytes = root_seed.to_bytes(8, "big")
    framed_path = framed(path)
    preimage = NAMESPACE + root_bytes + framed_path
    digest = hashlib.blake2b(preimage, digest_size=8)
    digest_bytes = digest.digest()
    return int.from_bytes(digest_bytes, "big")


class RecordingLeaf:
    """A stochastic leaf that records the calls the binder makes on it."""

    def __init__(self, name, events):
        self.name = name
        self.events = events

    def reserve_run_seed(self, seed):
        self.events.append(("reserve", self.name, seed))
        return seed_records.RunSeedReservation("derived", seed, self.name)

    def commit_run_seed(self, reservation):
        self.events.append(("commit", self.name, reservation.proposed_seed))

    def cancel_run_seed(self, reservation):
        self.events.append(("cancel", self.name, reservation.proposed_seed))


class GeneratorLeaf(seeding._RandomSeedConsumer):
    """A stochastic leaf that draws from the generator its binding installs."""

    def __init__(self, seed):
        self._initialize_run_seed_state(seed)


class RecordingComposite:
    """A component whose children are bound under its own path."""

    def __init__(self, children):
        self.children = children

    def run_seed_children(self):
        return self.children


def child_at(name, component):
    """The component as a child one field below its parent."""
    segment = field(name)
    return seed_records.RunSeedChild((segment,), component)


def root_at(name, component):
    """The component as a root of the walk, one field below the machine."""
    segment = field(name)
    return ((segment,), component)


def reserved_names(events):
    names = []
    for action, name, _seed in events:
        if action == "reserve":
            names.append(name)
    return names


def test_a_component_seed_is_blake2b_over_the_namespace_root_and_path():
    path = (field("metrics"), field("latency"))
    independent = blake2b_seed(23, path)
    derived = seeding.derive_component_seed(23, path)
    assert derived == independent


def test_a_substream_seed_is_the_component_law_over_its_framed_keys():
    path = (
        seed_records.RunSeedPathSegment("string_key", "stream"),
        seed_records.RunSeedPathSegment("integer_key", 4),
    )
    independent = blake2b_seed(23, path)
    derived = seeding.substream_seed(23, ("stream", 4))
    assert derived == independent


def test_a_reservation_that_fails_cancels_the_earlier_ones_in_reverse():
    events = []

    class FailingLeaf(RecordingLeaf):
        def reserve_run_seed(self, seed):
            self.events.append(("reserve", self.name, seed))
            raise RuntimeError("this leaf cannot prepare its state")

    first = RecordingLeaf("a", events)
    second = RecordingLeaf("b", events)
    failing = FailingLeaf("c", events)
    roots = [
        root_at("a", first),
        root_at("b", second),
        root_at("c", failing),
    ]

    with pytest.raises(RuntimeError, match="cannot prepare its state"):
        seeding.bind_run_seed(8, roots)

    named_actions = [(action, name) for action, name, _seed in events]
    assert named_actions == [
        ("reserve", "a"),
        ("reserve", "b"),
        ("reserve", "c"),
        ("cancel", "b"),
        ("cancel", "a"),
    ]


def test_children_are_bound_in_path_order_whatever_order_they_arrive_in():
    """The order is the encoded path's, so the seeds do not move."""
    forward_events = []
    early = RecordingLeaf("a", forward_events)
    late = RecordingLeaf("z", forward_events)
    forward_children = (child_at("a_child", early), child_at("z_child", late))
    forward = RecordingComposite(forward_children)
    reverse_events = []
    reverse_early = RecordingLeaf("a", reverse_events)
    reverse_late = RecordingLeaf("z", reverse_events)
    reverse_children = (
        child_at("z_child", reverse_late),
        child_at("a_child", reverse_early),
    )
    reverse = RecordingComposite(reverse_children)

    forward_roots = [root_at("root", forward)]
    reverse_roots = [root_at("root", reverse)]
    seeding.bind_run_seed(41, forward_roots)
    seeding.bind_run_seed(41, reverse_roots)

    assert forward_events == reverse_events
    assert reserved_names(forward_events) == ["a", "z"]


def test_a_leaf_reached_by_two_paths_is_bound_once_at_the_first_one():
    events = []
    shared = RecordingLeaf("shared", events)
    children = (child_at("z_path", shared), child_at("a_path", shared))
    composite = RecordingComposite(children)
    roots = [root_at("root", composite)]

    seeding.bind_run_seed(7, roots)

    first_path = (field("root"), field("a_path"))
    expected_seed = seeding.derive_component_seed(7, first_path)
    assert events == [
        ("reserve", "shared", expected_seed),
        ("commit", "shared", expected_seed),
    ]


def test_two_components_at_one_path_are_refused_naming_the_path():
    """One path is one seed, so two components on it would share a stream."""
    events = []
    first = RecordingLeaf("first", events)
    second = RecordingLeaf("second", events)
    children = (child_at("leaf", first), child_at("leaf", second))
    composite = RecordingComposite(children)
    roots = [root_at("root", composite)]

    with pytest.raises(ValueError, match="duplicate seed path root.leaf"):
        seeding.bind_run_seed(1, roots)


def test_a_cycle_in_the_component_graph_is_refused_naming_both_paths():
    outer = RecordingComposite(())
    inner_children = (child_at("outer", outer),)
    inner = RecordingComposite(inner_children)
    outer.children = (child_at("inner", inner),)
    roots = [root_at("root", outer)]

    with pytest.raises(ValueError) as caught:
        seeding.bind_run_seed(1, roots)

    assert str(caught.value) == "seed cycle from root.inner.outer to root"


def test_a_components_own_seed_conflicts_with_the_runs_root_seed():
    """A component seeded by hand cannot also take a derived seed.

    The run would silently overwrite the seed the caller asked for, so
    the binding is refused instead.
    """
    leaf = GeneratorLeaf(seed=5)
    with pytest.raises(ValueError, match="explicit seed that conflicts"):
        leaf.reserve_run_seed(9)


def seeded_generator_leaf():
    """A stochastic leaf that draws from one seeded random.Random."""
    return GeneratorLeaf(seed=None)


def test_a_cancelled_reservation_leaves_the_live_generator_alone():
    leaf = seeded_generator_leaf()
    live_state = leaf._rng.getstate()

    reservation = leaf.reserve_run_seed(29)
    leaf.cancel_run_seed(reservation)

    assert leaf._rng.getstate() == live_state

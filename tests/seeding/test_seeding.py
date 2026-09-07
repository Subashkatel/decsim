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
import random
import types

import pytest

import decsim.decoders.relay_belief_propagation.window_decoder as relay
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.records.seeds as seed_records
import decsim.seeding as seeding

MAX_SEED = (1 << 64) - 1
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


def actions_of(events):
    return [action for action, _name, _seed in events]


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


def test_the_root_seed_alone_derives_the_seed_of_the_empty_path():
    independent = blake2b_seed(0, ())
    derived = seeding.derive_component_seed(0, ())
    assert derived == independent


def test_the_widest_root_seed_derives_a_seed_of_its_own():
    path = (field("device"),)
    independent = blake2b_seed(MAX_SEED, path)
    derived = seeding.derive_component_seed(MAX_SEED, path)
    assert derived == independent


def test_two_components_of_one_run_draw_from_different_seeds():
    """Two leaves of one root differ, so no two components share a stream."""
    device_path = (field("device"),)
    decoder_path = (field("weak_decoder"),)
    device_seed = seeding.derive_component_seed(23, device_path)
    decoder_seed = seeding.derive_component_seed(23, decoder_path)
    assert device_seed != decoder_seed


def test_every_leaf_reserves_before_any_leaf_commits():
    events = []
    roots = []
    for name in ("a", "b", "c"):
        leaf = RecordingLeaf(name, events)
        root = root_at(name, leaf)
        roots.append(root)

    seeding.bind_run_seed(5, roots)

    assert actions_of(events) == ["reserve"] * 3 + ["commit"] * 3
    assert reserved_names(events) == ["a", "b", "c"]


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

    A relay-BP decoder built with its own gamma-table seed is the case
    that reaches this: the run would silently overwrite the seed the
    caller asked for, so the binding is refused instead.
    """
    decoder = relay.RelayBeliefPropagationWindowDecoder(gamma_table_seed=5)
    with pytest.raises(ValueError, match="explicit gamma-table seed"):
        decoder.reserve_run_seed(9)


def seeded_bit_device():
    """A stochastic leaf of the machine: the fake-bit syndrome device."""
    code = code_geometry.SurfaceCodeModel(distance=3)
    return syndrome_devices.SyndromeBitDevice(code)


def test_a_reserved_generator_is_installed_only_at_commit():
    device = seeded_bit_device()
    live_generator = device._rng

    reservation = device.reserve_run_seed(29)
    assert device._rng is live_generator

    device.commit_run_seed(reservation)
    assert device._rng is not live_generator


def test_a_committed_seed_reproduces_the_drawn_numbers():
    device = seeded_bit_device()
    reservation = device.reserve_run_seed(29)
    device.commit_run_seed(reservation)
    reference = random.Random(29)

    drawn = [device._rng.random() for _ in range(4)]
    expected = [reference.random() for _ in range(4)]

    assert drawn == expected


def test_a_cancelled_reservation_leaves_the_live_generator_alone():
    device = seeded_bit_device()
    live_state = device._rng.getstate()

    reservation = device.reserve_run_seed(29)
    device.cancel_run_seed(reservation)

    assert device._rng.getstate() == live_state


def test_a_component_that_draws_nothing_is_walked_and_skipped():
    """A plain component takes no seed; the walk carries on past it."""
    events = []
    plain = types.SimpleNamespace()
    leaf = RecordingLeaf("leaf", events)
    roots = [root_at("plain", plain), root_at("leaf", leaf)]

    seeding.bind_run_seed(3, roots)

    assert reserved_names(events) == ["leaf"]

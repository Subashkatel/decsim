"""Links and fabric rules, against the cards decsim ships (fabrics.py).

The referent is each card as link_profiles.py, the presets and the
shipped run files build it: a rule over today's routes gives the card
those give, record for record, so a task built from a rule keeps its id.
A link's cycle is gem5's cyclesToTicks (src/sim/clocked_object.hh:227).
"""

import fractions
import pathlib

import pytest

import decsim.config as config
import decsim.engine
import decsim.experiments.experiment as experiment
import decsim.links.channel as channel_module
import decsim.links.link_profiles as link_profiles
import decsim.presets.clocks as clocks
import decsim.presets.link_cards as link_cards
import decsim.presets.machines as machines
import decsim.records.transfers as transfer_records
import decsim.topology.fabrics as fabrics
import decsim.topology.legacy as legacy

FRIDGE = clocks.FRIDGE_CLOCK
ROOM = clocks.ROOM_CLOCK
THIS_FILE = pathlib.Path(__file__)
EXPERIMENTS = THIS_FILE.parents[2] / "experiments"
ONE_FRIDGE_CYCLE = fabrics.ClockedLink(clock=FRIDGE, latency_cycles=1)
ONE_ROOM_CYCLE = fabrics.ClockedLink(clock=ROOM, latency_cycles=1)
# the weak base's hops, each one fridge cycle (settings.py)
WEAK_BASE = fabrics.PathLinks(
    qpu_to_controller=ONE_FRIDGE_CYCLE,
    controller_to_weak_buffer=ONE_FRIDGE_CYCLE,
    weak_buffer_to_weak_decoder=ONE_FRIDGE_CYCLE,
    decoder_to_decoder=ONE_FRIDGE_CYCLE,
    weak_decoder_to_frame=ONE_FRIDGE_CYCLE,
    frame_to_controller=ONE_FRIDGE_CYCLE,
    controller_to_qpu=ONE_FRIDGE_CYCLE,
)
# the strong base's hops: the chip's on the fridge clock, the host's on
# the room clock
STRONG_BASE = fabrics.PathLinks(
    qpu_to_controller=ONE_FRIDGE_CYCLE,
    controller_to_weak_buffer=ONE_FRIDGE_CYCLE,
    controller_to_strong_buffer=ONE_ROOM_CYCLE,
    strong_buffer_to_strong_decoder=ONE_ROOM_CYCLE,
    strong_decoder_to_frame=ONE_ROOM_CYCLE,
    frame_to_controller=ONE_FRIDGE_CYCLE,
    controller_to_qpu=ONE_FRIDGE_CYCLE,
)
# Backline's CPU echo on the 100 Gb/s cable, the host polling its memory
ROCE_V2_CPU_LEG = fabrics.RoundTripLeg(2.305, 128, 100_000)
HOST_POLL = fabrics.TimedLink(0.0)


# the reference fabric, which picks no link for any route
REFERENCE = fabrics.ReferenceFabric()
# a clocked link of three cycles of the machine's clock
THREE_MACHINE_CYCLES = fabrics.ClockedLink(latency_cycles=3)


def _card(rule, wire_links=None):
    routes = fabrics.legacy_routes({})
    links = wire_links or {}
    return fabrics.card(rule, routes, links, None)


def _strong_side(base, weak_to_strong=ONE_ROOM_CYCLE):
    """The base with the strong side's five hops (settings.py)."""
    return fabrics.PathLinks(
        controller_to_strong_buffer=ONE_ROOM_CYCLE,
        weak_decoder_to_strong_decoder=weak_to_strong,
        strong_buffer_to_strong_decoder=ONE_ROOM_CYCLE,
        strong_decoder_to_frame=ONE_ROOM_CYCLE,
        strong_decoder_to_weak_decoder=ONE_ROOM_CYCLE,
        base=base,
    )


def _tasks(name: str) -> list:
    run_file = EXPERIMENTS / name / "run.py"
    loaded = experiment.load(run_file)
    return loaded.tasks


def _experiment_one_wire_links() -> dict:
    """Experiment 1's fiber loop on the weak side's wires."""
    uplink = fabrics.ClockedLink(
        clock=FRIDGE, latency_cycles=40, bits_per_cycle=38.79, lane_count=4
    )
    frame_read = fabrics.ClockedLink(
        clock=FRIDGE, latency_cycles=1, excludes_receiver_processing=False
    )
    downlink = fabrics.ClockedLink(
        clock=FRIDGE,
        latency_cycles=41,
        bits_per_cycle=38.79,
        lane_count=4,
        excludes_receiver_processing=False,
    )
    return {
        "controller_to_weak_buffer": uplink,
        "weak_buffer_to_weak_decoder": frame_read,
        "weak_decoder_to_frame": downlink,
    }


def test_the_reference_fabric_gives_the_reference_card():
    card = _card(REFERENCE)

    assert card == link_profiles.logical_reference_profile()


@pytest.mark.parametrize(
    ("measured_fabric", "round_trip", "echo_bits"),
    [
        (link_cards.ROCE_V2_CPU_FABRIC, 2.305, 128),
        (link_cards.ROCE_V2_GPU_FABRIC, 4.5, 128),
        (link_cards.NVQLINK_GPU_FABRIC, 3.839, 256),
    ],
)
def test_a_cable_between_fridge_and_host_gives_each_measured_card(
    measured_fabric, round_trip, echo_bits
):
    leg = fabrics.RoundTripLeg(round_trip, echo_bits, 100_000)
    rule = fabrics.FridgeHostFabric(cable=leg, host_local=HOST_POLL)

    card = _card(rule)

    assert card == _card(measured_fabric)


def test_the_weak_and_strong_bases_are_links_by_path():
    weak_graph = machines.weak_baseline(3, 0.001, 1.0)
    strong_graph = machines.strong_baseline(3, 0.001, 1.0)
    weak = legacy.resolved(weak_graph)
    strong = legacy.resolved(strong_graph)

    assert _card(WEAK_BASE) == weak.links
    assert _card(STRONG_BASE) == strong.links


def test_experiment_one_is_a_cable_rule_and_the_fiber_loops_wires():
    rule = fabrics.FridgeHostFabric(cable=ROCE_V2_CPU_LEG, host_local=HOST_POLL)
    wire_links = _experiment_one_wire_links()
    tasks = _tasks("switching_baseline")
    per_window_tasks = _tasks("switching_per_window")

    card = _card(rule, wire_links)

    for task in [*tasks, *per_window_tasks]:
        assert task.machine.links == card, task.name


def test_the_switching_study_is_the_weak_base_and_the_strong_side():
    ten_microseconds = fabrics.ClockedLink(clock=ROOM, latency_cycles=2500)
    cluster_gap_rule = _strong_side(WEAK_BASE, ten_microseconds)
    redo_window_rule = _strong_side(WEAK_BASE)
    cluster_gap = _card(cluster_gap_rule)
    redo_window = _card(redo_window_rule)

    for task in _tasks("switching"):
        expected = redo_window
        if task.name.startswith("cluster_gap"):
            expected = cluster_gap
        assert task.machine.links == expected, task.name


def test_the_data_movement_study_is_the_weak_base_with_a_slower_write():
    one_microsecond = fabrics.ClockedLink(clock=FRIDGE, latency_cycles=250)
    ten_microseconds = fabrics.ClockedLink(clock=ROOM, latency_cycles=2500)
    weak_tier = fabrics.PathLinks(
        controller_to_weak_buffer=one_microsecond, base=WEAK_BASE
    )
    weak_alone = _card(weak_tier)
    switching_rule = _strong_side(weak_tier, ten_microseconds)
    switching = _card(switching_rule)

    for task in _tasks("data_movement"):
        expected = weak_alone
        if task.name.startswith("switching"):
            expected = switching
        assert task.machine.links == expected, task.name


def test_a_wires_own_link_wins_over_the_rule():
    wire_link = fabrics.TimedLink(2.0)
    rule = fabrics.PathLinks(qpu_to_controller=ONE_FRIDGE_CYCLE)

    card = _card(rule, {"qpu_to_controller": wire_link})

    channel = card.qpu_to_controller.channel
    assert channel.propagation_latency_ticks == 2_000_000
    assert channel.name == "qpu_to_controller"


def test_a_route_inside_the_fridge_takes_the_base_rules_link():
    rule = fabrics.FridgeHostFabric(
        cable=ROCE_V2_CPU_LEG, host_local=HOST_POLL, base=WEAK_BASE
    )

    card = _card(rule)
    weak_base = _card(WEAK_BASE)

    assert card.decoder_to_decoder == weak_base.decoder_to_decoder


def test_a_clocked_link_counts_on_the_machines_clock_when_it_names_none():
    routes = fabrics.legacy_routes({})
    rule = fabrics.PathLinks(qpu_to_controller=THREE_MACHINE_CYCLES)

    card = fabrics.card(rule, routes, {}, FRIDGE)

    channel = card.qpu_to_controller.channel
    assert channel.propagation_latency_ticks == 3 * FRIDGE.period_ticks


def test_a_strong_pool_placed_in_the_fridge_leaves_the_cable():
    rule = fabrics.FridgeHostFabric(cable=ROCE_V2_CPU_LEG, host_local=HOST_POLL)
    sites = {
        "strong_decoder": fabrics.FRIDGE,
        "strong_syndrome_buffer": fabrics.FRIDGE,
    }
    routes = fabrics.legacy_routes(sites)

    card = fabrics.card(rule, routes, {}, None)

    reference = link_profiles.logical_reference_profile()
    assert card.strong_decoder_to_frame == reference.strong_decoder_to_frame


def test_a_clocked_links_cycles_cost_its_clocks_period_in_whole_ticks():
    """gem5 cyclesToTicks, clockPeriod() * c (clocked_object.hh:227).

    At 300 MHz the period is 3333 ticks, so three cycles are 9999 ticks
    and eight bits at eight bits per cycle take one period, the same
    ticks every other component on the domain charges.
    """
    fridge = config.Clock.from_megahertz(300.0)
    reference = link_profiles.logical_reference_profile()
    link = fabrics.ClockedLink(
        clock=fridge,
        latency_cycles=3,
        bits_per_cycle=8.0,
        setup_cycles_per_transfer=3,
        source="three cycles at 300 MHz",
    )
    path = link.priced(
        "controller_to_weak_buffer", reference.controller_to_weak_buffer, None
    )
    engine = decsim.engine.Engine()
    channel = channel_module.Channel(path.channel, engine)
    framed = transfer_records.FramedPayload(8)
    delivered = []
    channel.send(framed, 0, 0, delivered.append)
    engine.run()
    transfer = delivered[0]
    assert path.channel.propagation_latency_ticks == 9999
    assert path.setup_ticks == 9999
    assert transfer.serialization_ticks == 3333


def test_a_clocked_links_rate_is_the_exact_fraction_of_its_decimal():
    """The rate stays the exact fraction of the decimal.

    So two cards of the same numbers name one task (collect.json_value,
    which a task's id hashes).
    """
    reference = link_profiles.logical_reference_profile()
    fridge = config.Clock.from_megahertz(250.0)
    link = fabrics.ClockedLink(
        clock=fridge,
        latency_cycles=40,
        bits_per_cycle=38.79,
        lane_count=4,
        setup_cycles_per_transfer=2,
        header_bits_per_transfer=16,
        source="2603.16203 lines 895-897",
    )

    path = link.priced(
        "controller_to_weak_buffer", reference.controller_to_weak_buffer, None
    )

    rate = path.channel.capacity.input_bits_per_microsecond
    assert rate == fractions.Fraction(38790)
    assert path.excludes_receiver_processing is True


def test_a_clocked_link_with_no_rate_is_an_unbounded_wire():
    reference = link_profiles.logical_reference_profile()
    fridge = config.Clock.from_megahertz(250.0)
    link = fabrics.ClockedLink(
        clock=fridge, latency_cycles=1, source="an unbounded wire"
    )

    path = link.priced("qpu_to_controller", reference.qpu_to_controller, None)

    assert path.channel.capacity is None


def test_a_timed_link_on_one_path_moves_that_path_alone():
    half_a_microsecond = fabrics.TimedLink(
        0.5, excludes_receiver_processing=True
    )
    rule = fabrics.PathLinks(frame_to_controller=half_a_microsecond)
    reference = link_profiles.logical_reference_profile()

    links = _card(rule)

    changed = links.frame_to_controller.channel
    assert changed.propagation_latency_ticks == 500_000
    assert links.controller_to_qpu == reference.controller_to_qpu

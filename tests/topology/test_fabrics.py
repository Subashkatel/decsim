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
import decsim.records.routes as route_records
import decsim.records.transfers as transfer_records
import decsim.topology.fabrics as fabrics
import decsim.topology.graph_layout as graph_layout
import tests.experiments.run_files as run_files

FRIDGE = clocks.FRIDGE_CLOCK
ROOM = clocks.ROOM_CLOCK
THIS_FILE = pathlib.Path(__file__)
EXPERIMENTS = THIS_FILE.parents[2] / "experiments"
ONE_FRIDGE_CYCLE = fabrics.ClockedLink(clock=FRIDGE, latency_cycles=1)
ONE_ROOM_CYCLE = fabrics.ClockedLink(clock=ROOM, latency_cycles=1)
# Backline's CPU echo on the 100 Gb/s cable, the host polling its memory
ROCE_V2_CPU_LEG = fabrics.RoundTripLeg(2.305, 128, 100_000)
HOST_POLL = fabrics.TimedLink(0.0)
# the reference fabric, which picks no link for any route
REFERENCE = fabrics.ReferenceFabric()
# a clocked link of three cycles of the machine's clock
THREE_MACHINE_CYCLES = fabrics.ClockedLink(latency_cycles=3)
WRITE_WRITTEN = route_records.HopRole(route_records.WRITE, "written")
READ_WRITTEN = route_records.HopRole(route_records.READ, "written")
ANSWER_FRIDGE = route_records.HopRole(route_records.ANSWER, fabrics.FRIDGE)
READOUT = route_records.HopRole(route_records.READOUT)


def _layout(machine):
    at_task = machine.at_task()
    return graph_layout.layout_of(at_task)


def _routes_of(card) -> tuple:
    """The card's hops as routes, each with its sites and its role."""
    routes = []
    for path in card.paths:
        route = fabrics.Route(
            path.name, path.source_site, path.target_site, path.role
        )
        routes.append(route)
    return tuple(routes)


def _switching_routes() -> tuple:
    """Every hop of a weak pool escalating to a strong pool on the host."""
    switching = run_files.switching_machine()
    layout = _layout(switching)
    return _routes_of(layout.card)


SWITCHING_ROUTES = _switching_routes()


def _card(rule, wire_links=None):
    links = wire_links or {}
    return fabrics.card(rule, SWITCHING_ROUTES, links, None)


def _tasks(name: str) -> list:
    run_file = EXPERIMENTS / name / "run.py"
    loaded = experiment.load(run_file)
    return loaded.tasks


def _task_card(task, rule, wire_links=None):
    """The card the rule gives over the task's own hops."""
    routes = _routes_of(task.layout.card)
    links = wire_links or {}
    clock = task.machine.clock
    return fabrics.card(rule, routes, links, clock, task.machine.readout_routes)


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
        "weak_buffer_to_weak": frame_read,
        "weak_to_frame": downlink,
    }


def test_the_reference_fabric_gives_the_reference_card():
    card = _card(REFERENCE)

    for path in card.paths:
        reference = link_profiles.reference_path(path.role, path.name)
        assert path.settings == reference, path.name


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


def test_every_hop_of_the_weak_base_is_one_fridge_cycle():
    weak_graph = machines.weak_baseline(3, 0.001, 1.0)

    layout = _layout(weak_graph)

    for path in layout.card.paths:
        channel = path.settings.channel
        latency_ticks = channel.propagation_latency_ticks
        assert latency_ticks == FRIDGE.period_ticks, path.name


def test_the_strong_bases_loop_is_on_the_chip_and_its_pool_on_the_host():
    strong_graph = machines.strong_baseline(3, 0.001, 1.0)

    layout = _layout(strong_graph)

    latency_ticks_by_hop = {}
    for path in layout.card.paths:
        channel = path.settings.channel
        latency_ticks_by_hop[path.name] = channel.propagation_latency_ticks
    room = ROOM.period_ticks
    fridge = FRIDGE.period_ticks
    assert latency_ticks_by_hop == {
        "qpu_to_controller": fridge,
        "controller_to_qpu": fridge,
        "frame_to_controller": fridge,
        "controller_to_strong_buffer": room,
        "strong_buffer_to_strong": room,
        "strong_to_frame": room,
        "strong_to_strong": room,
    }


def test_experiment_one_is_a_cable_rule_and_the_fiber_loops_wires():
    rule = fabrics.FridgeHostFabric(cable=ROCE_V2_CPU_LEG, host_local=HOST_POLL)
    wire_links = _experiment_one_wire_links()
    tasks = _tasks("switching_baseline")
    per_window_tasks = _tasks("switching_per_window")

    for task in [*tasks, *per_window_tasks]:
        card = _task_card(task, rule, wire_links)
        assert task.layout.card == card, task.name


def test_the_switching_study_is_the_weak_base_and_the_strong_side():
    ten_microseconds = fabrics.ClockedLink(clock=ROOM, latency_cycles=2500)
    weak_base = link_cards.WEAK_BASELINE_FABRIC
    cluster_gap_rule = link_cards.strong_side(weak_base, ten_microseconds)
    redo_window_rule = link_cards.strong_side(weak_base)

    for task in _tasks("switching"):
        rule = redo_window_rule
        if task.name.startswith("cluster_gap"):
            rule = cluster_gap_rule
        expected = _task_card(task, rule)
        assert task.layout.card == expected, task.name


def test_the_data_movement_study_is_the_weak_base_with_a_slower_write():
    one_microsecond = fabrics.ClockedLink(clock=FRIDGE, latency_cycles=250)
    ten_microseconds = fabrics.ClockedLink(clock=ROOM, latency_cycles=2500)
    weak_tier = fabrics.RoleLinks(
        write_written=one_microsecond, base=link_cards.WEAK_BASELINE_FABRIC
    )
    switching_rule = link_cards.strong_side(weak_tier, ten_microseconds)

    for task in _tasks("data_movement"):
        rule = weak_tier
        if task.name.startswith("switching"):
            rule = switching_rule
        expected = _task_card(task, rule)
        assert task.layout.card == expected, task.name


def test_a_wires_own_link_wins_over_the_rule():
    wire_link = fabrics.TimedLink(2.0)
    rule = fabrics.RoleLinks(readout=ONE_FRIDGE_CYCLE)

    card = _card(rule, {"qpu_to_controller": wire_link})

    path = card.path_settings("qpu_to_controller")
    channel = path.channel
    assert channel.propagation_latency_ticks == 2_000_000
    assert channel.name == "qpu_to_controller"


def test_a_hop_takes_the_link_of_the_role_it_carries():
    rule = fabrics.RoleLinks(read_written=ONE_FRIDGE_CYCLE)
    read = fabrics.Route(
        "weak_buffer_to_weak", fabrics.FRIDGE, fabrics.FRIDGE, READ_WRITTEN
    )
    answer = fabrics.Route(
        "weak_to_frame", fabrics.FRIDGE, fabrics.FRIDGE, ANSWER_FRIDGE
    )

    read_link = rule.link_for(read)
    answer_link = rule.link_for(answer)

    assert read_link is ONE_FRIDGE_CYCLE
    assert answer_link is None


def test_a_route_inside_the_fridge_takes_the_base_rules_link():
    weak_base = link_cards.WEAK_BASELINE_FABRIC
    rule = fabrics.FridgeHostFabric(
        cable=ROCE_V2_CPU_LEG, host_local=HOST_POLL, base=weak_base
    )

    card = _card(rule)
    weak_base_card = _card(weak_base)

    same_pool = card.path_settings("weak_to_weak")
    assert same_pool == weak_base_card.path_settings("weak_to_weak")


def test_a_clocked_link_counts_on_the_machines_clock_when_it_names_none():
    rule = fabrics.RoleLinks(readout=THREE_MACHINE_CYCLES)

    card = fabrics.card(rule, SWITCHING_ROUTES, {}, FRIDGE)

    path = card.path_settings("qpu_to_controller")
    latency_ticks = path.channel.propagation_latency_ticks
    assert latency_ticks == 3 * FRIDGE.period_ticks


def test_a_strong_pool_placed_in_the_fridge_leaves_the_cable():
    weak_graph = machines.weak_baseline(3, 0.001, 1.0)
    switching = run_files.switching_machine()
    edge = switching.part(machines.SWITCHING)
    in_the_fridge = machines.with_escalation(
        weak_graph,
        machines.WEAK_POOL,
        machines.STRONG_POOL,
        machines.STRONG_BASELINE_POOL,
        edge.settings,
        site=fabrics.FRIDGE,
    )
    rule = fabrics.FridgeHostFabric(cable=ROCE_V2_CPU_LEG, host_local=HOST_POLL)
    layout = _layout(in_the_fridge)
    routes = _routes_of(layout.card)

    card = fabrics.card(rule, routes, {}, None)

    answer = card.path("strong_to_frame")
    reference = link_profiles.reference_path(answer.role, answer.name)
    assert answer.settings == reference


def test_a_clocked_links_cycles_cost_its_clocks_period_in_whole_ticks():
    """gem5 cyclesToTicks, clockPeriod() * c (clocked_object.hh:227).

    At 300 MHz the period is 3333 ticks, so three cycles are 9999 ticks
    and eight bits at eight bits per cycle take one period, the same
    ticks every other component on the domain charges.
    """
    fridge = config.Clock.from_megahertz(300.0)
    reference = link_profiles.reference_path(
        WRITE_WRITTEN, "controller_to_weak_buffer"
    )
    link = fabrics.ClockedLink(
        clock=fridge,
        latency_cycles=3,
        bits_per_cycle=8.0,
        setup_cycles_per_transfer=3,
        source="three cycles at 300 MHz",
    )
    path = link.priced("controller_to_weak_buffer", reference, None)
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
    reference = link_profiles.reference_path(
        WRITE_WRITTEN, "controller_to_weak_buffer"
    )
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

    path = link.priced("controller_to_weak_buffer", reference, None)

    rate = path.channel.capacity.input_bits_per_microsecond
    assert rate == fractions.Fraction(38790)
    assert path.excludes_receiver_processing is True


def test_a_clocked_link_with_no_rate_is_an_unbounded_wire():
    reference = link_profiles.reference_path(READOUT, "qpu_to_controller")
    fridge = config.Clock.from_megahertz(250.0)
    link = fabrics.ClockedLink(
        clock=fridge, latency_cycles=1, source="an unbounded wire"
    )

    path = link.priced("qpu_to_controller", reference, None)

    assert path.channel.capacity is None


def test_a_timed_link_on_one_role_moves_that_hop_alone():
    half_a_microsecond = fabrics.TimedLink(
        0.5, excludes_receiver_processing=True
    )
    rule = fabrics.RoleLinks(decision=half_a_microsecond)
    reference = _card(REFERENCE)

    links = _card(rule)

    decision = links.path_settings("frame_to_controller")
    assert decision.channel.propagation_latency_ticks == 500_000
    command = links.path_settings("controller_to_qpu")
    assert command == reference.path_settings("controller_to_qpu")


GPU_SITE = "gpu0"
CABLE = fabrics.TimedLink(1.0, 100_000.0)
FORWARD = route_records.HopRole(route_records.FORWARD)


def _priced_channel(rule, route) -> str:
    """The channel the rule's link puts the route on."""
    reference = link_profiles.reference_path(route.role, route.path)
    link = rule.link_for(route)
    priced = link.priced(route.path, reference, FRIDGE)
    return priced.channel.name


def test_a_site_link_prices_every_route_between_its_two_sites():
    rule = fabrics.SiteLinks(
        (fabrics.SiteLink((fabrics.FRIDGE, GPU_SITE), CABLE),)
    )
    up = fabrics.Route("weak_to_strong", fabrics.FRIDGE, GPU_SITE, FORWARD)
    inside = fabrics.Route(
        "weak_to_weak", fabrics.FRIDGE, fabrics.FRIDGE, FORWARD
    )

    assert rule.link_for(up) is CABLE
    assert rule.link_for(inside) is None


def test_a_shared_site_link_puts_each_way_on_one_channel():
    """Two pools' hops into the GPU wait for one another; replies apart."""
    shared = fabrics.SiteLink((fabrics.FRIDGE, GPU_SITE), CABLE, shared=True)
    rule = fabrics.SiteLinks((shared,))
    weak_up = fabrics.Route("weak_to_strong", fabrics.FRIDGE, GPU_SITE, FORWARD)
    other_up = fabrics.Route(
        "fpga_to_strong", fabrics.FRIDGE, GPU_SITE, FORWARD
    )
    down = fabrics.Route("strong_to_weak", GPU_SITE, fabrics.FRIDGE, FORWARD)

    channels = [
        _priced_channel(rule, weak_up),
        _priced_channel(rule, other_up),
        _priced_channel(rule, down),
    ]
    assert channels == ["fridge_to_gpu0", "fridge_to_gpu0", "gpu0_to_fridge"]


def test_two_site_links_for_one_pair_of_sites_are_refused():
    first = fabrics.SiteLink((fabrics.FRIDGE, GPU_SITE), CABLE)
    second = fabrics.SiteLink((GPU_SITE, fabrics.FRIDGE), CABLE)

    with pytest.raises(ValueError, match="two site links join"):
        fabrics.SiteLinks((first, second))

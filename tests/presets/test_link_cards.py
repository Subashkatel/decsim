"""The presets' link cards carry their sources' numbers (link_cards.py).

Sources: the provisioning rule of the bandwidth card (each path carries
its nominal traffic in one commit region, ns-3's per-device DataRate,
and ns-3's txTime, bits over the DataRate, point-to-point-net-device.cc
:243); Backline 2609.09270 Table III, the CPU and GPU echo rows, and
NVQLink 2510.25213 line 485 for the three measured round trips; the
weak base's one-cycle hops for the strong side.
"""

import ast
import fractions
import pathlib
import re

import pytest

import decsim.config as config
import decsim.engine
import decsim.links.channel as channel_module
import decsim.links.link_profiles as link_profiles
import decsim.presets.link_cards as link_cards
import decsim.records.transfers as transfer_records
import decsim.topology.fabrics as fabrics

# the four hops a strong-node card prices as a cable or chassis crossing
STRONG_NODE_CROSSINGS = (
    transfer_records.CONTROLLER_TO_STRONG_BUFFER,
    transfer_records.WEAK_DECODER_TO_STRONG_DECODER,
    transfer_records.STRONG_DECODER_TO_FRAME,
    transfer_records.STRONG_DECODER_TO_WEAK_DECODER,
)
# The hops the measured RoCE v2 rows reprice: the write into the strong
# syndrome buffer, the escalation, the strong window's input, and the
# reply to the frame or to the weak chip.
STRONG_SIDE_PATHS = (
    "controller_to_strong_buffer",
    "weak_decoder_to_strong_decoder",
    "strong_buffer_to_strong_decoder",
    "strong_decoder_to_frame",
    "strong_decoder_to_weak_decoder",
)
# every path of today's card, each end where the reference card puts it
LEGACY_ROUTES = fabrics.legacy_routes(fabrics.LEGACY_SITES)
ROCE_V2_CPU_CARD = fabrics.card(
    link_cards.ROCE_V2_CPU_FABRIC, LEGACY_ROUTES, {}, None
)
ROCE_V2_GPU_CARD = fabrics.card(
    link_cards.ROCE_V2_GPU_FABRIC, LEGACY_ROUTES, {}, None
)
NVQLINK_GPU_CARD = fabrics.card(
    link_cards.NVQLINK_GPU_FABRIC, LEGACY_ROUTES, {}, None
)

TESTS_FILE = pathlib.Path(__file__)
TESTS_PATH = TESTS_FILE.resolve()
PACKAGE_ROOT = TESTS_PATH.parent.parent.parent
DECSIM_ROOT = PACKAGE_ROOT / "decsim"

# A payload source written as Record.field, or Record.method(), states
# where a transfer's bit count came from; anything else is prose about
# what the card assumes.
NAMED_FIELD = re.compile(r"^([A-Z]\w+)\.(\w+)(\(\))?$")
# an arXiv identifier as the cards write it: four digits, a dot, four or
# five digits
ARXIV = re.compile(r"\b\d{4}\.\d{4,5}\b")

# One distance-5 patch: 24 syndrome bits per 1.0 us round, commit and
# buffer regions of 5 rounds.
DISTANCE_5_GEOMETRY = dict(
    syndrome_bits_per_round=24,
    round_microseconds=1.0,
    commit_rounds=5,
    buffer_rounds=5,
)


def latencies_of(profile):
    latencies = {}
    for path in transfer_records.LEGACY_PATHS:
        path_settings = profile.path_settings(path)
        channel = path_settings.channel
        latencies[path] = channel.propagation_latency_ticks
    return latencies


def capacities_of(profile):
    capacities = {}
    for path in transfer_records.LEGACY_PATHS:
        path_settings = profile.path_settings(path)
        capacity = path_settings.channel.capacity
        rate = capacity.input_bits_per_microsecond
        capacities[path] = rate
    return capacities


def sources_of(profile):
    """The configuration source of each path's channel, by path name."""
    sources = {}
    for path in transfer_records.LEGACY_PATHS:
        path_settings = profile.path_settings(path)
        sources[path] = path_settings.channel.configuration_source
    return sources


def payload_sources_of(profile) -> list:
    """Every payload source string one card states, path by path."""
    sources = []
    for path in transfer_records.LEGACY_PATHS:
        path_settings = profile.path_settings(path)
        _add_payload_sources(path_settings, sources)
    return sources


def names_by_class() -> dict:
    """Every class decsim declares, with the names it carries."""
    declared = {}
    modules = DECSIM_ROOT.rglob("*.py")
    for module in sorted(modules):
        text = module.read_text()
        tree = ast.parse(text)
        _add_module_classes(tree, declared)
    return declared


def paths_outside_the_strong_side(profile) -> dict:
    """What every hop but the five strong-side ones is priced at."""
    priced = {}
    for path in transfer_records.LEGACY_PATHS:
        if path in STRONG_SIDE_PATHS:
            continue
        path_settings = profile.path_settings(path)
        priced[path] = _priced(path_settings)
    return priced


def test_every_payload_source_that_names_a_field_names_a_real_one():
    """A card that names a runtime quantity names one the tree carries.

    The string travels into the traffic ledger as the transfer's
    payload_source, so a record or field the tree does not have is a
    false statement about what crossed. A card that carries no runtime
    count says so in prose instead, and this law leaves prose alone.
    """
    declared = names_by_class()
    reference = link_profiles.logical_reference_profile()
    stated = payload_sources_of(reference)
    bandwidth = link_cards.bandwidth_limited_card(**DISTANCE_5_GEOMETRY)
    provisioned = payload_sources_of(bandwidth)
    stated.extend(provisioned)
    measured_on_the_cpu_path = payload_sources_of(ROCE_V2_CPU_CARD)
    stated.extend(measured_on_the_cpu_path)
    measured_on_the_gpu_path = payload_sources_of(ROCE_V2_GPU_CARD)
    stated.extend(measured_on_the_gpu_path)
    unknown = _unknown_fields(stated, declared)
    assert unknown == {}


def test_the_bandwidth_card_provisions_each_path_for_one_commit_region():
    profile = link_cards.bandwidth_limited_card(**DISTANCE_5_GEOMETRY)
    assert capacities_of(profile) == {
        "qpu_to_controller": 24.0,
        "controller_to_weak_buffer": 24.0,
        "controller_to_strong_buffer": 24.0,
        "weak_buffer_to_weak_decoder": 48.0,
        # a selection with one crossing-commit bit and a region of 360
        # bits, each behind a 64-bit name
        "weak_decoder_to_strong_decoder": fractions.Fraction("97.8"),
        "strong_buffer_to_strong_decoder": 72.0,
        "weak_decoder_to_frame": fractions.Fraction("0.2"),
        "decoder_to_decoder": fractions.Fraction("4.8"),
        # one flip behind the same name, home or to the weak chip
        "strong_decoder_to_frame": 13,
        "frame_to_controller": fractions.Fraction("6.4"),
        "controller_to_qpu": fractions.Fraction("25.6"),
        "strong_decoder_to_weak_decoder": 13,
    }
    assert profile.profile_name == "bandwidth_limited"


def test_a_nominal_round_serializes_in_exactly_one_round_period():
    """ns-3 txTime = bits / rate (point-to-point-net-device.cc:243).

    At 8 bits per 1.1 us a float rate falls below 80/11 bits per
    microsecond and the round would take one tick more than its period.
    """
    profile = link_cards.bandwidth_limited_card(
        syndrome_bits_per_round=8,
        round_microseconds=1.1,
        commit_rounds=3,
        buffer_rounds=3,
    )
    engine = decsim.engine.Engine()
    readout = profile.qpu_to_controller.channel
    channel = channel_module.Channel(readout, engine)
    framed = transfer_records.FramedPayload(8)
    delivered = []
    channel.send(framed, 0, 0, delivered.append)
    engine.run()
    transfer = delivered[0]
    assert transfer.serialization_ticks == 1_100_000


def test_the_bandwidth_card_keeps_the_reference_latencies():
    reference = link_profiles.logical_reference_profile()
    bandwidth = link_cards.bandwidth_limited_card(**DISTANCE_5_GEOMETRY)
    assert latencies_of(bandwidth) == latencies_of(reference)


def test_the_bandwidth_cards_default_payloads_are_one_regions_traffic():
    profile = link_cards.bandwidth_limited_card(**DISTANCE_5_GEOMETRY)
    assert profile.qpu_to_controller.default_payload.input_bits == 24
    assert profile.weak_buffer_to_weak_decoder.default_payload.input_bits == 240
    assert (
        profile.strong_buffer_to_strong_decoder.default_payload.input_bits
        == 360
    )
    assert (
        profile.weak_decoder_to_strong_decoder.default_payload.input_bits
        == 64 + 360
    )
    assert profile.decoder_to_decoder.default_payload.input_bits == 24


def test_the_capacity_scale_multiplies_every_rate():
    halved = link_cards.bandwidth_limited_card(
        **DISTANCE_5_GEOMETRY, capacity_scale=0.5
    )
    capacities = capacities_of(halved)
    assert capacities["qpu_to_controller"] == 12.0
    assert capacities["controller_to_qpu"] == fractions.Fraction("12.8")


def test_a_capacity_scale_of_zero_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        link_cards.bandwidth_limited_card(
            **DISTANCE_5_GEOMETRY, capacity_scale=0.0
        )


@pytest.mark.parametrize(
    ("profile", "echo_bits", "round_trip_microseconds"),
    [
        (ROCE_V2_CPU_CARD, 128, 2.305),
        (ROCE_V2_GPU_CARD, 128, 4.5),
        (NVQLINK_GPU_CARD, 256, 3.839),
    ],
)
def test_an_echo_of_the_measured_payload_takes_the_measured_round_trip(
    profile, echo_bits, round_trip_microseconds
):
    """The paper's own echo, replayed on the card, is the paper's median.

    Backline echoes a 16-byte payload (2609.09270 line 1611) and NVQLink
    a 32-byte one (2510.25213 lines 402-403), and each times the whole
    round trip, the payload's own time on the 100 Gb/s cable included.
    An echo is the escalation out, the poll of the coprocessor's own
    memory, and the reply back, each its latency and its bits' time on
    its wire.
    """
    out = profile.weak_decoder_to_strong_decoder
    poll = profile.strong_buffer_to_strong_decoder
    back = profile.strong_decoder_to_frame
    echo_ticks = (
        _crossing_ticks(out, echo_bits)
        + _crossing_ticks(poll, echo_bits)
        + _crossing_ticks(back, echo_bits)
    )
    measured_ticks = config.microseconds_to_ticks(round_trip_microseconds)
    write = profile.controller_to_strong_buffer

    assert echo_ticks == measured_ticks
    assert write.channel.propagation_latency_ticks == (
        out.channel.propagation_latency_ticks
    )
    assert poll.channel.propagation_latency_ticks == 0


@pytest.mark.parametrize("profile", [ROCE_V2_CPU_CARD, ROCE_V2_GPU_CARD])
def test_a_measured_rows_cable_legs_serialize_at_backlines_100_gbps(
    profile,
):
    """Backline 2609.09270 lines 1229-1230: a 100 Gb direct-attach cable.

    The write, the escalation and both replies cross it at 100000 bits
    per microsecond; the poll reads the coprocessor's own memory, so it
    crosses no cable and is unbounded.
    """
    write = profile.controller_to_strong_buffer.channel.capacity
    escalation = profile.weak_decoder_to_strong_decoder.channel.capacity
    reply = profile.strong_decoder_to_frame.channel.capacity
    reply_to_chip = profile.strong_decoder_to_weak_decoder.channel.capacity
    poll = profile.strong_buffer_to_strong_decoder.channel.capacity
    write_rate = write.input_bits_per_microsecond
    escalation_rate = escalation.input_bits_per_microsecond
    reply_rate = reply.input_bits_per_microsecond
    reply_to_chip_rate = reply_to_chip.input_bits_per_microsecond
    rates = (write_rate, escalation_rate, reply_rate, reply_to_chip_rate)

    assert rates == (100000,) * 4
    assert poll is None


def test_the_measured_rows_keep_the_reference_card_off_the_strong_side():
    """Only the five strong-side hops move: the rest is the default card."""
    reference = link_profiles.logical_reference_profile()
    unchanged = paths_outside_the_strong_side(reference)
    assert paths_outside_the_strong_side(ROCE_V2_CPU_CARD) == unchanged
    assert paths_outside_the_strong_side(ROCE_V2_GPU_CARD) == unchanged


def test_the_cpu_rows_strong_side_sources_cite_backline():
    """Five hops read one measurement, and none of them is a choice now.

    On the reference card the weak-to-strong hop names no paper and says
    "repository weak-to-strong model choice"; on this row it is a leg of
    a measured round trip and cites the paper it was read from.
    """
    sources = sources_of(ROCE_V2_CPU_CARD)
    cited = _named_where(sources, _cites_backline)
    assert cited == [
        "controller_to_strong_buffer",
        "strong_buffer_to_strong_decoder",
        "strong_decoder_to_frame",
        "strong_decoder_to_weak_decoder",
        "weak_decoder_to_strong_decoder",
    ]
    declared = _named_where(sources, _declares_a_choice)
    assert declared == []


def test_the_gpu_rows_strong_side_sources_cite_backline():
    """The same five hops, read from the GPU echo row."""
    sources = sources_of(ROCE_V2_GPU_CARD)
    cited = _named_where(sources, _cites_backline)
    assert cited == [
        "controller_to_strong_buffer",
        "strong_buffer_to_strong_decoder",
        "strong_decoder_to_frame",
        "strong_decoder_to_weak_decoder",
        "weak_decoder_to_strong_decoder",
    ]
    declared = _named_where(sources, _declares_a_choice)
    assert declared == []


def test_the_nvqlink_rows_strong_side_sources_cite_nvqlink():
    """The five strong-side hops read one measurement, 2510.25213."""
    sources = sources_of(NVQLINK_GPU_CARD)
    cited = _named_where(sources, _cites_nvqlink)
    assert cited == [
        "controller_to_strong_buffer",
        "strong_buffer_to_strong_decoder",
        "strong_decoder_to_frame",
        "strong_decoder_to_weak_decoder",
        "weak_decoder_to_strong_decoder",
    ]


def test_the_nvqlink_rows_strong_side_retransmits_nothing():
    """An unreliable connection by choice (2510.25213 lines 376-388)."""
    profile = NVQLINK_GPU_CARD
    strong_side = (
        profile.controller_to_strong_buffer,
        profile.weak_decoder_to_strong_decoder,
        profile.strong_buffer_to_strong_decoder,
        profile.strong_decoder_to_frame,
        profile.strong_decoder_to_weak_decoder,
    )
    protocols = [path.channel.protocol for path in strong_side]
    assert protocols == [None] * 5


@pytest.mark.parametrize("path", STRONG_NODE_CROSSINGS)
def test_the_nvqlink_rows_cable_legs_serialize_at_100_gbps(path):
    """NVQLink's 100 Gb Ethernet link (2510.25213 line 382, Fig. 2)."""
    path_settings = NVQLINK_GPU_CARD.path_settings(path)
    capacity = path_settings.channel.capacity
    rate = capacity.input_bits_per_microsecond
    assert rate == 100000


def test_the_nvqlink_row_keeps_the_reference_card_off_the_strong_side():
    reference = link_profiles.logical_reference_profile()
    unchanged = paths_outside_the_strong_side(reference)
    assert paths_outside_the_strong_side(NVQLINK_GPU_CARD) == unchanged


def test_the_strong_side_puts_its_five_hops_one_room_cycle_on_the_base():
    """The weak base's own hops stay; the strong side's are 4 ns, unbounded.

    One cycle of the 250 MHz room clock is 4000 ticks.
    """
    rule = link_cards.strong_side(link_cards.WEAK_BASELINE_FABRIC)
    weak_base_rule = link_cards.WEAK_BASELINE_FABRIC

    card = fabrics.card(rule, LEGACY_ROUTES, {}, None)
    weak_base = fabrics.card(weak_base_rule, LEGACY_ROUTES, {}, None)

    strong_side = [getattr(card, name) for name in STRONG_SIDE_PATHS]
    latencies = [path.channel.propagation_latency_ticks for path in strong_side]
    capacities = [path.channel.capacity for path in strong_side]
    assert latencies == [4000] * 5
    assert capacities == [None] * 5
    assert card.weak_decoder_to_frame == weak_base.weak_decoder_to_frame
    assert card.qpu_to_controller == weak_base.qpu_to_controller


def test_the_strong_side_prices_the_escalation_where_a_study_sets_it():
    ten_microseconds = fabrics.TimedLink(10.0)
    rule = link_cards.strong_side(
        link_cards.WEAK_BASELINE_FABRIC, ten_microseconds
    )

    card = fabrics.card(rule, LEGACY_ROUTES, {}, None)

    escalation = card.weak_decoder_to_strong_decoder.channel
    reply = card.strong_decoder_to_weak_decoder.channel
    assert escalation.propagation_latency_ticks == 10_000_000
    assert reply.propagation_latency_ticks == 4000


def _add_payload_sources(path_settings, sources: list) -> None:
    """Add one path's actual source and its default payload's source."""
    actual = path_settings.actual_payload_source
    if actual is not None:
        sources.append(actual)
    default_payload = path_settings.default_payload
    if default_payload is not None:
        sources.append(default_payload.source)


def _add_module_classes(tree: ast.Module, declared: dict) -> None:
    """Add one module's classes to the map of declared names."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            carried = declared.setdefault(node.name, set())
            carried |= _names_of_class(node)


def _names_of_class(node: ast.ClassDef) -> set:
    """The fields and methods one class body declares."""
    names = set()
    for statement in node.body:
        _add_statement_name(statement, names)
    return names


def _add_statement_name(statement, names: set) -> None:
    """Add the name one class-body statement declares, if it declares one."""
    if isinstance(statement, ast.FunctionDef):
        names.add(statement.name)
        return
    if isinstance(statement, ast.AnnAssign):
        target = statement.target
        if isinstance(target, ast.Name):
            names.add(target.id)
        return
    if isinstance(statement, ast.Assign):
        _add_assigned_names(statement, names)


def _add_assigned_names(statement: ast.Assign, names: set) -> None:
    """Add the plain names one assignment in a class body writes to."""
    for target in statement.targets:
        if isinstance(target, ast.Name):
            names.add(target.id)


def _unknown_fields(sources: list, declared: dict) -> dict:
    """Each source that names a field the tree does not have: its class."""
    unknown = {}
    for source in sources:
        _add_unknown_field(source, declared, unknown)
    return unknown


def _add_unknown_field(source: str, declared: dict, unknown: dict) -> None:
    """Record a payload source that names a field the tree does not have."""
    match = NAMED_FIELD.match(source)
    if match is None:
        return
    class_name, field_name = match.group(1), match.group(2)
    carried = declared.get(class_name, set())
    if field_name in carried:
        return
    unknown[source] = class_name


def _priced(path_settings) -> tuple:
    """One path's latency, capacity and payload rule."""
    channel = path_settings.channel
    return (
        channel.propagation_latency_ticks,
        channel.capacity,
        path_settings.default_payload,
        path_settings.actual_payload_source,
    )


def _crossing_ticks(path_settings, bits: int) -> int:
    """One transfer of `bits` alone on the path: latency and wire time."""
    channel = path_settings.channel
    latency = channel.propagation_latency_ticks
    if channel.capacity is None:
        return latency
    wire_ticks = channel_module.serialization_ticks(bits, channel.capacity)
    return latency + wire_ticks


def _named_where(sources: dict, holds) -> list:
    """The path names whose source satisfies a predicate, in order."""
    names = []
    carded = sources.items()
    items = sorted(carded)
    for name, source in items:
        if holds(source):
            names.append(name)
    return names


def _declares_a_choice(source: str) -> bool:
    """Whether one source says the number is decsim's own."""
    return "repository" in source


def _cites_backline(source: str) -> bool:
    """Whether one source carries Backline's arXiv identifier."""
    found = ARXIV.search(source)
    if found is None:
        return False
    identifier = found.group()
    return identifier == "2609.09270"


def _cites_nvqlink(source: str) -> bool:
    """Whether one source carries NVQLink's arXiv identifier."""
    found = ARXIV.search(source)
    if found is None:
        return False
    identifier = found.group()
    return identifier == "2510.25213"

"""Building the two syndrome buffers, the strong receiver and the frame.

The strong syndrome buffer exists only when a tier reads from it, and a
separate controller readout cost needs a link card that leaves that cost
out, a refusal that sits on the one card
it is about.
"""

import dataclasses
import pathlib
import shutil

import pytest

import decsim.build.parts as build_parts
import decsim.build.stores as store_build
import decsim.config as config
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.engine as engine_module
import decsim.escalation.policies as escalation_policies
import decsim.escalation.settings as escalation_settings
import decsim.experiments.experiment as experiment
import decsim.links.settings as link_settings
import decsim.machine as machine_module
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.syndrome_buffer.ported_syndrome_buffer as ported_syndrome_buffer
import decsim.syndrome_buffer.settings as store_settings
import tests.declared_run as declared_run

THIS_FILE = pathlib.Path(__file__)
CONFIGS = THIS_FILE.parents[2] / "configs"
# the data-movement grid's fourth block, the one with the strong tier
DATA_MOVEMENT_SWITCHING_BLOCK = 3


class _Policy:
    """The two facts the store build reads off an escalation policy."""

    def __init__(self, *, may_escalate, primary_tier) -> None:
        self.requires_strong_context = may_escalate
        self.primary_tier = primary_tier


def _parts(settings):
    """The fixtures a store seat's builder reads, and nothing else."""
    engine = engine_module.Engine()
    return build_parts.Parts(
        settings=settings,
        engine=engine,
        plan=None,
        escalation_policy=None,
        pool=None,
        detection_events=None,
    )


def test_buffer_zero_is_built_from_the_kind_the_section_names():
    settings = _machine_settings()

    parts = _parts(settings)
    store = store_build.build_weak_syndrome_buffer(parts)

    kind = settings.weak_syndrome_buffer.kind
    assert isinstance(store, ported_syndrome_buffer.SYNDROME_BUFFERS[kind])


def test_a_syndrome_buffer_kind_that_names_no_row_is_refused():
    weak_syndrome_buffer = store_settings.SyndromeBufferSettings(kind="tape")
    settings = _machine_settings(weak_syndrome_buffer=weak_syndrome_buffer)

    with pytest.raises(ValueError) as refusal:
        store_build.check_store_kinds(settings)

    assert "weak_syndrome_buffer.kind" in str(refusal.value)


def test_a_room_side_kind_that_names_no_row_is_refused_though_unused():
    """A weak-only run builds no strong syndrome buffer but reads its yaml."""
    strong_syndrome_buffer = store_settings.SyndromeBufferSettings(kind="tape")
    settings = _machine_settings(strong_syndrome_buffer=strong_syndrome_buffer)

    with pytest.raises(ValueError) as refusal:
        store_build.check_store_kinds(settings)

    assert "strong_syndrome_buffer.kind" in str(refusal.value)


# A ported strong store's row settings: the default byte FIFO, and AFS's
# 32-bit word at four cycles an access (ported_syndrome_buffer.py).
PORTED_ROW_SETTINGS = [
    ported_syndrome_buffer.PortedSyndromeBuffer.Settings(),
    ported_syndrome_buffer.PortedSyndromeBuffer.Settings(
        word_bits=32, cycles_per_access=4
    ),
]


@pytest.mark.parametrize("row_settings", PORTED_ROW_SETTINGS)
@pytest.mark.parametrize("plan", ["weak_only", "strong_only"])
def test_a_ported_strong_store_is_refused_at_build(row_settings, plan):
    """Its writes land unbooked, so its reads alone would be priced.

    The build refuses it whether or not a tier reads the room side, as
    it refuses a kind off the table.
    """
    storage_clock = config.Clock(1000)
    strong_syndrome_buffer = store_settings.SyndromeBufferSettings(
        kind="ported_syndrome_buffer",
        clock=storage_clock,
        row_settings=row_settings,
    )
    settings = _machine_settings(strong_syndrome_buffer=strong_syndrome_buffer)
    planned = _with_one_tier(settings, plan)

    with pytest.raises(ValueError, match="ported_syndrome_buffer prices"):
        machine_module.Machine.build(planned)


def test_a_ported_strong_store_is_refused_alike_from_yaml_and_python(
    tmp_path,
):
    """Both routes ask the one check, so they refuse in one sentence."""
    configs = tmp_path / "configs"
    shutil.copytree(CONFIGS, configs)
    path = configs / "experiments" / "data_movement" / "data_movement.yaml"
    text = path.read_text()
    config = experiment.load_experiment(path)
    switching_block = config.sweep[DATA_MOVEMENT_SWITCHING_BLOCK]
    points = switching_block.points()
    values = dict(points[0])
    values["workload.arguments.physical_error_probability"] = 0.001
    point = config.point_task(values)
    settings = point.settings
    ported_strong = dataclasses.replace(
        settings.strong_syndrome_buffer, kind="ported_syndrome_buffer"
    )
    python_built = dataclasses.replace(
        settings, strong_syndrome_buffer=ported_strong
    )
    ported_section = "strong_syndrome_buffer:\n  kind: ported_syndrome_buffer\n"
    ported_text = text + ported_section
    path.write_text(ported_text)

    with pytest.raises(ValueError) as yaml_refusal:
        experiment.load_experiment(path)
    with pytest.raises(ValueError) as python_refusal:
        machine_module.Machine.build(python_built)

    yaml_sentence = str(yaml_refusal.value)
    python_sentence = str(python_refusal.value)
    assert yaml_sentence.endswith(python_sentence)


def test_a_weak_only_run_reads_nothing_from_the_room_side():
    """No tier reads from the room side, so no seat is built for it."""
    weak_only = _Policy(
        may_escalate=False, primary_tier=window_records.DecoderTier.WEAK
    )

    assert not store_build.uses_strong_store(weak_only)


def test_a_run_that_may_escalate_reads_the_room_side():
    switching = _Policy(
        may_escalate=True, primary_tier=window_records.DecoderTier.WEAK
    )

    assert store_build.uses_strong_store(switching)


def test_a_strong_primary_run_reads_the_room_side():
    strong_primary = _Policy(
        may_escalate=False, primary_tier=window_records.DecoderTier.STRONG
    )

    assert store_build.uses_strong_store(strong_primary)
    assert store_build.uses_the_room_side(strong_primary)


def test_a_readout_cost_on_the_controller_needs_a_card_that_excludes_it():
    """Otherwise the reference latency charges the same work twice."""
    settings = _settings_with_readout_cost(
        readout_to_bits_cycles=6, card_excludes_it=False
    )

    with pytest.raises(ValueError) as refusal:
        store_build.check_readout_cost_is_priced(settings)

    sentence = str(refusal.value)
    assert "separate controller readout cost" in sentence
    assert "excludes that cost" in sentence


def test_a_readout_cost_beside_a_card_that_excludes_it_is_allowed():
    settings = _settings_with_readout_cost(
        readout_to_bits_cycles=6, card_excludes_it=True
    )

    store_build.check_readout_cost_is_priced(settings)


def test_no_readout_cost_asks_nothing_of_the_card():
    settings = _settings_with_readout_cost(
        readout_to_bits_cycles=0, card_excludes_it=False
    )

    store_build.check_readout_cost_is_priced(settings)


@pytest.mark.parametrize(
    "processing_exclusions", [(False, True), (True, False)]
)
def test_every_readout_route_must_exclude_separately_charged_processing(
    processing_exclusions: tuple[bool, bool],
) -> None:
    settings = _settings_with_readout_cost(
        readout_to_bits_cycles=6, card_excludes_it=True
    )
    first_path = dataclasses.replace(
        settings.links.qpu_to_controller,
        excludes_receiver_processing=processing_exclusions[0],
    )
    second_path = dataclasses.replace(
        settings.links.qpu_to_controller,
        excludes_receiver_processing=processing_exclusions[1],
    )
    first_route = link_settings.ReadoutRoute((0,), first_path)
    second_route = link_settings.ReadoutRoute((1,), second_path)
    links = dataclasses.replace(
        settings.links, readout_routes=(first_route, second_route)
    )
    settings = dataclasses.replace(settings, links=links)

    with pytest.raises(ValueError, match="latency excludes that cost"):
        store_build.check_readout_cost_is_priced(settings)


@pytest.mark.parametrize(
    "readout_cycles, excludes_processing", [(0, False), (6, True)]
)
def test_readout_routes_allow_processing_to_be_charged_once(
    readout_cycles: int, excludes_processing: bool
) -> None:
    settings = _settings_with_readout_cost(
        readout_to_bits_cycles=readout_cycles, card_excludes_it=True
    )
    path = dataclasses.replace(
        settings.links.qpu_to_controller,
        excludes_receiver_processing=excludes_processing,
    )
    route = link_settings.ReadoutRoute((0,), path)
    links = dataclasses.replace(settings.links, readout_routes=(route,))
    settings = dataclasses.replace(settings, links=links)

    store_build.check_readout_cost_is_priced(settings)


def _settings_with_readout_cost(*, readout_to_bits_cycles, card_excludes_it):
    """A machine whose controller readout cost and card are set by hand."""
    controller = declared_run.declared_controller()
    controller = dataclasses.replace(
        controller, readout_to_bits_cycles=readout_to_bits_cycles
    )
    links = declared_run.declared_profile()
    card = links.qpu_to_controller
    links = _links_with_readout_flag(links, card, card_excludes_it)
    workload = declared_run.declared_workload(None, 6)
    qpu = declared_run.declared_qpu()
    frame = declared_run.declared_frame()
    return machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        links=links,
        controller=controller,
        pauli_frame=frame,
    )


def _links_with_readout_flag(links, card, excludes_receiver_processing):
    """The fabric settings with one card's flag replaced."""
    changed_card = dataclasses.replace(
        card, excludes_receiver_processing=excludes_receiver_processing
    )
    return dataclasses.replace(links, qpu_to_controller=changed_card)


def _machine_settings(**changes):
    """A weak-only machine's settings, with the named sections replaced."""
    workload = declared_run.declared_workload(None, 6)
    qpu = declared_run.declared_qpu()
    links = declared_run.declared_profile()
    controller = declared_run.declared_controller()
    frame = declared_run.declared_frame()
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        links=links,
        controller=controller,
        pauli_frame=frame,
    )
    return dataclasses.replace(settings, **changes)


def _ported_run_with_a_rate(bits_per_microsecond):
    """A weak-only machine on a ported store, its read link rated or not."""
    links = declared_run.declared_profile()
    path = links.weak_buffer_to_weak_decoder
    capacity = None
    if bits_per_microsecond is not None:
        capacity = link_settings.CapacitySettings(bits_per_microsecond, "card")
    channel = dataclasses.replace(path.channel, capacity=capacity)
    path = dataclasses.replace(path, channel=channel)
    links = dataclasses.replace(links, weak_buffer_to_weak_decoder=path)
    ported = store_settings.SyndromeBufferSettings(
        kind="ported_syndrome_buffer"
    )
    return _machine_settings(weak_syndrome_buffer=ported, links=links)


def test_a_rate_out_of_a_ported_store_is_refused_as_a_second_price():
    """The store's read port prices the bits; the link keeps its latency."""
    settings = _ported_run_with_a_rate(2000.0)

    with pytest.raises(ValueError, match="set its bits_per_cycle to null"):
        store_build.check_one_price_for_a_read(settings)


def test_a_ported_store_beside_an_unrated_link_is_accepted():
    settings = _ported_run_with_a_rate(None)

    store_build.check_one_price_for_a_read(settings)


def _with_one_tier(settings, plan: str):
    """The settings with one decoding tier: the weak one or the strong one."""
    decoder = decoders.PresetLatencyDecoder(1.0)
    tier = decoder_settings.DecoderSettings(decoder=decoder)
    if plan == "weak_only":
        return dataclasses.replace(settings, weak_decoder=tier)
    policy = escalation_policies.StrongOnly(escalation_policies.NO_CONFIDENCE)
    escalation = escalation_settings.EscalationSettings(policy=policy)
    return dataclasses.replace(
        settings, strong_decoder=tier, escalation=escalation
    )

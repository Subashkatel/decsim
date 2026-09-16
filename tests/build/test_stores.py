"""Building the two round stores, the strong writer and the frame.

The room-side store exists only when a tier reads from it, and a
separate controller readout cost needs a link card that leaves that cost
out, which is the finding I7 Part 1 slice 4(d) moved onto the one card
it is about.
"""

import dataclasses

import pytest

import decsim.assembly as assembly
import decsim.build.stores as store_build
import decsim.engine as engine_module
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as store_settings
import tests.declared_run as declared_run


class _Policy:
    """The two facts the store build reads off an escalation policy."""

    def __init__(self, *, may_escalate, primary_tier) -> None:
        self.requires_strong_context = may_escalate
        self.primary_tier = primary_tier


def _parts(settings):
    """The fixtures a store seat's builder reads, and nothing else."""
    engine = engine_module.Engine()
    return assembly.Parts(
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
    store = store_build.build_round_store(parts)

    kind = settings.round_store.kind
    assert isinstance(store, round_store_module.ROUND_STORES[kind])


def test_a_round_store_kind_that_names_no_row_is_refused():
    round_store = store_settings.RoundStoreSettings(kind="tape")
    settings = _machine_settings(round_store=round_store)

    with pytest.raises(ValueError) as refusal:
        store_build.check_store_kinds(settings)

    assert "round_store.kind" in str(refusal.value)


def test_a_room_side_kind_that_names_no_row_is_refused_though_unused():
    """A weak-only run builds no room-side store; its yaml is read anyway."""
    strong_round_store = store_settings.RoundStoreSettings(kind="tape")
    settings = _machine_settings(strong_round_store=strong_round_store)

    with pytest.raises(ValueError) as refusal:
        store_build.check_store_kinds(settings)

    assert "strong_round_store.kind" in str(refusal.value)


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

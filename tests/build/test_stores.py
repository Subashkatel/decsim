"""Building the two round stores, the strong writer and the frame.

The room-side store exists only when a tier reads from it, and a
separate controller readout cost needs a link card that leaves that cost
out, which is the finding I7 Part 1 slice 4(d) moved onto the one card
it is about.
"""

import dataclasses

import pytest

import decsim.build.stores as store_build
import decsim.controller.round_writes as round_writes
import decsim.controller.settings as controller_settings
import decsim.engine as engine_module
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as store_settings
import tests.declared_run as declared_run


def _held_rounds():
    """The hold a refused round waits in, on a free engine."""
    engine = engine_module.Engine()
    on_full = controller_settings.PackingOverflowPolicy.STALL
    return round_writes.HeldRounds(engine, on_full)


class _Policy:
    """The two facts the store build reads off an escalation policy."""

    def __init__(self, *, may_escalate, primary_tier) -> None:
        self.requires_strong_context = may_escalate
        self.primary_tier = primary_tier


def test_buffer_zero_is_built_from_the_kind_the_section_names():
    settings = store_settings.RoundStoreSettings()
    held = _held_rounds()

    store = store_build.build_round_store(settings, held)

    assert isinstance(store, round_store_module.ROUND_STORES[settings.kind])


def test_a_round_store_kind_that_names_no_row_is_refused():
    settings = store_settings.RoundStoreSettings(kind="tape")
    held = _held_rounds()

    with pytest.raises(ValueError) as refusal:
        store_build.build_round_store(settings, held)

    assert "round_store.kind" in str(refusal.value)


def test_a_weak_only_run_gets_no_room_side_store():
    """No tier reads from the room side, so nothing is built for it."""
    weak_only = _Policy(
        may_escalate=False, primary_tier=window_records.DecoderTier.WEAK
    )
    settings = store_settings.RoundStoreSettings()
    held = _held_rounds()

    store = store_build.build_strong_round_store(settings, weak_only, held)

    assert store is None


def test_a_run_that_may_escalate_gets_the_room_side_store():
    switching = _Policy(
        may_escalate=True, primary_tier=window_records.DecoderTier.WEAK
    )
    settings = store_settings.RoundStoreSettings()
    held = _held_rounds()

    store = store_build.build_strong_round_store(settings, switching, held)

    assert store is not None


def test_a_strong_primary_run_gets_the_room_side_store():
    strong_primary = _Policy(
        may_escalate=False, primary_tier=window_records.DecoderTier.STRONG
    )
    settings = store_settings.RoundStoreSettings()
    held = _held_rounds()

    store = store_build.build_strong_round_store(settings, strong_primary, held)

    assert store is not None


def test_no_room_side_store_means_no_strong_writer():
    writer = store_build.build_strong_round_writer(None, None, None, None)

    assert writer is None


def test_a_readout_cost_on_the_controller_needs_a_card_that_excludes_it():
    """Otherwise the reference latency charges the same work twice."""
    settings = _settings_with_readout_cost(
        readout_to_bits_microseconds=3.0, card_excludes_it=False
    )

    with pytest.raises(ValueError) as refusal:
        store_build.check_readout_cost_is_priced(settings)

    sentence = str(refusal.value)
    assert "separate controller readout cost" in sentence
    assert "excludes that cost" in sentence


def test_a_readout_cost_beside_a_card_that_excludes_it_is_allowed():
    settings = _settings_with_readout_cost(
        readout_to_bits_microseconds=3.0, card_excludes_it=True
    )

    store_build.check_readout_cost_is_priced(settings)


def test_no_readout_cost_asks_nothing_of_the_card():
    settings = _settings_with_readout_cost(
        readout_to_bits_microseconds=0.0, card_excludes_it=False
    )

    store_build.check_readout_cost_is_priced(settings)


def _settings_with_readout_cost(
    *, readout_to_bits_microseconds, card_excludes_it
):
    """A machine whose controller readout cost and card are set by hand."""
    controller = declared_run.declared_controller()
    controller = dataclasses.replace(
        controller,
        readout_to_bits_microseconds=readout_to_bits_microseconds,
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

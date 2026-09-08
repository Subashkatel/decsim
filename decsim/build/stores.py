"""Build the two round stores, the strong writer and the Pauli frame."""

from typing import Optional

import decsim.controller.round_writes as round_writes
import decsim.engine as engine_module
import decsim.links.fabric as fabric
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings
import decsim.syndrome_buffer.strong_round_writer as strong_round_writer_module


def build_round_store(
    settings: round_store_settings.RoundStoreSettings,
    held_rounds: round_writes.HeldRounds,
):
    """Buffer 0; a freed slot retries the rounds held for room."""
    row = machine_settings.row(
        round_store_module.ROUND_STORES, "round_store.kind", settings.kind
    )
    return row(settings, on_slot_freed=held_rounds.retry)


def build_strong_round_store(
    settings: round_store_settings.RoundStoreSettings,
    escalation_policy,
    held_rounds: round_writes.HeldRounds,
):
    """The room-side store, only when a tier reads from the room side."""
    row = machine_settings.row(
        round_store_module.ROUND_STORES,
        "strong_round_store.kind",
        settings.kind,
    )
    uses_strong_store = (
        escalation_policy.requires_strong_context
        or escalation_policy.primary_tier is window_records.DecoderTier.STRONG
    )
    if not uses_strong_store:
        return None
    return row(settings, on_slot_freed=held_rounds.retry)


def build_strong_round_writer(
    engine: engine_module.Engine,
    links: fabric.LinkFabric,
    strong_round_store,
    window_manager,
):
    """The crossing into the room-side store; the window manager hears it."""
    if strong_round_store is None:
        return None
    return strong_round_writer_module.StrongRoundWriter(
        engine,
        links,
        strong_round_store,
        on_round_stored=window_manager.accept_room_round,
    )


def build_pauli_frame(
    settings: Optional[pauli_frame_module.PauliFrameConfig],
    engine: engine_module.Engine,
) -> Optional[pauli_frame_module.PauliFrame]:
    if settings is None:
        return None
    return settings.resolve(engine)


def check_readout_cost_is_priced(
    settings: machine_settings.MachineSettings,
) -> None:
    """A readout cost on the controller needs a card that leaves it out."""
    readout_ticks = settings.controller.readout_to_bits_ticks()
    links = settings.links
    if readout_ticks > 0 and not (
        links.is_controller_processing_outside_qpu_to_controller
    ):
        raise ValueError(
            "a separate controller readout cost requires a link profile "
            "whose QC latency excludes that cost"
        )

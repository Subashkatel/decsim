"""Build the two round stores, their ports, the strong writer and the frame."""

from typing import Optional

import decsim.controller.round_writes as round_writes
import decsim.engine as engine_module
import decsim.links.fabric as fabric
import decsim.links.window_transfers as window_transfers
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.syndrome_buffer.round_input as round_input
import decsim.syndrome_buffer.round_output as round_output
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings
import decsim.syndrome_buffer.strong_round_writer as strong_round_writer_module
import decsim.tables as tables


def build_round_store(
    settings: round_store_settings.RoundStoreSettings,
    held_rounds: round_writes.HeldRounds,
):
    """Buffer 0; a freed slot retries the rounds held for room."""
    row = tables.row(
        round_store_module.ROUND_STORES, "round_store.kind", settings.kind
    )
    return row(settings, on_slot_freed=held_rounds.retry)


def build_strong_round_store(
    settings: round_store_settings.RoundStoreSettings,
    escalation_policy,
    held_rounds: round_writes.HeldRounds,
):
    """The room-side store, only when a tier reads from the room side."""
    row = tables.row(
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


def build_store_outputs(
    engine: engine_module.Engine,
    links: fabric.LinkFabric,
    round_store,
    strong_round_store,
) -> tuple:
    """Each store's outgoing port, built beside the store it belongs to.

    A round leaves by the end that holds it, so the port that executes
    the send is the store's and is wired here rather than by whoever
    asks for the round.
    """
    transfers = window_transfers.WindowTransfers(engine, links)
    weak_output = round_output.RoundStoreOutput(
        transfers,
        transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER,
        "Buffer 0",
        round_store,
    )
    strong_output = round_output.RoundStoreOutput(
        transfers,
        transfer_records.LinkPath.STRONG_BUFFER_TO_STRONG_DECODER,
        "Buffer 1",
        strong_round_store,
    )
    return weak_output, strong_output


def build_round_store_input(
    engine: engine_module.Engine,
    round_store,
    weak_output: round_output.RoundStoreOutput,
    window_manager,
) -> round_input.RoundStoreInput:
    """Buffer 0's port toward the controller, built after the windows.

    The port announces a published round to the window manager, so it is
    built once the manager exists, as the strong writer is.
    """
    return round_input.RoundStoreInput(
        engine, round_store, weak_output, window_manager
    )


def build_pauli_frame(
    settings: Optional[pauli_frame_module.PauliFrameConfig],
    engine: engine_module.Engine,
) -> Optional[pauli_frame_module.PauliFrame]:
    """The Pauli frame the run commits into; None when it has no frame."""
    if settings is None:
        return None
    return settings.resolve(engine)


def check_readout_cost_is_priced(
    settings: machine_settings.MachineSettings,
) -> None:
    """A readout cost on the controller needs a card that leaves it out.

    The claim belongs to the one card it is about: a yaml that leaves
    qpu_to_controller null keeps the reference number, which already
    covers the controller turning the readout into bits, so a second
    charge for that work would count it twice.
    """
    readout_ticks = settings.controller.readout_to_bits_ticks()
    readout_hop = settings.links.qpu_to_controller
    if readout_ticks > 0 and not readout_hop.excludes_receiver_processing:
        raise ValueError(
            "a separate controller readout cost requires a "
            "qpu_to_controller card whose latency excludes that cost"
        )

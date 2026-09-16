"""Build one buffer-side seat each, from the run's settings.

Every function here reads the settings its seat needs and returns the
seat; which seats a run has and what each is wired to is the assembly
file's (decsim/assembly.py).
"""

import decsim.controller.round_writes as round_writes
import decsim.links.link_profiles as link_profiles
import decsim.links.window_transfers as window_transfers
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.syndrome_buffer.round_input as round_input
import decsim.syndrome_buffer.round_output as round_output
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.strong_round_writer as strong_round_writer_module
import decsim.tables as tables


def build_links(parts):
    """The link fabric of the run's kind, carded by the links section."""
    row = tables.row(
        link_profiles.LINK_FABRICS, "links.kind", parts.settings.links.kind
    )
    return row.build(parts.settings.links, parts.engine)


def build_held_rounds(parts):
    """The waiting line in front of the stores, and what a full store does."""
    return round_writes.HeldRounds(
        parts.engine, parts.settings.controller.packing_overflow
    )


def build_round_store(parts):
    """Buffer 0, the store every finished round is published into."""
    settings = parts.settings.round_store
    row = tables.row(
        round_store_module.ROUND_STORES, "round_store.kind", settings.kind
    )
    return row(settings)


def build_strong_round_store(parts):
    """The room-side store, Buffer 1."""
    settings = parts.settings.strong_round_store
    row = tables.row(
        round_store_module.ROUND_STORES,
        "strong_round_store.kind",
        settings.kind,
    )
    return row(settings)


def check_store_kinds(settings: machine_settings.MachineSettings) -> None:
    """Both store sections name a row, including the one nothing reads.

    A run that never reads the room side builds no store for it, so the
    kind its yaml names would otherwise go unread; a kind off the table
    is a mistake in the file either way.
    """
    tables.row(
        round_store_module.ROUND_STORES,
        "round_store.kind",
        settings.round_store.kind,
    )
    tables.row(
        round_store_module.ROUND_STORES,
        "strong_round_store.kind",
        settings.strong_round_store.kind,
    )


def build_store_transfers(parts):
    """The buffer side's transfers, which execute its sends."""
    return window_transfers.WindowTransfers(parts.engine)


def build_weak_output(parts):
    """Buffer 0's outgoing end, which executes every send of its rounds."""
    del parts
    return round_output.RoundStoreOutput(
        transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER, "Buffer 0"
    )


def build_strong_output(parts):
    """Buffer 1's outgoing end."""
    del parts
    return round_output.RoundStoreOutput(
        transfer_records.LinkPath.STRONG_BUFFER_TO_STRONG_DECODER, "Buffer 1"
    )


def build_primary_output(parts):
    """The outgoing end the tier that decodes the plan's windows reads.

    A weak-primary plan reads Buffer 0 and a strong-primary plan reads
    Buffer 1, so this row names a seat another row built rather than
    building a second end onto the same store.
    """
    if uses_the_room_side(parts.escalation_policy):
        return parts.seats["strong_output"]
    return parts.seats["weak_output"]


def build_strong_round_writer(parts):
    """The room-side end of the crossing out of the fridge."""
    return strong_round_writer_module.StrongRoundWriter(parts.engine)


def build_store_input(parts):
    """Buffer 0's room and landing."""
    return round_input.RoundStoreInput(parts.engine, parts.settings.round_store)


def build_pauli_frame(parts):
    """The Pauli frame the run commits into."""
    return parts.settings.pauli_frame.resolve(parts.engine)


def uses_strong_store(escalation_policy) -> bool:
    """Whether a tier of this run reads its rounds from the room side."""
    if escalation_policy.requires_strong_context:
        return True
    return uses_the_room_side(escalation_policy)


def uses_the_room_side(escalation_policy) -> bool:
    """Whether the tier that decodes the plan's windows reads Buffer 1."""
    strong = window_records.DecoderTier.STRONG
    return escalation_policy.primary_tier is strong


def check_readout_cost_is_priced(
    settings: machine_settings.MachineSettings,
) -> None:
    """A readout cost on the controller needs a card that leaves it out.

    The claim belongs to the one card it is about: a yaml that leaves
    qpu_to_controller null keeps the reference number, which already
    covers the controller turning the readout into bits, so a second
    charge for that work would count it twice.
    """
    readout_cycles = settings.controller.readout_to_bits_cycles
    readout_hop = settings.links.qpu_to_controller
    if readout_cycles > 0 and not readout_hop.excludes_receiver_processing:
        raise ValueError(
            "a separate controller readout cost requires a "
            "qpu_to_controller card whose latency excludes that cost"
        )

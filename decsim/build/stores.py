"""Build one buffer-side seat each, from the run's settings.

Every function here reads the settings its seat needs and returns the
seat; which seats a run has and what each is wired to is the assembly
file's (decsim/assembly.py).
"""

import decsim.build.parts as build_parts
import decsim.controller.syndrome_round_sender as syndrome_round_sender
import decsim.links.link_profiles as link_profiles
import decsim.links.window_transfers as window_transfers
import decsim.ports as ports
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.syndrome_buffer.round_output as round_output
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import decsim.tables as tables
from decsim.syndrome_buffer import (
    strong_syndrome_round_receiver as strong_syndrome_round_receiver_module,
)
from decsim.syndrome_buffer import (
    weak_syndrome_round_receiver as weak_syndrome_round_receiver,
)


def build_links(parts: build_parts.Parts) -> ports.Link:
    """The link fabric of the run's kind, carded by the links section."""
    row = tables.row(
        link_profiles.LINK_FABRICS, "links.kind", parts.settings.links.kind
    )
    return row.build(parts.settings.links, parts.engine)


def build_held_rounds(
    parts: build_parts.Parts,
) -> syndrome_round_sender.HeldRounds:
    """The waiting line in front of the stores, and what a full store does."""
    return syndrome_round_sender.HeldRounds(
        parts.engine, parts.settings.controller.packing_overflow
    )


def build_weak_syndrome_buffer(
    parts: build_parts.Parts,
) -> ports.SyndromeBuffer:
    """The weak syndrome buffer, where every finished round is published."""
    settings = parts.settings.weak_syndrome_buffer
    row = tables.row(
        syndrome_buffer_module.SYNDROME_BUFFERS,
        "weak_syndrome_buffer.kind",
        settings.kind,
    )
    return row(settings)


def build_strong_syndrome_buffer(
    parts: build_parts.Parts,
) -> ports.SyndromeBuffer:
    """The strong syndrome buffer."""
    settings = parts.settings.strong_syndrome_buffer
    row = tables.row(
        syndrome_buffer_module.SYNDROME_BUFFERS,
        "strong_syndrome_buffer.kind",
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
        syndrome_buffer_module.SYNDROME_BUFFERS,
        "weak_syndrome_buffer.kind",
        settings.weak_syndrome_buffer.kind,
    )
    tables.row(
        syndrome_buffer_module.SYNDROME_BUFFERS,
        "strong_syndrome_buffer.kind",
        settings.strong_syndrome_buffer.kind,
    )


def build_store_transfers(
    parts: build_parts.Parts,
) -> window_transfers.WindowTransfers:
    """The buffer side's transfers, which execute its sends."""
    return window_transfers.WindowTransfers(parts.engine)


def build_weak_output(
    parts: build_parts.Parts,
) -> round_output.SyndromeBufferOutput:
    """The weak syndrome buffer's outgoing end, which executes every send."""
    del parts
    return round_output.SyndromeBufferOutput(
        transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER,
        "weak syndrome buffer",
    )


def build_strong_output(
    parts: build_parts.Parts,
) -> round_output.SyndromeBufferOutput:
    """The strong syndrome buffer's outgoing end."""
    del parts
    return round_output.SyndromeBufferOutput(
        transfer_records.LinkPath.STRONG_BUFFER_TO_STRONG_DECODER,
        "strong syndrome buffer",
    )


def build_primary_output(
    parts: build_parts.Parts,
) -> round_output.SyndromeBufferOutput:
    """The outgoing end the tier that decodes the plan's windows reads.

    A weak-primary plan reads the weak syndrome buffer and a strong-primary
    plan reads the strong syndrome buffer, so this row names a seat another row
    built rather than building a second end onto the same store.
    """
    if uses_the_room_side(parts.escalation_policy):
        return parts.seats["strong_output"]
    return parts.seats["weak_output"]


def build_strong_syndrome_round_receiver(
    parts: build_parts.Parts,
) -> strong_syndrome_round_receiver_module.StrongSyndromeRoundReceiver:
    """The room-side end of the crossing out of the fridge."""
    return strong_syndrome_round_receiver_module.StrongSyndromeRoundReceiver(
        parts.engine
    )


def build_weak_syndrome_round_receiver(
    parts: build_parts.Parts,
) -> weak_syndrome_round_receiver.WeakSyndromeRoundReceiver:
    """The weak syndrome buffer's room and landing."""
    return weak_syndrome_round_receiver.WeakSyndromeRoundReceiver(
        parts.engine, parts.settings.weak_syndrome_buffer
    )


def build_pauli_frame(parts: build_parts.Parts) -> ports.Frame:
    """The Pauli frame the run commits into."""
    return parts.settings.pauli_frame.resolve(parts.engine)


def uses_strong_store(escalation_policy) -> bool:
    """Whether a tier of this run reads its rounds from the room side."""
    if escalation_policy.requires_strong_context:
        return True
    return uses_the_room_side(escalation_policy)


def uses_the_room_side(escalation_policy) -> bool:
    """Whether the plan's decoding tier reads the strong syndrome buffer."""
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

"""Build one buffer-side seat each, from the run's settings.

Every function here reads the settings its seat needs and returns the
seat; which seats a run has and what each is wired to is the assembly
file's (decsim/assembly.py).
"""

import decsim.build.parts as build_parts
import decsim.controller.syndrome_round_sender as syndrome_round_sender
import decsim.decoders.settings as decoder_settings
import decsim.links.link_profiles as link_profiles
import decsim.links.window_transfers as window_transfers
import decsim.ports as ports
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.syndrome_buffer.ported_syndrome_buffer as ported_syndrome_buffer
import decsim.syndrome_buffer.round_output as round_output
import decsim.syndrome_buffer.settings as syndrome_buffer_settings
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
    """The waiting line in front of the stores."""
    return syndrome_round_sender.HeldRounds(parts.engine)


def build_weak_syndrome_buffer(
    parts: build_parts.Parts,
) -> ports.SyndromeBuffer:
    """The weak syndrome buffer, where every finished round is published."""
    settings = parts.settings.weak_syndrome_buffer
    row = tables.row(
        ported_syndrome_buffer.SYNDROME_BUFFERS,
        "weak_syndrome_buffer.kind",
        settings.kind,
    )
    return row(settings, parts.engine)


def build_strong_syndrome_buffer(
    parts: build_parts.Parts,
) -> ports.SyndromeBuffer:
    """The strong syndrome buffer."""
    settings = parts.settings.strong_syndrome_buffer
    row = tables.row(
        ported_syndrome_buffer.SYNDROME_BUFFERS,
        "strong_syndrome_buffer.kind",
        settings.kind,
    )
    return row(settings, parts.engine)


def check_store_kinds(settings: machine_settings.MachineSettings) -> None:
    """Both store sections name a row, including the one nothing reads.

    A run that never reads the room side builds no store for it, so the
    kind its yaml names would otherwise go unread; a kind off the table
    is a mistake in the file either way, and so is a ported strong store,
    which a Python-built settings record reaches without the yaml.
    """
    tables.row(
        ported_syndrome_buffer.SYNDROME_BUFFERS,
        "weak_syndrome_buffer.kind",
        settings.weak_syndrome_buffer.kind,
    )
    tables.row(
        ported_syndrome_buffer.SYNDROME_BUFFERS,
        "strong_syndrome_buffer.kind",
        settings.strong_syndrome_buffer.kind,
    )
    syndrome_buffer_settings.check_strong_store_kind(
        settings.strong_syndrome_buffer.kind
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
    reads_in_place = _reads_in_place(parts.settings.weak_decoder, "weak")
    return round_output.SyndromeBufferOutput(
        parts.engine,
        transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER,
        "weak syndrome buffer",
        reads_in_place,
        "weak_decoder",
    )


def build_strong_output(
    parts: build_parts.Parts,
) -> round_output.SyndromeBufferOutput:
    """The strong syndrome buffer's outgoing end."""
    reads_in_place = _reads_in_place(parts.settings.strong_decoder, "strong")
    return round_output.SyndromeBufferOutput(
        parts.engine,
        transfer_records.LinkPath.STRONG_BUFFER_TO_STRONG_DECODER,
        "strong syndrome buffer",
        reads_in_place,
        "strong_decoder",
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
    return weak_syndrome_round_receiver.WeakSyndromeRoundReceiver(parts.engine)


def build_pauli_frame(parts: build_parts.Parts) -> ports.Frame:
    """The Pauli frame the run commits into."""
    return parts.settings.pauli_frame.resolve(parts.engine)


def uses_strong_store(escalation_policy) -> bool:
    """Whether a tier of this run reads its rounds from the room side."""
    if escalation_policy.requires_strong_context:
        return True
    return uses_the_room_side(escalation_policy)


def uses_the_room_side(escalation_policy: ports.EscalationPolicy) -> bool:
    """Whether the plan's decoding tier reads the strong syndrome buffer."""
    strong = window_records.DecoderTier.STRONG
    return escalation_policy.primary_tier is strong


def check_one_price_for_a_read(
    settings: machine_settings.MachineSettings,
) -> None:
    """A ported weak store prices its reads; the link out of it may not.

    The store's read port moves the bits by words, so a rate on
    weak_buffer_to_weak_decoder as well would charge the same bits twice.
    The link keeps its latency.
    """
    if settings.weak_syndrome_buffer.kind != "ported_syndrome_buffer":
        return
    path = settings.links.weak_buffer_to_weak_decoder
    if path.channel.capacity is None:
        return
    raise ValueError(
        "weak_syndrome_buffer kind ported_syndrome_buffer prices the read "
        "out of the store by its words, and links.weak_buffer_to_weak_decoder "
        "has a rate that would charge the same bits again; set its "
        "bits_per_cycle to null"
    )


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
    if readout_cycles == 0:
        return
    readout_hops = [settings.links.qpu_to_controller]
    readout_hops.extend(
        route.settings for route in settings.links.readout_routes
    )
    for readout_hop in readout_hops:
        if not readout_hop.excludes_receiver_processing:
            raise ValueError(
                "a separate controller readout cost requires a "
                "qpu_to_controller card whose latency excludes that cost"
            )


def _reads_in_place(tier_settings, tier: str) -> bool:
    """Whether the tier a store feeds reads the rounds where they sit.

    The rule is the tier's input row (<tier>.input); a run without that
    tier builds the store's end and never sends on it.
    """
    if tier_settings is None:
        return False
    copies = tables.row(
        decoder_settings.DECODER_INPUTS,
        f"{tier}_decoder.input",
        tier_settings.input,
    )
    return not copies

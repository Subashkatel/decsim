"""The readout part: the path a round takes from the controller to the stores.

The controller takes each readout, the packing stage turns its fragments
into one packed round, the sender writes the round into every store that
must hold it, and each store's outgoing end hands the rounds on to the
decoder that reads them. A run that never reads the room side has no
strong syndrome buffer and no ends for one.
"""

import dataclasses
from typing import Optional

import decsim.controller.controller as controller_module
import decsim.controller.round_assembly as round_assembly
import decsim.controller.round_transmission as round_transmission
import decsim.controller.syndrome_round_sender as syndrome_round_sender
import decsim.decoders.memory_rounds as memory_rounds
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.detection_event_formation as formation
import decsim.detector_error_model.settings as event_settings
import decsim.engine as engine_module
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
    strong_syndrome_round_receiver as strong_receiver_module,
)
from decsim.syndrome_buffer import (
    weak_syndrome_round_receiver as weak_receiver_module,
)


@dataclasses.dataclass(frozen=True)
class Readout:
    """Every component a round passes on its way into the stores.

    The three strong fields are None on a run that never reads the room
    side. primary_output is one of the two outgoing ends: the one the
    tier that decodes the plan's windows reads.
    """

    controller: controller_module.Controller
    detection_events: ports.DetectionEventPlacement
    assembler: round_assembly.RoundAssembler
    packing_line: syndrome_round_sender.HeldRounds
    rounds_in_flight: round_assembly.RoundsInFlight
    syndrome_round_sender: syndrome_round_sender.SyndromeRoundSender
    transmitter: round_transmission.RoundTransmitter
    held_rounds: syndrome_round_sender.HeldRounds
    store_transfers: window_transfers.WindowTransfers
    memory_arrivals: memory_rounds.MemoryRoundArrivals
    weak_syndrome_buffer: ports.SyndromeBuffer
    weak_output: round_output.SyndromeBufferOutput
    weak_syndrome_round_receiver: weak_receiver_module.WeakSyndromeRoundReceiver
    strong_syndrome_buffer: Optional[ports.SyndromeBuffer]
    strong_output: Optional[round_output.SyndromeBufferOutput]
    strong_syndrome_round_receiver: Optional[
        strong_receiver_module.StrongSyndromeRoundReceiver
    ]
    primary_output: round_output.SyndromeBufferOutput

    @classmethod
    def build(
        cls,
        settings: machine_settings.MachineSettings,
        engine: engine_module.Engine,
        escalation_policy: ports.EscalationPolicy,
        detection_events: ports.DetectionEventPlacement,
        links: ports.Link,
    ) -> "Readout":
        """Every component of the readout path, wired to one another.

        One line per component, in the order a round meets them, then
        the wires inside the part.
        """
        _check_readout_cost_is_priced(settings)
        _check_one_price_for_a_read(settings)
        _check_store_kinds(settings)
        controller_settings = settings.controller
        controller = controller_module.Controller(engine, controller_settings)
        assembler = round_assembly.RoundAssembler(engine, controller_settings)
        packing_line = syndrome_round_sender.HeldRounds(engine)
        in_flight_bound = controller_settings.packing_rounds_in_flight
        rounds_in_flight = round_assembly.RoundsInFlight(in_flight_bound)
        sender = syndrome_round_sender.SyndromeRoundSender(engine)
        transmitter = round_transmission.RoundTransmitter(engine)
        held_rounds = syndrome_round_sender.HeldRounds(engine)
        store_transfers = window_transfers.WindowTransfers(engine)
        memory_arrivals = memory_rounds.MemoryRoundArrivals(engine)
        weak_store = _store(settings.weak_syndrome_buffer, "weak", engine)
        weak_output = _weak_output(settings, engine)
        weak_receiver = weak_receiver_module.WeakSyndromeRoundReceiver(engine)
        strong_store, strong_output, strong_receiver = _strong_store(
            settings, engine, escalation_policy
        )
        primary_output = weak_output
        if _uses_the_room_side(escalation_policy):
            primary_output = strong_output
        readout = cls(
            controller=controller,
            detection_events=detection_events,
            assembler=assembler,
            packing_line=packing_line,
            rounds_in_flight=rounds_in_flight,
            syndrome_round_sender=sender,
            transmitter=transmitter,
            held_rounds=held_rounds,
            store_transfers=store_transfers,
            memory_arrivals=memory_arrivals,
            weak_syndrome_buffer=weak_store,
            weak_output=weak_output,
            weak_syndrome_round_receiver=weak_receiver,
            strong_syndrome_buffer=strong_store,
            strong_output=strong_output,
            strong_syndrome_round_receiver=strong_receiver,
            primary_output=primary_output,
        )
        readout._wire_the_controller_path(links)
        readout._wire_the_weak_store(links)
        readout._wire_the_strong_store(links)
        readout._wire_the_primary_store()
        return readout

    def connect(
        self,
        windows: ports.WindowInput,
        retention: ports.WindowRetention,
    ) -> None:
        """Tell the window side of every round that lands."""
        self.syndrome_round_sender.windows = windows
        self.memory_arrivals.windows = windows
        self.weak_syndrome_round_receiver.windows = windows
        self.store_transfers.retention = retention
        if self.strong_syndrome_round_receiver is not None:
            self.strong_syndrome_round_receiver.windows = windows

    def start(self) -> None:
        """Let the sender read which store the plan's windows come from."""
        self.syndrome_round_sender.start()

    def check_settled(self) -> None:
        """No round is still on its way; the stores before the waiting line.

        A round held for room is the symptom, the hold that keeps the
        store full is the cause, so the stores are asked first; the
        seats last, since every round has left the stores by then.
        """
        self.assembler.check_settled()
        self.weak_syndrome_round_receiver.check_settled()
        if self.strong_syndrome_round_receiver is not None:
            self.strong_syndrome_round_receiver.check_settled()
        self.syndrome_round_sender.check_settled()
        self.transmitter.check_settled()
        self.detection_events.check_settled()

    def seed_roots(self) -> tuple:
        """This part's stochastic owners, each by the name its seed hashes."""
        return (("controller", self.controller),)

    def _wire_the_controller_path(self, links: ports.Link) -> None:
        self.controller.link = links
        self.controller.assembler = self.assembler
        self.assembler.detection_events = self.detection_events
        self.assembler.rounds_in_flight = self.rounds_in_flight
        self.assembler.packing_line = self.packing_line
        self.assembler.syndrome_round_sender = self.syndrome_round_sender
        self.rounds_in_flight.held_rounds = self.held_rounds
        self.rounds_in_flight.transmitter = self.transmitter
        self.rounds_in_flight.syndrome_round_sender = self.syndrome_round_sender
        self.syndrome_round_sender.link = links
        self.syndrome_round_sender.transmitter = self.transmitter
        self.syndrome_round_sender.packing_line = self.packing_line
        self.syndrome_round_sender.held_rounds = self.held_rounds
        self.transmitter.link = links
        self.transmitter.packing_line = self.packing_line
        self.transmitter.memory_arrivals = self.memory_arrivals
        self.store_transfers.link = links

    def _wire_the_weak_store(self, links: ports.Link) -> None:
        self.weak_syndrome_buffer.held_rounds = self.held_rounds
        self.weak_output.transfers = self.store_transfers
        self.weak_output.store = self.weak_syndrome_buffer
        self.weak_output.link = links
        self.weak_output.detection_events = self.detection_events
        receiver = self.weak_syndrome_round_receiver
        receiver.store = self.weak_syndrome_buffer
        receiver.output = self.weak_output
        receiver.detection_events = self.detection_events
        self.transmitter.weak_receiver = receiver
        self.syndrome_round_sender.weak_receiver = receiver
        self.syndrome_round_sender.weak_store = self.weak_syndrome_buffer

    def _wire_the_strong_store(self, links: ports.Link) -> None:
        receiver = self.strong_syndrome_round_receiver
        if receiver is None:
            return
        self.strong_syndrome_buffer.held_rounds = self.held_rounds
        self.strong_output.transfers = self.store_transfers
        self.strong_output.store = self.strong_syndrome_buffer
        self.strong_output.link = links
        self.strong_output.detection_events = self.detection_events
        receiver.store = self.strong_syndrome_buffer
        receiver.output = self.strong_output
        receiver.memory_arrivals = self.memory_arrivals
        receiver.detection_events = self.detection_events
        self.syndrome_round_sender.strong_receiver = receiver

    def _wire_the_primary_store(self) -> None:
        """The store the plan's windows read tells the seats what leaves it.

        Every read, the escalation's included, holds its rounds there
        until they have landed and formed, so a round that leaves it is
        formed nowhere after.
        """
        primary_store = self.weak_syndrome_buffer
        if self.primary_output is self.strong_output:
            primary_store = self.strong_syndrome_buffer
        primary_store.detection_events = self.detection_events


def build_detection_events(
    settings: machine_settings.MachineSettings,
    device,
    escalation_policy: ports.EscalationPolicy,
    burst_detector: Optional[ports.BurstDetector] = None,
) -> ports.DetectionEventPlacement:
    """Where this machine forms its detection events, as one component.

    Every path a round of this run takes to a decoder crosses exactly
    one seat of detection_events.formed_at: a path with none would
    decode raw outcomes, and a path with two would form events of
    events, a wrong answer either way. A source that does not answer
    the DetectionEventFormer port forms nothing. A burst detector
    counts the rounds the first seat on the primary tier's path forms.
    The decoder units are compiled from this placement, so the machine
    builds it before the parts.
    """
    formed_at = settings.detection_events.formed_at
    paths = _paths_of_the_run(escalation_policy)
    for path in paths:
        _check_one_seat_on(path, formed_at)
    source = None
    if isinstance(device, ports.DetectionEventFormer):
        source = device
    observed_seat = None
    if burst_detector is not None:
        _check_forms_events(source)
        observed_seat = _first_seat_on(paths[0], formed_at)
    return formation.SeatedFormation(
        source, settings.detection_events, observed_seat, burst_detector
    )


def _uses_strong_store(escalation_policy: ports.EscalationPolicy) -> bool:
    """Whether a tier of this run reads its rounds from the room side."""
    if escalation_policy.requires_strong_context:
        return True
    return _uses_the_room_side(escalation_policy)


def _uses_the_room_side(escalation_policy: ports.EscalationPolicy) -> bool:
    """Whether the plan's decoding tier reads the strong syndrome buffer."""
    strong = window_records.DecoderTier.STRONG
    return escalation_policy.primary_tier is strong


def _strong_store(
    settings: machine_settings.MachineSettings,
    engine: engine_module.Engine,
    escalation_policy: ports.EscalationPolicy,
) -> tuple:
    """The strong syndrome buffer, its outgoing end and its receiving end.

    Three Nones on a run that never reads the room side.
    """
    if not _uses_strong_store(escalation_policy):
        return None, None, None
    store = _store(settings.strong_syndrome_buffer, "strong", engine)
    output = _strong_output(settings, engine)
    receiver = strong_receiver_module.StrongSyndromeRoundReceiver(engine)
    return store, output, receiver


def _store(
    store_settings: syndrome_buffer_settings.SyndromeBufferSettings,
    tier: str,
    engine: engine_module.Engine,
) -> ports.SyndromeBuffer:
    """One syndrome buffer, of the row its section names."""
    row = tables.row(
        ported_syndrome_buffer.SYNDROME_BUFFERS,
        f"{tier}_syndrome_buffer.kind",
        store_settings.kind,
    )
    return row(store_settings, engine)


def _weak_output(
    settings: machine_settings.MachineSettings, engine: engine_module.Engine
) -> round_output.SyndromeBufferOutput:
    """The weak syndrome buffer's outgoing end, which executes every send."""
    reads_in_place = _reads_in_place(settings.weak_decoder, "weak")
    return round_output.SyndromeBufferOutput(
        engine,
        transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER,
        "weak syndrome buffer",
        reads_in_place,
        "weak_decoder",
    )


def _strong_output(
    settings: machine_settings.MachineSettings, engine: engine_module.Engine
) -> round_output.SyndromeBufferOutput:
    """The strong syndrome buffer's outgoing end."""
    reads_in_place = _reads_in_place(settings.strong_decoder, "strong")
    return round_output.SyndromeBufferOutput(
        engine,
        transfer_records.LinkPath.STRONG_BUFFER_TO_STRONG_DECODER,
        "strong syndrome buffer",
        reads_in_place,
        "strong_decoder",
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


def _check_store_kinds(settings: machine_settings.MachineSettings) -> None:
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


def _check_one_price_for_a_read(
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


def _check_readout_cost_is_priced(
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


def _check_forms_events(source) -> None:
    """A burst detector counts detection events, so the source forms some."""
    if source is not None:
        return
    raise ValueError(
        "burst_detector counts detection events, and this qpu.kind "
        "forms none; name a source that forms them, or write "
        "burst_detector: {kind: none}"
    )


def _paths_of_the_run(escalation_policy: ports.EscalationPolicy) -> tuple:
    """The paths this run's rounds take to a decoder, the primary first."""
    if _uses_the_room_side(escalation_policy):
        return (event_settings.STRONG_PATH,)
    if escalation_policy.requires_strong_context:
        return (event_settings.WEAK_PATH, event_settings.ESCALATION_PATH)
    return (event_settings.WEAK_PATH,)


def _check_one_seat_on(path: tuple, formed_at: tuple) -> None:
    """The path crosses exactly one seat that forms, or the run is refused."""
    seats = []
    for seat in path:
        if seat in formed_at:
            seats.append(seat)
    if len(seats) == 1:
        return
    listed = ", ".join(path)
    raise ValueError(
        f"detection_events.formed_at {list(formed_at)} forms the rounds "
        f"on the path {listed} at {seats}: every path to a decoder "
        "crosses exactly one seat, since a path with none decodes raw "
        "outcomes and a path with two forms events of events"
    )


def _first_seat_on(path: tuple, formed_at: tuple) -> str:
    """The seat of formed_at the path crosses; the path crosses one."""
    for seat in path:
        if seat in formed_at:
            return seat
    raise AssertionError("every path crosses one seat")

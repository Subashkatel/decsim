"""The readout part: the path a round takes from the controller to the stores.

The controller takes each readout, the packing stage turns its fragments
into one packed round, the sender writes it into every store that must
hold it, and each store's outgoing end hands it on to the decoder that
reads it. A store exists only when a decoder reads it.
"""

import dataclasses
from typing import Optional, Union

import decsim.config as config
import decsim.controller.controller as controller_module
import decsim.controller.round_assembly as round_assembly
import decsim.controller.round_transmission as round_transmission
import decsim.controller.settings as controller_settings_module
import decsim.controller.syndrome_round_sender as syndrome_round_sender
import decsim.decoders.memory_rounds as memory_rounds
import decsim.detector_error_model.detection_event_formation as formation
import decsim.detector_error_model.settings as event_settings
import decsim.engine as engine_module
import decsim.links.settings as link_settings
import decsim.links.window_transfers as window_transfers
import decsim.ports as ports
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.syndrome_buffer.ported_syndrome_buffer as ported_syndrome_buffer
import decsim.syndrome_buffer.round_output as round_output
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
from decsim.syndrome_buffer import (
    strong_syndrome_round_receiver as strong_receiver_module,
)
from decsim.syndrome_buffer import (
    weak_syndrome_round_receiver as weak_receiver_module,
)


@dataclasses.dataclass(frozen=True)
class StoreSlot:
    """One decoder's read access to a syndrome buffer.

    reads_in_place: the unit reads the rounds where the store keeps them
    (DecoderPoolSettings.copies_input False).
    """

    settings: Union[
        syndrome_buffer_module.SyndromeBufferSettings,
        ported_syndrome_buffer.PortedSyndromeBufferSettings,
    ]
    reads_in_place: bool


@dataclasses.dataclass(frozen=True)
class Readout:
    """Every component a round passes on its way into the stores.

    A store no decoder reads has its three fields None. primary_output is
    the outgoing end of the store the plan's windows read.
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
    weak_syndrome_buffer: Optional[ports.SyndromeBuffer]
    weak_output: Optional[round_output.SyndromeBufferOutput]
    weak_syndrome_round_receiver: Optional[
        weak_receiver_module.WeakSyndromeRoundReceiver
    ]
    strong_syndrome_buffer: Optional[ports.SyndromeBuffer]
    strong_output: Optional[round_output.SyndromeBufferOutput]
    strong_syndrome_round_receiver: Optional[
        strong_receiver_module.StrongSyndromeRoundReceiver
    ]
    primary_output: Optional[round_output.SyndromeBufferOutput]

    @classmethod
    def build(
        cls,
        controller_settings: controller_settings_module.ControllerSettings,
        link_card: link_settings.FabricSettings,
        weak_store_slot: Optional[StoreSlot],
        strong_store_slot: Optional[StoreSlot],
        machine_clock: Optional[config.Clock],
        engine: engine_module.Engine,
        detection_events: ports.DetectionEventPlacement,
        links: ports.Link,
    ) -> "Readout":
        """Every component of the readout path, wired to one another."""
        _check_readout_cost_is_priced(controller_settings, link_card)
        _check_one_price_for_a_read(weak_store_slot, link_card)
        _check_strong_store_charges_nothing(strong_store_slot)
        clocked_controller = config.with_machine_clock(
            controller_settings, machine_clock
        )
        controller = controller_module.Controller(engine, clocked_controller)
        assembler = round_assembly.RoundAssembler(engine, clocked_controller)
        packing_line = syndrome_round_sender.HeldRounds(engine)
        in_flight_bound = controller_settings.packing_rounds_in_flight
        rounds_in_flight = round_assembly.RoundsInFlight(in_flight_bound)
        sender = syndrome_round_sender.SyndromeRoundSender(engine)
        transmitter = round_transmission.RoundTransmitter(engine)
        held_rounds = syndrome_round_sender.HeldRounds(engine)
        store_transfers = window_transfers.WindowTransfers(engine)
        memory_arrivals = memory_rounds.MemoryRoundArrivals(engine)
        weak_store, weak_output, weak_receiver = _weak_store(
            weak_store_slot, machine_clock, engine
        )
        strong_store, strong_output, strong_receiver = _strong_store(
            strong_store_slot, engine
        )
        primary_output = weak_output
        if weak_output is None:
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
        if self.weak_syndrome_round_receiver is not None:
            self.weak_syndrome_round_receiver.windows = windows
        self.store_transfers.retention = retention
        if self.strong_syndrome_round_receiver is not None:
            self.strong_syndrome_round_receiver.windows = windows

    def start(self) -> None:
        """Let the sender read which store the plan's windows come from."""
        self.syndrome_round_sender.start()

    def check_settled(self) -> None:
        """No round is still on its way; the stores before the waiting line.

        A round held for room is the symptom and the hold that keeps the
        store full the cause, so the stores are asked first.
        """
        self.assembler.check_settled()
        if self.weak_syndrome_round_receiver is not None:
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
        receiver = self.weak_syndrome_round_receiver
        if receiver is None:
            return
        self.weak_syndrome_buffer.held_rounds = self.held_rounds
        self.weak_output.transfers = self.store_transfers
        self.weak_output.store = self.weak_syndrome_buffer
        self.weak_output.link = links
        self.weak_output.detection_events = self.detection_events
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

        Every read holds its rounds there until they have landed and
        formed, so a round that leaves it is formed nowhere after.
        """
        primary_store = self.weak_syndrome_buffer
        if self.primary_output is self.strong_output:
            primary_store = self.strong_syndrome_buffer
        if primary_store is None:
            return
        primary_store.detection_events = self.detection_events


def build_detection_events(
    detection_event_settings: event_settings.DetectionEventSettings,
    machine_clock: Optional[config.Clock],
    device: ports.SyndromeSource,
    window_tier: window_records.DecoderTier,
    escalates: bool,
) -> ports.DetectionEventPlacement:
    """Where this machine forms its detection events, as one component.

    Every path a round takes to a decoder crosses exactly one seat of
    detection_events.formed_at: with none it would decode raw outcomes,
    with two it would form events of events. A source that does not
    answer DetectionEventFormer forms nothing.
    """
    formed_at = detection_event_settings.formed_at
    paths = _paths_of_the_run(window_tier, escalates)
    for path in paths:
        _check_one_seat_on(path, formed_at)
    source = None
    if isinstance(device, ports.DetectionEventFormer):
        source = device
    clocked_settings = config.with_machine_clock(
        detection_event_settings, machine_clock
    )
    return formation.SeatedFormation(source, clocked_settings)


def _weak_store(
    slot: Optional[StoreSlot],
    machine_clock: Optional[config.Clock],
    engine: engine_module.Engine,
) -> tuple:
    """The weak syndrome buffer, its outgoing end and its receiving end."""
    if slot is None:
        return None, None, None
    store_settings = config.with_machine_clock(slot.settings, machine_clock)
    store = store_settings.build(engine)
    output = round_output.SyndromeBufferOutput(
        engine,
        transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER,
        "weak syndrome buffer",
        slot.reads_in_place,
        "weak_decoder",
    )
    receiver = weak_receiver_module.WeakSyndromeRoundReceiver(engine)
    return store, output, receiver


def _strong_store(
    slot: Optional[StoreSlot], engine: engine_module.Engine
) -> tuple:
    """The strong syndrome buffer, its outgoing end and its receiving end."""
    if slot is None:
        return None, None, None
    store = slot.settings.build(engine)
    output = round_output.SyndromeBufferOutput(
        engine,
        transfer_records.LinkPath.STRONG_BUFFER_TO_STRONG_DECODER,
        "strong syndrome buffer",
        slot.reads_in_place,
        "strong_decoder",
    )
    receiver = strong_receiver_module.StrongSyndromeRoundReceiver(engine)
    return store, output, receiver


def _check_strong_store_charges_nothing(slot: Optional[StoreSlot]) -> None:
    """Refuse a cost on the strong syndrome buffer: it is never paid."""
    if slot is None:
        return
    settings = slot.settings
    if settings.prices_read_bits:
        raise ValueError(
            "the strong syndrome buffer takes SyndromeBufferSettings, not "
            "PortedSyndromeBufferSettings: it stores a round as it lands "
            "and books no port access"
        )
    charged = _charged_cost_fields(settings)
    if not charged:
        return
    raise ValueError(
        f"the strong syndrome buffer's settings set {charged}, which only "
        "the weak syndrome buffer charges; the strong syndrome buffer "
        "stores a round as it lands and charges nothing, so leave them out"
    )


def _charged_cost_fields(settings) -> list:
    """The cost fields of a plain store's settings set away from free."""
    charged = []
    if settings.clock is not None:
        charged.append("clock")
    if settings.write_cycles != 0:
        charged.append("write_cycles")
    if settings.read_cycles != 0:
        charged.append("read_cycles")
    return charged


def _check_one_price_for_a_read(
    weak_store_slot: Optional[StoreSlot],
    link_card: link_settings.FabricSettings,
) -> None:
    """A ported weak store prices its reads; a link rate would charge twice."""
    if weak_store_slot is None:
        return
    if not weak_store_slot.settings.prices_read_bits:
        return
    path = link_card.weak_buffer_to_weak_decoder
    if path.channel.capacity is None:
        return
    raise ValueError(
        "weak_syndrome_buffer kind ported_syndrome_buffer prices the read "
        "out of the store by its words, and links.weak_buffer_to_weak_decoder "
        "has a rate that would charge the same bits again; set its "
        "bits_per_cycle to null"
    )


def _check_readout_cost_is_priced(
    controller_settings: controller_settings_module.ControllerSettings,
    link_card: link_settings.FabricSettings,
) -> None:
    """A readout cost on the controller needs a card that leaves it out.

    A qpu_to_controller card that keeps the reference number already
    covers turning the readout into bits.
    """
    readout_cycles = controller_settings.readout_to_bits_cycles
    if readout_cycles == 0:
        return
    readout_hops = [link_card.qpu_to_controller]
    readout_hops.extend(route.settings for route in link_card.readout_routes)
    for readout_hop in readout_hops:
        if not readout_hop.excludes_receiver_processing:
            raise ValueError(
                "a separate controller readout cost requires a "
                "qpu_to_controller card whose latency excludes that cost"
            )


def _paths_of_the_run(
    window_tier: window_records.DecoderTier, escalates: bool
) -> tuple:
    """The paths this run's rounds take to a decoder, the primary first."""
    if window_tier is window_records.DecoderTier.STRONG:
        return (event_settings.STRONG_PATH,)
    if escalates:
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

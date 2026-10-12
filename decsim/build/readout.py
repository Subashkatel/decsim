"""The readout part: the path a round takes from the controller to the stores.

The controller takes each readout, the packing stage turns its fragments
into one packed round, the sender writes it into every store that must
hold it, and each store's outgoing end hands it on to the decoder that
reads it. A store exists only when a decoder reads it.
"""

import dataclasses
from typing import Optional

import decsim.build.layout as layout_module
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
import decsim.syndrome_buffer.round_output as round_output
import decsim.syndrome_buffer.syndrome_round_receiver as receiver_module


@dataclasses.dataclass(frozen=True)
class StoreParts:
    """One store, its outgoing end and its receiving end, and its plan.

    other_outputs holds an outgoing end per entry pool past the first a
    router feeds from this store, as (pool, output) pairs.
    """

    plan: layout_module.StorePlan
    store: ports.SyndromeBuffer
    output: round_output.SyndromeBufferOutput
    receiver: receiver_module.SyndromeRoundReceiver
    other_outputs: tuple = ()

    def output_for(
        self, pool: window_records.PoolId
    ) -> round_output.SyndromeBufferOutput:
        """The outgoing end that sends to one of the other entry pools."""
        for reader_pool, output in self.other_outputs:
            if reader_pool == pool:
                return output
        raise KeyError(pool)


@dataclasses.dataclass(frozen=True)
class Readout:
    """Every component a round passes on its way into the stores.

    stores holds each store the layout reads, the entry store first: the
    one the controller writes, whose outgoing end serves the plan's
    windows.
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
    stores: tuple

    @classmethod
    def build(
        cls,
        controller_settings: controller_settings_module.ControllerSettings,
        link_card: link_settings.FabricSettings,
        store_plans: tuple,
        machine_clock: Optional[config.Clock],
        engine: engine_module.Engine,
        detection_events: ports.DetectionEventPlacement,
        links: ports.Link,
    ) -> "Readout":
        """Every component of the readout path, wired to one another."""
        _check_readout_cost_is_priced(controller_settings, link_card)
        for plan in store_plans:
            _check_the_store_prices(plan, link_card)
        clocked_controller = config.with_machine_clock(
            controller_settings, machine_clock
        )
        controller = controller_module.Controller(engine, clocked_controller)
        assembler = round_assembly.RoundAssembler(engine, clocked_controller)
        packing_line = syndrome_round_sender.HeldRounds(engine)
        in_flight_bound = controller_settings.packing_rounds_in_flight
        rounds_in_flight = round_assembly.RoundsInFlight(in_flight_bound)
        landing_path = _landing_path(store_plans)
        sender = syndrome_round_sender.SyndromeRoundSender(engine, landing_path)
        transmitter = round_transmission.RoundTransmitter(engine)
        held_rounds = syndrome_round_sender.HeldRounds(engine)
        store_transfers = window_transfers.WindowTransfers(engine)
        memory_arrivals = memory_rounds.MemoryRoundArrivals(engine)
        stores = []
        for plan in store_plans:
            parts = _store_parts(plan, machine_clock, engine)
            stores.append(parts)
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
            stores=tuple(stores),
        )
        readout._wire_the_controller_path(links)
        for parts in readout.stores:
            readout._wire_the_store(parts, links)
        readout._wire_the_entry_store()
        return readout

    def store_parts(self, name: str) -> Optional[StoreParts]:
        """The parts of the store of this name; None when it was not built."""
        for parts in self.stores:
            if parts.plan.name == name:
                return parts
        return None

    def store_named(
        self, name: Optional[str]
    ) -> Optional[ports.SyndromeBuffer]:
        """The store of this name; None for no name or a store not built."""
        if name is None:
            return None
        parts = self.store_parts(name)
        if parts is None:
            return None
        return parts.store

    @property
    def primary_output(self) -> Optional[round_output.SyndromeBufferOutput]:
        """The entry store's outgoing end; None with no store."""
        if not self.stores:
            return None
        entry = self.stores[0]
        return entry.output

    def connect(
        self,
        windows: ports.WindowInput,
        retention: ports.WindowRetention,
    ) -> None:
        """Tell the window side of every round that lands."""
        self.memory_arrivals.windows = windows
        self.store_transfers.retention = retention
        for parts in self.stores:
            parts.receiver.windows = windows

    def check_settled(self) -> None:
        """No round is still on its way; the stores before the waiting line.

        A round held for room is the symptom and the hold that keeps the
        store full the cause, so the stores are asked first, entry first.
        """
        self.assembler.check_settled()
        for parts in self.stores:
            parts.receiver.check_settled()
        self.syndrome_round_sender.check_settled()
        self.transmitter.check_settled()
        self.detection_events.check_settled()

    def seed_roots(self) -> tuple:
        """This part's stochastic owners, each by the name its seed hashes."""
        return (("controller", self.controller),)

    def statistics_parts(self) -> tuple:
        """Each store the run built, at its settings path."""
        parts_by_path = []
        for parts in self.stores:
            parts_by_path.append((parts.plan.name, parts.store))
        return tuple(parts_by_path)

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

    def _wire_the_store(self, parts: StoreParts, links: ports.Link) -> None:
        """One store between its receiving end and its outgoing end.

        A written store takes its rounds from the transmitter; a landed
        one tells the memory arrivals of each round it stores.
        """
        store = parts.store
        output = parts.output
        receiver = parts.receiver
        store.held_rounds = self.held_rounds
        outputs = [output]
        for _, other_output in parts.other_outputs:
            outputs.append(other_output)
        for each_output in outputs:
            each_output.transfers = self.store_transfers
            each_output.store = store
            each_output.link = links
            each_output.detection_events = self.detection_events
        receiver.store = store
        receiver.output = output
        receiver.detection_events = self.detection_events
        if parts.plan.intake.shape == receiver_module.WRITTEN:
            self.transmitter.receiver = receiver
            return
        receiver.memory_arrivals = self.memory_arrivals

    def _wire_the_entry_store(self) -> None:
        """The store the plan's windows read: the one the sender writes.

        It tells the seats what leaves it: every read holds its rounds
        there until they have landed and formed, so a round that leaves
        it is formed nowhere after.
        """
        if not self.stores:
            return
        entry = self.stores[0]
        entry.store.detection_events = self.detection_events
        self.syndrome_round_sender.receiver = entry.receiver


def build_detection_events(
    detection_event_settings: event_settings.DetectionEventSettings,
    machine_clock: Optional[config.Clock],
    device: ports.SyndromeSource,
    paths: tuple,
) -> ports.DetectionEventPlacement:
    """Where this machine forms its detection events, as one component.

    paths is each path a round takes to a decoder, as seats
    (Layout.formation_paths). Every one crosses exactly one seat of
    detection_events.formed_at: with none it would decode raw outcomes,
    with two it would form events of events. A source that does not
    answer DetectionEventFormer forms nothing.
    """
    formed_at = detection_event_settings.formed_at
    for path in paths:
        _check_one_seat_on(path, formed_at)
    source = None
    if isinstance(device, ports.DetectionEventFormer):
        source = device
    clocked_settings = config.with_machine_clock(
        detection_event_settings, machine_clock
    )
    return formation.SeatedFormation(source, clocked_settings)


def _store_parts(
    plan: layout_module.StorePlan,
    machine_clock: Optional[config.Clock],
    engine: engine_module.Engine,
) -> StoreParts:
    """One store, its outgoing end and its receiving end."""
    store_settings = plan.settings
    if plan.takes_machine_clock:
        store_settings = config.with_machine_clock(
            store_settings, machine_clock
        )
    store = store_settings.build(engine)
    output = round_output.SyndromeBufferOutput(
        engine,
        plan.read_hop,
        plan.label,
        plan.reads_in_place,
        plan.reader,
    )
    receiver = receiver_module.SyndromeRoundReceiver(engine, plan.intake)
    other_outputs = []
    for reader in plan.other_readers:
        other_output = round_output.SyndromeBufferOutput(
            engine,
            reader.read_hop,
            plan.label,
            reader.reads_in_place,
            reader.reader,
        )
        other_outputs.append((reader.pool, other_output))
    return StoreParts(plan, store, output, receiver, tuple(other_outputs))


def _landing_path(store_plans: tuple) -> Optional[transfer_records.LinkPath]:
    """The hop the entry store's rounds land over; None for a written one.

    An entry store the controller writes takes its rounds through the
    transmitter; one that takes them landed, as a strong-only run's
    strong store does, is reached over its controller hop.
    """
    if not store_plans:
        return None
    entry = store_plans[0]
    intake = entry.intake
    if intake.shape == receiver_module.WRITTEN:
        return None
    return intake.controller_write.path_name


def _check_the_store_prices(
    plan: layout_module.StorePlan, link_card: link_settings.FabricSettings
) -> None:
    """A written store may price its reads; a landed one charges nothing."""
    if plan.intake.shape == receiver_module.WRITTEN:
        _check_one_price_for_a_read(plan, link_card)
        return
    _check_landed_store_charges_nothing(plan)


def _check_landed_store_charges_nothing(plan: layout_module.StorePlan) -> None:
    """Refuse a cost on a landed store: it is never paid."""
    settings = plan.settings
    label = plan.label
    if settings.prices_read_bits:
        raise ValueError(
            f"the {label} takes SyndromeBufferSettings, not "
            "PortedSyndromeBufferSettings: it stores a round as it lands "
            "and books no port access"
        )
    charged = _charged_cost_fields(settings)
    if not charged:
        return
    raise ValueError(
        f"the {label}'s settings set {charged}, which only the store the "
        f"controller writes charges; the {label} stores a round as it "
        "lands and charges nothing, so leave them out"
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
    plan: layout_module.StorePlan, link_card: link_settings.FabricSettings
) -> None:
    """A ported store prices its reads; a link rate would charge twice."""
    if not plan.settings.prices_read_bits:
        return
    path = link_card.path_settings(plan.read_hop)
    if path.channel.capacity is None:
        return
    raise ValueError(
        f"{plan.name} kind ported_syndrome_buffer prices the read out of "
        f"the store by its words, and links.{plan.read_hop} has a rate "
        "that would charge the same bits again; set its bits_per_cycle "
        "to null"
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

"""The controller's syndrome packing stage.

Raw measurement fragments of one round are assembled in this stage's own
workspace, merged, formed into detection events (the device's formation
table), and the finished round is written into the stores (Buffer 0, and
syndrome buffer 1 when wired) before it is arbitrated onto its route:
controller_to_weak_buffer to the window input, or
weak_buffer_to_weak_decoder as a feedback-memory round. The stores hold
finished rounds only; a context is PARTIAL while fragments are missing,
STALLED while its store is full, PACKED_WAIT while it waits for its
route, and DRAINING once transmission has started.
"""

import dataclasses
import enum
import functools
from typing import Callable, Optional

import decsim.message as message
import decsim.ports as ports
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module


class ReassemblyQueueAdmission(enum.Enum):
    """When a round starts competing for its route.

    As soon as its first fragment allocates a context, or once the round
    is complete.
    """

    ON_ALLOCATION = "on_allocation"
    ON_COMPLETION = "on_completion"


class PackingOverflowPolicy(enum.Enum):
    """What the controller does with a finished round when its store is full.

    STALL holds the round in the packing workspace until a slot frees and
    publishes it in order, the backpressure real-time systems apply to
    their source: the Rigetti sequencer polls the decoder's status
    register and stalls (Caune et al. 2410.05202), Helios's input port is
    valid/ready and asserts ready only when it can take data, QubiC's
    cores block in WAIT_MEAS, and credit-based flow control never loses
    a flit. LILLIPUT's readout buffer and QubiC's measurement register
    instead overwrite the latest value, a storage choice this policy does
    not model. DROP_ROUND and FAIL_STOP are study knobs.
    """

    STALL = "stall"
    FAIL_STOP = "fail_stop"
    DROP_ROUND = "drop_round"


@dataclasses.dataclass(frozen=True)
class SyndromePackingPolicy:
    """The packing stage's Python-only knobs."""

    queue_admission: ReassemblyQueueAdmission = (
        ReassemblyQueueAdmission.ON_COMPLETION
    )
    overflow: PackingOverflowPolicy = PackingOverflowPolicy.STALL
    reassembly_timeout_ticks: Optional[int] = None


_DEFAULT_POLICY = SyndromePackingPolicy()


@dataclasses.dataclass(frozen=True)
class SyndromePackingSnapshot:
    """Live packing contexts by state."""

    packing_context_capacity: Optional[int]
    packing_contexts: int
    partial_identities: tuple
    stalled_identities: tuple
    packed_wait_identities: tuple
    draining_identities: tuple


class SyndromeReassemblyTimeoutError(RuntimeError):
    """An incomplete round exceeded the configured reassembly timeout."""

    status = "syndrome_reassembly_timeout"

    def __init__(
        self, *, tick, identity, received_fragments, expected_fragments
    ):
        self.tick = tick
        self.identity = identity
        self.received_fragments = received_fragments
        self.expected_fragments = expected_fragments
        text = (
            f"syndrome reassembly {identity!r} timed out at tick {tick}: "
            f"received {received_fragments}/{expected_fragments} fragments"
        )
        RuntimeError.__init__(self, text)


class SyndromePackingOverflowError(RuntimeError):
    """A new round found no free assembly context, or no slot in Buffer 0."""

    status = "controller_packing_overflow"

    def __init__(self, *, tick, route, incoming_identity, capacity, snapshot):
        self.tick = tick
        self.route = route
        self.incoming_packet_or_fragment_identity = incoming_identity
        self.packing_context_capacity = capacity
        self.packing_contexts = snapshot.packing_contexts
        self.partial_identities = snapshot.partial_identities
        self.packed_wait_identities = snapshot.packed_wait_identities
        text = f"controller packing capacity {capacity} is full at tick {tick}"
        RuntimeError.__init__(self, text)


@dataclasses.dataclass(frozen=True)
class SyndromeRoundEvent:
    """One recorded transition of one syndrome round through the controller.

    The flight-recorder rows the run ledger is built from. Append-only
    and passive; recording never schedules or decides. Reassembly expiry
    raises instead of recording, so a finished run's terminal states are
    PUBLISHED, DROPPED, or FEEDBACK_MEMORY_DELIVERED. kind is one of
    EMITTED, BINARY_AVAILABLE, PACKED, STALLED, CWB_SENT, PUBLISHED,
    DROPPED, FEEDBACK_MEMORY_DELIVERED.
    """

    kind: str
    tick: int
    operation_id: object
    round_index: int
    patch_id: object = None
    route: str = ""


class _PackingSlotState(enum.Enum):
    PARTIAL = enum.auto()
    # packed, waiting for a free store slot
    STALLED = enum.auto()
    PACKED_WAIT = enum.auto()
    DRAINING = enum.auto()


@dataclasses.dataclass
class _PackingContext:
    """One round on its way through the controller.

    Its route, how far its assembly has come, and its packed packet once
    complete.
    """

    identity: tuple
    round_key: tuple
    route: message.SyndromePacketRoute
    fragment_count: int
    fragments: list = dataclasses.field(default_factory=list)
    received_fragments: int = 0
    state: _PackingSlotState = _PackingSlotState.PARTIAL
    packet: Optional[message.SyndromeRoundPacket] = None
    packet_bits: Optional[int] = None
    cwb_reserved: bool = False
    cwb_delivered: bool = False


class SyndromePacking:
    """The packing stage: assemble, store, arbitrate and transmit rounds."""

    def __init__(
        self,
        engine,
        links: ports.Link,
        packing_ticks: int = 0,
        *,
        packing_context_capacity: Optional[int],
        window_input_receiver,
        feedback_memory_receiver,
        syndrome_buffer: Optional[syndrome_buffer_module.SyndromeBuffer] = None,
        syndrome_buffer_1=None,
        window_input_store=None,
        policy: SyndromePackingPolicy = _DEFAULT_POLICY,
        detector_formation=None,
    ):
        self.policy = policy
        # the source that forms a complete round's detection events from
        # its raw packet (the device, which holds the formation table);
        # None for timing-only or synthetic sources
        self.detector_formation = detector_formation
        self.engine = engine
        self.links = links
        self.packing_ticks = packing_ticks
        # assembly workspace: how many rounds may be in flight through
        # this stage at once; retention capacity is the stores' own knob
        self.packing_context_capacity = packing_context_capacity
        self.window_input_receiver = window_input_receiver
        self.feedback_memory_receiver = feedback_memory_receiver
        if syndrome_buffer is None:
            syndrome_buffer = syndrome_buffer_module.SyndromeBuffer(
                capacity=packing_context_capacity
            )
        self.syndrome_buffer = syndrome_buffer
        self.syndrome_buffer_1 = syndrome_buffer_1
        self.window_input_store = self._window_input_store(window_input_store)
        self._contexts: dict = {}
        # finished rounds waiting for store room, in completion order
        self._stalled_identities: list = []
        self._connect_store_releases()
        self._route_queues = {}
        for kind in message.SyndromePacketRouteKind:
            self._route_queues[kind] = []
        self._next_route_index = 0
        self._arbitration_pending = False
        self._dropped_rounds: set = set()
        # the flight-recorder rows (SyndromeRoundEvent), append-only
        self.round_events: list = []
        self.reassembly_timeouts = 0
        self.packing_drops = 0
        self._connect_window_input_ready(window_input_receiver)

    def _window_input_store(self, window_input_store):
        """The store whose arrival wakes windows.

        Buffer 0 for a weak-primary run, syndrome buffer 1 when the
        strong tier decodes the plan's windows itself. A one-tier system
        has one path from the controller to its decoder buffer, so a
        strong-primary round never crosses controller_to_weak_buffer.
        """
        if window_input_store is None:
            return self.syndrome_buffer
        return window_input_store

    def _connect_store_releases(self) -> None:
        """A store that frees a slot wakes the stalled rounds."""
        self.syndrome_buffer.on_round_released = self._retry_stalled_rounds
        if self.syndrome_buffer_1 is not None:
            room_store = self.syndrome_buffer_1.store
            room_store.on_round_released = self._retry_stalled_rounds

    def _connect_window_input_ready(self, window_input_receiver) -> None:
        connect_ready = getattr(
            window_input_receiver, "connect_window_input_ready_receiver", None
        )
        if callable(connect_ready):
            connect_ready(self.notify_window_input_ready)

    # ---- arrival: qpu_to_controller delivery, processing, reassembly

    def relay_qpu_readout(
        self,
        payload,
        route: message.SyndromePacketRoute,
        *,
        processing_ticks: int,
    ) -> None:
        """Carry one readout over qpu_to_controller, then receive it.

        The fragment is received after the controller's processing time.
        """
        fragment = message.RetainedSyndromeFragment.from_payload(payload)
        fragment_count = payload.n_fragments
        self._record(
            "EMITTED",
            fragment.operation_id,
            fragment.round_index,
            route,
            patch_id=fragment.patch_id,
        )
        attribution = _round_attribution(
            fragment.operation_id, (fragment.patch_id,), fragment.round_index
        )

        def receive():
            self._receive_fragment(fragment, fragment_count, route)

        def at_controller():
            if processing_ticks == 0:
                receive()
                return
            self.engine.schedule(
                processing_ticks,
                receive,
                label="controller-binary-availability",
            )

        self._send(
            message.LinkPath.QPU_TO_CONTROLLER,
            payload_bits=payload.size_bits,
            attribution=attribution,
            on_delivered=at_controller,
        )

    def _receive_fragment(
        self,
        fragment: message.RetainedSyndromeFragment,
        fragment_count: int,
        route: message.SyndromePacketRoute,
    ) -> None:
        self._record(
            "BINARY_AVAILABLE",
            fragment.operation_id,
            fragment.round_index,
            route,
            patch_id=fragment.patch_id,
        )
        round_key = (fragment.operation_id, fragment.round_index)
        if round_key in self._dropped_rounds:
            return
        context = self._context_for(fragment, fragment_count, route)
        if context is None:
            return
        context.fragments.append(fragment)
        context.received_fragments = len(context.fragments)
        if context.received_fragments != context.fragment_count:
            return
        # every completed round pays the assembly time once: packetization
        # and framing cost the controller per syndrome word, however many
        # fragments the word arrived in. Caune et al. 2410.05202 measure
        # 250 to 370 FPGA cycles for packetization, bus transfer, result
        # return and the conditional together, an upper bound for this term
        if self.packing_ticks > 0:
            finish = functools.partial(self._finish_packing, context)
            self.engine.schedule(
                self.packing_ticks, finish, label="controller pack"
            )
            return
        self._finish_packing(context)

    def _context_for(
        self, fragment, fragment_count, route
    ) -> Optional[_PackingContext]:
        """The live context of the fragment's round, opened on its first.

        None when the round was dropped for want of a context.
        """
        round_key = (fragment.operation_id, fragment.round_index)
        identity = (
            route.kind.name,
            route.source_operation_id,
            fragment.operation_id,
            fragment.round_index,
        )
        context = self._contexts.get(identity)
        if context is not None:
            return context
        capacity = self.packing_context_capacity
        if capacity is not None and len(self._contexts) >= capacity:
            return self._workspace_is_full(fragment, route, identity, round_key)
        return self._open_context(identity, round_key, route, fragment_count)

    def _workspace_is_full(
        self, fragment, route, identity, round_key
    ) -> Optional[_PackingContext]:
        """Apply the overflow policy to a round that finds no free context."""
        if self.policy.overflow is PackingOverflowPolicy.DROP_ROUND:
            self.packing_drops += 1
            self._dropped_rounds.add(round_key)
            self._record(
                "DROPPED",
                fragment.operation_id,
                fragment.round_index,
                route,
                patch_id=fragment.patch_id,
            )
            return None
        snapshot = self.packing_snapshot()
        raise SyndromePackingOverflowError(
            tick=self.engine.now,
            route=route,
            incoming_identity=identity,
            capacity=self.packing_context_capacity,
            snapshot=snapshot,
        )

    def _open_context(
        self, identity, round_key, route, fragment_count: int
    ) -> _PackingContext:
        context = _PackingContext(identity, round_key, route, fragment_count)
        self._contexts[identity] = context
        if (
            self.policy.queue_admission
            is ReassemblyQueueAdmission.ON_ALLOCATION
        ):
            self._route_queues[route.kind].append(identity)
        timeout_ticks = self.policy.reassembly_timeout_ticks
        if timeout_ticks is not None:
            expire = functools.partial(self._expire_reassembly, identity)
            self.engine.schedule(
                timeout_ticks, expire, label="syndrome reassembly timeout"
            )
        return context

    def _forget_context(self, context: _PackingContext) -> None:
        self._contexts.pop(context.identity, None)
        route_queue = self._route_queues[context.route.kind]
        if context.identity in route_queue:
            route_queue.remove(context.identity)

    def _finish_packing(self, context: _PackingContext) -> None:
        """The round is complete: merge, form detection events, store."""
        operation_id, round_index = context.round_key
        self._record("PACKED", operation_id, round_index, context.route)
        raw_fragments = _merge_fragments_by_patch(context.fragments)
        # controller_to_weak_buffer carries the raw measurement bits;
        # detection events exist only from the decoder input (Buffer 0)
        # onward
        context.packet_bits = _fragment_bits(raw_fragments)
        formed_fragments = self._form_detection_events(raw_fragments)
        context.packet = message.SyndromeRoundPacket(
            operation_id=operation_id,
            round_index=round_index,
            fragments=formed_fragments,
        )
        context.fragments = []
        self._admit_packed_round(context)

    def _admit_packed_round(self, context: _PackingContext) -> bool:
        """Store the packed round and let it compete for its route.

        Its publication tick is set now unless a controller_to_weak_buffer
        hop is priced later. Returns False when the store is full and the
        overflow policy keeps the round waiting.
        """
        packet = context.packet
        if self._stores_window_input_in_syndrome_buffer_1(context):
            if not self.syndrome_buffer_1.has_room():
                return self._store_is_full(context)
            self._store_in_syndrome_buffer_1(packet, context)
            return True
        if not self.syndrome_buffer.has_operation(packet.operation_id):
            self.syndrome_buffer.open_operation(packet.operation_id)
        if not self._strong_store_has_room():
            return self._store_is_full(context)
        publication_tick = self._publication_tick_at_storage(context)
        admission = self.syndrome_buffer.accept_packed_round(
            packet, publication_tick=publication_tick
        )
        if admission.refused:
            return self._store_is_full(context)
        self._stored_in_buffer_0(context, publication_tick)
        context.state = _PackingSlotState.PACKED_WAIT
        if (
            self.policy.queue_admission
            is ReassemblyQueueAdmission.ON_COMPLETION
        ):
            self._route_queues[context.route.kind].append(context.identity)
        self._schedule_arbitration()
        return True

    def _strong_store_has_room(self) -> bool:
        if self.syndrome_buffer_1 is None:
            return True
        return self.syndrome_buffer_1.has_room()

    def _publication_tick_at_storage(
        self, context: _PackingContext
    ) -> Optional[int]:
        """The window-input route's publication tick, known at storage.

        Publication is the route's arrival in Buffer 0: now on a fabric
        without controller_to_weak_buffer, at its delivery otherwise; a
        feedback-memory round is stored but never published, its
        terminal is FEEDBACK_MEMORY_DELIVERED.
        """
        cwb_is_priced = self.links.is_wired(
            message.LinkPath.CONTROLLER_TO_WEAK_BUFFER
        )
        window_input = message.SyndromePacketRouteKind.WINDOW_INPUT
        on_window_route = context.route.kind is window_input
        if on_window_route and not cwb_is_priced:
            return self.engine.now
        return None

    def _stored_in_buffer_0(
        self, context: _PackingContext, publication_tick: Optional[int]
    ) -> None:
        """Record the storage; the dual write leaves for the room side."""
        packet = context.packet
        operation_id, round_index = context.round_key
        if publication_tick is not None:
            self._record(
                "PUBLISHED",
                operation_id,
                round_index,
                context.route,
                tick=publication_tick,
            )
        self.engine.log_io("Buffer 0", lambda: self._received_text(packet))
        if self.syndrome_buffer_1 is not None:
            # the dual write: the same packed round leaves for the
            # room-side store in parallel with its Buffer 0 publication
            attribution = _packet_attribution(packet)
            self.syndrome_buffer_1.write(
                packet, packet_bits=context.packet_bits, attribution=attribution
            )

    def _received_text(self, packet: message.SyndromeRoundPacket) -> str:
        defects = packet.defects_text()
        holds = self.syndrome_buffer.held_rounds_description()
        return (
            f"received round {packet.round_index} of "
            f"op {packet.operation_id} from packing; {defects}; holds {holds}"
        )

    def _store_is_full(self, context: _PackingContext) -> bool:
        """Apply the overflow policy to a packed round its store cannot take."""
        policy = self.policy.overflow
        operation_id, round_index = context.round_key
        if policy is PackingOverflowPolicy.STALL:
            if context.state is not _PackingSlotState.STALLED:
                context.state = _PackingSlotState.STALLED
                self._stalled_identities.append(context.identity)
                self._record(
                    "STALLED", operation_id, round_index, context.route
                )
            return False
        self._forget_context(context)
        if policy is PackingOverflowPolicy.DROP_ROUND:
            self.packing_drops += 1
            self._dropped_rounds.add(context.round_key)
            self._record("DROPPED", operation_id, round_index, context.route)
            return False
        snapshot = self.packing_snapshot()
        raise SyndromePackingOverflowError(
            tick=self.engine.now,
            route=context.route,
            incoming_identity=context.identity,
            capacity=self.syndrome_buffer.capacity,
            snapshot=snapshot,
        )

    def _retry_stalled_rounds(self) -> None:
        """A store freed a slot: admit waiting rounds in completion order.

        The walk stops at the first round that still finds no room.
        """
        while self._stalled_identities:
            head = self._stalled_identities[0]
            context = self._contexts[head]
            admitted = self._admit_packed_round(context)
            if not admitted:
                return
            self._stalled_identities.pop(0)

    def _stores_window_input_in_syndrome_buffer_1(self, context) -> bool:
        window_input = message.SyndromePacketRouteKind.WINDOW_INPUT
        on_window_route = context.route.kind is window_input
        strong_is_window_store = (
            self.window_input_store is self.syndrome_buffer_1
        )
        return on_window_route and strong_is_window_store

    def _store_in_syndrome_buffer_1(self, packet, context) -> None:
        """A strong-primary round's only hop: into syndrome buffer 1.

        The store's arrival signal wakes the window.
        """
        attribution = _packet_attribution(packet)
        self.syndrome_buffer_1.write(
            packet, packet_bits=context.packet_bits, attribution=attribution
        )
        self._forget_context(context)
        if self._has_queued_rounds():
            self._schedule_arbitration()

    def _form_detection_events(self, raw_fragments) -> tuple:
        """Form the complete round's detection events from its raw packet.

        Sources without a formation table (timing-only or synthetic bits)
        are retained as they arrived.
        """
        form_round = getattr(self.detector_formation, "form_round", None)
        if form_round is None:
            return raw_fragments
        has_bits = all(fragment.bits is not None for fragment in raw_fragments)
        if not has_bits:
            return raw_fragments
        assert len(raw_fragments) == 1, (
            "detector formation expects one merged raw fragment per round"
        )
        (raw,) = raw_fragments
        events = form_round(raw.operation_id, raw.round_index, raw.bits)
        formed = tuple(events)
        fragment = dataclasses.replace(raw, bits=formed, size_bits=len(formed))
        return (fragment,)

    # ---- arbitration onto the routes

    def _has_queued_rounds(self) -> bool:
        queues = self._route_queues.values()
        return any(queues)

    def _schedule_arbitration(self) -> None:
        if self._arbitration_pending:
            return
        self._arbitration_pending = True
        self.engine.schedule(
            0, self._arbitrate, label="syndrome packing arbitration"
        )

    def _arbitrate(self) -> None:
        """Try each route once, round robin.

        The route after the last one that progressed goes first next
        time.
        """
        self._arbitration_pending = False
        kinds = tuple(message.SyndromePacketRouteKind)
        first = self._next_route_index
        ordered_kinds = kinds[first:] + kinds[:first]
        for kind in ordered_kinds:
            route_queue = self._route_queues[kind]
            if not route_queue:
                continue
            progressed = self._drain_route_queue(route_queue)
            if progressed:
                position = kinds.index(kind)
                self._next_route_index = (position + 1) % len(kinds)

    def _drain_route_queue(self, route_queue) -> bool:
        """Rounds pipeline onto their link.

        Every waiting round behind in-flight ones is sent in completion
        order, on both routes; a refused window-input round stops the
        walk so Buffer 0 sees rounds in order. The sender never waits for
        a round to land before sending the next: the DAQs of Yang et al.
        (arXiv 2605.04892) and the control electronics of Google (arXiv
        2408.13687) stream every round to the decoder, Caune et al.
        (arXiv 2410.05202) publish each classified result as it is
        produced, gem5's DmaPort queues the next request behind the front
        of transmitList (src/dev/dma_device.cc) and ns-3's point-to-point
        device starts the next packet at TransmitComplete
        (point-to-point-net-device.cc); the FIFO channel keeps delivery
        order.
        """
        progressed = False
        for identity in list(route_queue):
            context = self._contexts[identity]
            if context.state is not _PackingSlotState.PACKED_WAIT:
                continue
            sent = self._attempt_head(context)
            if not sent:
                break
            progressed = True
        return progressed

    def _attempt_head(self, context: _PackingContext) -> bool:
        if context.state is not _PackingSlotState.PACKED_WAIT:
            return False
        window_input = message.SyndromePacketRouteKind.WINDOW_INPUT
        if context.route.kind is window_input:
            return self._transmit_window_input_round(context)
        context.state = _PackingSlotState.DRAINING
        self._transmit_feedback_memory_round(context)
        return True

    # ---- window input: controller_to_weak_buffer to Buffer 0

    def _transmit_window_input_round(self, context: _PackingContext) -> bool:
        """Send on controller_to_weak_buffer once.

        A packet delivered but backpressured is retried without a second
        send, as is every packet on a fabric without the path.
        """
        cwb_is_priced = self.links.is_wired(
            message.LinkPath.CONTROLLER_TO_WEAK_BUFFER
        )
        if context.cwb_reserved or not cwb_is_priced:
            return self._deliver_window_input_round(context)
        packet = context.packet
        context.cwb_reserved = True
        operation_id, round_index = context.round_key
        self._record("CWB_SENT", operation_id, round_index, context.route)
        context.state = _PackingSlotState.DRAINING
        attribution = _packet_attribution(packet)
        deliver = functools.partial(self._deliver_window_input_round, context)
        self._send(
            message.LinkPath.CONTROLLER_TO_WEAK_BUFFER,
            payload_bits=context.packet_bits,
            attribution=attribution,
            on_delivered=deliver,
        )
        return True

    def _deliver_window_input_round(self, context: _PackingContext) -> bool:
        cwb_is_priced = self.links.is_wired(
            message.LinkPath.CONTROLLER_TO_WEAK_BUFFER
        )
        if cwb_is_priced and not context.cwb_delivered:
            self.syndrome_buffer.mark_publication_tick(
                context.round_key, self.engine.now
            )
            context.cwb_delivered = True
            operation_id, round_index = context.round_key
            self._record("PUBLISHED", operation_id, round_index, context.route)
        if self._refused_round_ahead_of(context):
            # keep round order
            context.state = _PackingSlotState.PACKED_WAIT
            return False
        accepted = self.window_input_receiver.accept_window_input(
            context.packet
        )
        if not accepted:
            context.state = _PackingSlotState.PACKED_WAIT
            return False
        context.state = _PackingSlotState.DRAINING
        release = functools.partial(self._release_context, context)
        self.engine.schedule(
            0, release, label="window input publication complete"
        )
        return True

    def _refused_round_ahead_of(self, context: _PackingContext) -> bool:
        route_queue = self._route_queues[context.route.kind]
        position = route_queue.index(context.identity)
        ahead = route_queue[:position]
        states = [self._contexts[identity].state for identity in ahead]
        return _PackingSlotState.PACKED_WAIT in states

    def notify_window_input_ready(self) -> None:
        """Buffer 0 has room again: retry backpressured packets."""
        self._schedule_arbitration()

    # ---- feedback memory: weak_buffer_to_weak_decoder

    def _transmit_feedback_memory_round(self, context: _PackingContext) -> None:
        """Send on weak_buffer_to_weak_decoder once.

        The next memory round follows at its own completion tick, not at
        this one's landing (_drain_route_queue).
        """
        packet = context.packet
        source_operation_id = context.route.source_operation_id
        attribution = _packet_attribution(packet)
        deliver = functools.partial(
            self._deliver_feedback_memory_round, context, source_operation_id
        )
        self._send(
            message.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER,
            payload_bits=context.packet_bits,
            attribution=attribution,
            on_delivered=deliver,
        )

    def _deliver_feedback_memory_round(
        self, context: _PackingContext, source_operation_id
    ) -> None:
        self.feedback_memory_receiver.accept_feedback_memory_round(
            source_operation_id
        )
        operation_id, round_index = context.round_key
        self._record(
            "FEEDBACK_MEMORY_DELIVERED",
            operation_id,
            round_index,
            context.route,
        )
        self.syndrome_buffer.release_round(context.round_key)
        self._release_context(context)

    # ---- context end

    def _expire_reassembly(self, identity) -> None:
        context = self._contexts.get(identity)
        if context is None:
            return
        if context.state is not _PackingSlotState.PARTIAL:
            return
        self.reassembly_timeouts += 1
        self._forget_context(context)
        raise SyndromeReassemblyTimeoutError(
            tick=self.engine.now,
            identity=identity,
            received_fragments=context.received_fragments,
            expected_fragments=context.fragment_count,
        )

    def _release_context(self, context: _PackingContext) -> None:
        route_queue = self._route_queues[context.route.kind]
        assert route_queue, "the controller released an empty route"
        head = route_queue.pop(0)
        assert head == context.identity, (
            "the controller released a non-head route slot"
        )
        del self._contexts[context.identity]
        if self._has_queued_rounds():
            self._schedule_arbitration()

    def check_work_settled(self) -> None:
        """Fail if the run ended with a partial or blocked packing context."""
        snapshot = self.packing_snapshot()
        if not snapshot.packing_contexts:
            return
        live = (
            snapshot.partial_identities
            + snapshot.stalled_identities
            + snapshot.packed_wait_identities
            + snapshot.draining_identities
        )
        raise RuntimeError(
            f"run ended with incomplete syndrome packing contexts: {live}"
        )

    def packing_snapshot(self) -> SyndromePackingSnapshot:
        """The live contexts by state."""
        identities_by_state = {}
        for state in _PackingSlotState:
            identities_by_state[state] = []
        for context in self._contexts.values():
            identities_by_state[context.state].append(context.identity)
        partial = identities_by_state[_PackingSlotState.PARTIAL]
        stalled = identities_by_state[_PackingSlotState.STALLED]
        packed_wait = identities_by_state[_PackingSlotState.PACKED_WAIT]
        draining = identities_by_state[_PackingSlotState.DRAINING]
        return SyndromePackingSnapshot(
            packing_context_capacity=self.packing_context_capacity,
            packing_contexts=len(self._contexts),
            partial_identities=tuple(partial),
            stalled_identities=tuple(stalled),
            packed_wait_identities=tuple(packed_wait),
            draining_identities=tuple(draining),
        )

    # ---- the flight recorder and the links

    def _record(
        self,
        kind: str,
        operation_id,
        round_index: int,
        route: message.SyndromePacketRoute,
        *,
        patch_id=None,
        tick: Optional[int] = None,
    ) -> None:
        if tick is None:
            tick = self.engine.now
        event = SyndromeRoundEvent(
            kind, tick, operation_id, round_index, patch_id, route.kind.name
        )
        self.round_events.append(event)

    def _send(
        self,
        path: message.LinkPath,
        *,
        payload_bits,
        attribution: message.TransferAttribution,
        on_delivered: Callable[[], None],
    ) -> None:
        """Send one packet on a path; on_delivered runs at its delivery."""

        def delivered(_transfer) -> None:
            on_delivered()

        self.links.send(
            path, payload_bits, self.engine.now, attribution, delivered
        )


def _packet_attribution(
    packet: message.SyndromeRoundPacket,
) -> message.TransferAttribution:
    patch_ids = tuple(fragment.patch_id for fragment in packet.fragments)
    return _round_attribution(
        packet.operation_id, patch_ids, packet.round_index
    )


def _round_attribution(
    operation_id, patch_ids: tuple, round_index: int
) -> message.TransferAttribution:
    ordered_patch_ids = tuple(
        sorted(patch_ids, key=message.stable_identity_order_key)
    )
    return message.TransferAttribution(
        operation_id=operation_id,
        patch_ids=ordered_patch_ids,
        window_id=None,
        first_round=round_index,
        last_round=round_index,
    )


def _fragment_bits(fragments) -> Optional[int]:
    """The fragments' wire size, None when any fragment has no known size."""
    fragment_sizes = [fragment.size_bits for fragment in fragments]
    if None in fragment_sizes:
        return None
    return sum(fragment_sizes)


def _fragment_index(fragment: message.RetainedSyndromeFragment) -> int:
    return fragment.fragment_index


def _merge_fragments_by_patch(fragments) -> tuple:
    """Order fragments by index, merging parts from the same patch.

    SyndromeRoundPacket requires distinct patch identities, so parts of
    one patch concatenate bits and sizes in fragment-index order. Distinct
    patches keep their own immutable fragments untouched.
    """
    merged: list = []
    for fragment in sorted(fragments, key=_fragment_index):
        prior_index = _index_of_patch(merged, fragment.patch_id)
        if prior_index is None:
            merged.append(fragment)
            continue
        prior = merged[prior_index]
        merged[prior_index] = _concatenated(prior, fragment)
    return tuple(merged)


def _index_of_patch(fragments: list, patch_id) -> Optional[int]:
    for index, fragment in enumerate(fragments):
        if message.same_stable_identity(fragment.patch_id, patch_id):
            return index
    return None


def _concatenated(
    prior: message.RetainedSyndromeFragment,
    fragment: message.RetainedSyndromeFragment,
) -> message.RetainedSyndromeFragment:
    """The two parts of one patch as one fragment; unknown sizes stay so."""
    bits = None
    if prior.bits is not None and fragment.bits is not None:
        bits = prior.bits + fragment.bits
    size_bits = None
    if prior.size_bits is not None and fragment.size_bits is not None:
        size_bits = prior.size_bits + fragment.size_bits
    return dataclasses.replace(prior, bits=bits, size_bits=size_bits)

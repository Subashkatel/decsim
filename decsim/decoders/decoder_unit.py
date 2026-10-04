"""One decoder unit's occupancy: slots, memory, compute claim.

The unit is Smith's decoupled access-execute machine with two input slots
(Smith 1982; TI EDMA ping-pong, SPRAAN4A Example D; gem5-Aladdin's
full/empty bits, tracked per cache line or per half array for double
buffering, Shao et al. MICRO 2016 Sec. IV-B-2):
the next window's transfer lands in the second slot while the current
decode computes. Compute is claimed apart from the slots (Tomasulo's
rule: an instruction whose operands are not ready waits in its
reservation station, never on the functional unit; gem5 O3 issues only
ready work, inst_queue.hh scheduleReadyInsts). The compute takes one
decode at a time, as Helios's controller refuses input while it decodes:
input_ready is high only while it is idle or preparing the next
measurement (Helios_scalable_QEC control_node_single_FPGA.v lines
234-243).
The unit records who holds what, gem5's functional unit
(fu_pool.hh:64-75); the service starts and ends the decodes.

A finished result nobody has asked for yet waits in the unit's output
slot, not in the manager: AFS keeps the finished error log in the unit
(2001.06598 lines 833-840), ChipCheck's output sits in programmable
registers until it is read (2309.05558 lines 484-485), and gem5's sender
keeps the packet and sends it again when the far side accepts it
(port.hh:244-255). Holding it costs nothing and blocks nothing: the
compute and the input slots are freed at the decode's end as before.
"""

import dataclasses
import math
from collections.abc import Callable
from typing import Optional

import decsim.decoders.decoder_memory as decoder_memory_module
import decsim.records.decoding as decoding_records

# the decode computing and the next window's input landing beside it
INPUT_SLOT_COUNT = 2


@dataclasses.dataclass
class ComputeClaim:
    """Who holds or reserves the unit's compute, and when it is expected free.

    The holder runs from assignment through decode end; None means the
    compute is free or back in the pool. The expected free tick comes
    from the running decode's declared occupancy; a decoder measured on
    the host clock declares none, and infinity keeps the unit last in
    the pool's least-work-left order.
    """

    holder: Optional[decoding_records.DecodeJob] = None
    expected_free_ticks: float = math.inf


@dataclasses.dataclass
class UnitSlots:
    """What the unit's slots hold, in and out.

    residents are the jobs whose input occupies or reserves an input
    slot (in transfer or landed), in dispatch order; at most the
    resident capacity. finished are the results this unit has produced
    that nobody has taken yet, by destination window.
    """

    residents: list = dataclasses.field(default_factory=list)
    finished: dict = dataclasses.field(default_factory=dict)


class DecoderUnit:
    """One numbered engine of a pool with its own input memory."""

    def __init__(
        self,
        pool: str,
        index: int,
        memory: decoder_memory_module.DecoderMemory,
    ) -> None:
        self.pool = pool
        self.index = index
        self.memory = memory
        self.slots = UnitSlots()
        self.compute = ComputeClaim()

    @property
    def residents(self) -> list:
        """The jobs holding this unit's input slots, in dispatch order."""
        return self.slots.residents

    @property
    def name(self) -> str:
        """The unit as the trace names it: pool#index."""
        return f"{self.pool}#{self.index}"

    @property
    def holder(self) -> Optional[decoding_records.DecodeJob]:
        """The job holding or reserving the compute; None when free."""
        return self.compute.holder

    # ------------------------------------------------------- the slots

    def has_room(
        self,
        job: decoding_records.DecodeJob,
        resident_capacity: int,
        memory_demand_of: Callable[[decoding_records.DecodeJob], Optional[int]],
    ) -> bool:
        """Whether a slot is free and the memory holds the job beside the rest.

        A second resident joins only if the unit's memory holds both
        inputs; a unit sized for one window keeps serial residency (the
        doubled-SRAM price of overlap is paid explicitly, never assumed).
        A first resident is always admitted, so a genuinely oversized
        window still stops loudly at its deposit. A job whose rounds
        this unit already holds needs no memory of its own: it is one
        more reader of the copy that is here, and the same holds for
        rounds still on their way to a resident that reads them, which
        land as one copy. A resident whose input has landed carries its
        rounds in the memory rather than in its payloads, so the
        memory's own occupancy answers for it.
        """
        if len(self.residents) >= resident_capacity:
            return False
        if self.memory.holds(job):
            return True
        live = self.live_residents()
        if not live:
            return True
        capacity = self.memory.capacity_bits
        if capacity is None:
            return True
        readers = live + [job]
        arriving_bits = self._arriving_bits(readers, memory_demand_of)
        demand = self.memory.occupied_bits + arriving_bits
        return demand <= capacity

    def live_residents(self) -> list:
        """The residents that are neither cancelled nor completed."""
        live = []
        for resident in self.residents:
            if resident.cancelled or resident.completed:
                continue
            live.append(resident)
        return live

    def admit(self, job: decoding_records.DecodeJob) -> None:
        """The job takes a slot; its input moves into this unit's memory."""
        self.slots.residents.append(job)
        job.unit = self

    def evict(self, job: decoding_records.DecodeJob) -> None:
        """The job leaves its slot."""
        if job in self.slots.residents:
            self.slots.residents.remove(job)
        job.unit = None

    def oldest_landed_resident_ready_to_start(
        self,
    ) -> Optional[decoding_records.DecodeJob]:
        """The first resident whose input landed and that may compute now."""
        for resident in self.residents:
            if is_past_start(resident):
                continue
            if not resident.input_landed:
                continue
            if resident.is_parked:
                continue
            return resident
        return None

    def oldest_landing_resident_that_may_start(
        self,
    ) -> Optional[decoding_records.DecodeJob]:
        """The first resident in flight whose window owes no boundary."""
        for resident in self.residents:
            if is_past_start(resident):
                continue
            if resident.input_landed:
                continue
            if not is_startable(resident):
                continue
            return resident
        return None

    def residents_awaiting_compute_count(self) -> int:
        """How many residents still need this unit's compute.

        A resident that has not started and is neither cancelled nor
        completed is work waiting at this unit: it holds an input slot
        and takes the compute as soon as it may. A decode in flight or
        finished waits for nothing.
        """
        awaiting_count = 0
        for resident in self.residents:
            if is_past_start(resident):
                continue
            awaiting_count += 1
        return awaiting_count

    def work_left_ticks(
        self, now: int, occupancy_ticks_of: Callable[..., float]
    ) -> float:
        """Ticks of compute this unit may still owe the jobs it holds.

        The time to the tick the holder is expected to free the
        compute, then the declared cost of every other resident that
        has not started. Nobody knows when a parked resident is
        released, so its decode is counted whole: that is the most a
        newcomer can wait behind it, and the least such total over the
        units bounds the newcomer's worst start. A holder that outlived
        its prediction, its result not yet read, is held for a time
        nobody declared, so the work is unbounded.
        """
        work_left = 0.0
        holder = self.holder
        if holder is not None:
            work_left = self.compute.expected_free_ticks - now
        if work_left < 0:
            return math.inf
        for resident in self.residents:
            if resident is holder or is_past_start(resident):
                continue
            work_left += occupancy_ticks_of(resident)
        return work_left

    def parked_residents(self) -> list:
        """The residents landed with a boundary still owed."""
        parked = []
        for resident in self.residents:
            if resident.is_parked:
                parked.append(resident)
        return parked

    def describe_residents(self) -> str:
        """One compact line of the residents and their phase."""
        if not self.residents:
            return "empty"
        parts = []
        for resident in self.residents:
            phase = self._resident_phase(resident)
            parts.append(
                f"{resident.label} {phase}, {resident.round_count} rounds"
            )
        return "; ".join(parts)

    # -------------------------------------------------- the output slot

    def hold_output(self, window_key: tuple, completion) -> None:
        """Keep one finished result until its destination asks for it.

        A destination window has at most one unconsumed strong result,
        so a second one for the same window is a defect, not a queue.
        """
        self.slots.finished[window_key] = completion

    def take_output(self, window_key: tuple):
        """Take the result waiting for that destination, or None."""
        return self.slots.finished.pop(window_key, None)

    def output_windows(self) -> list:
        """The destinations whose results are still waiting here."""
        return list(self.slots.finished)

    # ----------------------------------------------------- the compute

    def claim_compute(self, job: decoding_records.DecodeJob) -> None:
        """The job holds the compute from now until its decode ends."""
        self.compute.holder = job

    def release_compute(self) -> None:
        """The holder gives the compute back to the pool or a resident."""
        self.compute.holder = None
        self.compute.expected_free_ticks = math.inf

    def expect_compute_free(self, ticks: Optional[float]) -> None:
        """Record when the running decode is expected to free the compute."""
        if ticks is None:
            self.compute.expected_free_ticks = math.inf
            return
        self.compute.expected_free_ticks = ticks

    def _arriving_bits(
        self,
        readers: list,
        memory_demand_of: Callable[[decoding_records.DecodeJob], Optional[int]],
    ) -> int:
        """The bits not yet in the memory, each input counted once."""
        bits_by_input = {}
        for reader in readers:
            if self.memory.holds(reader):
                continue
            input_identity = self.memory.landing_key(reader)
            bits = memory_demand_of(reader)
            self.memory.check_input_size(reader, bits)
            bits_by_input[input_identity] = bits
        arriving = bits_by_input.values()
        return sum(arriving)

    def _resident_phase(self, resident: decoding_records.DecodeJob) -> str:
        if self.compute.holder is resident and resident.service_started:
            return "computing"
        if resident.is_parked:
            return "parked"
        if resident.input_landed:
            return "ready"
        return "capturing"


def is_startable(job: decoding_records.DecodeJob) -> bool:
    """A job whose window owes no boundary may hold compute.

    Anything windowless (external, strong context, merged batch) always
    may.
    """
    window = job.window
    if window is None:
        return True
    return window.deps_remaining <= 0


def is_past_start(job: decoding_records.DecodeJob) -> bool:
    """A job that never starts again; an in-flight decode stays resident."""
    if job.cancelled:
        return True
    if job.completed:
        return True
    return job.service_started

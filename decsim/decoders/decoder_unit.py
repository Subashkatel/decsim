"""One decoder unit's occupancy: slots, memory, compute claim, flights.

The unit is Smith's decoupled access-execute machine with two input slots
(Smith 1982; TI EDMA ping-pong, SPRAAN4A Example D; gem5-Aladdin ready
bits at whole-buffer granularity, Shao et al. MICRO 2016 Sec. IV-B-2):
the next window's transfer lands in the second slot while the current
decode computes. Compute is claimed apart from the slots (Tomasulo's
rule: an instruction whose operands are not ready waits in its
reservation station, never on the functional unit; gem5 O3 issues only
ready work, inst_queue.hh scheduleReadyInsts). A pipelined unit keeps
every in-flight decode resident and retires them in issue order
(Hennessy and Patterson, Computer Architecture, App. C; Helios
2301.08419). The unit records who holds what, gem5's functional unit
(fu_pool.hh:64-75); the service starts and ends the decodes.
"""

import dataclasses
import math
from typing import Callable, Optional

import decsim.decoders.decoder_memory as decoder_memory_module
import decsim.message as message


@dataclasses.dataclass
class ComputeClaim:
    """Who holds or reserves the unit's compute, and when it is expected free.

    The holder runs from assignment through decode end; None means the
    compute is free or back in the pool. The expected free tick comes
    from the running decode's declared occupancy; a decoder measured on
    the host clock declares none, and infinity keeps the unit last in
    the pool's least-work-left order.
    """

    holder: Optional[message.DecodeJob] = None
    expected_free_ticks: float = math.inf


@dataclasses.dataclass
class PipelineFlights:
    """A pipelined unit's decodes in flight and its intake.

    flights holds (job, latency ticks) of every decode started and not
    yet finished. The intake job holds the unit's compute until its
    initiation interval ends; a full pipeline keeps that claim on the
    stalled owner until the next completion. All three stay empty on a
    unit whose occupancy is its latency.
    """

    flights: list = dataclasses.field(default_factory=list)
    intake_job: Optional[message.DecodeJob] = None
    stalled_owner: Optional[message.DecodeJob] = None
    stalled_depth: int = 0


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
        # jobs whose input occupies or reserves a slot (in transfer or
        # landed), in dispatch order; at most the resident capacity
        self.residents: list[message.DecodeJob] = []
        self.compute = ComputeClaim()
        self.pipeline = PipelineFlights()

    @property
    def name(self) -> str:
        """The unit as the trace names it: pool#index."""
        return f"{self.pool}#{self.index}"

    @property
    def holder(self) -> Optional[message.DecodeJob]:
        """The job holding or reserving the compute; None when free."""
        return self.compute.holder

    # ------------------------------------------------------- the slots

    def has_room(
        self,
        job: message.DecodeJob,
        resident_capacity: int,
        memory_demand_of: Callable[[message.DecodeJob], int],
    ) -> bool:
        """Whether a slot is free and the memory holds the job beside the rest.

        A second resident joins only if the unit's memory holds both
        inputs; a unit sized for one window keeps serial residency (the
        doubled-SRAM price of overlap is paid explicitly, never assumed).
        A first resident is always admitted, so a genuinely oversized
        window still stops loudly at its deposit.
        """
        if len(self.residents) >= resident_capacity:
            return False
        live = self.live_residents()
        if not live:
            return True
        capacity = self.memory.capacity_rounds
        if capacity is None:
            return True
        demand = 0
        for resident in live:
            demand += memory_demand_of(resident)
        demand += memory_demand_of(job)
        return demand <= capacity

    def live_residents(self) -> list:
        """The residents that are neither cancelled nor completed."""
        live = []
        for resident in self.residents:
            if resident.cancelled or resident.completed:
                continue
            live.append(resident)
        return live

    def admit(self, job: message.DecodeJob) -> None:
        """The job takes a slot; its input moves into this unit's memory."""
        self.residents.append(job)
        job.unit = self

    def evict(self, job: message.DecodeJob) -> None:
        """The job leaves its slot."""
        if job in self.residents:
            self.residents.remove(job)
        job.unit = None

    def oldest_landed_resident_ready_to_start(
        self,
    ) -> Optional[message.DecodeJob]:
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
    ) -> Optional[message.DecodeJob]:
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
                f"{resident.label} {phase}, {resident.n_rounds} rounds"
            )
        return "; ".join(parts)

    # ----------------------------------------------------- the compute

    def claim_compute(self, job: message.DecodeJob) -> None:
        """The job holds the compute from now until its decode ends."""
        self.compute.holder = job

    def release_compute(self) -> None:
        """The holder gives the compute back to the pool or a resident."""
        self.compute.holder = None

    def expect_compute_free(self, ticks: Optional[float]) -> None:
        """Record when the running decode is expected to free the compute."""
        if ticks is None:
            self.compute.expected_free_ticks = math.inf
            return
        self.compute.expected_free_ticks = ticks

    # ---------------------------------------------------- the pipeline

    def add_flight(self, job: message.DecodeJob, latency_ticks: int) -> None:
        """A pipelined decode started; it retires in issue order.

        A hardware pipeline retires in issue order, so every in-flight
        decode on one unit must declare the same latency; mixed
        latencies refuse loudly.
        """
        flights = self.pipeline.flights
        for flight_job, flight_latency in flights:
            if flight_latency != latency_ticks:
                raise RuntimeError(
                    f"decode job {job.label!r} declares latency "
                    f"{latency_ticks} ticks while {flight_job.label!r} is "
                    "in flight with a different latency: a pipelined unit "
                    "completes in order and takes one latency per unit"
                )
        flights.append((job, latency_ticks))
        if self.pipeline.intake_job is not None:
            raise RuntimeError(
                f"pipelined unit {self.name!r} started {job.label!r} before "
                "its prior initiation interval completed"
            )
        self.pipeline.intake_job = job

    def take_flight(self, job: message.DecodeJob) -> bool:
        """Retire the job's flight; whether it was in flight here."""
        flights = self.pipeline.flights
        for flight in flights:
            if flight[0] is job:
                flights.remove(flight)
                return True
        return False

    def flight_count(self) -> int:
        """Decodes started on this unit and not yet finished."""
        return len(self.pipeline.flights)

    def flight_labels(self) -> list:
        """The labels of the decodes in flight."""
        labels = []
        for flight_job, _latency_ticks in self.pipeline.flights:
            labels.append(flight_job.label)
        return labels

    def stall(self, owner: message.DecodeJob, depth: int) -> None:
        """The pipeline is full: the owner keeps the compute claim."""
        self.pipeline.stalled_owner = owner
        self.pipeline.stalled_depth = depth

    def lift_stall(self) -> Optional[message.DecodeJob]:
        """The stalled owner whose claim may go, once a flight retired.

        None when nothing was stalled, the pipeline is still full, or
        the owner no longer holds the compute or is past its end.
        """
        owner = self.pipeline.stalled_owner
        if owner is None:
            return None
        if self.flight_count() >= self.pipeline.stalled_depth:
            return None
        self.pipeline.stalled_owner = None
        self.pipeline.stalled_depth = 0
        if self.compute.holder is not owner:
            return None
        if owner.cancelled or owner.completed:
            return None
        return owner

    def _resident_phase(self, resident: message.DecodeJob) -> str:
        if self.compute.holder is resident and resident.service_started:
            return "computing"
        if resident.is_parked:
            return "parked"
        if resident.input_landed:
            return "ready"
        return "capturing"


def is_startable(job: message.DecodeJob) -> bool:
    """A job whose window owes no boundary may hold compute.

    Anything windowless (external, strong context, merged batch) always
    may.
    """
    window = job.window
    if window is None:
        return True
    return window.deps_remaining <= 0


def is_past_start(job: message.DecodeJob) -> bool:
    """A job that never starts again; an in-flight decode stays resident."""
    if job.cancelled:
        return True
    if job.completed:
        return True
    return job.service_started

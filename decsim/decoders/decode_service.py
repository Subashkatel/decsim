"""One decode on one unit: staged, started once landed and allowed, freed.

gem5's IEW stage (src/cpu/o3/iew.hh:70-87) executes what the queue
issued and writes the result back; here the service takes the unit the
dispatcher chose, moves the job's rounds into that unit's memory (the
accelerator pattern: invoke the unit, then DMA its input into the
unit's memory, then compute; gem5-Aladdin aladdin_sys_connection.h and
dma_interface.h), starts the routed decoder when every transfer landed
and the window owes no boundary, and frees the unit at the decode's end.
A landed job whose window still owes a boundary parks in its slot and
releases its compute claim (Tomasulo's rule at the boundary hazard), so
a dependent that fills early never deadlocks the unit against its own
predecessor. A pipelined unit issues one decode per initiation interval
and keeps at most its depth in flight (Hennessy and Patterson App. C;
rowD4).
"""

import dataclasses
import functools
from typing import Callable

import numpy

import decsim.config as config
import decsim.decoders.decode_queue as decode_queue
import decsim.decoders.decoder_memory_transfer as staging_module
import decsim.decoders.decoder_pool as decoder_pool_module
import decsim.decoders.decoder_unit as decoder_unit_module
import decsim.decoders.strong_requests as strong_requests_module
import decsim.records.decoding as decoding_records
import decsim.records.log_sources as log_sources
import decsim.trace_source as trace_source

# the job kinds a pipelined unit serves; every other kind holds its unit
# for the whole decode until it gets its own design pass
PIPELINED_JOB_KINDS = frozenset(
    {
        decoding_records.DecodeJobKind.WINDOW,
        decoding_records.DecodeJobKind.SELF_CONTAINED,
    }
)


class DecodeService:
    """Stages, starts, prices and frees every decode on its unit.

    Six attributes: the engine, the pool that routes and holds the
    units, the staging, the strong requests (a merged batch's members
    are the ledger's knowledge), and the two calls back to the manager:
    on_completed(job, result) at every decode's end, and dispatch()
    wherever compute frees inside an engine event, so the non-reentrant
    dispatch loop runs from the same points it always did. Four trace
    sources, each carrying (job, unit):
    job_dispatched when the job takes its slot, input_landed when every
    transfer of its input has landed, job_started when its decode
    begins, job_finished when its compute ends.
    """

    def __init__(
        self,
        engine,
        pool: decoder_pool_module.DecoderPool,
        staging: staging_module.DecoderInputStaging,
        strong_requests: strong_requests_module.StrongRequests,
        on_completed: Callable[[decoding_records.DecodeJob, object], None],
        dispatch: Callable[[], None],
        dispatch_ticks: int = 0,
    ) -> None:
        self.engine = engine
        self.pool = pool
        self.staging = staging
        self.strong_requests = strong_requests
        self.manager = _ManagerSide(on_completed, dispatch, dispatch_ticks)
        self.trace = _TraceSources()

    # ------------------------------------------ what the dispatcher asks

    def resident_capacity(self, job: decoding_records.DecodeJob) -> int:
        """Residents a unit holds for this job's route.

        Two (the depth-1 access-execute machine) unless the routed decoder
        pipelines: then every in-flight decode stays resident (its input
        lives in the unit's memory until its result emerges) plus one
        landing next. Memory admission still gates every resident, so a
        deep pipeline pays its SRAM price visibly or refuses loudly.
        """
        decoder = self.pool.decoder_for(job)
        depth = decoder.pipeline_depth(job)
        return depth + 1

    def carries_input(self, job: decoding_records.DecodeJob) -> bool:
        """The job moves syndrome data into a unit's memory."""
        if job.send_input is not None:
            return True
        demand = self.memory_demand(job)
        return demand > 0

    def memory_demand(self, job: decoding_records.DecodeJob) -> int:
        """The rounds a job's input occupies in unit memory.

        The distinct rounds of its payloads, the same count the deposit
        charges. An external job carries no syndrome data and stores
        nothing; a merged batch stores every member's input.
        """
        if job.on_done is not None:
            return 0
        if strong_requests_module.is_merged_batch(job):
            demand = 0
            for member in self.strong_requests.members_of(job):
                demand += decoding_records.distinct_round_count(member.payloads)
            return demand
        return decoding_records.distinct_round_count(job.payloads)

    # -------------------------------------------------- dispatch and start

    def dispatch_to(
        self,
        pool: str,
        job: decoding_records.DecodeJob,
        unit: decoder_unit_module.DecoderUnit,
        claim_compute: bool,
    ) -> None:
        """The job takes a slot of the unit; its input starts moving.

        The unit is assigned at DMA start (gem5-Aladdin's invocation
        model: invoke the unit, then DMA its input); compute is claimed
        only when this unit's compute is actually free.
        """
        job.pool = pool
        self._assign_service_key(job)
        unit.admit(job)
        # kept past the job's eviction: the confidence its evidence
        # feeds is charged on this unit and names it (decision D8)
        job.decoding_unit_name = unit.name
        if claim_compute:
            self.pool.claim(unit, job)
        if job.window is not None:
            job.window.t_dispatch = self.engine.now
        self._log_assignment(pool, job, claim_compute)
        self.trace.job_dispatched.fire(job, unit)
        self._charge_dispatch(job, unit, claim_compute)

    def _charge_dispatch(
        self,
        job: decoding_records.DecodeJob,
        unit: decoder_unit_module.DecoderUnit,
        claim_compute: bool,
    ) -> None:
        """The manager's own work, before this job's input is asked for.

        Caune et al. 2410.05202 (lines 519-526 and 636-641) measure 250
        to 370 cycles of the control system's own clock between a
        decode's arrival and its dispatch;
        decoder_manager.dispatch_cycles prices that, zero by default.
        """
        if self.manager.dispatch_ticks == 0:
            self._ask_for_input(job, unit, claim_compute)
            return
        ask = functools.partial(self._ask_for_input, job, unit, claim_compute)
        label = f"dispatch_cost({job.label})"
        self.engine.schedule(self.manager.dispatch_ticks, ask, label=label)

    def _ask_for_input(
        self,
        job: decoding_records.DecodeJob,
        unit: decoder_unit_module.DecoderUnit,
        claim_compute: bool,
    ) -> None:
        """Say the job's input may move; a job cancelled meanwhile does not."""
        if job.cancelled:
            return
        self._stage_input(job, unit)
        if claim_compute:
            self._predict_compute_free(job)

    def begin(
        self, job: decoding_records.DecodeJob, gated: bool = True
    ) -> None:
        """The unit's memory holds the input: start the decode."""
        if job.cancelled:  # cancelled while its input was in flight
            return
        if gated and self._is_boundary_owed(job):
            self._park(job)
            return
        if job.gate is not None:
            # the seam mask is XORed into the landed input exactly once,
            # at the moment the decode actually starts
            job.gate.mask_input(job)
        job.service_started = True
        if job.window is not None:
            job.window.service_began = True
            job.window.t_compute_start = self.engine.now
        decoder = self.pool.decoder_for(job)
        self.engine.log(
            log_sources.DECODER_MANAGER, f"START DECODE {job.label}"
        )
        self.trace.job_started.fire(job, job.unit)
        self._predict_compute_free(job)
        pipeline = self._pipeline_of(decoder, job)
        if job.decoder_input is not None:
            # the decoder reads this unit's memory now
            job.payloads = job.decoder_input.fragments()
        decoder.start(
            job,
            self.engine,
            lambda result: self.manager.on_completed(job, result),
        )
        if pipeline is None:
            return
        interval_ticks, latency_ticks, depth = pipeline
        self._track_pipelined_start(job, latency_ticks, interval_ticks, depth)

    def restart_parked(self, job: decoding_records.DecodeJob) -> None:
        """A released job takes the unit's compute if it can have it now."""
        unit = job.unit
        holder = unit.holder
        if holder is None and self.pool.is_free(unit):
            self.pool.claim(unit, job)
            self.begin(job, gated=False)
            return
        if holder is None:
            return
        if holder.service_started:
            # the unit is decoding: no longer parked, so the compute
            # offer starts this job at the next compute end
            return
        if holder.input_landed:
            return
        # steal a reservation held for an input still in flight:
        # ready work issues first; the in-flight job re-competes
        # at its own landing
        unit.claim_compute(job)
        self.begin(job, gated=False)

    # ------------------------------------------------------- the unit's end

    def free(self, job: decoding_records.DecodeJob) -> None:
        """Compute finished: drop the job from its slot and offer the compute.

        The ping-pong swap at compute end.
        """
        self._end_flight(job, offer_now=False)
        unit = job.unit
        unit.evict(job)
        intake_owner = unit.pipeline.intake_job
        if unit.holder is job and intake_owner is not job:
            unit.release_compute()
        self.engine.log_io(
            f"unit {unit.name} SRAM",
            lambda: _emitted_description(job, unit),
        )
        self.trace.job_finished.fire(job, unit)
        self._offer_compute(unit)

    def evict(self, job: decoding_records.DecodeJob) -> None:
        """Drop a job from its slot before its decode started.

        Compute it held or reserved passes onward.
        """
        unit = job.unit
        unit.evict(job)
        job.is_parked = False
        self.staging.cancel(job)
        if unit.holder is job:
            unit.release_compute()
            self._offer_compute(unit)

    def abort(self, job: decoding_records.DecodeJob) -> None:
        """Stop a running decode: the decoder, its input, its slot."""
        decoder = self.pool.decoder_for(job)
        decoder.cancel(job)
        self.staging.cancel(job)
        self.free(job)

    def discard_cancelled(self, job: decoding_records.DecodeJob) -> None:
        """A cancelled decode reported its end: retire its flight and input."""
        self._end_flight(job, offer_now=True)
        self.staging.release(job)

    def release_input(self, job: decoding_records.DecodeJob) -> None:
        """Free the job's rounds from its unit's memory; none held is fine."""
        self.staging.release(job)

    def cancel_input(self, job: decoding_records.DecodeJob) -> None:
        """Drop the job from transport and storage, then free its hold."""
        self.staging.cancel(job)

    def release_inputs(self, members) -> None:
        """Return the credits of every request one decode still serves."""
        self.staging.release_service_members(members)

    def free_unit_count(self, pool: str) -> int:
        """Units of the pool with free compute now."""
        return self.pool.free_count(pool)

    # ------------------------------------------------- the units' state

    def parked_jobs(self) -> list:
        """Every resident landed with a boundary still owed, unit by unit."""
        parked = []
        for unit in self.pool.units():
            unit_parked = unit.parked_residents()
            parked.extend(unit_parked)
        return parked

    def resident_jobs(self) -> list:
        """Every job holding a slot of any unit, unit by unit."""
        residents = []
        for unit in self.pool.units():
            residents.extend(unit.residents)
        return residents

    def take_strong_output(self, window_key: tuple):
        """Take the finished result waiting for that destination, if any.

        The result waits in the output slot of the unit that produced
        it, and a destination has at most one, so the first unit holding
        one for this window is the one.
        """
        for unit in self.pool.units():
            completion = unit.take_output(window_key)
            if completion is not None:
                return completion
        return None

    def windows_holding_output(self) -> list:
        """The destinations whose results are still waiting in a unit."""
        waiting = []
        for unit in self.pool.units():
            windows = unit.output_windows()
            waiting.extend(windows)
        return sorted(waiting)

    def units_holding_rounds(self) -> list:
        """The names of the units whose memory still holds rounds."""
        held = []
        for unit in self.pool.units():
            if unit.memory.occupied_rounds:
                held.append(unit.name)
        return held

    def check_settled(self) -> None:
        """Refuse a quiescent run with a decode parked, in flight or intaking.

        A parked decode, a flight, or an intake interval still active
        means an event never came.
        """
        parked = []
        for job in self.parked_jobs():
            parked.append(job.label)
        if parked:
            parked = sorted(parked)
            raise RuntimeError(
                f"run ended with parked decodes never released: {parked}"
            )
        in_flight = []
        busy = []
        for unit in self.pool.units():
            flight_labels = unit.flight_labels()
            in_flight.extend(flight_labels)
            if unit.pipeline.intake_job is not None:
                busy.append(unit.name)
        if in_flight:
            raise RuntimeError(
                "run ended with pipelined decodes still in flight: "
                f"{sorted(in_flight)}"
            )
        if busy:
            raise RuntimeError(
                "run ended with pipelined decoder intake intervals still "
                f"active: {sorted(busy)}"
            )

    # ------------------------------------------------- dispatch, private

    def _assign_service_key(self, job: decoding_records.DecodeJob) -> None:
        """One service key per decode, shared by every request it serves."""
        members = job.service_original_request_keys
        if not members and job.request_key is not None:
            members = (job.request_key,)
            job.service_original_request_keys = members
        if not members:
            return
        run_sequences = []
        for key in members:
            run_sequences.append(key.run_sequence)
        first_sequence = min(run_sequences)
        job.service_key = decoding_records.DecoderServiceKey(first_sequence)
        job.service_dispatch_ticks = self.engine.now
        for member in self.strong_requests.members_of(job):
            member.service_key = job.service_key
            member.service_dispatch_ticks = self.engine.now

    def _log_assignment(
        self, pool: str, job: decoding_records.DecodeJob, claim_compute: bool
    ) -> None:
        waited_ticks = self.engine.now - job.ready_time
        waited = config.format_ticks(waited_ticks)
        waited = waited.strip()
        slot_note = ""
        if not claim_compute:
            slot_note = "staged, "
        pool_tag = decode_queue.pool_tag_of(pool)
        free_now = self.pool.free_count(pool)
        self.engine.log(
            log_sources.DECODER_MANAGER,
            f"ASSIGN UNIT {job.label} "
            f"({slot_note}waited {waited} in queue, "
            f"{pool_tag}units free now {free_now})",
        )

    def _stage_input(
        self,
        job: decoding_records.DecodeJob,
        unit: decoder_unit_module.DecoderUnit,
    ) -> None:
        """Move every member request's rounds into this unit's memory.

        One transfer per request (a merged strong batch has several); the
        decode starts when all have landed.
        """
        members = self._transfer_members(job)
        memory = unit.memory
        source = f"unit {unit.name} SRAM"
        remaining = len(members)

        def landed(_member: decoding_records.DecodeJob) -> None:
            nonlocal remaining
            remaining -= 1
            if remaining > 0:
                return
            self._input_landed(job, unit)

        for member in members:
            self._log_input_receiving(source, member)
            self.staging.stage(member, memory, landed)

    def _log_input_receiving(
        self, source: str, member: decoding_records.DecodeJob
    ) -> None:
        self.engine.log_io(source, lambda: _receiving_text(member))

    def _transfer_members(self, job: decoding_records.DecodeJob) -> list:
        members = []
        for member in self.strong_requests.members_of(job):
            if member is not job:
                members.append(member)
        if not members:
            members = [job]
        return members

    def _input_landed(
        self,
        job: decoding_records.DecodeJob,
        unit: decoder_unit_module.DecoderUnit,
    ) -> None:
        """Every transfer landed: start now, or wait for the unit's compute."""
        job.input_landed = True
        self.engine.log_io(
            f"unit {unit.name} SRAM",
            lambda: _landed_description(job, unit),
        )
        self.trace.input_landed.fire(job, unit)
        holder = unit.holder
        if holder is job:
            # this job holds or was reserved the unit's compute
            self.begin(job)
            return
        if holder is None and self.pool.is_free(unit):
            # compute went back to the pool (a resident parked and
            # released its claim); take it now
            self.pool.claim(unit, job)
            self.begin(job)
        # otherwise the compute is busy: the compute offer picks this
        # job up at the next compute end

    def _is_boundary_owed(self, job: decoding_records.DecodeJob) -> bool:
        if job.gate is None:
            return False
        return not job.gate.may_start(job)

    def _park(self, job: decoding_records.DecodeJob) -> None:
        job.is_parked = True
        self.engine.log(
            log_sources.DECODER_MANAGER,
            f"PARK DECODE {job.label} (boundary pending)",
        )
        self._release_compute_claim(job)

    # ------------------------------------------------- the compute, private

    def _release_compute_claim(self, job: decoding_records.DecodeJob) -> None:
        """Tomasulo's rule at the boundary hazard.

        A parked job keeps its input slot but never the unit's compute.
        """
        unit = job.unit
        if unit.holder is job:
            unit.release_compute()
            self._offer_compute(unit)
            self.manager.dispatch()

    def _offer_compute(self, unit: decoder_unit_module.DecoderUnit) -> None:
        """Free compute goes to the oldest startable resident.

        Or it stays reserved for the oldest one still in flight, or it
        returns to the pool. gem5 O3's scheduleReadyInsts is the reference
        rule: only a ready instruction acquires a functional unit
        (fu_pool->getUnit at issue), and blocked work waits in the queue,
        never on the unit.
        """
        if unit.holder is not None:
            return
        ready = unit.oldest_landed_resident_ready_to_start()
        if ready is not None:
            unit.claim_compute(ready)
            self.begin(ready)
            return
        landing = unit.oldest_landing_resident_that_may_start()
        if landing is not None:
            unit.claim_compute(landing)  # starts at its landing
            self._predict_compute_free(landing)
            return
        if self.pool.is_free(unit):
            # a pipelined unit's intake went back to the pool at the end of
            # its initiation interval; the decode's completion offers again
            return
        self.pool.release(unit)

    def _predict_compute_free(self, job: decoding_records.DecodeJob) -> None:
        """Record when the compute this job holds frees.

        The decode starts once its input has landed and runs for the
        declared latency (the initiation interval on a pipelined unit). A
        decoder measured on the host clock declares no latency, so its
        unit stays unpredicted.
        """
        unit = job.unit
        decoder = self.pool.decoder_for(job)
        occupancy = decoder.occupancy(job)
        if occupancy is None:
            unit.expect_compute_free(None)
            return
        start = self.engine.now
        landing = job.input_landing_ticks
        if landing is not None:
            start = max(start, landing)
        free_ticks = start + occupancy
        unit.expect_compute_free(free_ticks)

    # ------------------------------------------------- the pipelined unit

    def _pipeline_of(self, decoder, job: decoding_records.DecodeJob):
        """(interval, latency, depth) of a pipelined start, else None.

        The row declares both halves and this asks for both. The Decoder
        port's pipeline_depth is the decodes that may be in flight on
        one unit, and one is no pipeline; a unit also runs the pipelined
        model when it accepts work on its own interval rather than at
        the rate it answers, which is a row whose occupancy differs from
        its latency (gem5's FuncUnit declares an issue latency beside
        its op latency, fu_pool.hh; the intake rate is the initiation
        interval's alone, Hennessy and Patterson App. C). The two agree
        on every row: a depth above one needs an interval shorter than
        the latency, which the assert below says. The pipelined model
        serves the job kinds in PIPELINED_JOB_KINDS; the strong tier and
        merged batches are not pipelined yet, and a pipelined route
        there refuses loudly rather than silently serializing.
        """
        occupancy = decoder.occupancy(job)
        if occupancy is None:
            return None
        latency_ticks = decoder.latency(job)
        depth = decoder.pipeline_depth(job)
        accepts_on_its_own_interval = occupancy != latency_ticks
        if depth == 1 and not accepts_on_its_own_interval:
            return None
        assert accepts_on_its_own_interval, (
            f"decoder for {job.label!r} declares a pipeline depth of "
            f"{depth} and an occupancy equal to its latency; a depth "
            "above one needs an intake interval shorter than the answer"
        )
        if _is_outside_pipelined_model(job):
            raise RuntimeError(
                f"decode job {job.label!r}: a pipelined decoder serves "
                "window and self-contained decodes only; the strong tier "
                "and merged batches are not pipelined yet"
            )
        return occupancy, latency_ticks, depth

    def _track_pipelined_start(
        self,
        job: decoding_records.DecodeJob,
        latency_ticks: int,
        interval_ticks: int,
        depth: int,
    ) -> None:
        unit = job.unit
        unit.add_flight(job, latency_ticks)
        self.engine.schedule(
            interval_ticks,
            lambda: self._initiation_complete(unit, job, depth),
            label=f"initiation_complete({job.label})",
        )

    def _initiation_complete(
        self,
        unit: decoder_unit_module.DecoderUnit,
        job: decoding_records.DecodeJob,
        depth: int,
    ) -> None:
        """The pipelined unit's intake is free again.

        Release the compute claim so the next start may begin, unless the
        pipeline is full; a full pipeline keeps the claim until the next
        completion.
        """
        if unit.pipeline.intake_job is not job:
            return
        unit.pipeline.intake_job = None
        if unit.holder is not job:
            return
        if job.cancelled or job.completed:
            unit.release_compute()
            self._offer_compute(unit)
            self.manager.dispatch()
            return
        if unit.flight_count() >= depth:
            unit.stall(job, depth)
            return
        unit.release_compute()
        self._offer_compute(unit)
        self.manager.dispatch()

    def _end_flight(
        self, job: decoding_records.DecodeJob, offer_now: bool
    ) -> None:
        """A pipelined decode left the unit (done or cancelled).

        Retire its flight and lift a full-pipeline stall. The caller's
        normal flow performs the compute offer unless offer_now says
        otherwise.
        """
        for unit in self.pool.units():
            if not unit.take_flight(job):
                continue
            self._lift_pipeline_stall(unit, offer_now)
            return

    def _lift_pipeline_stall(
        self, unit: decoder_unit_module.DecoderUnit, offer_now: bool
    ) -> None:
        owner = unit.lift_stall()
        if owner is None:
            return
        unit.release_compute()
        if offer_now:
            self._offer_compute(unit)
            self.manager.dispatch()


def _is_outside_pipelined_model(job: decoding_records.DecodeJob) -> bool:
    if job.kind not in PIPELINED_JOB_KINDS:
        return True
    return len(job.service_original_request_keys) > 1


def _emitted_description(
    job: decoding_records.DecodeJob, unit: decoder_unit_module.DecoderUnit
) -> str:
    holds = unit.describe_residents()
    return f"emitted {job.label} result; holds {holds}"


def _landed_description(
    job: decoding_records.DecodeJob, unit: decoder_unit_module.DecoderUnit
) -> str:
    defects = job_defects_text(job)
    holds = unit.describe_residents()
    return f"{job.label} input landed; {defects}; holds {holds}"


def _receiving_text(member: decoding_records.DecodeJob) -> str:
    return (
        f"receiving {member.label} input "
        f"({member.round_count} rounds from the round store)"
    )


def job_defects_text(job: decoding_records.DecodeJob) -> str:
    """The landed window input's cargo, sparse, for the I/O trace.

    The set detection-event indices of the rounds now in this unit's
    memory (the algorithm stage reads the same fragments).
    """
    if job.decoder_input is not None:
        fragments = job.decoder_input.fragments()
    else:
        fragments = job.payloads
        if fragments is None:
            fragments = []
    bit_arrays = []
    for fragment in fragments:
        if fragment.bits is None:
            continue
        bits = numpy.asarray(fragment.bits, dtype=numpy.uint8)
        bit_arrays.append(bits)
    if not bit_arrays:
        return "no payload bits"
    all_bits = numpy.concatenate(bit_arrays)
    defects = numpy.flatnonzero(all_bits)
    if defects.size == 0:
        return "no defects"
    defect_texts = []
    for defect in defects.tolist():
        defect_texts.append(str(defect))
    listed = ", ".join(defect_texts)
    return f"defects {{{listed}}}"


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the decode service reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (tmp/resources/gem5/src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    job_dispatched: trace_source.TraceSource = trace_source.new_source()
    input_landed: trace_source.TraceSource = trace_source.new_source()
    job_started: trace_source.TraceSource = trace_source.new_source()
    job_finished: trace_source.TraceSource = trace_source.new_source()


@dataclasses.dataclass(frozen=True)
class _ManagerSide:
    """The service's two calls back to the manager, and the manager's cost.

    on_completed(job, result) at every decode's end; dispatch() wherever
    compute frees inside an engine event, so the non-reentrant dispatch
    loop runs from the same points it always did; dispatch_ticks is the
    manager's own work per dispatch, charged before the input is asked
    for (decoder_manager.dispatch_cycles).
    """

    on_completed: Callable
    dispatch: Callable
    dispatch_ticks: int

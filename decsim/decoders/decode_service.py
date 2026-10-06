"""One decode on one unit: staged, started once landed, freed.

As gem5's IEW stage executes what the queue issued
(src/cpu/o3/iew.hh:70-87), the service takes the unit the dispatcher
chose, moves the job's rounds into that unit's memory (invoke the unit,
DMA its input, then compute; gem5-Aladdin aladdin_sys_connection.h and
dma_interface.h), starts the decoder when every transfer landed and the
unit's compute is free, and frees the unit when the decode, and any
confidence walk charged on it, has ended. The dispatcher hands over a
job only once its window owes no boundary, so a landed input never
waits for one.
"""

import dataclasses
import functools
from typing import TYPE_CHECKING, Optional

import numpy

import decsim.config as config
import decsim.decoders.decode_queue as decode_queue
import decsim.decoders.decoder_memory_transfer as staging_module
import decsim.decoders.decoder_pool as decoder_pool_module
import decsim.decoders.decoder_unit as decoder_unit_module
import decsim.decoders.strong_requests as strong_requests_module
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.log_sources as log_sources
import decsim.trace_source as trace_source

if TYPE_CHECKING:
    import decsim.decoders.decoder_manager as decoder_manager_module


class DecodeService:
    """Serves every decode on its unit, from staging to release.

    Every decode's end goes to the manager's decode_completed, and
    dispatch runs wherever compute frees inside an engine event. Four
    trace sources, each carrying (job, unit): job_dispatched when the
    job takes its slot, input_landed when every transfer of its input
    has landed, job_started when its decode begins, job_finished when
    its compute ends.
    """

    def __init__(
        self,
        engine: engine_module.Engine,
        pool: decoder_pool_module.DecoderPool,
        staging: staging_module.DecoderInputStaging,
        manager: "decoder_manager_module.DecoderManager",
        clock: Optional[config.Clock] = None,
        dispatch_cycles: int = 0,
    ) -> None:
        self.engine = engine
        self.pool = pool
        self.staging = staging
        self.manager = manager
        self.dispatch_cost = _DispatchCost(clock, dispatch_cycles)
        self.trace = _TraceSources()

    # ------------------------------------------ what the dispatcher asks

    def carries_input(self, job: decoding_records.DecodeJob) -> bool:
        """The job moves syndrome data into a unit's memory.

        Its rounds, not their width: a timing-only device's rounds state
        no size and still move.
        """
        if job.send_input is not None:
            return True
        round_count = self._input_round_count(job)
        return round_count > 0

    def input_is_on_the_unit(
        self,
        job: decoding_records.DecodeJob,
        unit: decoder_unit_module.DecoderUnit,
    ) -> bool:
        """Whether this job's rounds are in that unit's memory or on their way.

        Either way the job moves nothing: the staging makes it one more
        reader of the rounds that are here, or one more job joining the
        landing of the transfer that is bringing them.
        """
        memory = unit.memory
        if memory.holds(job):
            return True
        return self.staging.is_landing_into(memory, job)

    def memory_demand(self, job: decoding_records.DecodeJob) -> Optional[int]:
        """The bits a job's input occupies in unit memory.

        The bits of the payloads that land for it, at the width they
        cross the input link; None when a payload states no size. A tier
        that forms its own detection events holds the rounds at this
        same width.
        """
        demand = 0
        for input_job in self._input_jobs(job):
            bits = input_job.payload_bits()
            if bits is None:
                return None
            demand += bits
        return demand

    def _input_jobs(self, job: decoding_records.DecodeJob) -> list:
        """The jobs whose payloads land in the unit's memory for this one.

        An external job carries no syndrome data and stores nothing; a
        merged batch stores every member's input; every other job stores
        its own.
        """
        if job.on_done is not None:
            return []
        if strong_requests_module.is_merged_batch(job):
            return self.manager.strong_requests.members_of(job)
        return [job]

    def _input_round_count(self, job: decoding_records.DecodeJob) -> int:
        """The syndrome rounds a job moves into the unit's memory."""
        round_count = 0
        for input_job in self._input_jobs(job):
            payloads = input_job.payloads
            round_count += decoding_records.distinct_round_count(payloads)
        return round_count

    # -------------------------------------------------- dispatch and start

    def dispatch_to(
        self,
        job: decoding_records.DecodeJob,
        unit: decoder_unit_module.DecoderUnit,
        claim_compute: bool,
    ) -> None:
        """The job takes a slot of the unit; its input starts moving.

        The unit is assigned at DMA start (gem5-Aladdin's invocation
        model); compute is claimed only when this unit's compute is
        free.
        """
        job.pool = self.pool.name
        self._assign_service_key(job)
        unit.admit(job)
        # kept past the job's eviction: the confidence its evidence
        # feeds is charged on this unit and names it
        job.decoding_unit_name = unit.name
        if claim_compute:
            self.pool.claim(unit, job)
        job.dispatch_ticks = self.engine.now
        if job.window is not None:
            job.window.t_dispatch = self.engine.now
        self._log_assignment(job, claim_compute)
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
        to 370 cycles of the control system's clock between a decode's
        arrival and its dispatch; decoder_manager.dispatch_cycles prices
        that, zero by default.
        """
        cycles = self.dispatch_cost.cycles
        if cycles == 0:
            self._ask_for_input(job, unit, claim_compute)
            return
        now = self.engine.now
        edge = self.dispatch_cost.clock.edge(cycles, now)
        delay = edge - now
        ask = functools.partial(self._ask_for_input, job, unit, claim_compute)
        label = f"dispatch_cost({job.label})"
        self.engine.schedule(delay, ask, label=label)

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

    def begin(self, job: decoding_records.DecodeJob) -> None:
        """The unit's memory holds the input: start the decode."""
        if job.cancelled:  # cancelled while its input was in flight
            return
        if job.gate is not None:
            # the seam mask is XORed into the landed input exactly once,
            # at the moment the decode actually starts
            job.gate.mask_input(job)
        job.service_started = True
        if job.window is not None:
            job.window.service_began = True
            job.window.t_compute_start = self.engine.now
        self._start_decoder(job)

    # ------------------------------------------------------- the unit's end

    def free(self, job: decoding_records.DecodeJob) -> None:
        """Compute finished: free the job's slot and offer the compute."""
        unit = job.unit
        unit.evict(job)
        if unit.holder is job:
            unit.release_compute()
        self.engine.log_io(
            f"unit {unit.name} SRAM",
            lambda: _emitted_description(job, unit),
        )
        self.trace.job_finished.fire(job, unit)
        self._offer_compute(unit)

    def evict(self, job: decoding_records.DecodeJob) -> None:
        """Drop a job from its slot before its decode started.

        Compute it held or reserved passes onward. The staging cancels
        first, because it finds the job's landing through its unit.
        """
        unit = job.unit
        self.staging.cancel(job)
        unit.evict(job)
        if unit.holder is job:
            unit.release_compute()
            self._offer_compute(unit)

    def abort(self, job: decoding_records.DecodeJob) -> None:
        """Stop a running decode: the decoder, its input, its slot."""
        decoder = self.manager.decoder
        decoder.cancel(job)
        self.staging.cancel(job)
        self.free(job)

    def release_input(self, job: decoding_records.DecodeJob) -> None:
        """Free the job's rounds from its unit's memory; none held is fine."""
        self.staging.release(job)

    def cancel_input(self, job: decoding_records.DecodeJob) -> None:
        """Drop the job from transport and storage, then free its hold."""
        self.staging.cancel(job)

    def release_inputs(self, members: list) -> None:
        """Return the credits of every request one decode still serves."""
        self.staging.release_service_members(members)

    def free_unit_count(self) -> int:
        """Units of the pool with free compute now."""
        return len(self.pool.free)

    # ------------------------------------------------- the units' state

    def resident_jobs(self) -> list:
        """Every job holding a slot of any unit, unit by unit."""
        residents = []
        for unit in self.pool.units:
            residents.extend(unit.residents)
        return residents

    def take_strong_output(
        self, window_key: tuple
    ) -> Optional[strong_requests_module.StrongCompletion]:
        """Take the finished result waiting for that destination, if any.

        A destination has at most one, so the first unit holding one for
        this window is the one.
        """
        for unit in self.pool.units:
            completion = unit.take_output(window_key)
            if completion is not None:
                return completion
        return None

    def windows_holding_output(self) -> list:
        """The destinations whose results are still waiting in a unit."""
        waiting = []
        for unit in self.pool.units:
            windows = unit.output_windows()
            waiting.extend(windows)
        return sorted(waiting)

    def units_holding_rounds(self) -> list:
        """The names of the units whose memory still holds rounds."""
        held = []
        for unit in self.pool.units:
            if unit.memory.resident_input_count:
                held.append(unit.name)
        return held

    # ------------------------------------------------- dispatch, private

    def _start_decoder(self, job: decoding_records.DecodeJob) -> None:
        """Hand the started job to the decoder on this unit."""
        decoder = self.manager.decoder
        self.engine.log(
            log_sources.DECODER_MANAGER, f"START DECODE {job.label}"
        )
        self.trace.job_started.fire(job, job.unit)
        self._predict_compute_free(job)
        if job.decoder_input is not None:
            # the decoder reads this unit's memory now
            job.payloads = job.decoder_input.fragments()
        decoder.start(
            job,
            self.engine,
            lambda result: self.manager.decode_completed(job, result),
        )

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
        for member in self.manager.strong_requests.members_of(job):
            member.service_key = job.service_key
            member.service_dispatch_ticks = self.engine.now

    def _log_assignment(
        self, job: decoding_records.DecodeJob, claim_compute: bool
    ) -> None:
        waited_ticks = self.engine.now - job.ready_time
        waited = config.format_ticks(waited_ticks)
        waited = waited.strip()
        slot_note = ""
        if not claim_compute:
            slot_note = "staged, "
        pool_tag = decode_queue.pool_tag_of(self.pool.name)
        free_now = self.free_unit_count()
        self.engine.log(
            log_sources.DECODER_MANAGER,
            f"ASSIGN UNIT {job.decoding_unit_name} to {job.label} "
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
        for member in self.manager.strong_requests.members_of(job):
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
        job.ready_ticks = self.engine.now
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
        # the compute offer picks this job up at the next compute end
        assert holder is not None, f"{job.label} landed on a free unit"

    # ------------------------------------------------- the compute, private

    def _offer_compute(self, unit: decoder_unit_module.DecoderUnit) -> None:
        """Free compute goes to the oldest landed resident.

        Otherwise it stays reserved for the oldest one still in flight,
        or returns to the pool. As in gem5 O3's scheduleReadyInsts, only
        ready work acquires a functional unit, and blocked work waits in
        the queue, never on the unit.
        """
        if unit.holder is not None:
            return
        ready = unit.oldest_landed_resident_ready_to_start()
        if ready is not None:
            unit.claim_compute(ready)
            self.begin(ready)
            return
        landing = unit.oldest_landing_resident()
        if landing is not None:
            unit.claim_compute(landing)  # starts at its landing
            self._predict_compute_free(landing)
            return
        self.pool.release(unit)

    def _predict_compute_free(self, job: decoding_records.DecodeJob) -> None:
        """Record when the compute this job holds frees.

        The decode starts once its input has landed and runs for the
        declared latency. A decoder measured on the host clock declares
        no latency, so its unit stays unpredicted.
        """
        unit = job.unit
        decoder = self.manager.decoder
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


def job_defects_text(job: decoding_records.DecodeJob) -> str:
    """The landed window input's set detection-event indices, for the trace."""
    fragments = _landed_fragments(job)
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
        f"({member.round_count} rounds from the syndrome buffer)"
    )


def _landed_fragments(job: decoding_records.DecodeJob) -> list:
    """What the job reads: its landed input, else the payloads it carries."""
    if job.decoder_input is not None:
        return job.decoder_input.fragments()
    if job.payloads is None:
        return []
    return job.payloads


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the decode service reports, as one member."""

    job_dispatched: trace_source.TraceSource = trace_source.new_source()
    input_landed: trace_source.TraceSource = trace_source.new_source()
    job_started: trace_source.TraceSource = trace_source.new_source()
    job_finished: trace_source.TraceSource = trace_source.new_source()


@dataclasses.dataclass(frozen=True)
class _DispatchCost:
    """The manager's own work per dispatch, in cycles of its clock."""

    clock: Optional[config.Clock]
    cycles: int

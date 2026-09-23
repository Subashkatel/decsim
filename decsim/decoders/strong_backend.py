"""The strong decoder on a device: a FIFO queue in front of its capacity.

StrongBackendDecoder answers the Decoder port for a row whose decode
runs on a StrongBackend (decsim/ports.py). The decoder manager's units
hand it decodes once their input has landed; up to the backend's
capacity run at once and the rest wait in arrival order. Arrival order
is what every referent here serves in: an IonQ decoding core
"processes those blocks sequentially" (2608.25027 lines 509-511), and a
CUDA-Q dispatcher serves its ring's slots in turn, `current_slot =
(current_slot + 1) % num_slots` (cuda-quantum
realtime/lib/daemon/dispatcher/dispatch_kernel.cu:213-265).
"""

import collections
from typing import Any, Optional

import decsim.decoders.decoder as decoder_module
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.seeding as seeding


class StrongBackendDecoder(decoder_module.DecoderBase):
    """The Decoder port served by a StrongBackend, FIFO past its capacity.

    A decode's time is known only once the device has it, so the unit
    declares no occupancy in advance, the measured decoder's shape
    (DecoderBase). A cancelled waiting decode leaves the queue. A
    cancelled running decode holds the device until its service time
    ends and its result is dropped: a GPU does not stop a running
    kernel, it schedules other work "as the currently running ...
    kernel's thread blocks finish" (CUDA C++ Programming Guide,
    preemption).

    The row is transparent to the seed walk: the backend's seeded parts
    sit at the paths they would hold without it, so a backend that
    answers with decsim's own decoder draws what that decoder's row
    draws under the same run seed.
    """

    def __init__(self, backend: ports.StrongBackend) -> None:
        self.backend = backend
        # (job, on_result) in arrival order, waiting for the device
        self._waiting: collections.deque = collections.deque()
        self._running_count = 0
        # id() of each job cancelled while it ran; the job stays alive in
        # its finish event until then, so its id is not reused meanwhile
        self._cancelled_job_ids: set = set()

    def run_seed_children(self) -> tuple:
        """The backend's own children, at the paths it names."""
        if not isinstance(self.backend, seeding.RunSeedComposite):
            return ()
        return self.backend.run_seed_children()

    def decode(
        self, job: decoding_records.DecodeJob
    ) -> decoding_records.DecodeResult:
        """The device's answer for one region, outside the queue."""
        ticket = self.backend.submit(job, 0)
        return self.backend.result(ticket)

    def latency(self, job: decoding_records.DecodeJob) -> int:
        """A device's time is its own; there is none before it runs."""
        raise NotImplementedError(
            "a strong decode's time is the device's answer; the queue holds "
            "the unit until the device says"
        )

    def occupancy(self, job: decoding_records.DecodeJob) -> Optional[int]:
        """None: the device prices the decode when it runs it."""
        del job
        return None

    def start(
        self,
        job: decoding_records.DecodeJob,
        engine,
        on_result: decoder_module.OnResult,
    ) -> None:
        """Queue the decode; it runs once the device has room."""
        self._waiting.append((job, on_result))
        self._run_waiting(engine)

    def cancel(self, job: decoding_records.DecodeJob) -> None:
        """Drop a waiting decode; a running one finishes and says nothing."""
        for index, entry in enumerate(self._waiting):
            if entry[0] is job:
                del self._waiting[index]
                return
        job_id = id(job)
        self._cancelled_job_ids.add(job_id)

    def _run_waiting(self, engine) -> None:
        """Start waiting decodes in arrival order while the device has room."""
        capacity = self.backend.capacity()
        while self._waiting and self._running_count < capacity:
            job, on_result = self._waiting.popleft()
            self._run(job, on_result, engine)

    def _run(
        self,
        job: decoding_records.DecodeJob,
        on_result: decoder_module.OnResult,
        engine,
    ) -> None:
        ticket = self.backend.submit(job, self._running_count)
        self._running_count += 1
        service_ticks = self.backend.service_ticks(ticket)
        engine.schedule(
            service_ticks,
            lambda: self._finish(job, ticket, on_result, engine),
            label=f"decode_done({job.label})",
        )

    def _finish(
        self,
        job: decoding_records.DecodeJob,
        ticket: Any,
        on_result: decoder_module.OnResult,
        engine,
    ) -> None:
        """The device is free again; the decode's answer goes out."""
        self._running_count -= 1
        job_id = id(job)
        was_cancelled = job_id in self._cancelled_job_ids
        self._cancelled_job_ids.discard(job_id)
        self._run_waiting(engine)
        if was_cancelled:
            return
        result = self.backend.result(ticket)
        on_result(result)

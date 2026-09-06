"""The split-gap rendezvous: a sibling half spawned, two weights joined.

The complementary gap is the weight difference between the two forced
logical classes (Toshio et al. 2510.25222; Gidney et al. 2312.04522
Fig. 10). In split-pair mode the primary decode solves one class on its
unit and a sibling job solves the other on the gap pool; the weak
outcome is processed only when both halves have reported, an AND of two
completions, the same shape as the landed input join in the staging.
The weak unit itself is freed at its own solve end; only the outcome
waits. The sibling sends its own copy of the primary's landed rounds on
the Link port in the window's name.
"""

import dataclasses
from typing import Callable, Optional

import decsim.confidence.complementary as complementary
import decsim.decoders.decode_queue as decode_queue
import decsim.message as message
import decsim.observe.trace_source as trace_source

# the structure the sibling's own copy of the rounds lands in
GAP_SIBLING_INPUT = "gap sibling input"


@dataclasses.dataclass
class _GapJoin:
    """One window's rendezvous: the sibling's weight, the held primary."""

    sibling_weight: Optional[float] = None
    sibling_reported: bool = False
    held_weak_job: Optional[message.DecodeJob] = None
    held_weak_result: Optional[message.DecodeResult] = None


class GapJoins:
    """Spawns each window's sibling half and joins the two weights.

    Trace source: copy_made(primary, bits, memory_name, "gap sibling
    input") where the pair is formed, because the sibling takes its own
    copy of the primary's landed rounds instead of holding Buffer 0 for
    a second read (data_path.md section 8's copy table).
    """

    def __init__(
        self,
        engine,
        link,
        enqueue: Callable,
        is_enabled: bool,
    ) -> None:
        self.engine = engine
        # the link the sibling's input rides; None sends nothing
        self.link = link
        # the decode queue's enqueue(job, send_input, on_decoded)
        self.enqueue = enqueue
        # a router with a gap route turns the joins on
        self.is_enabled = is_enabled
        # (op_id, window_id) -> the join, from the sibling's spawn until
        # the join concludes the window
        self.joins_by_window: dict[tuple, _GapJoin] = {}
        self.copy_made = trace_source.TraceSource()

    def spawn(self, job: message.DecodeJob) -> None:
        """Submit the other forced-class solve to the gap pool.

        Fired at the primary weak decode's service start: the boundary
        mask is applied by then, so the sibling reads the same masked
        rounds the primary decodes (both units receive the same adjusted
        stream, the strong-buffer dual-write pattern on the strong side).
        The sibling carries its own copy of the rounds, pays its own
        transfer to the weak decoder and queues for its own unit. The
        copy is taken here instead of holding Buffer 0 for a second
        read, so the sibling's transfer is priced but never blocks on
        round retention.
        """
        if not self._wants_sibling(job):
            return
        key = (job.op_id, job.window_id)
        if key in self.joins_by_window:
            return
        sibling = self._sibling_job(job, key)
        self.joins_by_window[key] = _GapJoin()
        payload_bits = sibling.payload_bits()
        self.copy_made.fire(
            job, payload_bits, job.memory.name, GAP_SIBLING_INPUT
        )
        self.engine.log(
            decode_queue.LOG_SOURCE,
            f"SPLIT GAP {job.label}: sibling submitted to the gap pool",
        )
        self.enqueue(
            sibling,
            lambda on_landed: self._send_sibling_input(on_landed, sibling, job),
        )

    def take_weak_result(
        self, job: message.DecodeJob, result: message.DecodeResult
    ) -> Optional[message.DecodeResult]:
        """The primary decode finished: its result with the gap attached.

        None while the sibling half is still out; the result is held
        and comes back from sibling_done.
        """
        key = (job.op_id, job.window_id)
        join = self.joins_by_window.get(key)
        if join is None:
            return result
        if not join.sibling_reported:
            join.held_weak_job = job
            join.held_weak_result = result
            self.engine.log(
                decode_queue.LOG_SOURCE,
                f"GAP JOIN {job.label}: holding for the sibling half",
            )
            return None
        del self.joins_by_window[key]
        _attach_gap(result, join.sibling_weight)
        return result

    def sibling_done(
        self, key: tuple, sibling_weight: Optional[float]
    ) -> Optional[tuple]:
        """One gap half landed: (job, result) of the window if the other is in.

        None while the primary decode is still running.
        """
        join = self.joins_by_window.get(key)
        if join is None:
            raise RuntimeError(
                f"gap sibling finished for window {key} with no join entry"
            )
        join.sibling_reported = True
        join.sibling_weight = sibling_weight
        if join.held_weak_job is None:
            return None
        del self.joins_by_window[key]
        _attach_gap(join.held_weak_result, sibling_weight)
        return join.held_weak_job, join.held_weak_result

    def unresolved_windows(self) -> list:
        """The windows whose join has not concluded, sorted."""
        return sorted(self.joins_by_window)

    def _wants_sibling(self, job: message.DecodeJob) -> bool:
        if not self.is_enabled:
            return False
        if job.strong_decode_for is not None:
            return False
        if job.on_done is not None:
            return False
        if job.gap_sibling_for is not None:
            return False
        if job.window is None:
            return False
        if job.dem is None:
            return False
        return job.decoder_input is not None

    def _sibling_job(
        self, job: message.DecodeJob, key: tuple
    ) -> message.DecodeJob:
        """The other forced-class solve over the primary's landed rounds."""
        masked_fragments = job.decoder_input.fragments()
        round_count = len(job.decoder_input.rounds)
        return message.DecodeJob(
            op_id=job.op_id,
            window_id=job.window_id,
            n_rounds=round_count,
            dem=job.dem,
            payloads=masked_fragments,
            ready_time=self.engine.now,
            label=f"gap({job.label})",
            hint="gap",
            spatial_nodes=job.spatial_nodes,
            code=job.code,
            gap_sibling_for=key,
        )

    def _send_sibling_input(
        self,
        on_landed: Callable[[], None],
        sibling: message.DecodeJob,
        primary: message.DecodeJob,
    ) -> int:
        if self.link is None:
            on_landed()
            return 0
        payload_bits = sibling.payload_bits()
        # the link attribution is the window's, so the transfer rides
        # the primary job's window identity; the landing is the
        # sibling's own
        path = message.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER
        attribution = message.TransferAttribution.for_job(
            primary, primary.request_key
        )
        now_ticks = self.engine.now
        expected_delay_ticks = self.link.expected_delay_ticks(
            path, payload_bits, now_ticks
        )
        self.link.send(
            path, payload_bits, now_ticks, attribution, lambda _t: on_landed()
        )
        return expected_delay_ticks


def _attach_gap(
    result: message.DecodeResult, sibling_weight: Optional[float]
) -> None:
    """Build the SoftOutput from the two forced-class weights.

    Either half missing leaves soft_output None, and the policy then
    escalates (the same behavior a metric-less serial decode has).
    """
    primary_weight = result.gap_half_weight
    if primary_weight is None or sibling_weight is None:
        return
    w_min = min(primary_weight, sibling_weight)
    w_comp = max(primary_weight, sibling_weight)
    gap = w_comp - w_min
    result.soft_output = message.SoftOutput(
        gap=gap,
        source=complementary.COMPLEMENTARY_GAP_SOURCE,
        w_min=w_min,
        w_comp=w_comp,
    )

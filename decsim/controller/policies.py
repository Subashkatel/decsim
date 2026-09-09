"""How an idle round of a waiting patch travels: the idle policy rows.

idle_policy names one of Ignore, ExtendStream or SeparateDecodeJobs; each
fills the IdlePolicy seam (decsim/ports.py). The boundary policy rows
live beside the windows they ship for (windows/boundary_policies.py).

Idle rounds are real decoder workload. Terhal's backlog bound sets the
rate syndrome bits are generated, rgen, against the rate they are
processed, rproc, and every generated bit counts: "the decoding should
never lead to a increasing backlog of syndrome data" (1302.3428 lines
3151-3159). Battistel et al. say the same per logical qubit: "the
decoder needs to process that data at the acquisition rate or close to
it to avoid an exponential slowdown due to an ever-growing data backlog"
(2303.00054 line 144). Deferring them is legitimate, deleting them is a
modeling choice: only data feeding the next non-Clifford decision is
latency-critical (Skoric 2209.08552), so each policy below is valid for a
different claim.
"""


class Ignore:
    """Idle rounds travel as feedback-memory rounds and cost no decode work.

    An optimistic card: valid for latency studies of the active path, an
    undercount of decoder throughput, utilization, and unit counts on
    multi-operation workloads (every reference decodes idle volume).
    """

    def relay(self, idle_rounds, operation, patch, round_index: int) -> None:
        """Send the round as a memory round."""
        idle_rounds.emit_memory_round(operation, patch, round_index)

    def end_idle_period(self, idle_rounds, operation, patch) -> None:
        """Nothing was charged, so nothing settles."""
        del idle_rounds
        del operation
        del patch


class ExtendStream:
    """Idle rounds extend the operation's live stream when it has one.

    They travel as memory rounds otherwise. The stream rounds carry
    sampled content and are decoded: the XQsim continuous-stream shape.
    """

    def relay(self, idle_rounds, operation, patch, round_index: int) -> None:
        """Extend the live stream, or send a memory round."""
        extended = idle_rounds.extend_live_stream(operation, patch)
        if not extended:
            idle_rounds.emit_memory_round(operation, patch, round_index)

    def end_idle_period(self, idle_rounds, operation, patch) -> None:
        """Nothing was charged, so nothing settles."""
        del idle_rounds
        del operation
        del patch


class SeparateDecodeJobs:
    """Idle rounds travel as memory rounds and are charged as decode jobs.

    Every commit region of them costs one synthetic load-only decode job,
    sized to the region plus the buffer rounds and carrying no real
    syndrome contents. The rounds left over when an operation claims the
    patch cost one shorter job: a final window may be smaller than a
    regular one (Tan et al. 2209.09219; Skoric et al. 2209.08552), and no
    validated system leaves the end of a stream undecoded (Google's
    streaming decoder, LILLIPUT's per-cycle decode, Bombin's modular
    decoding of idle memory). The honest default for throughput,
    utilization, backlog, or unit-count claims.
    """

    def relay(self, idle_rounds, operation, patch, round_index: int) -> None:
        """Send the memory round and count it toward the next job."""
        idle_rounds.emit_memory_round(operation, patch, round_index)
        idle_rounds.submit_idle_decode_if_due(operation, patch, round_index)

    def end_idle_period(self, idle_rounds, operation, patch) -> None:
        """Charge the rounds left after the last full commit region."""
        idle_rounds.submit_idle_decode_for_remaining_rounds(operation, patch)

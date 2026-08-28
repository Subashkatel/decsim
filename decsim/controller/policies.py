"""The two policy seams a RunSpec fills: ``boundary_policy`` tells the window
manager when a committed boundary ships (Eager, Held); ``idle_policy`` tells
the controller how an idle round of a waiting patch travels (Ignore,
ExtendStream, SeparateDecodeJobs). Any object with the same methods works.

Idle rounds are real decoder workload in every reference system: SWIPER
windows idle syndrome exactly like operation syndrome (ISCA 2025, 2412.05115,
device_manager emits an UNWANTED_IDLE round per unused patch per cycle),
XQsim decodes every patch under each RUN_ESM (ISCA 2022), and Terhal's
backlog bound charges the decoder for every generated round (via Battistel
2303.00054). Deferring them is legitimate, deleting them is a modeling
choice: only data feeding the next non-Clifford decision is latency-critical
(Skoric 2209.08552), so each policy below is valid for a different claim.
"""


class Eager:
    """Ships every committed boundary, final or provisional."""

    def on_commit(self, window, final: bool) -> bool:
        return True


class Held:
    """Opt-in: ship only when the committing result is final."""

    def on_commit(self, window, final: bool) -> bool:
        return final


class Ignore:
    """Idle rounds travel as ordinary feedback-memory rounds and cost no extra
    decode work. An optimistic card: valid for latency studies of the active
    path, an undercount of decoder throughput, utilization, and unit counts
    on multi-op workloads (every reference decodes idle volume)."""

    def relay(self, controller, operation, patch, round_index: int) -> None:
        controller.emit_memory_round(operation, patch, round_index)



class ExtendStream:
    """Idle rounds extend the op's live dynamic stream when it has one, and
    travel as memory rounds otherwise. The stream rounds carry sampled
    content and are decoded: the XQsim continuous-stream shape."""

    def relay(self, controller, operation, patch, round_index: int) -> None:
        if not controller.extend_live_stream(operation, patch):
            controller.emit_memory_round(operation, patch, round_index)



class SeparateDecodeJobs:
    """Idle rounds travel as memory rounds and every completed commit region
    of them costs one synthetic load-only decode job, sized to commit plus
    buffer rounds and carrying no real syndrome contents. The honest default
    for throughput, utilization, backlog, or unit-count claims: the same
    cost shape as SWIPER's idle windows and XQsim's decode-everything."""

    def relay(self, controller, operation, patch, round_index: int) -> None:
        controller.emit_memory_round(operation, patch, round_index)
        controller.submit_idle_decode_if_due(operation, patch, round_index)

"""Policies supplied through ``RunSpec`` at the two runtime seams.

``RunSpec.boundary_policy`` reaches ``WindowManager`` for ``on_commit`` and
optional speculative recovery. ``RunSpec.idle_policy`` reaches ``Controller``
for mode routing and account callbacks. Compatible external objects enter
through ``RunSpec`` without registration.
"""


class Eager:
    """Ships every committed boundary and requests replay when a later strong
    result revises it."""

    speculative = True

    def on_commit(self, window, final: bool) -> bool:
        return True


class Held:
    """Opt-in: ship only when the committing result is final."""

    def on_commit(self, window, final: bool) -> bool:
        return final



class Ignore:
    """Idle rounds travel as ordinary feedback-memory rounds and cost no extra decode work."""

    def relay(self, controller, operation, patch, round_index: int) -> None:
        controller.emit_memory_round(operation, patch, round_index)

    def account(self, idle_rounds: int, op) -> None:
        pass


class ExtendStream:
    """Idle rounds extend the op's live dynamic stream when it has one, and
    travel as memory rounds otherwise."""

    def relay(self, controller, operation, patch, round_index: int) -> None:
        if not controller.extend_live_stream(operation, patch):
            controller.emit_memory_round(operation, patch, round_index)

    def account(self, idle_rounds: int, op) -> None:
        pass


class SeparateDecodeJobs:
    """Idle rounds travel as memory rounds and every completed commit region
    of them costs one synthetic load-only decode job, sized to commit plus
    buffer rounds and carrying no real syndrome contents."""

    def relay(self, controller, operation, patch, round_index: int) -> None:
        controller.emit_memory_round(operation, patch, round_index)
        controller.submit_idle_decode_if_due(operation, patch, round_index)

    def account(self, idle_rounds: int, op) -> None:
        pass

"""Policies supplied through ``RunSpec`` at the two runtime seams.

``RunSpec.boundary_policy`` reaches ``WindowManager`` for ``on_commit`` and
optional speculative recovery. ``RunSpec.idle_policy`` reaches ``Controller``
for mode routing and account callbacks. Compatible external objects enter
through ``RunSpec`` without registration.
"""


# BoundaryPolicy, protocol port 16

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


# IdlePolicy, protocol port 17

class Ignore:
    """Uses ordinary feedback-memory rounds without extra idle decode demand."""

    mode = "ignore"

    def account(self, idle_rounds: int, op) -> None:
        pass


class ExtendStream:
    """Inject idle rounds into the op's live dynamic stream when one exists;
    falls back to memory rounds otherwise (mode 'extend_stream')."""

    mode = "extend_stream"

    def account(self, idle_rounds: int, op) -> None:
        pass


class SeparateDecodeJobs:
    """Requests one synthetic load-only demand per completed commit region,
    sized to commit plus buffer rounds and carrying no real syndrome contents."""

    mode = "separate_decode_jobs"

    def account(self, idle_rounds: int, op) -> None:
        pass

"""The ticks of every operation's life, heard from the execution runtime.

A listener on the runtime's operation_issued, operation_started,
body_finished, decode_released and result_returned sources, each carrying
(operation_id, tick). The runtime keeps which operations have started,
finished and been released; the ticks live here, keyed by operation id,
and last_finish is the latest body done.
"""


class RuntimeStamps:
    """Five maps of operation id to tick, and the last body's finish."""

    def __init__(self) -> None:
        self.op_start: dict = {}
        self.body_done: dict = {}
        self.decode_release: dict = {}
        self.result_return: dict = {}
        self.last_finish = 0

    def operation_issued(self, operation_id, tick: int) -> None:
        """The issuer took the operation; its QPU start replaces this stamp."""
        self.op_start[operation_id] = tick

    def operation_started(self, operation_id, tick: int) -> None:
        """The QPU started the operation's body on this boundary."""
        self.op_start[operation_id] = tick

    def body_finished(self, operation_id, tick: int) -> None:
        """The operation's body is physically complete."""
        self.body_done[operation_id] = tick
        self.last_finish = max(self.last_finish, tick)

    def decode_released(self, operation_id, tick: int) -> None:
        """The blocked operation's release reached the controller."""
        self.decode_release[operation_id] = tick

    def result_returned(self, operation_id, tick: int) -> None:
        """The operation's result return reached the controller."""
        self.result_return[operation_id] = tick

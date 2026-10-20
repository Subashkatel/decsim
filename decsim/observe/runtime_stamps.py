"""The ticks of every operation's life, heard from the execution runtime."""


class RuntimeStamps:
    """Six maps of operation id to tick, the last body's finish, the waits.

    otherwise_ready holds when every start gate but a feedback release
    was open; magic_state_wait holds the ticks each operation waited for
    its magic state, the factory's supply stall.
    """

    def __init__(self) -> None:
        self.otherwise_ready: dict = {}
        self.op_start: dict = {}
        self.body_done: dict = {}
        self.decode_release: dict = {}
        self.result_return: dict = {}
        self.magic_state_wait: dict = {}

    @property
    def last_finish(self) -> int:
        """The latest body done, 0 before any."""
        finishes = self.body_done.values()
        return max(finishes, default=0)

    def operation_ready(self, operation_id: int, tick: int) -> None:
        """Every start gate but a feedback release opened at this tick."""
        self.otherwise_ready[operation_id] = tick

    def operation_issued(self, operation_id: int, tick: int) -> None:
        """The issuer took the operation; its QPU start replaces this stamp."""
        self.op_start[operation_id] = tick

    def operation_started(self, operation_id: int, tick: int) -> None:
        """The QPU started the operation's body on this boundary."""
        self.op_start[operation_id] = tick

    def body_finished(self, operation_id: int, tick: int) -> None:
        """The operation's body is physically complete."""
        self.body_done[operation_id] = tick

    def decode_released(self, operation_id: int, tick: int) -> None:
        """The blocked operation's release reached the controller."""
        self.decode_release[operation_id] = tick

    def result_returned(self, operation_id: int, tick: int) -> None:
        """The operation's result return reached the controller."""
        self.result_return[operation_id] = tick

    def magic_state_delivered(
        self, operation_id: int, waited_ticks: int
    ) -> None:
        """The factory handed the operation its state after that wait."""
        self.magic_state_wait[operation_id] = waited_ticks

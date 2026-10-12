"""The decode backlog: rounds read out and not yet finally corrected.

DecodeBacklog is built only when the observation section asks for
backlog_trace. The busy units and the jobs waiting are statistics the
pools and queues declare (decsim/statistics).
"""

import decsim.observe.round_events as round_events_module
import decsim.observe.run_views as run_views
import decsim.ports as ports


class DecodeBacklog:
    """The peak count of rounds read out whole and not finally corrected.

    Sampled after the last action of every tick, since the rounds
    waiting are spread over the window manager and the queues and no one
    source reports them. A
    window counts once its rounds are read out, so one whose rounds the
    controller holds for room in a full store is in the backlog too. The
    peak is over the counts ticks end at, the rule of a Level's peak
    (statistics/kinds.py): a round made whole and corrected at one tick
    never waited.
    """

    def __init__(
        self,
        window_manager: ports.WindowBacklog,
        decoder_managers: tuple,
        round_events: round_events_module.RoundEventRecorder,
    ) -> None:
        self.window_manager = window_manager
        self.decoder_managers = decoder_managers
        self.round_events = round_events
        self.ended_peak = 0
        self.tick = 0
        self.rounds = 0

    def reset(self, tick: int) -> None:
        """Begin the region of interest: no tick before this one counts.

        The rounds waiting now stay, a level held into the region.
        """
        self.ended_peak = 0
        self.tick = tick

    def observe(self, tick: int) -> None:
        """Sample the backlog after a tick's last action.

        A later tick ends this one.
        """
        view = run_views.backlog_view(
            self.window_manager,
            self.decoder_managers,
            self.round_events.rounds_read_out_by_operation,
        )
        if tick != self.tick:
            self.ended_peak = max(self.ended_peak, self.rounds)
            self.tick = tick
        self.rounds = view.total_rounds

    @property
    def peak(self) -> int:
        """The most rounds a tick ended at, the last tick's count included."""
        return max(self.ended_peak, self.rounds)

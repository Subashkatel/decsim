"""The decoders' end for a timing-only round: it counts and tells.

A feedback-memory round has no syndrome to decode, so what its landing
does is account for the stream stage it occupies, which both stream
decoders read for this hop run per round as the stream fills (LILLIPUT
2108.06569, Yang et al. 2605.04892), and tell the window side, which
counts the operation's idle rounds for window readiness. The receiving
end handles the landing: gem5's requesting port hands the packet to the
peer's own receive method (tmp/resources/gem5/src/mem/port.hh:603-614
into src/mem/protocol/timing.cc:49-53).
"""

import decsim.decoders.memory_rounds as memory_rounds
import decsim.engine as engine_module
import decsim.observe.log_writers as log_writers


class _Windows:
    """The WindowInput port, recording what the decoder side tells it."""

    def __init__(self) -> None:
        self.told = []

    def accept_feedback_memory_round(self, source_operation_id) -> None:
        self.told.append(source_operation_id)


def test_a_landed_memory_round_is_counted_and_the_window_side_is_told():
    engine = engine_module.Engine()
    windows = _Windows()
    arrivals = memory_rounds.MemoryRoundArrivals(engine, windows)

    arrivals.receive_memory_round(7)
    arrivals.receive_memory_round(7)
    arrivals.receive_memory_round(9)

    assert arrivals.landed_by_operation == {7: 2, 9: 1}
    assert windows.told == [7, 7, 9]


def test_the_arrival_is_narrated_on_the_decoder_sides_own_line():
    engine = engine_module.Engine()
    log = log_writers.LogWriter()
    engine.line.connect(log.write)
    windows = _Windows()
    arrivals = memory_rounds.MemoryRoundArrivals(engine, windows)

    arrivals.receive_memory_round(7)
    arrivals.receive_memory_round(7)

    assert log.lines == [
        "[  0.000 us] Decoder manager: memory round for op 7 "
        "(idle buffer rounds: 1)",
        "[  0.000 us] Decoder manager: memory round for op 7 "
        "(idle buffer rounds: 2)",
    ]

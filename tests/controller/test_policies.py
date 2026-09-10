"""How an idle round of a waiting patch travels, and what it costs.

idle_policy answers one question for the idle accounting: how the rounds
of a patch that waits travel and what they cost. The boundary policy rows
are tested beside them, in tests/windows/test_boundary_policies.py.

The three idle rows are three defensible cards, not one right answer.
Idle rounds are decoder workload, so the charged row is the default:
the backlog bound counts every generated syndrome bit against the
decoder's processing rate (Terhal 1302.3428 lines 3151-3159; Battistel
et al. 2303.00054 line 144). Ignore is the optimistic card, valid for
latency studies of the active path, since only data feeding the next
non-Clifford decision is latency critical (Skoric 2209.08552).
ExtendStream folds an idle patch's rounds into the live stream beside it.

The charged row's own arithmetic (one job per commit region, one shorter
job for the remainder) is pinned in test_idle_rounds.py, where the
accounting that counts the regions lives.
"""

import decsim.controller.policies as policies


class RecordingIdleRounds:
    """The accounting the policies call: what each policy asked for."""

    def __init__(self, is_stream_live=False):
        self.is_stream_live = is_stream_live
        self.memory_rounds = []
        self.stream_rounds = []

    def emit_memory_round(self, operation, patch, round_index):
        self.memory_rounds.append((operation, patch, round_index))

    def extend_live_stream(self, operation, patch):
        if not self.is_stream_live:
            return False
        self.stream_rounds.append((operation, patch))
        return True


def test_ignore_sends_the_idle_round_as_a_memory_round():
    ignore = policies.Ignore()
    idle_rounds = RecordingIdleRounds()
    ignore.relay(idle_rounds, "operation", "patch-a", 3)
    assert idle_rounds.memory_rounds == [("operation", "patch-a", 3)]
    assert idle_rounds.stream_rounds == []


def test_extend_stream_sends_the_idle_round_into_a_live_stream():
    extend = policies.ExtendStream()
    idle_rounds = RecordingIdleRounds(is_stream_live=True)
    extend.relay(idle_rounds, "operation", "patch-a", 3)
    assert idle_rounds.stream_rounds == [("operation", "patch-a")]
    assert idle_rounds.memory_rounds == []


def test_extend_stream_sends_a_memory_round_when_no_stream_is_live():
    extend = policies.ExtendStream()
    idle_rounds = RecordingIdleRounds(is_stream_live=False)
    extend.relay(idle_rounds, "operation", "patch-a", 3)
    assert idle_rounds.memory_rounds == [("operation", "patch-a", 3)]
    assert idle_rounds.stream_rounds == []

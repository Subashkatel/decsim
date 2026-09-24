"""How an idle round of a waiting patch travels, and what it costs.

idle_policy answers one question for the idle accounting: how the rounds
of a patch that waits travel and what they cost. The boundary policy rows
are tested beside them, in tests/windows/test_boundary_policies.py.

The two idle rows are two defensible cards, not one right answer.
Idle rounds are decoder workload, so the charged row is the default:
the backlog bound counts every generated syndrome bit against the
decoder's processing rate (Terhal 1302.3428 lines 3151-3159; Battistel
et al. 2303.00054 line 144). Ignore is the optimistic card, valid for
latency studies of the active path, since only data feeding the next
non-Clifford decision is latency critical (Skoric 2209.08552).

The charged row's own arithmetic (one job per commit region, one shorter
job for the remainder) is pinned in test_idle_rounds.py, where the
accounting that counts the regions lives.
"""

import decsim.controller.policies as policies


class RecordingIdleRounds:
    """The accounting the policies call: what each policy asked for."""

    def __init__(self):
        self.memory_rounds = []

    def emit_memory_round(self, operation, patch, round_index):
        self.memory_rounds.append((operation, patch, round_index))


def test_ignore_sends_the_idle_round_as_a_memory_round():
    ignore = policies.Ignore()
    idle_rounds = RecordingIdleRounds()
    ignore.relay(idle_rounds, "operation", "patch-a", 3)
    assert idle_rounds.memory_rounds == [("operation", "patch-a", 3)]

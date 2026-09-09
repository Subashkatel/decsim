"""The controller's idle-round counter, over every patch of a run.

A listener on the idle accounting's idle_round_emitted source; the count
is what the frozen gate pins as controller_idle_rounds.
"""

import decsim.observe.controller_counters as controller_counters


def test_a_run_that_emitted_no_idle_round_counts_zero():
    counters = controller_counters.ControllerCounters()

    assert counters.idle_rounds == 0


def test_every_idle_round_of_every_patch_is_counted_once():
    """One counter over all patches, which is what the gate pins."""
    counters = controller_counters.ControllerCounters()

    counters.idle_round_emitted(1, "p0", 1)
    counters.idle_round_emitted(1, "p1", 1)
    counters.idle_round_emitted(2, "p0", 2)

    assert counters.idle_rounds == 3

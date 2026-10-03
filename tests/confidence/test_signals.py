"""The soft-output rows a switching run's weak decoder can report.

CONFIDENCE_SIGNALS is the table escalation.confidence names a row of,
and every row's Settings record builds the row from the weak decoder's
own settings and the threshold, so a row written outside decsim
reaches the root through that one call. Each row declares what evidence
it needs from the decode and which logical classes the window must be
decoded in, so the window side asks the weak decoder for exactly those
solves and never for the row's class.
"""

import pytest

import decsim.confidence.signals as confidence_signals


def test_the_shipped_rows_are_reachable_by_the_name_the_yaml_writes():
    rows = confidence_signals.CONFIDENCE_SIGNALS

    assert sorted(rows) == [
        "cluster_gap",
        "complementary_gap",
        "extra_cluster_gap",
    ]


@pytest.mark.parametrize("name", sorted(confidence_signals.CONFIDENCE_SIGNALS))
def test_every_rows_record_refuses_a_negative_walk_card(name):
    """A negative walk would shorten the unit's hold without a stop."""
    row = confidence_signals.CONFIDENCE_SIGNALS[name]

    with pytest.raises(ValueError):
        row.Settings(walk_microseconds=-1.0)

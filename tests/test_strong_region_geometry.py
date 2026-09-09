"""The forward strong region's width is declared once and read everywhere.

The escalated window's strong region is its commit region with one
buffer region folded in on each side, rcom + 2 rbuf. Toshio et al.
2510.25222 line 1352 parameterises that two as alpha, so the width is a
sweep axis: the interaction row that plans the region and the card that
provisions the link carrying it must move together when it changes, and
not one release apart.
"""

import decsim.links.link_profiles as link_profiles
import decsim.records.windows as window_records
import decsim.windows.boundary_payloads as boundary_payloads
import decsim.windows.window_interactions as window_interactions

COMMIT_ROUNDS = 5
BUFFER_ROUNDS = 5
SYNDROME_BITS_PER_ROUND = 24
OPERATION_ROUND_COUNT = 400

GEOMETRY = dict(
    syndrome_bits_per_round=SYNDROME_BITS_PER_ROUND,
    round_microseconds=1.0,
    commit_rounds=COMMIT_ROUNDS,
    buffer_rounds=BUFFER_ROUNDS,
)


def _escalated_window() -> window_records.WindowInfo:
    """One weak window of five commit rounds and five buffer rounds."""
    return window_records.WindowInfo(
        operation_id=1,
        window_index=3,
        commit_lo=11,
        commit_hi=15,
        buffer_hi=20,
        round_count=10,
        buffer_lo=None,
        deps=(),
        dependents=(),
    )


def _planned_region_round_count() -> int:
    """The rounds the shipped interaction row's strong region commits."""
    payload = boundary_payloads.DenseSeamMask()
    interaction = window_interactions.DefaultWindowInteraction(0, payload)
    window = _escalated_window()
    plan = interaction.plan_strong_region(window, [], OPERATION_ROUND_COUNT)
    return plan.commit_hi - plan.commit_lo + 1


def _provisioned_region_round_count() -> float:
    """The rounds the bandwidth card provisions the strong input for."""
    profile = link_profiles.bandwidth_limited_profile(**GEOMETRY)
    path = profile.strong_buffer_to_strong_decoder
    bits = path.default_payload.aggregate_bits
    return bits / SYNDROME_BITS_PER_ROUND


def test_the_declared_width_is_commit_plus_two_buffer_regions():
    declared = window_records.strong_region_round_count(
        COMMIT_ROUNDS, BUFFER_ROUNDS
    )
    assert window_records.STRONG_REGION_BUFFER_REGIONS == 2
    assert declared == 15


def test_the_restart_reread_width_counts_whole_buffer_regions():
    """A restart window re-reads whole buffer regions of the region."""
    none = window_records.restart_reread_round_count(0, BUFFER_ROUNDS)
    one = window_records.restart_reread_round_count(1, BUFFER_ROUNDS)
    assert none == 0
    assert one == BUFFER_ROUNDS


def test_the_row_and_the_card_read_the_declared_width():
    declared = window_records.strong_region_round_count(
        COMMIT_ROUNDS, BUFFER_ROUNDS
    )
    assert _planned_region_round_count() == declared
    assert _provisioned_region_round_count() == declared


def test_a_wider_declared_region_moves_the_row_and_the_card_together(
    monkeypatch,
):
    """The sweep axis is one number; nobody carries a second copy of it."""
    monkeypatch.setattr(window_records, "STRONG_REGION_BUFFER_REGIONS", 3)
    widened = window_records.strong_region_round_count(
        COMMIT_ROUNDS, BUFFER_ROUNDS
    )
    assert widened == 20
    assert _planned_region_round_count() == widened
    assert _provisioned_region_round_count() == widened

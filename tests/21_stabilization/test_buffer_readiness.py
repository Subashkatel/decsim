"""Buffer 0 and syndrome buffer 1 readiness under declared ticks.

Contract: ../validation/responsibility_audit_2026_08_30/buffer_contract.md.
Round r of a 1.0 us cadence is emitted at tick r us; every stage below
adds its declared cost, so each assertion is exact arithmetic.
"""

import pytest

from decsim.config import us
from decsim.engine import Engine
from decsim.links.link_profiles import logical_reference_profile
from decsim.links.links import TrafficAttribution
from decsim.message import RetainedSyndromeFragment, SyndromeRoundPacket
from decsim.syndrome_buffer.syndrome_buffer_1 import SyndromeBuffer1


def test_weak_only_pipeline_arithmetic(fabric):
    """qc 2 + binary 3 + cwb 4 publishes round r at r+9; wbd 5 + weak 10
    finish the single window; wdo 2 + frame 1 commit the correction."""
    completed = fabric["weak_only_run"](rounds=6)
    window = completed.window_manager.windows[(1, 0)]

    assert window.t_first_round == us(1 + 2 + 3 + 4)          # round 1 published
    assert window.t_data_complete == us(6 + 2 + 3 + 4)        # round 6 published
    assert window.t_queued == window.t_data_complete
    assert window.t_dispatch == window.t_data_complete
    assert window.t_done == window.t_data_complete + us(5 + 10)
    (record,) = completed.pauli_frame.snapshot().records
    assert record.tier == "weak"
    assert record.accepted_ticks == window.t_done + us(2)     # wdo
    assert record.committed_ticks == record.accepted_ticks + us(1)  # frame write


def test_unpriced_cwb_publishes_at_packing(fabric):
    """Without the CWB edge the round is public at packing completion."""
    completed = fabric["weak_only_run"](rounds=6, cwb=False)
    window = completed.window_manager.windows[(1, 0)]
    assert window.t_first_round == us(1 + 2 + 3)
    assert window.t_data_complete == us(6 + 2 + 3)


def test_notification_without_publication_refuses(fabric):
    """accept_window_input for a round the store does not hold fails loudly."""
    completed = fabric["weak_only_run"](rounds=6)
    fragment = RetainedSyndromeFragment(
        operation_id=1, patch_id=1, round_index=3, bits=None,
        size_bits=None, fragment_index=0)
    packet = SyndromeRoundPacket(1, 3, (fragment,))
    with pytest.raises(RuntimeError):
        completed.window_manager.accept_window_input(packet)


def test_cwb_delivers_rounds_in_order(fabric):
    """Buffer 0 receives rounds 1..6 in strictly increasing order."""
    completed = fabric["weak_only_run"](rounds=6, io_trace=True)
    arrival_order = [
        fabric["log_index"](completed.engine.log_lines,
                            f"round {round_index} of mem1 arrived")
        for round_index in range(1, 7)
    ]
    assert arrival_order == sorted(arrival_order)


def test_weak_primary_readiness_is_buffer0_not_sb1(fabric):
    """With csb 7 slower than cwb 4, weak readiness completes at the
    Buffer 0 publication; the room-side landing follows later."""
    completed = fabric["switching_run"](escalation_probability=0.0,
                                        rounds=9, io_trace=True)
    first_window = completed.window_manager.windows[(1, 0)]
    # window (1,0) reads rounds 1..6: complete at 6 + qc 2 + binary 3 + cwb 4
    assert first_window.t_data_complete == us(6 + 2 + 3 + 4)
    buffer0_complete = fabric["log_index"](
        completed.engine.log_lines, "round 6 of mem1 arrived")
    sb1_landing = fabric["log_index"](
        completed.engine.log_lines, "received round 6 of op 1 from csb")
    assert buffer0_complete < sb1_landing


def test_strong_primary_readiness_is_sb1(fabric):
    """Strong-primary readiness waits for the csb landing: round r is
    ready at r + qc 2 + binary 3 + csb 7."""
    completed = fabric["strong_only_run"](rounds=6)
    window = completed.window_manager.windows[(1, 0)]
    assert window.t_first_round == us(1 + 2 + 3 + 7)
    assert window.t_data_complete == us(6 + 2 + 3 + 7)
    # sbd 6 then the strong decode 30
    assert window.t_done == window.t_data_complete + us(6 + 30)
    (record,) = completed.pauli_frame.snapshot().records
    assert record.tier == "strong"
    assert record.accepted_ticks == window.t_done + us(4)     # do
    assert record.committed_ticks == record.accepted_ticks + us(1)


def test_sb1_gap_cannot_be_served(fabric):
    """A missing interior round is never hidden by the stored-through
    counter: exact reads refuse."""
    sb1 = SyndromeBuffer1(Engine(), logical_reference_profile().resolve())
    sb1.register_hold("reader", [(1, 1), (1, 2), (1, 3)])

    def write(round_index):
        fragment = RetainedSyndromeFragment(
            operation_id=1, patch_id=0, round_index=round_index,
            bits=(1,), size_bits=1, fragment_index=0)
        sb1.write(SyndromeRoundPacket(1, round_index, (fragment,)),
                  packet_bits=1,
                  attribution=TrafficAttribution(
                      operation_id=1, patch_ids=(0,), window_id=None,
                      round_lo=round_index, round_hi=round_index))

    write(1)
    write(3)
    assert sb1.rounds_arrived[1] == 3          # the max counter reads 3
    assert sb1.retained_fragments((1, 2)) is None
    with pytest.raises(RuntimeError, match="not stored in syndrome buffer 1"):
        sb1.ready_tick([(1, 2)])


def test_stores_settle_empty(fabric):
    """At the end of a run neither store holds a round or a hold."""
    completed = fabric["switching_run"](escalation_probability=1.0, rounds=9)
    buffer0 = completed.syndrome_buffer.snapshot()
    assert buffer0.occupancy == 0
    completed.syndrome_buffer_1.check_settled()
    assert completed.syndrome_buffer_1.peak_occupancy_rounds() > 0


def test_upstream_rounds_survive_until_the_input_transfer_lands(fabric):
    """Buffer 0 occupancy holds all six rounds through the WBD transfer
    and frees them exactly at its landing (dispatch 15 + wbd 5)."""
    make_metrics, probes = fabric["occupancy_metrics"]()
    fabric["weak_only_run"](rounds=6, make_metrics=make_metrics)
    (probe,) = probes
    timeline = probe.buffer0_timeline

    assert max(occupancy for _, occupancy in timeline) == 6
    first_release_tick = next(
        tick for (tick, occupancy), (_, previous) in
        zip(timeline[1:], timeline) if occupancy < previous)
    assert first_release_tick == us(15 + 5)


def test_room_side_rounds_survive_until_the_final_strong_commit(fabric):
    """SB1 occupancy holds the escalated window's context through the
    whole strong path; the refcounted holds free it only after the final
    commit, never at the SBD landing."""
    make_metrics, probes = fabric["occupancy_metrics"]()
    fabric["switching_run"](escalation_probability=1.0, rounds=6,
                            make_metrics=make_metrics)
    (probe,) = probes
    timeline = probe.sb1_timeline

    assert max(occupancy for _, occupancy in timeline) == 6
    first_release_tick = next(
        tick for (tick, occupancy), (_, previous) in
        zip(timeline[1:], timeline) if occupancy < previous)
    # strong committed at 71, held boundary and courier resolve after;
    # the empirically pinned release tick of this configuration
    assert first_release_tick == us(87.5)

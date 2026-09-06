"""Buffer 0 and syndrome buffer 1 readiness under declared ticks.

Contract: ../validation/responsibility_audit_2026_08_30/buffer_contract.md.
Round r of a 1.0 us cadence is emitted at tick r us; every stage below
adds its declared cost, so each assertion is exact arithmetic.
"""

import pytest

from decsim.config import microseconds_to_ticks
from decsim.engine import Engine
from decsim.links.fabric import LinkFabric
from decsim.links.link_profiles import logical_reference_profile
from decsim.message import (DecoderTier, RetainedSyndromeFragment,
                            SyndromeRoundPacket)


def test_weak_only_pipeline_arithmetic(fabric):
    """qc 2 + binary 3 + cwb 4 publishes round r at r+9; wbd 5 + weak 10
    finish the single window; wdo 2 + frame 1 commit the correction."""
    completed = fabric["weak_only_run"](rounds=6)
    window = completed.window_manager.windows[(1, 0)]

    assert window.t_first_round == microseconds_to_ticks(1 + 2 + 3 + 4)          # round 1 published
    assert window.t_data_complete == microseconds_to_ticks(6 + 2 + 3 + 4)        # round 6 published
    assert window.t_queued == window.t_data_complete
    assert window.t_dispatch == window.t_data_complete
    assert window.t_done == window.t_data_complete + microseconds_to_ticks(5 + 10)
    (record,) = completed.pauli_frame.snapshot().records
    assert record.tier == "weak"
    assert record.accepted_ticks == window.t_done + microseconds_to_ticks(2)     # wdo
    assert record.committed_ticks == record.accepted_ticks + microseconds_to_ticks(1)  # frame write


def test_unpriced_cwb_publishes_at_packing(fabric):
    """Without the CWB edge the round is public at packing completion."""
    completed = fabric["weak_only_run"](rounds=6, controller_to_weak_buffer=False)
    window = completed.window_manager.windows[(1, 0)]
    assert window.t_first_round == microseconds_to_ticks(1 + 2 + 3)
    assert window.t_data_complete == microseconds_to_ticks(6 + 2 + 3)



def test_cwb_delivers_rounds_in_order(fabric):
    """Buffer 0 receives rounds 1..6 in strictly increasing order."""
    completed = fabric["weak_only_run"](rounds=6, io_trace=True)
    arrival_order = [
        fabric["log_index"](completed.observation.log.lines,
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
    assert first_window.t_data_complete == microseconds_to_ticks(6 + 2 + 3 + 4)
    buffer0_complete = fabric["log_index"](
        completed.observation.log.lines, "round 6 of mem1 arrived")
    sb1_landing = fabric["log_index"](
        completed.observation.log.lines, "received round 6 of op 1 from controller_to_strong_buffer")
    assert buffer0_complete < sb1_landing


def test_strong_primary_readiness_is_sb1(fabric):
    """Strong-primary readiness waits for the csb landing: round r is
    ready at r + qc 2 + binary 3 + csb 7."""
    completed = fabric["strong_only_run"](rounds=6)
    window = completed.window_manager.windows[(1, 0)]
    assert window.t_first_round == microseconds_to_ticks(1 + 2 + 3 + 7)
    assert window.t_data_complete == microseconds_to_ticks(6 + 2 + 3 + 7)
    # sbd 6 then the strong decode 30
    assert window.t_done == window.t_data_complete + microseconds_to_ticks(6 + 30)
    (record,) = completed.pauli_frame.snapshot().records
    assert record.tier == "strong"
    assert record.accepted_ticks == window.t_done + microseconds_to_ticks(4)     # do
    assert record.committed_ticks == record.accepted_ticks + microseconds_to_ticks(1)


def test_strong_primary_rounds_never_enter_the_weak_path(fabric):
    """When the strong tier is the only decoder, a round travels once: over
    csb into syndrome buffer 1. Nothing crosses cwb and Buffer 0 never
    allocates, the single-path streaming of a one-tier system (LILLIPUT's
    readout-to-decoder FIFO, Das et al. ASPLOS 2022; Google's streaming
    decoder, Nature 2024)."""
    completed = fabric["strong_only_run"](rounds=6, record=True)
    link_traffic = completed.traffic_ledger.traffic_json_value()
    transfers_by_path = {edge["path"]: edge["counters"]["transfer_count"]
                         for edge in link_traffic["semantic_edges"]}
    assert transfers_by_path["controller_to_weak_buffer"] == 0
    assert transfers_by_path["controller_to_strong_buffer"] == 6
    (record,) = completed.pauli_frame.snapshot().records
    assert record.tier == "strong"
    (request,) = completed.decode_records.requests
    assert request.request_key.tier is DecoderTier.STRONG
    assert request.terminal_processing_outcome.value == "primary_forwarded_for_delivery"


def test_a_full_buffer_0_stalls_the_controller_instead_of_failing(fabric):
    """When Buffer 0 is full, the controller holds the finished round in
    its packing workspace and publishes it once a slot frees, in round
    order; nothing is dropped and the run completes. Real-time decoders
    apply backpressure to the source rather than discarding syndromes:
    the Rigetti sequencer polls the decoder's status register and stalls
    (Caune et al. 2410.05202), Helios asserts input ready only when it can
    take data, and QubiC's cores block in WAIT_MEAS until readout data is
    consumed."""
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.pauli_frame.pauli_frame import PauliFrameConfig
    from decsim.qpu.round_policies import FixedRounds
    from decsim.decoders.settings import DecoderSettings
    from decsim.frontends.settings import WorkloadSettings
    from decsim.machine import MachineSettings
    from decsim.syndrome_buffer.settings import RoundStoreSettings
    declared = fabric["DECLARED_US"]
    settings = MachineSettings(
        workload=WorkloadSettings(operations=[fabric["memory_op"](1)],
                                  rounds_policy=FixedRounds(12)),
        qpu=fabric["declared_qpu"](),
        weak_decoder=DecoderSettings(decoder=PresetLatencyDecoder(declared["weak"])),
        links=fabric["declared_profile"](controller_to_weak_buffer=True, controller_to_strong_buffer=False),
        controller=fabric["declared_timing"](),
        round_store=RoundStoreSettings(rounds=7),
        pauli_frame=PauliFrameConfig(commit_microseconds=declared["frame"]))
    completed = fabric["run_machine"](settings, 0)
    recorder = completed.round_events
    assert recorder.packing_drops == 0
    published = [(event.round_index, event.tick) for event in recorder.events
                 if event.kind == "PUBLISHED"]
    assert [round_index for round_index, _ in published] == list(range(1, 13))
    ticks = [tick for _, tick in published]
    assert ticks == sorted(ticks)
    # round 8 finds the store full (rounds 1..7 are held for the first
    # window) and waits for the first window's input to land
    publication = dict(published)
    assert publication[7] == microseconds_to_ticks(7 + 2 + 3 + 4)
    assert publication[8] > microseconds_to_ticks(8 + 2 + 3 + 4)
    records = completed.pauli_frame.snapshot().records
    assert [record.tier for record in records] == ["weak"] * 3


def test_stores_settle_empty(fabric):
    """At the end of a run neither store holds a round or a hold."""
    completed = fabric["switching_run"](escalation_probability=1.0, rounds=9)
    assert completed.round_store.occupancy == 0
    completed.strong_round_writer.check_settled()


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
    assert first_release_tick == microseconds_to_ticks(15 + 5)


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
    assert first_release_tick == microseconds_to_ticks(87.5)

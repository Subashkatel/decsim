"""Advisor-requested least-claim baseline: weak-only, FIFO, unbounded storage."""
from decsim import RunSpec
from decsim.decoders import PerRoundDecoder
from decsim.message import DecoderTier, Operation
from decsim.planner import FixedRounds
from decsim.schedulers import FifoScheduler
from decsim.switching import Baseline
from decsim.syndrome_buffer import SyndromeBufferingConfig


def test_simple_weak_only_baseline_is_complete_and_observable():
    completed = RunSpec(
        ops=[Operation(0, "memory", (0,))], d=3,
        rounds_policy=FixedRounds(9), decoder=PerRoundDecoder(0.5),
        scheduler=FifoScheduler(), strategy=Baseline(),
        syndrome_buffering=SyndromeBufferingConfig(),
        record_switching_windows=True,
    ).build(verbose=False)

    syndrome_buffer = completed.syndrome_buffer
    assert completed.result.terminal_status == "complete"
    assert syndrome_buffer.capacity is None
    assert type(completed.decoder_manager.scheduler) is FifoScheduler
    local_memory = completed.decoder_manager.decoder_input_transfer.local_memory
    assert local_memory.capacity is None
    assert local_memory.slots_in_use == 0
    assert completed.window_manager.links.config.profile_name == "logical_reference"
    edge_counts = {
        row["path"]: row["counters"]["transfer_count"]
        for row in completed.result.link_traffic["semantic_edges"]
    }
    # Phase A: window-input rounds publish from the retained upstream round
    # without a per-round CWD transfer; the decoder-input CWD charge is paid
    # once per weak decode submission instead.
    assert edge_counts["cwd"] == completed.window_manager.total_windows == 2
    assert edge_counts["wsd"] == edge_counts["csd"] == 0

    requests = completed.decoder_manager.terminal_request_records_snapshot()
    assert requests
    assert all(record.request_key.tier is DecoderTier.WEAK for record in requests)
    assert all(record.terminal_processing_outcome.value ==
               "weak_forwarded_for_delivery" for record in requests)
    assert syndrome_buffer.peak_payloads > 0
    assert syndrome_buffer.payloads_held == 0
    assert completed.syndrome_ingress.ingress_snapshot().ingress_contexts == 0

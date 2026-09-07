"""The gap join's law: the weight difference, once both halves report.

The gap is complementary_class_weight minus decoded_class_weight, and
neither half of the pair can name it alone. Toshio et al. 2510.25222
(the complementary gap between the two forced logical classes); Gidney
et al. 2312.04522 Fig. 10.
"""

import decsim.decoders.decoder_memory as decoder_memory
import decsim.decoders.gap_joins as gap_joins_module
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records


class _Enqueued:
    """Keeps every sibling the joins enqueue."""

    def __init__(self):
        self.jobs = []

    def __call__(self, job, send_input=None, on_decoded=None):
        del send_input
        del on_decoded
        self.jobs.append(job)


def _primary():
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_id="p",
        round_index=1,
        bits=(1, 0),
        size_bits=2,
        fragment_index=0,
    )
    landed_round = decoder_memory.MaterializedSyndromeRound(1, 1, (fragment,))
    decoder_input = decoder_memory.DecoderInput(1, 0, None, (landed_round,))
    window = window_records.Window(
        operation_id=1,
        window_index=0,
        commit_lo=1,
        commit_hi=1,
        buffer_hi=1,
        round_count=1,
    )
    model = object()
    # a job whose input has landed sits in some unit's memory, and the
    # sibling's copy is named after it
    memory = decoder_memory.DecoderMemory("default", 0, None)
    return decoding_records.DecodeJob(
        operation_id=1,
        window_id=0,
        round_count=1,
        detector_error_model=model,
        decoder_input=decoder_input,
        label="mem W0",
        window=window,
        memory=memory,
    )


def _joins(enqueue):
    engine = engine_module.Engine()
    return gap_joins_module.GapJoins(engine, None, enqueue, is_enabled=True)


def test_the_sibling_is_enqueued_at_service_start_with_the_landed_rounds():
    enqueued = _Enqueued()
    joins = _joins(enqueued)
    primary = _primary()
    joins.spawn(primary)
    (sibling,) = enqueued.jobs
    assert sibling.label == "gap(mem W0)"
    assert sibling.hint == "gap"
    assert sibling.gap_sibling_for == (1, 0)
    assert sibling.payloads == primary.decoder_input.fragments()
    assert joins.unresolved_windows() == [(1, 0)]


def test_a_window_has_a_join_from_the_spawn_until_both_halves_report():
    enqueued = _Enqueued()
    joins = _joins(enqueued)
    primary = _primary()
    assert joins.has_join(primary) is False
    joins.spawn(primary)
    assert joins.has_join(primary) is True
    result = decoding_records.DecodeResult(1, 0, gap_half_weight=3.0)
    joins.take_weak_result(primary, result)
    joins.sibling_done((1, 0), 5.0)
    assert joins.has_join(primary) is False


def test_the_gap_is_the_two_class_weights_difference_once_both_report():
    enqueued = _Enqueued()
    joins = _joins(enqueued)
    primary = _primary()
    joins.spawn(primary)
    result = decoding_records.DecodeResult(1, 0, gap_half_weight=3.0)
    assert joins.take_weak_result(primary, result) is None
    joined_job, joined_result = joins.sibling_done((1, 0), 5.0)
    assert joined_job is primary
    assert joined_result.soft_output.gap == 2.0
    assert joined_result.soft_output.decoded_class_weight == 3.0
    assert joined_result.soft_output.complementary_class_weight == 5.0
    assert joins.unresolved_windows() == []


def test_a_sibling_that_reports_first_joins_at_the_primary_result():
    enqueued = _Enqueued()
    joins = _joins(enqueued)
    primary = _primary()
    joins.spawn(primary)
    assert joins.sibling_done((1, 0), 4.0) is None
    result = decoding_records.DecodeResult(1, 0, gap_half_weight=6.0)
    joined = joins.take_weak_result(primary, result)
    assert joined is result
    assert joined.soft_output.gap == 2.0


def test_a_missing_half_leaves_no_soft_output():
    enqueued = _Enqueued()
    joins = _joins(enqueued)
    primary = _primary()
    joins.spawn(primary)
    result = decoding_records.DecodeResult(1, 0, gap_half_weight=3.0)
    joins.take_weak_result(primary, result)
    _joined_job, joined_result = joins.sibling_done((1, 0), None)
    assert joined_result.soft_output is None

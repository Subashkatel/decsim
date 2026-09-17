"""The referee wrapper's timing is the inner row's in every respect.

The wrapper re-decodes every window with Tesseract and charges nothing
for it (decsim/decoders/verify_windows.py); the unit is held for what
the inner row says, its own cycle count included.
"""

import pytest

import decsim.config as config
import decsim.decoders.union_find.cycle_count as cycle_count_module
import decsim.decoders.union_find.decoder as union_find
import decsim.decoders.verify_windows as verify_windows
import decsim.records.decoder_evidence as evidence_records
import decsim.records.decoding as decoding_records

CYCLE_TICKS = config.microseconds_to_ticks(0.01)
CLOCK = config.Clock(CYCLE_TICKS)


def test_the_checked_decoder_holds_the_unit_for_the_inner_rows_count():
    pytest.importorskip("tesseract_decoder")
    count = cycle_count_module.CycleCount(CLOCK, delay_cycles=3)
    inner = union_find.UnionFindDecoder(cycle_count=count)
    checked = verify_windows.TesseractCheckedDecoder(inner)
    graph = evidence_records.UnionFindGraph(
        detector_count=1,
        fault_count=0,
        edges=(),
        baseline_faults=(),
        baseline_syndrome=(0,),
    )
    evidence = evidence_records.UnionFindHardEvidence(
        graph=graph,
        syndrome=(),
        residual_syndrome=(),
        selected_faults=(),
        contact_faults=(),
        edge_intervals=(),
        erasure_forest_faults=(),
        logical_observables=(),
    )
    result = decoding_records.DecodeResult(1, 0)
    result.cluster_evidence = evidence
    host_nanoseconds = 5
    # the quiet machine: one iteration and a peel that peels nothing
    assert checked.ticks_after_decode(result, host_nanoseconds, 0) == (
        11 * CYCLE_TICKS
    )

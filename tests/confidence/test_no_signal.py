"""No confidence: an edge whose verdict reads none (confidence/no_signal.py).

The row asks the decode for nothing and reports no gap, so any decoder
serves it, and a verdict that reads the threshold is refused it rather
than escalating every window.
"""

import dataclasses

import pytest

import decsim.build.escalation as escalation_build
import decsim.confidence.no_signal as no_signal
import decsim.engine as engine_module
import decsim.escalation.verdicts as verdicts
import decsim.records.decoding as decoding_records
import tests.topology.test_graph_layout as layout_tests

STATUS = verdicts.Status.Settings()
NO_SIGNAL = no_signal.NoSignal.Settings()


def test_the_none_row_asks_for_nothing_and_reports_no_gap():
    signal = NO_SIGNAL.build(None, None)
    result = decoding_records.DecodeResult(1, 0)

    computation = signal.compute((result,))

    assert signal.decoder_evidence_requirement == frozenset()
    assert signal.forced_logical_classes == ()
    assert computation.soft_output is None
    assert computation.ticks == 0


def test_a_deep_edge_on_its_sources_status_needs_no_confidence():
    deep_switching = dataclasses.replace(
        layout_tests.DEEP_SWITCHING_SLOT, confidence=NO_SIGNAL, verdict=STATUS
    )
    chain = layout_tests.chain_machine(deep_switching)
    layout = layout_tests.laid_out(chain)
    engine = engine_module.Engine()

    parts = escalation_build.build_switching(layout, engine)

    deep = parts[1]
    assert deep.confidence_signal.source == no_signal.NO_SIGNAL_SOURCE


def test_a_verdict_that_reads_the_threshold_is_refused_no_confidence():
    deep_switching = dataclasses.replace(
        layout_tests.DEEP_SWITCHING_SLOT, confidence=NO_SIGNAL
    )
    chain = layout_tests.chain_machine(deep_switching)
    layout = layout_tests.laid_out(chain)
    engine = engine_module.Engine()

    with pytest.raises(ValueError, match="confidence none gives none"):
        escalation_build.build_switching(layout, engine)

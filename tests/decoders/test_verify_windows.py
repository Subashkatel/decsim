"""The referee wrapper's timing is the inner row's in every respect.

The wrapper re-decodes every window with Tesseract and charges nothing
for it (decsim/decoders/verify_windows.py): the inner row starts every
job, so a strong backend's measured line prices the decode whether the
referee checks it or not.
"""

import dataclasses

import pytest

import decsim.decoders.decoders as decoders
import decsim.decoders.measured_table.decoder as measured_table
import decsim.decoders.staged_decoder as staged_decoder
import decsim.decoders.verify_windows as verify_windows
import decsim.engine as engine_module
import decsim.machine as machine_module
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models
import decsim.settings as machine_settings
import tests.decoders.windows as windows
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)


def _algorithm_spans(strong_algorithm) -> list:
    """(start, end) ticks of every algorithm stage of one strong-only run."""
    base = machine_settings.strong_decoder_baseline(5, 0.001, 1.0)
    strong_decoder = dataclasses.replace(
        base.strong_decoder, algorithm=strong_algorithm
    )
    settings = dataclasses.replace(base, strong_decoder=strong_decoder)
    machine = machine_module.Machine.build(settings)
    machine.run()
    records = machine.observation.stages.records
    return [
        (record.start_ticks, record.end_ticks)
        for record in records
        if record.stage == staged_decoder.ALGORITHM_STAGE
    ]


def test_the_referee_leaves_a_measured_table_tiers_spans_unchanged():
    pytest.importorskip("relay_bp")
    pytest.importorskip("tesseract_decoder")
    measured = measured_table.MeasuredTableDecoder.Settings(device="a100")
    checked = verify_windows.TesseractCheckedDecoder.Settings(inner=measured)

    unchecked_spans = _algorithm_spans(measured)
    checked_spans = _algorithm_spans(checked)

    assert len(unchecked_spans) > 0
    assert checked_spans == unchecked_spans


def test_a_cancelled_job_delivers_nothing_and_is_not_checked():
    """A cancelled job's decode delivers None (DecoderBase.start).

    The referee passes that None on and reaches no verdict, although the
    job carries a window model it could have checked.
    """
    pytest.importorskip("tesseract_decoder")
    circuit = windows.memory_circuit(3, 3, 0.001)
    requirement = fault_models.LINKED_FAULT_MODELS_REQUIRED
    model = windows.whole_circuit_window(circuit, 3, requirement)
    events, _observables = windows.sampled_shots(circuit, 1, 3)
    job = windows.job_for(model, events[0])
    job.cancelled = True
    inner = decoders.PresetLatencyDecoder(2.0)
    referee = verify_windows.TesseractCheckedDecoder(inner)
    verdicts = []
    delivered = []

    def note_verdict(window_key, is_agreement) -> None:
        verdicts.append((window_key, is_agreement))

    referee.window_checked.connect(note_verdict)
    engine = engine_module.Engine()

    referee.start(job, engine, delivered.append)
    engine.run()

    assert delivered == [None]
    assert verdicts == []


def test_the_referee_record_reads_as_the_decoder_it_checks():
    """The results name the tier by the inner decoder's word."""
    inner = decoders.PresetLatencyDecoder.Settings(2.0)
    checked = verify_windows.TesseractCheckedDecoder.Settings(inner=inner)

    assert checked.name == inner.name
    assert checked.weight_step is None


def test_the_referee_record_builds_the_referee_around_its_decoder():
    pytest.importorskip("tesseract_decoder")
    inner = decoders.PresetLatencyDecoder.Settings(2.0)
    checked = verify_windows.TesseractCheckedDecoder.Settings(inner=inner)

    referee = checked.build()

    assert isinstance(referee, verify_windows.TesseractCheckedDecoder)
    assert isinstance(referee.inner, decoders.PresetLatencyDecoder)


class _SizeBlindDecoder(decoders.PresetLatencyDecoder):
    """A user row that predicts without reading the syndrome's size."""

    def decode(self, job):
        return decoding_records.DecodeResult(
            job.operation_id, job.window_id, logical_observables=(0,)
        )


def test_the_referee_refuses_a_syndrome_that_does_not_fit_its_model():
    """A skipped window would leave the audit short with nothing said."""
    circuit = windows.memory_circuit(3, 3, 0.001)
    requirement = fault_models.LINKED_FAULT_MODELS_REQUIRED
    model = windows.whole_circuit_window(circuit, 3, requirement)
    events, _observables = windows.sampled_shots(circuit, 1, 3)
    job = windows.job_for(model, events[0])
    fragment = job.payloads[0]
    short_bits = fragment.bits[:-1]
    job.payloads[0] = dataclasses.replace(
        fragment, bits=short_bits, size_bits=len(short_bits)
    )
    inner = _SizeBlindDecoder(1.0)
    referee = verify_windows.TesseractCheckedDecoder(inner)

    with pytest.raises(ValueError, match="do not match the window error"):
        referee.decode(job)


def test_the_referee_keeps_the_checked_decoders_seed_paths():
    """A checked decoder draws what it draws unchecked, under one run seed.

    The seed walk derives a component's seed from its path
    (decsim/seeding.py), so the inner decoder's children keep theirs.
    """
    pytest.importorskip("tesseract_decoder")
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=1.0
    )
    inner = matching.build()
    referee = verify_windows.TesseractCheckedDecoder(inner)

    inner_paths = _child_paths(inner)
    referee_paths = _child_paths(referee)

    assert inner_paths
    assert inner_paths <= referee_paths


def _child_paths(component) -> set:
    paths = set()
    for child in component.run_seed_children():
        paths.add(child.relative_path)
    return paths

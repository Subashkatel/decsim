"""The confidence wrappers' law: the base decode, with the signal's soft output.

Toshio et al. 2510.25222 Sec. III A: the weak decoder computes its soft
output during decoding; the correction and the observables stay the
base decoder's, the signal only supplies the confidence.
"""

import decsim.confidence.decoder as confidence_decoder
import decsim.decoders.decoder as decoder_module
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoding as decoding_records

SOURCE = decoding_records.SoftOutputSource(
    method="stub",
    cluster_origin="stub",
    growth_schedule="stub",
    gap_units="natural-log",
    correction="none",
    weight_step_natural_log=None,
    references=(),
)


class _Base(decoder_module.DecoderBase):
    """A measured row that answers one prediction in no time at all."""

    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED

    def latency(self, job) -> int:
        del job
        return 1

    def occupancy(self, job):
        del job
        return None

    def decode(self, job) -> decoding_records.DecodeResult:
        return decoding_records.DecodeResult(
            job.op_id, job.window_id, logical_observables=(1,)
        )

    def decode_timed(self, job) -> tuple:
        result = self.decode(job)
        return result, 0


class _Metric:
    def __init__(self, gap: float) -> None:
        self.gap = gap

    def evaluate(self, syndrome) -> decoding_records.SoftOutput:
        del syndrome
        return decoding_records.SoftOutput(gap=self.gap, source=SOURCE)


class _Signal:
    """A signal with one metric for every model, or none at all."""

    source = SOURCE
    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED

    def __init__(self, metric) -> None:
        self.metric = metric

    def metric_for(self, model):
        del model
        return self.metric


class _Model:
    """A window model the wrapper caches its metric by (weakly referenced)."""


def _job() -> decoding_records.DecodeJob:
    model = _Model()
    return decoding_records.DecodeJob(
        op_id=1, window_id=0, n_rounds=3, dem=model
    )


def test_the_wrapper_attaches_the_signals_soft_output_to_the_base_result():
    metric = _Metric(3.5)
    signal = _Signal(metric)
    base = _Base()
    wrapper = confidence_decoder.SoftOutputDecoder(base, signal)
    job = _job()
    result = wrapper.decode(job)
    assert result.logical_observables == (1,)
    assert result.soft_output.gap == 3.5
    assert result.soft_output.source is SOURCE


def test_a_model_the_signal_cannot_measure_leaves_the_result_without_one():
    signal = _Signal(None)
    base = _Base()
    wrapper = confidence_decoder.SoftOutputDecoder(base, signal)
    job = _job()
    result = wrapper.decode(job)
    assert result.logical_observables == (1,)
    assert result.soft_output is None


def test_the_wrapper_seeds_its_base_and_its_signal_under_the_recorded_names():
    signal = _Signal(None)
    base = _Base()
    wrapper = confidence_decoder.SoftOutputDecoder(base, signal)
    children = wrapper.run_seed_children()
    names = []
    for child in children:
        segment = child.relative_path[0]
        names.append(segment.value)
    assert names == ["base", "metric_cls"]
    assert children[0].child is base
    assert children[1].child is signal

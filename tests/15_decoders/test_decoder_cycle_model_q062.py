"""Blind behavior tests for the Q-062(c) decoder cycle model."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

import decsim.decoder_cycle_model as cycle_model_module
from decsim.decoder_cycle_model import (
    CycleQuantityBasis,
    DecoderCycleModel,
    DecoderPipelineStage,
    StageCycleConfig,
    delegated_latency_model,
    lookup_table_published_total_model,
)
from decsim.decoders import CycleModelDecoder, NO_FAULT_MODEL_REQUIRED


def _stage(cycles, basis=CycleQuantityBasis.PER_JOB, label="configured"):
    if cycles == 0:
        return StageCycleConfig(
            cycles=cycles,
            basis=basis,
            zero_justification=f"{label} is deliberately free in this card.",
        )
    return StageCycleConfig(
        cycles=cycles,
        basis=basis,
        source=f"{label} cycle budget source.",
    )


def _model(
    *,
    stages=None,
    frequency_mhz=250.0,
    frequency_source="Configured FPGA clock source.",
    include_inner_latency=True,
    inner_latency_source="Wrapped decoder latency source.",
    inner_latency_zero_justification=None,
    model_name="test_card",
):
    stages = stages or (_stage(0),) * 5
    return DecoderCycleModel(
        fetch=stages[0],
        decode=stages[1],
        execute=stages[2],
        memory=stages[3],
        writeback=stages[4],
        frequency_mhz=frequency_mhz,
        frequency_source=frequency_source,
        include_inner_latency=include_inner_latency,
        inner_latency_source=inner_latency_source,
        inner_latency_zero_justification=inner_latency_zero_justification,
        model_name=model_name,
    )


def test_stage_and_basis_vocabularies_are_closed_and_ordered():
    """The card exposes exactly five ordered stages and two cycle bases."""
    assert tuple(DecoderPipelineStage) == (
        DecoderPipelineStage.FETCH,
        DecoderPipelineStage.DECODE,
        DecoderPipelineStage.EXECUTE,
        DecoderPipelineStage.MEMORY,
        DecoderPipelineStage.WRITEBACK,
    )
    assert tuple(stage.value for stage in DecoderPipelineStage) == (
        "fetch",
        "decode",
        "execute",
        "memory",
        "writeback",
    )
    assert tuple(CycleQuantityBasis) == (
        CycleQuantityBasis.PER_JOB,
        CycleQuantityBasis.PER_ROUND,
    )


def test_stage_rows_charge_job_and_round_bases_as_exact_integers():
    """Per-job work is charged once while per-round work scales with job size."""
    card = _model(
        stages=(
            _stage(2),
            _stage(3, CycleQuantityBasis.PER_ROUND),
            _stage(5),
            _stage(7, CycleQuantityBasis.PER_ROUND),
            _stage(11),
        )
    )

    rows = card.stage_cycles(SimpleNamespace(n_rounds=4.0))

    assert rows == (
        (DecoderPipelineStage.FETCH, 2),
        (DecoderPipelineStage.DECODE, 12),
        (DecoderPipelineStage.EXECUTE, 5),
        (DecoderPipelineStage.MEMORY, 28),
        (DecoderPipelineStage.WRITEBACK, 11),
    )
    assert all(type(charged_cycles) is int for _, charged_cycles in rows)
    assert card.total_cycles(SimpleNamespace(n_rounds=4)) == 58


@pytest.mark.parametrize("cycles", [-1, 1.5, float("nan"), float("inf")])
def test_stage_cycle_counts_refuse_non_whole_or_negative_values(cycles):
    """A stage refuses cycle counts that cannot be nonnegative whole cycles."""
    with pytest.raises(ValueError):
        StageCycleConfig(
            cycles=cycles,
            basis=CycleQuantityBasis.PER_JOB,
            source="Measured positive cycle source.",
        )


def test_stage_cycle_count_normalizes_a_semantically_whole_value():
    """A semantically whole numeric cycle count is stored as an integer."""
    stage = StageCycleConfig(
        cycles=3.0,
        basis=CycleQuantityBasis.PER_JOB,
        source="Measured positive cycle source.",
    )
    assert stage.cycles == 3
    assert type(stage.cycles) is int


@pytest.mark.parametrize("round_count", [-1, 2.5, float("nan"), float("inf")])
def test_per_round_work_refuses_invalid_job_sizes(round_count):
    """Per-round pricing refuses negative, fractional, or nonfinite job sizes."""
    card = _model(
        stages=(
            _stage(0),
            _stage(1, CycleQuantityBasis.PER_ROUND),
            _stage(0),
            _stage(0),
            _stage(0),
        )
    )
    with pytest.raises(ValueError):
        card.total_cycles(SimpleNamespace(n_rounds=round_count))


def test_stage_basis_must_be_one_of_the_closed_basis_members():
    """A stage refuses strings and other values outside the basis enumeration."""
    with pytest.raises(ValueError):
        StageCycleConfig(
            cycles=1,
            basis="per_job",
            source="Measured positive cycle source.",
        )


@pytest.mark.parametrize(
    "cycles,source,zero_justification",
    [
        (0, None, None),
        (0, "Unexpected source.", "Explicit zero reason."),
        (0, "Unexpected source.", None),
        (1, None, None),
        (1, "Measured source.", "Unexpected zero reason."),
    ],
)
def test_stage_source_and_zero_justification_are_exclusive(
    cycles,
    source,
    zero_justification,
):
    """Zero and positive stages require their matching exclusive attribution."""
    with pytest.raises(ValueError):
        StageCycleConfig(
            cycles=cycles,
            basis=CycleQuantityBasis.PER_JOB,
            source=source,
            zero_justification=zero_justification,
        )


@pytest.mark.parametrize("bad_line", ["", "   ", "source\nsecond", "source\rsecond"])
def test_stage_provenance_refuses_empty_or_multiline_text(bad_line):
    """Stage attribution must be a nonempty single line."""
    with pytest.raises(ValueError):
        StageCycleConfig(
            cycles=1,
            basis=CycleQuantityBasis.PER_JOB,
            source=bad_line,
        )
    with pytest.raises(ValueError):
        StageCycleConfig(
            cycles=0,
            basis=CycleQuantityBasis.PER_JOB,
            zero_justification=bad_line,
        )


def test_provenance_lines_are_stripped_before_storage():
    """Valid provenance is normalized to its stripped one-line form."""
    positive = StageCycleConfig(
        1,
        CycleQuantityBasis.PER_JOB,
        source="  measured source  ",
    )
    zero = StageCycleConfig(
        0,
        CycleQuantityBasis.PER_JOB,
        zero_justification="  explicit zero  ",
    )
    card = _model(
        frequency_source="  clock source  ",
        inner_latency_source="  inner source  ",
    )
    assert positive.source == "measured source"
    assert zero.zero_justification == "explicit zero"
    assert card.frequency_source == "clock source"
    assert card.inner_latency_source == "inner source"


@pytest.mark.parametrize("frequency_mhz", [0, -1, float("nan"), float("inf")])
def test_frequency_must_be_finite_and_positive(frequency_mhz):
    """The hardware clock refuses nonpositive and nonfinite frequencies."""
    with pytest.raises(ValueError):
        _model(frequency_mhz=frequency_mhz)


@pytest.mark.parametrize("bad_line", ["", "   ", "clock\nsource", "clock\rsource"])
def test_frequency_source_must_be_a_nonempty_single_line(bad_line):
    """Every configured frequency carries readable one-line attribution."""
    with pytest.raises(ValueError):
        _model(frequency_source=bad_line)


def test_frequency_refuses_a_clock_where_one_cycle_rounds_to_zero_ticks():
    """A configured positive cycle can never become a free simulated hop."""
    with pytest.raises(ValueError, match="one cycle must convert to at least one tick"):
        _model(frequency_mhz=2_000_000.0)


@pytest.mark.parametrize(
    "include_inner_latency,inner_source,zero_reason",
    [
        (True, None, None),
        (True, "Inner source.", "Unexpected zero reason."),
        (True, None, "Wrong mode reason."),
        (False, None, None),
        (False, "Wrong mode source.", None),
        (False, "Wrong mode source.", "Zero reason."),
    ],
)
def test_inner_latency_attribution_matches_the_flag_exclusively(
    include_inner_latency,
    inner_source,
    zero_reason,
):
    """Included and excluded inner latency each require only their matching text."""
    with pytest.raises(ValueError):
        _model(
            include_inner_latency=include_inner_latency,
            inner_latency_source=inner_source,
            inner_latency_zero_justification=zero_reason,
        )


@pytest.mark.parametrize("include_inner_latency", [0, 1, None, "yes"])
def test_inner_latency_flag_must_be_an_exact_boolean(include_inner_latency):
    """The inner-latency decision refuses truthy and falsey substitutes."""
    with pytest.raises(ValueError):
        _model(
            include_inner_latency=include_inner_latency,
            inner_latency_source=None,
            inner_latency_zero_justification="Explicit exclusion reason.",
        )


@pytest.mark.parametrize("bad_line", ["", "   ", "inner\nsource", "inner\rsource"])
def test_inner_latency_provenance_must_be_a_nonempty_single_line(bad_line):
    """Both inner-latency modes require readable one-line attribution."""
    with pytest.raises(ValueError):
        _model(inner_latency_source=bad_line)
    with pytest.raises(ValueError):
        _model(
            include_inner_latency=False,
            inner_latency_source=None,
            inner_latency_zero_justification=bad_line,
        )


def test_cycles_are_summed_before_exactly_one_time_conversion(monkeypatch):
    """The complete cycle budget is converted once rather than stage by stage."""
    card = _model(
        stages=tuple(_stage(cycles) for cycles in (1, 2, 3, 4, 5)),
        frequency_mhz=100.0,
        include_inner_latency=False,
        inner_latency_source=None,
        inner_latency_zero_justification="The stage card owns all latency.",
    )
    conversion_inputs = []

    def recording_conversion(microseconds):
        conversion_inputs.append(microseconds)
        return 12345

    monkeypatch.setattr(cycle_model_module, "us", recording_conversion)

    assert card.latency_ticks(SimpleNamespace(n_rounds=9), 0) == 12345
    assert conversion_inputs == [15 / 100.0]


def test_runtime_guard_rejects_positive_work_that_converts_to_zero(monkeypatch):
    """A standing runtime check prevents future rounding changes from making work free."""
    card = _model(stages=(_stage(1),) + (_stage(0),) * 4)
    monkeypatch.setattr(cycle_model_module, "us", lambda _microseconds: 0)

    with pytest.raises(RuntimeError, match="positive decoder work converted to zero ticks"):
        card.latency_ticks(SimpleNamespace(n_rounds=1), 0)


def test_excluded_inner_latency_cannot_be_supplied_as_hidden_work():
    """A pure stage card refuses any nonzero hidden inner latency."""
    card = _model(
        stages=(_stage(1),) + (_stage(0),) * 4,
        include_inner_latency=False,
        inner_latency_source=None,
        inner_latency_zero_justification="The stage card owns all latency.",
    )
    with pytest.raises(ValueError, match="inner_latency_ticks must be zero"):
        card.latency_ticks(SimpleNamespace(n_rounds=1), 1)


def test_json_reports_every_knob_and_each_stage_attribution():
    """The JSON card fully reports cycles, bases, sources, zeros, clock, and inner mode."""
    card = _model(
        stages=(
            _stage(2, label="fetch"),
            _stage(0, label="decode"),
            _stage(3, CycleQuantityBasis.PER_ROUND, label="execute"),
            _stage(0, label="memory"),
            _stage(0, label="writeback"),
        ),
        frequency_mhz=500.0,
        frequency_source="Measured ASIC clock.",
        include_inner_latency=False,
        inner_latency_source=None,
        inner_latency_zero_justification="This card owns the full decode price.",
        model_name="fully_attributed",
    )

    assert card.to_json_value() == {
        "model_name": "fully_attributed",
        "stages": [
            {
                "stage": "fetch",
                "cycles": 2,
                "basis": "per_job",
                "source": "fetch cycle budget source.",
                "zero_justification": None,
            },
            {
                "stage": "decode",
                "cycles": 0,
                "basis": "per_job",
                "source": None,
                "zero_justification": "decode is deliberately free in this card.",
            },
            {
                "stage": "execute",
                "cycles": 3,
                "basis": "per_round",
                "source": "execute cycle budget source.",
                "zero_justification": None,
            },
            {
                "stage": "memory",
                "cycles": 0,
                "basis": "per_job",
                "source": None,
                "zero_justification": "memory is deliberately free in this card.",
            },
            {
                "stage": "writeback",
                "cycles": 0,
                "basis": "per_job",
                "source": None,
                "zero_justification": "writeback is deliberately free in this card.",
            },
        ],
        "frequency_mhz": 500.0,
        "frequency_source": "Measured ASIC clock.",
        "include_inner_latency": False,
        "inner_latency_zero_justification": "This card owns the full decode price.",
    }


def test_cycle_cards_are_frozen_after_validation():
    """Validated stage and model cards cannot change underneath a running decoder."""
    stage = _stage(1)
    card = _model(stages=(stage,) * 5)
    with pytest.raises(FrozenInstanceError):
        stage.cycles = 2
    with pytest.raises(FrozenInstanceError):
        card.frequency_mhz = 1000.0


def test_delegated_default_is_exactly_the_wrapped_decoder_latency():
    """The default five-zero-stage card preserves wrapped latency for every job size."""
    class InnerDecoder:
        fault_model_requirement = NO_FAULT_MODEL_REQUIRED

        def __init__(self):
            self.latency_calls = []

        def latency(self, job):
            self.latency_calls.append(job.n_rounds)
            return 7000 + job.n_rounds

        def decode(self, job):
            return job

    inner = InnerDecoder()
    card = delegated_latency_model()
    wrapped = CycleModelDecoder(inner, card)

    assert card.model_name == "delegated_latency"
    assert card.frequency_mhz == 250.0
    assert card.include_inner_latency is True
    json_value = card.to_json_value()
    assert json_value["inner_latency_source"] == card.inner_latency_source
    assert "inner_latency_zero_justification" not in json_value
    assert card.stage_cycles(SimpleNamespace(n_rounds=999)) == tuple(
        (stage, 0) for stage in DecoderPipelineStage
    )
    for round_count in (0, 1, 17, 1000):
        job = SimpleNamespace(n_rounds=round_count)
        assert wrapped.latency(job) == 7000 + round_count
    assert inner.latency_calls == [0, 1, 17, 1000]


def test_lilliput_card_reproduces_seven_cycles_at_250_mhz():
    """The published lookup card charges only aggregate execute and yields 28000 ticks."""
    card = lookup_table_published_total_model()
    job = SimpleNamespace(n_rounds=2)

    assert card.model_name == "lilliput_d3_m2_embedded_memory"
    assert card.frequency_mhz == 250.0
    assert card.include_inner_latency is False
    assert card.stage_cycles(job) == (
        (DecoderPipelineStage.FETCH, 0),
        (DecoderPipelineStage.DECODE, 0),
        (DecoderPipelineStage.EXECUTE, 7),
        (DecoderPipelineStage.MEMORY, 0),
        (DecoderPipelineStage.WRITEBACK, 0),
    )
    assert card.total_cycles(job) == 7
    assert card.latency_ticks(job, 0) == 28000
    for stage_row in card.to_json_value()["stages"]:
        if stage_row["stage"] == "execute":
            assert stage_row["source"] is not None
            assert stage_row["zero_justification"] is None
        else:
            assert stage_row["source"] is None
            assert stage_row["zero_justification"] is not None


def test_adapter_delegates_decode_fault_requirement_and_seed_child():
    """The timing adapter preserves functional output, fault needs, and seed reachability."""
    decode_result = object()

    class FunctionalInner:
        fault_model_requirement = NO_FAULT_MODEL_REQUIRED

        def latency(self, job):
            return 99

        def decode(self, job):
            self.decoded_job = job
            return decode_result

    inner = FunctionalInner()
    wrapped = CycleModelDecoder(inner, delegated_latency_model())
    job = SimpleNamespace(n_rounds=3)

    assert wrapped.decode(job) is decode_result
    assert inner.decoded_job is job
    assert wrapped.fault_model_requirement is inner.fault_model_requirement
    children = wrapped.run_seed_children()
    assert len(children) == 1
    assert children[0].child is inner
    assert len(children[0].relative_path) == 1
    assert children[0].relative_path[0].kind == "field"
    assert children[0].relative_path[0].value == "inner"


def test_adapter_never_calls_inner_latency_when_the_card_excludes_it():
    """A pure cycle card has no hidden call to the wrapped decoder's latency model."""
    class InnerWhoseLatencyMustStayUnused:
        fault_model_requirement = NO_FAULT_MODEL_REQUIRED

        def latency(self, job):
            raise AssertionError("inner latency must not be consulted")

        def decode(self, job):
            return job

    card = lookup_table_published_total_model()
    wrapped = CycleModelDecoder(InnerWhoseLatencyMustStayUnused(), card)

    assert wrapped.latency(SimpleNamespace(n_rounds=2)) == 28000

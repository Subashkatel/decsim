"""The run plan: the rows the yaml names and the defaults it is given.

Two window keys carry a null yaml default whose meaning the yaml reader
derives from whether the switching slot is filled and from its strong
window row's declared facts, never from a kind string. Both are pinned
here through build_plan, on settings read as a yaml would leave them.
"""

import dataclasses
from typing import Optional

import pytest
import stim

import decsim.build.escalation as escalation_build
import decsim.build.plan as plan_build
import decsim.confidence.complementary as complementary
import decsim.config as config
import decsim.controller.policies as policies
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.settings as event_settings
import decsim.engine as engine_module
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.frontends.settings as workload_settings
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.qpu.streaming_stim_device as streaming_stim_device
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.records.circuits as circuit_records
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.workload as workload_records
import decsim.settings as machine_settings
import decsim.trace_source as trace_source
import decsim.windows.boundary_policies as boundary_policies
import decsim.windows.schemes.sliding as sliding_scheme
import decsim.windows.settings as window_settings
import tests.declared_run as declared_run
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

# the decoder manager section a run gets when the test names none
NO_BULK_STRONG = decoder_settings.DecoderManagerSettings()
# the detection events section a run gets when the test names none
CONTROLLER_FORMS = event_settings.DetectionEventSettings()
# the windows section a yaml writes when it names the scheme alone
WINDOWS_SECTION = {
    "kind": "sliding",
    "commit_rounds": None,
    "buffer_rounds": None,
}
NO_CLOCKS = config.ClockSettings({})


def _windows(switching=None, **keys):
    """The windows a yaml section with these keys reads as, beside switching."""
    section = {**WINDOWS_SECTION, **keys}
    return window_settings.WindowSettings.from_yaml(
        section, NO_CLOCKS, switching
    )


def _plan(
    *,
    switching=None,
    windows=None,
    qpu=None,
    idle_policy=None,
    workload=None,
    decoder_manager=NO_BULK_STRONG,
    detection_events=CONTROLLER_FORMS,
):
    """The plan of a six-round memory run with the given sections."""
    if idle_policy is None:
        idle_policy = policies.SeparateDecodeJobsSettings()
    if windows is None:
        windows = _windows(switching)
    if qpu is None:
        qpu = declared_run.declared_qpu()
    if workload is None:
        workload = declared_run.declared_workload(None, 6)
    links = declared_run.declared_profile()
    controller = declared_run.declared_controller()
    frame = declared_run.declared_frame()
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        links=links,
        controller=controller,
        pauli_frame=frame,
        windows=windows,
        idle_policy=idle_policy,
        decoder_manager=decoder_manager,
        detection_events=detection_events,
    )
    if switching is not None:
        settings = _with_switching(settings, switching)
    engine = engine_module.Engine()
    switching_part = escalation_build.build_switching(
        settings.switching, settings.weak_decoder, engine
    )
    return plan_build.build_plan(
        settings.qpu,
        settings.workload,
        settings.windows,
        settings.idle_policy,
        settings.detection_events,
        settings.switching,
        settings.decoder_manager.bulk_strong,
        switching_part,
    )


def _with_switching(settings, switching):
    """The settings with both decoders and the switching slot filled."""
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings()
    decoder = decoder_settings.DecoderPoolSettings(algorithm=matching)
    return dataclasses.replace(
        settings,
        weak_decoder=decoder,
        strong_decoder=decoder,
        switching=switching,
    )


def _switching(**changes):
    """A switching slot with a fixed threshold and a gap signal."""
    confidence = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=20.0
    )
    return escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold, **changes
    )


def test_bulk_strong_is_refused_when_the_rounds_carry_bits():
    """A merged strong decode carries timing alone; bits would be dropped."""
    bits_source = syndrome_devices.SyndromeBitDevice.Settings()
    bits = qpu_settings.QpuSettings(source=bits_source, distance=3)
    switching = _switching()
    bulk = decoder_settings.DecoderManagerSettings(bulk_strong=True)

    with pytest.raises(ValueError, match="qpu.kind syndrome_bits"):
        _plan(qpu=bits, switching=switching, decoder_manager=bulk)


def test_bulk_strong_is_refused_where_no_strong_pool_merges():
    bulk = decoder_settings.DecoderManagerSettings(bulk_strong=True)

    with pytest.raises(ValueError, match="has no strong pool"):
        _plan(decoder_manager=bulk)


def test_bulk_strong_is_built_beside_an_explicitly_empty_model_source():
    empty_models = syndrome_devices.NO_WINDOW_MODELS
    models_record = declared_run.GivenSource(empty_models)
    timing = qpu_settings.QpuSettings(error_model_provider=models_record)
    switching = _switching()
    bulk = decoder_settings.DecoderManagerSettings(bulk_strong=True)

    plan = _plan(qpu=timing, switching=switching, decoder_manager=bulk)

    assert plan.error_model_provider is empty_models


def test_bulk_strong_is_built_when_the_rounds_carry_timing_alone():
    switching = _switching()
    bulk = decoder_settings.DecoderManagerSettings(bulk_strong=True)

    plan = _plan(switching=switching, decoder_manager=bulk)

    assert not plan.device.emits_bit_values


def test_a_row_shaped_by_the_code_card_is_built_with_the_runs_card():
    """The source record names no card, so the build supplies the run's."""
    bits_source = syndrome_devices.SyndromeBitDevice.Settings()
    bits = qpu_settings.QpuSettings(source=bits_source, distance=5)

    plan = _plan(qpu=bits)

    assert plan.device.code is plan.code


def test_a_stim_source_is_its_own_window_model_source():
    stim_record = stim_device.StimDevice.Settings()
    stim_source = qpu_settings.QpuSettings(source=stim_record, distance=3)

    plan = _plan(qpu=stim_source)

    assert plan.error_model_provider is plan.device


def test_a_circuit_less_source_wires_the_model_source_that_builds_nothing():
    plan = _plan()

    assert plan.error_model_provider is syndrome_devices.NO_WINDOW_MODELS


def _live_fragments_workload():
    """One live segment on a stream, its fragments a single measurement."""
    segment = program_records.Operation(
        1, "prefix", ("p",), patches=("p",), stream_id=100, stream_offset=0
    )
    fragment = stim.Circuit("M 0")
    program = circuit_records.RepeatedStimCircuit(
        fragment, fragment, fragment, fragment
    )
    workload = workload_records.Workload((segment,), {1: 3}, program)
    return workload_settings.WorkloadSettings.running(workload)


RECORDED_SOURCE = stim_device.RecordedStimDevice.Settings()
STREAMING_SOURCE = streaming_stim_device.StreamingStimDevice.Settings()


@pytest.mark.parametrize(
    "record, sentence",
    [
        (RECORDED_SOURCE, "required positional arguments: 'measurements'"),
        (STREAMING_SOURCE, "required positional argument: 'programs'"),
    ],
)
def test_a_source_the_workload_cannot_fill_stops_its_call(record, sentence):
    """Python's own call names the argument the source is not given."""
    source = qpu_settings.QpuSettings(source=record, distance=3)

    with pytest.raises(TypeError, match=sentence):
        _plan(qpu=source)


def test_live_fragments_under_a_finite_circuit_source_stop_its_call():
    stim_record = stim_device.StimDevice.Settings()
    source = qpu_settings.QpuSettings(source=stim_record, distance=3)
    workload = _live_fragments_workload()

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _plan(qpu=source, workload=workload)


def test_live_fragments_build_the_streaming_source_from_its_record():
    streaming_record = streaming_stim_device.StreamingStimDevice.Settings()
    source = qpu_settings.QpuSettings(source=streaming_record, distance=3)
    workload = _live_fragments_workload()

    plan = _plan(qpu=source, workload=workload)

    assert isinstance(plan.device, streaming_stim_device.StreamingStimDevice)


def _lookback_workload():
    """Ten rounds of one qubit; round r's detector is rec[-1] ^ rec[-3]."""
    circuit = stim.Circuit(
        "R 0\nM 0\nDETECTOR rec[-1]\nM 0\nDETECTOR rec[-1]\n"
        "REPEAT 8 {\nM 0\nDETECTOR rec[-1] rec[-3]\n}\n"
    )
    measurement_rounds = {index: index + 1 for index in range(10)}
    physical = workload_records.FiniteCircuit(circuit, measurement_rounds)
    operation = declared_run.memory_operation()
    workload = workload_records.Workload((operation,), {1: 10}, physical)
    return workload_settings.WorkloadSettings.running(workload)


def _second_window_strong_hold(plan) -> tuple:
    """The rounds the plan holds for a strong redo of the second window."""
    holds = dict(plan.run_plan.buffering.potential_holds)
    potential = decoding_records.PotentialStrong((1, 1))
    return holds[potential]


SWITCHING = declared_run.declared_switching()
STRONG_SIDE_FORMS = event_settings.DetectionEventSettings(
    formed_at=("weak_decoder", "strong_decoder")
)


def test_a_forming_strong_read_holds_what_its_circuit_reads_before_it():
    """The second window commits from round 4, whose detector reads 2."""
    stim_record = stim_device.StimDevice.Settings()
    source = qpu_settings.QpuSettings(source=stim_record, distance=3)
    workload = _lookback_workload()

    plan = _plan(
        qpu=source,
        workload=workload,
        switching=SWITCHING,
        detection_events=STRONG_SIDE_FORMS,
    )

    held = _second_window_strong_hold(plan)
    assert held[:3] == ((1, 2), (1, 3), (1, 4))


def test_a_source_with_no_recipes_holds_nothing_before_a_read():
    bits_record = syndrome_devices.SyndromeBitDevice.Settings()
    source = qpu_settings.QpuSettings(source=bits_record, distance=3)
    workload = _lookback_workload()

    plan = _plan(
        qpu=source,
        workload=workload,
        switching=SWITCHING,
        detection_events=STRONG_SIDE_FORMS,
    )

    held = _second_window_strong_hold(plan)
    assert held[0] == (1, 4)


def test_a_restart_read_holds_no_round_before_its_restart_start():
    """Window 3 restarts at round 7, re-reading one buffer of three.

    Round 7's detector reads round 5, inside the strong region, whose
    request keeps its rounds until the restart window commits, so the
    restart read holds none before round 7.
    """
    stim_record = stim_device.StimDevice.Settings()
    source = qpu_settings.QpuSettings(source=stim_record, distance=3)
    workload = _lookback_workload()
    double_window = strong_window_shapes.DoubleWindow.Settings()
    switching = declared_run.declared_switching(strong_window=double_window)
    weak_decoder_forms = event_settings.DetectionEventSettings(
        formed_at=("weak_decoder", "strong_decoder")
    )

    plan = _plan(
        qpu=source,
        workload=workload,
        switching=switching,
        detection_events=weak_decoder_forms,
    )

    holds = dict(plan.run_plan.buffering.weak_holds)
    restart = decoding_records.PotentialRestart((1, 3))
    assert holds[restart][0] == (1, 7)


def test_a_run_that_never_escalates_gets_the_flush_tail():
    plan = _plan()

    assert plan.scheme.terminal_policy == "flush"
    assert plan.scheme.has_trailing_tail_context is False


def test_a_run_that_may_escalate_gets_the_lookahead_tail():
    """A strong recovery reads past the last window's commit."""
    switching = _switching()

    plan = _plan(switching=switching)

    assert plan.scheme.terminal_policy == "lookahead"
    assert plan.scheme.has_trailing_tail_context is True


def test_a_terminal_policy_written_in_the_section_wins_over_the_default():
    windows = _windows(terminal_policy="lookahead")

    plan = _plan(windows=windows)

    assert plan.scheme.terminal_policy == "lookahead"


def test_a_run_that_never_escalates_ships_boundaries_eagerly():
    plan = _plan()

    assert isinstance(plan.boundary_policy, boundary_policies.Eager)


def test_a_serial_escalation_holds_its_boundaries():
    """Descendants wait out an escalation, so nothing ships until final."""
    switching = _switching()

    plan = _plan(switching=switching)

    assert isinstance(plan.boundary_policy, boundary_policies.Held)


def test_a_boundaries_row_written_in_the_section_wins_over_the_default():
    windows = _windows(boundaries="held")

    plan = _plan(windows=windows)

    assert isinstance(plan.boundary_policy, boundary_policies.Held)


def test_a_windows_kind_that_names_no_row_is_refused():
    with pytest.raises(ValueError) as refusal:
        _windows(kind="diagonal")

    assert "windows.kind" in str(refusal.value)


class _SettingsRecordingScheme(sliding_scheme.SlidingWindowScheme):
    """A sliding scheme that keeps the Settings record it is built with."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        commit_rounds: Optional[int] = None
        buffer_rounds: Optional[int] = None
        stride_rounds: int = 1
        name = "recording"

        def build(self, terminal_policy) -> "_SettingsRecordingScheme":
            del terminal_policy
            return _SettingsRecordingScheme(self)

    def __init__(self, settings) -> None:
        sliding_scheme.SlidingWindowScheme.__init__(self)
        self.settings = settings


def test_a_scheme_row_with_settings_is_built_with_its_record():
    own_settings = _SettingsRecordingScheme.Settings(stride_rounds=2)
    windows = window_settings.WindowSettings(scheme=own_settings)

    plan = _plan(windows=windows)

    assert plan.scheme.settings is own_settings


class _SettingsRecordingIdlePolicy(policies.Ignore):
    """An idle policy that keeps the record it is built from."""

    def __init__(self, settings) -> None:
        self.settings = settings


@dataclasses.dataclass(frozen=True)
class _RecordingIdlePolicySettings:
    every_nth_round: int = 1

    def build(self) -> _SettingsRecordingIdlePolicy:
        return _SettingsRecordingIdlePolicy(self)


def test_the_idle_policy_record_builds_the_plans_policy():
    own_settings = _RecordingIdlePolicySettings(every_nth_round=2)

    plan = _plan(idle_policy=own_settings)

    assert plan.idle_policy.settings is own_settings


def test_a_scheme_that_declares_none_of_the_three_facts_is_refused():
    """Every row answers what the plan and the policy read off it."""
    silent = _SilentScheme.Settings()
    windows = window_settings.WindowSettings(scheme=silent)

    with pytest.raises(ValueError) as refusal:
        _plan(windows=windows)

    assert "has_trailing_tail_context" in str(refusal.value)


class _SilentScheme:
    """A scheme row that declares nothing the plan reads."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        commit_rounds: Optional[int] = None
        buffer_rounds: Optional[int] = None
        name = "silent"

        def build(self, terminal_policy) -> "_SilentScheme":
            del terminal_policy
            return _SilentScheme()

    def plan_operation(self, *arguments, **sizes):
        """Never reached: the plan refuses this row first."""
        del arguments, sizes
        raise AssertionError("the plan should have refused this row")


# ---- the default boundary row lives on the rows that decide it


class _OutsideBoundaryPolicy:
    """A boundary policy row written outside decsim; it ships when final."""

    ships_provisional_boundaries = False

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The outside row's record, which takes no setting."""

        def build(self) -> "_OutsideBoundaryPolicy":
            return _OutsideBoundaryPolicy()

    def on_commit(self, window, *, final: bool) -> bool:
        """Ship when the committing result is final."""
        del window
        return final


class _OutsideShape:
    """A strong window shape written outside decsim that absorbs nothing."""

    absorbs_weak_windows = False
    default_boundary_policy = "outside_boundaries"
    window_absorbed = trace_source.SILENT

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The outside row's record: its two facts and its build."""

        name = "outside_shape"
        absorbs_weak_windows = False
        default_boundary_policy = "outside_boundaries"
        restart_reread_buffer_regions = 1

        def build(self, engine) -> "_OutsideShape":
            return _OutsideShape(engine)

    def __init__(self, engine) -> None:
        del engine

    def plan(self, weak_job):
        """Never reached: the test builds the plan and runs nothing."""
        del weak_job
        raise AssertionError("the test never escalates a window")


def _outside_shape_switching():
    """A switching section whose strong window is the outside shape."""
    confidence = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=20.0
    )
    outside_shape = _OutsideShape.Settings()
    return escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold, strong_window=outside_shape
    )


def test_an_outside_strong_window_row_names_the_escalations_default(
    monkeypatch,
):
    """A row that may escalate leaves the default to its strong window."""
    monkeypatch.setitem(
        window_settings.BOUNDARY_POLICIES,
        "outside_boundaries",
        _OutsideBoundaryPolicy,
    )
    switching = _outside_shape_switching()

    plan = _plan(switching=switching)

    assert isinstance(plan.boundary_policy, _OutsideBoundaryPolicy)

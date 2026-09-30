"""The run plan: the rows the yaml names and the defaults the plan derives.

Two window keys carry a null default whose meaning the plan derives
from the escalation row's declared facts, and the plan branches on a
declared fact, never on a kind string.
Both are pinned here through build_plan, on settings shaped as a yaml
would leave them.
"""

import dataclasses

import pytest
import stim

import decsim.build.escalation as escalation_build
import decsim.build.plan as plan_build
import decsim.controller.policies as policies
import decsim.controller.settings as controller_settings
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.settings as event_settings
import decsim.escalation.settings as escalation_settings
import decsim.qpu.settings as qpu_settings
import decsim.qpu.streaming_stim_device as streaming_stim_device
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.records.circuits as circuit_records
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.records.workload as workload_records
import decsim.settings as machine_settings
import decsim.trace_source as trace_source
import decsim.windows.boundary_policies as boundary_policies
import decsim.windows.schemes.sliding as sliding_scheme
import decsim.windows.settings as window_settings
import tests.declared_run as declared_run

# the decoder manager section a run gets when the test names none
NO_BULK_STRONG = decoder_settings.DecoderManagerSettings()
# the detection events section a run gets when the test names none
CONTROLLER_FORMS = event_settings.DetectionEventSettings()


def _plan(
    *,
    escalation=None,
    windows=None,
    qpu=None,
    idle_policy=None,
    workload=None,
    decoder_manager=NO_BULK_STRONG,
    detection_events=CONTROLLER_FORMS,
):
    """The plan of a six-round memory run with the given sections."""
    if idle_policy is None:
        idle_policy = controller_settings.IdlePolicySettings()
    if escalation is None:
        escalation = escalation_settings.EscalationSettings()
    if windows is None:
        windows = window_settings.WindowSettings()
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
        escalation=escalation,
        windows=windows,
        idle_policy=idle_policy,
        decoder_manager=decoder_manager,
        detection_events=detection_events,
    )
    policy = escalation_build.build_escalation_policy(
        escalation, settings.weak_decoder
    )
    return plan_build.build_plan(settings, policy)


def _switching():
    """A switching section with a fixed threshold and a gap signal."""
    return escalation_settings.EscalationSettings(
        kind="switching",
        threshold_source="fixed",
        gap_threshold_nats=2.0,
        confidence="complementary_gap",
    )


def test_bulk_strong_is_refused_when_the_rounds_carry_bits():
    """A merged strong decode carries timing alone; bits would be dropped."""
    bits = qpu_settings.QpuSettings(kind="syndrome_bits", distance=3)
    escalation = _switching()
    bulk = decoder_settings.DecoderManagerSettings(bulk_strong=True)

    with pytest.raises(ValueError, match="qpu.kind syndrome_bits"):
        _plan(qpu=bits, escalation=escalation, decoder_manager=bulk)


def test_bulk_strong_is_refused_where_no_strong_pool_merges():
    bulk = decoder_settings.DecoderManagerSettings(bulk_strong=True)

    with pytest.raises(ValueError, match="has no strong pool"):
        _plan(decoder_manager=bulk)


def test_bulk_strong_is_built_beside_an_explicitly_empty_model_source():
    empty_models = syndrome_devices.NO_WINDOW_MODELS
    timing = qpu_settings.QpuSettings(error_model_provider=empty_models)
    escalation = _switching()
    bulk = decoder_settings.DecoderManagerSettings(bulk_strong=True)

    plan = _plan(qpu=timing, escalation=escalation, decoder_manager=bulk)

    assert plan.error_model_provider is empty_models


def test_bulk_strong_is_built_when_the_rounds_carry_timing_alone():
    escalation = _switching()
    bulk = decoder_settings.DecoderManagerSettings(bulk_strong=True)

    plan = _plan(escalation=escalation, decoder_manager=bulk)

    assert not plan.device.emits_bit_values


def test_a_row_shaped_by_the_code_card_is_built_with_the_runs_card():
    """A yaml names the kind alone, so the build supplies the card."""
    named_by_kind = qpu_settings.QpuSettings(kind="syndrome_bits", distance=5)

    plan = _plan(qpu=named_by_kind)

    assert plan.device.code is plan.code


def test_a_stim_source_is_its_own_window_model_source():
    stim_source = qpu_settings.QpuSettings(kind="stim_device", distance=3)

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
    section = declared_run.declared_workload(None, 6)
    return section.running(workload)


@pytest.mark.parametrize(
    "kind, sentence",
    [
        ("recorded_stim", "required positional arguments: 'measurements'"),
        ("streaming_stim", "required positional argument: 'programs'"),
    ],
)
def test_a_source_the_workload_cannot_fill_stops_its_call(kind, sentence):
    """Python's own call names the argument the source is not given."""
    source = qpu_settings.QpuSettings(kind=kind, distance=3)

    with pytest.raises(TypeError, match=sentence):
        _plan(qpu=source)


def test_live_fragments_under_a_finite_circuit_source_stop_its_call():
    source = qpu_settings.QpuSettings(kind="stim_device", distance=3)
    workload = _live_fragments_workload()

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _plan(qpu=source, workload=workload)


def test_live_fragments_build_the_streaming_source_from_a_yaml_kind():
    source = qpu_settings.QpuSettings(kind="streaming_stim", distance=3)
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
    section = declared_run.declared_workload(None, 10)
    return section.running(workload)


def _second_window_reads(plan) -> tuple:
    """The rounds the plan holds for the second window's weak read."""
    holds = dict(plan.run_plan.buffering.weak_holds)
    reads = decoding_records.WindowReads((1, 1))
    return holds[reads]


def test_a_forming_decoders_read_holds_what_its_circuit_reads_before_it():
    """The second window starts at round 4, whose detector reads round 2."""
    source = qpu_settings.QpuSettings(kind="stim_device", distance=3)
    workload = _lookback_workload()
    seats = event_settings.DetectionEventSettings(formed_at=("weak_decoder",))

    plan = _plan(qpu=source, workload=workload, detection_events=seats)

    held = _second_window_reads(plan)
    assert held[:3] == ((1, 2), (1, 3), (1, 4))


def test_a_source_with_no_recipes_holds_nothing_before_a_read():
    source = qpu_settings.QpuSettings(kind="syndrome_bits", distance=3)
    workload = _lookback_workload()
    seats = event_settings.DetectionEventSettings(formed_at=("weak_decoder",))

    plan = _plan(qpu=source, workload=workload, detection_events=seats)

    held = _second_window_reads(plan)
    assert held[0] == (1, 4)


def test_a_run_that_never_escalates_gets_the_flush_tail():
    plan = _plan()

    assert plan.scheme.terminal_policy == "flush"
    assert plan.scheme.has_trailing_tail_context is False


def test_a_run_that_may_escalate_gets_the_lookahead_tail():
    """A strong recovery reads past the last window's commit."""
    switching = _switching()

    plan = _plan(escalation=switching)

    assert plan.scheme.terminal_policy == "lookahead"
    assert plan.scheme.has_trailing_tail_context is True


def test_a_terminal_policy_written_in_the_section_wins_over_the_default():
    windows = window_settings.WindowSettings(terminal_policy="lookahead")

    plan = _plan(windows=windows)

    assert plan.scheme.terminal_policy == "lookahead"


def test_a_run_that_never_escalates_ships_boundaries_eagerly():
    plan = _plan()

    assert isinstance(plan.boundary_policy, boundary_policies.Eager)


def test_a_serial_escalation_holds_its_boundaries():
    """Descendants wait out an escalation, so nothing ships until final."""
    switching = _switching()

    plan = _plan(escalation=switching)

    assert isinstance(plan.boundary_policy, boundary_policies.Held)


def test_a_boundaries_row_written_in_the_section_wins_over_the_default():
    windows = window_settings.WindowSettings(boundaries="held")

    plan = _plan(windows=windows)

    assert isinstance(plan.boundary_policy, boundary_policies.Held)


def test_a_windows_kind_that_names_no_row_is_refused():
    windows = window_settings.WindowSettings(kind="diagonal")

    with pytest.raises(ValueError) as refusal:
        _plan(windows=windows)

    assert "windows.kind" in str(refusal.value)


class _SettingsRecordingScheme(sliding_scheme.SlidingWindowScheme):
    """A sliding scheme that keeps the Settings record it is built with."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        stride_rounds: int = 1

    def __init__(self, card, settings) -> None:
        sliding_scheme.SlidingWindowScheme.__init__(self, card)
        self.settings = settings


def test_a_scheme_row_with_settings_is_built_with_its_record(monkeypatch):
    monkeypatch.setitem(
        window_settings.WINDOWING_SCHEMES, "recording", _SettingsRecordingScheme
    )
    own_settings = _SettingsRecordingScheme.Settings(stride_rounds=2)
    windows = window_settings.WindowSettings(
        kind="recording", row_settings=own_settings
    )

    plan = _plan(windows=windows)

    assert plan.scheme.settings is own_settings


class _SettingsRecordingIdlePolicy(policies.Ignore):
    """An idle row that keeps the Settings record it is built with."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        every_nth_round: int = 1

    def __init__(self, settings) -> None:
        self.settings = settings


def test_an_idle_policy_row_with_settings_is_built_with_its_record(
    monkeypatch,
):
    monkeypatch.setitem(
        controller_settings.IDLE_POLICIES,
        "recording",
        _SettingsRecordingIdlePolicy,
    )
    own_settings = _SettingsRecordingIdlePolicy.Settings(every_nth_round=2)
    idle_policy = controller_settings.IdlePolicySettings(
        kind="recording", row_settings=own_settings
    )

    plan = _plan(idle_policy=idle_policy)

    assert plan.idle_policy.settings is own_settings


def test_a_scheme_that_declares_none_of_the_three_facts_is_refused():
    """Every row answers what the plan and the policy read off it."""
    silent = _SilentScheme()
    windows = window_settings.WindowSettings(scheme=silent)

    with pytest.raises(ValueError) as refusal:
        _plan(windows=windows)

    assert "has_trailing_tail_context" in str(refusal.value)


class _SilentScheme:
    """A scheme row that declares nothing the plan reads."""

    def plan_operation(self, *arguments, **sizes):
        """Never reached: the plan refuses this row first."""
        del arguments, sizes
        raise AssertionError("the plan should have refused this row")


# ---- the default boundary row lives on the rows that decide it


class _OutsideBoundaryPolicy:
    """A boundary policy row written outside decsim; it ships when final."""

    ships_provisional_boundaries = False

    def on_commit(self, window, *, final: bool) -> bool:
        """Ship when the committing result is final."""
        del window
        return final


class _OutsideEscalation:
    """An escalation row written outside decsim that never escalates."""

    decides_on_a_confidence = False
    requires_strong_context = False
    primary_tier = window_records.DecoderTier.WEAK
    default_boundary_policy = "outside_boundaries"

    def check_plan(self, plan) -> None:
        """Every run shape is served."""
        del plan

    def tiers_for_ready_window(self, window) -> tuple:
        """The weak tier alone."""
        del window
        return (window_records.DecoderTier.WEAK,)

    def verdict_for_weak_result(self, job, result):
        """Every result is final."""
        del job
        del result
        return decoding_records.Verdict.KEEP

    def learn_from_strong_result(self, window_key, result) -> None:
        """Nothing is learned."""
        del window_key
        del result


class _OutsideShape:
    """A strong window shape written outside decsim that absorbs nothing."""

    absorbs_weak_windows = False
    default_boundary_policy = "outside_boundaries"
    window_absorbed = trace_source.SILENT

    def __init__(self, collaborators) -> None:
        del collaborators

    def plan(self, weak_job):
        """Never reached: the test builds the plan and runs nothing."""
        del weak_job
        raise AssertionError("the test never escalates a window")


def _outside_shape_switching():
    """A switching section whose strong window is the outside shape."""
    return escalation_settings.EscalationSettings(
        kind="switching",
        threshold_source="fixed",
        gap_threshold_nats=2.0,
        confidence="complementary_gap",
        strong_window="outside_shape",
    )


def test_an_outside_escalation_row_names_its_own_default_boundary_row(
    monkeypatch,
):
    """A row that never escalates names the default; plan.py holds none.

    So a row of BOUNDARY_POLICIES written outside decsim can be it.
    """
    monkeypatch.setitem(
        window_settings.BOUNDARY_POLICIES,
        "outside_boundaries",
        _OutsideBoundaryPolicy,
    )
    policy = _OutsideEscalation()
    escalation = escalation_settings.EscalationSettings(policy=policy)

    plan = _plan(escalation=escalation)

    assert isinstance(plan.boundary_policy, _OutsideBoundaryPolicy)


def test_an_outside_strong_window_row_names_the_escalations_default(
    monkeypatch,
):
    """A row that may escalate leaves the default to its strong window."""
    monkeypatch.setitem(
        window_settings.BOUNDARY_POLICIES,
        "outside_boundaries",
        _OutsideBoundaryPolicy,
    )
    monkeypatch.setitem(
        escalation_settings.STRONG_WINDOW_SHAPES, "outside_shape", _OutsideShape
    )
    switching = _outside_shape_switching()

    plan = _plan(escalation=switching)

    assert isinstance(plan.boundary_policy, _OutsideBoundaryPolicy)

"""The two strong-window shapes plan the region the paper gives.

The context window is the commit region with one buffer of context on
each side, clipped at the operation's edge; that geometry is decsim's
own, since escalation unpins the past face (Skoric 2209.08552 line 388,
Tan 2209.09219 line 1021). Toshio et al. 2510.25222: the forward window
starts at the
escalated commit, absorbs the windows it covers, and is decoded once
both of its boundaries are weak-determined: the restart window's commit, or the
terminal data (Sec. III C, Fig. 12). A d=3 sliding window commits 3
rounds and buffers 3, so r_strong = r_com + 2 r_buf is 9 rounds
(Sec. III C, lines 1250-1251).

The restart window's weak decode re-reads
escalation.restart_reread_buffer_regions buffer regions of the strong
region from Buffer 0 (Sec. III C: the weak decoder resumes past the
strong region once its rounds are stored), so at width 1 the last
absorbed window's commit rounds must still be stored when the plan
lands, in a backlog regime where the absorbed windows' inputs are in
flight or have already landed in a unit. At width 0 the restart begins
on the round after the strong region. The gate's switching card, priced
on both tiers so the run is deterministic, is the regime the reviewer
found.
"""

import copy
import dataclasses
import pathlib

import pytest

import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.machine as machine_module
import decsim.observe.run_views as run_views
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.trace_source as trace_source
import decsim.windows.decode_requests as decode_requests
import tests.escalation.declared_fabric as fabric

# The gate's switching card, section by section as the yaml reads.
ONE_FRIDGE_CYCLE = {
    "latency_cycles": 1,
    "clock": "fridge",
    "bits_per_cycle": None,
}
ONE_ROOM_CYCLE = {"latency_cycles": 1, "clock": "room", "bits_per_cycle": None}
GATE_SWITCHING_CARD = {
    "clocks": {"fridge": 250.0, "room": 250.0},
    "qpu": {"kind": "stim_device"},
    "controller": {
        "clock": "fridge",
        "readout_to_bits_cycles": 0,
        "decision_to_pulse_cycles": 0,
        "packing_cycles_per_round": 0,
        "packing_rounds_in_flight": None,
    },
    "idle_policy": "separate_decode_jobs",
    "links": {
        "qpu_to_controller": ONE_FRIDGE_CYCLE,
        "controller_to_weak_buffer": ONE_FRIDGE_CYCLE,
        "controller_to_strong_buffer": ONE_ROOM_CYCLE,
        "weak_buffer_to_weak_decoder": ONE_FRIDGE_CYCLE,
        "weak_decoder_to_strong_decoder": ONE_ROOM_CYCLE,
        "strong_buffer_to_strong_decoder": ONE_ROOM_CYCLE,
        "decoder_to_decoder": ONE_FRIDGE_CYCLE,
        "weak_decoder_to_frame": ONE_FRIDGE_CYCLE,
        "strong_decoder_to_frame": ONE_ROOM_CYCLE,
        "frame_to_controller": None,
        "controller_to_qpu": None,
    },
    "round_store": {"rounds": None},
    "strong_round_store": {"rounds": None},
    "windows": {
        "kind": "sliding",
        "commit_rounds": None,
        "buffer_rounds": None,
    },
    "weak_decoder": {
        "kind": "pymatching",
        "units": 1,
        "unit_memory_rounds": None,
        "engine": {
            "clock": "fridge",
            "fetch_cycles_per_round": 1,
            "release_cycles_per_job": 10,
        },
    },
    "strong_decoder": {
        "kind": "belief_matching",
        "units": 1,
        "unit_memory_rounds": None,
        "engine": {
            "clock": "room",
            "fetch_cycles_per_round": 1,
            "release_cycles_per_job": 10,
        },
    },
    "escalation": {"kind": "switching", "gap_threshold_db": 20.0},
    "pauli_frame": {"clock": "fridge", "write_cycles": 1},
    "workload": {
        "kind": "memory_circuit",
        "code_task": "surface_code:rotated_memory_z",
        "rounds_per_shot": "10d",
    },
    "observation": {
        "check_windows_with": "none",
        "log_component_io": True,
        "log": "off",
    },
}


def _strong_request_record(machine, window_id: int):
    view = run_views.switching_records_view(
        machine.observation.windows, machine.observation.decode_records
    )
    for record in view.requests:
        is_window = record.request_key.window_id == window_id
        is_strong = record.request_key.tier is window_records.DecoderTier.STRONG
        if is_window and is_strong:
            return record
    raise AssertionError(f"no strong request for window {window_id}")


def test_the_context_window_reads_commit_plus_one_buffer_each_side():
    machine = fabric.switching_machine(
        rounds=12, escalated_windows={1}, record=True
    )
    machine.run()
    strong = _strong_request_record(machine, 1)
    # W1 commits rounds 4-6; one 3-round buffer each side reads 1-9
    assert (strong.input_round_lo, strong.input_round_hi) == (1, 9)
    assert strong.input_round_count == 9
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 1), "strong"),
        ((1, 2), "weak"),
        ((1, 3), "weak"),
    ]


def test_the_context_window_is_clipped_at_the_operations_first_round():
    machine = fabric.switching_machine(
        rounds=9, escalated_windows={0}, record=True
    )
    machine.run()
    strong = _strong_request_record(machine, 0)
    # W0 commits rounds 1-3; the left buffer has no rounds before 1
    assert (strong.input_round_lo, strong.input_round_hi) == (1, 6)
    assert strong.input_round_count == 6


def test_the_forward_window_absorbs_the_windows_it_covers():
    machine = fabric.switching_machine(
        rounds=15,
        escalated_windows={1},
        strong_window="forward",
        round_microseconds=4.0,
    )
    machine.run()
    assigned = fabric.log_lines_containing(machine, "assigned")
    assert len(assigned) == 1
    # W1 commits 4-6; the strong window is 9 rounds, 4-12, so W2 and W3
    # (commits 7-9 and 10-12) are absorbed and W4 restarts the chain
    assert (
        "strong window rounds 4-12 assigned; weak chain skips 2 window(s); "
        "strong start deferred until the far-side weak boundary"
    ) in assigned[0]
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 4), "weak"),
        ((1, 1), "strong"),
    ]


def test_the_forward_window_is_submitted_once_at_the_far_boundary_commit():
    machine = fabric.switching_machine(
        rounds=15,
        escalated_windows={1},
        strong_window="forward",
        round_microseconds=4.0,
    )
    machine.run()
    lines = machine.observation.log.lines
    deferred_lines = fabric.log_lines_containing(
        machine, "deferred until the far-side weak boundary"
    )
    submitted_lines = fabric.log_lines_containing(
        machine, "far-side weak boundary determined -> strong window submitted"
    )
    restart_lines = fabric.log_lines_containing(machine, "DECODE DONE mem1 W4")
    assert len(submitted_lines) == 1
    deferred = lines.index(deferred_lines[0])
    restart_committed = lines.index(restart_lines[0])
    submitted = lines.index(submitted_lines[0])
    assert deferred < restart_committed < submitted
    assert not machine.window_manager.strong_redecode.has_pending()


def test_the_forward_window_at_the_operations_end_waits_for_terminal_data():
    machine = fabric.switching_machine(
        rounds=9, escalated_windows={2}, strong_window="forward"
    )
    machine.run()
    submitted = fabric.log_lines_containing(
        machine, "terminal data complete -> strong window submitted"
    )
    assert len(submitted) == 1
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 1), "weak"),
        ((1, 2), "strong"),
    ]
    assert not machine.window_manager.strong_redecode.has_pending()


def test_a_second_escalation_of_one_window_is_refused():
    machine = fabric.switching_machine(
        rounds=9, escalated_windows={2}, strong_window="forward"
    )
    machine.run()
    shape = machine.window_manager.strong_redecode.shape
    again = decoding_records.DecodeJob(
        operation_id=1,
        window_id=2,
        round_count=3,
        strong_label="strong(mem1 W2)",
    )
    with pytest.raises(RuntimeError, match="duplicate strong escalation"):
        shape.plan(again)


# ---- the restart window's Buffer 0 claim under a backlog


def _gate_forward_window_machine(
    commit_rounds: int,
    buffer_rounds: int,
    weak_microseconds: float,
    strong_microseconds: float,
    weak_units: int,
    reread_buffer_regions: int,
) -> machine_module.Machine:
    """The gate's switching card with the forward window, both tiers priced.

    The gate's point: p 0.008, d 3, 1 us rounds, 30 rounds, seed 0. The
    weak tier at 40 us per window against 3 us of rounds is the backlog
    regime: every later window is requested, and staged as units free,
    long before the escalated window's verdict.
    """
    sections = copy.deepcopy(GATE_SWITCHING_CARD)
    sections["windows"]["commit_rounds"] = commit_rounds
    sections["windows"]["buffer_rounds"] = buffer_rounds
    sections["weak_decoder"]["kind"] = weak_microseconds
    sections["weak_decoder"]["units"] = weak_units
    sections["strong_decoder"]["kind"] = strong_microseconds
    sections["escalation"]["strong_window"] = "forward"
    sections["escalation"]["restart_reread_buffer_regions"] = (
        reread_buffer_regions
    )
    base_directory = pathlib.Path(".")
    settings = machine_settings.MachineSettings.from_mapping(
        sections, name="switching_validation", base_directory=base_directory
    )
    qpu = dataclasses.replace(
        settings.qpu, distance=3, round_period_microseconds=1.0
    )
    workload = dataclasses.replace(
        settings.workload, physical_error_probability=0.008
    )
    settings = dataclasses.replace(settings, qpu=qpu, workload=workload)
    return machine_module.Machine.build(settings, 0)


def _run_statuses(result) -> list:
    statuses = []
    for row in result.operation_results:
        statuses.append((row.operation_id, row.result_status))
    return statuses


def _log_index(machine, needle: str) -> int:
    lines = fabric.log_lines_containing(machine, needle)
    assert lines, needle
    return machine.observation.log.lines.index(lines[0])


def _claim(machine, window_index: int):
    store = machine.window_manager.retention.weak_store
    claim = decoding_records.PotentialRestart((1, window_index))
    if not store.has_hold(claim):
        return None
    return store.hold_round_identities(claim)


def test_a_forward_window_plan_claims_the_rounds_a_restart_would_read():
    """The plan's claims at the default re-read width, which is 0.

    A bounded window claims exactly its own reads, since a restart of
    it would begin on its first committed round. The first window, and
    every window of an ordinary run, claims nothing.
    """
    forward = fabric.switching_machine(
        rounds=15, escalated_windows=set(), strong_window="forward"
    )
    # W1 commits 4-6 and reads to 9
    assert _claim(forward, 1) == tuple((1, index) for index in range(4, 10))
    # W4 commits 13-15 and reads to 18, clipped at the operation's end
    assert _claim(forward, 4) == tuple((1, index) for index in range(13, 16))
    assert _claim(forward, 0) is None
    ordinary = fabric.switching_machine(rounds=15, escalated_windows=set())
    for window_index in range(5):
        assert _claim(ordinary, window_index) is None


def test_the_restart_window_keeps_its_re_read_rounds_across_the_withdrawals():
    """The reviewer's reproduction: commit 3, buffer 3, weak 40 us.

    W3 escalates at 166 us with W4's input landed and W5's in flight;
    the strong window 10-18 absorbs W4 and W5, and the restart W6
    re-reads 16-18, W5's commit rounds, whose last Buffer 0 holder was
    W5's request. The run used to die there with the retention
    sentence; now W6's own claim carries the rounds across W5's
    withdrawal and W6's stale request is withdrawn and rebuilt.
    """
    machine = _gate_forward_window_machine(3, 3, 40.0, 5.0, 1, 1)
    result = machine.run()
    assert _run_statuses(result) == [(1, "logical_observables")]
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 1), "weak"),
        ((1, 2), "weak"),
        ((1, 3), "strong"),
        ((1, 9), "strong"),
        ((1, 6), "strong"),
    ]
    resliced = fabric.log_lines_containing(machine, "re-sliced")
    assert (
        "restart window (1, 6) re-sliced across strong window edge 18 "
        "(reads rounds 16-24; crossing faults owned by strong_region)"
    ) in resliced[0]
    withdrawn_restart = _log_index(machine, "WITHDRAW memory W6")
    assert withdrawn_restart < machine.observation.log.lines.index(resliced[0])
    assert not machine.window_manager.strong_redecode.has_pending()


def test_the_re_read_rounds_survive_with_commit_four_and_buffer_four():
    """The reviewer's second shape: W2 escalates, W5 re-reads 17-20."""
    machine = _gate_forward_window_machine(4, 4, 40.0, 5.0, 1, 1)
    result = machine.run()
    assert _run_statuses(result) == [(1, "logical_observables")]
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 1), "weak"),
        ((1, 5), "strong"),
        ((1, 2), "strong"),
    ]
    resliced = fabric.log_lines_containing(machine, "re-sliced")
    assert (
        "restart window (1, 5) re-sliced across strong window edge 20 "
        "(reads rounds 17-28; crossing faults owned by strong_region)"
    ) in resliced[0]


def test_the_re_read_rounds_survive_the_absorbed_inputs_landing_first():
    """Four weak units: the absorbed W5 and the restart W6 land first.

    Both inputs are in unit memory before W3's verdict. A landed
    input's Buffer 0 request hold ends at the landing, so with one
    holder the re-read rounds 16-18 and W6's own 19-21 would be gone
    before the plan runs; W6's claim keeps them past both landings.
    Four units hold two windows at once, since a window's confidence
    is two forced-class solves and each takes a unit.
    """
    machine = _gate_forward_window_machine(3, 3, 40.0, 5.0, 4, 1)
    result = machine.run()
    landed_absorbed = _log_index(
        machine, "memory W5 [commit 16-18] input landed"
    )
    landed_restart = _log_index(
        machine, "memory W6 [commit 19-21] input landed"
    )
    escalated = _log_index(machine, "WITHDRAW memory W4")
    assert landed_absorbed < escalated
    assert landed_restart < escalated
    assert _run_statuses(result) == [(1, "logical_observables")]
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 1), "weak"),
        ((1, 2), "weak"),
        ((1, 3), "strong"),
        ((1, 9), "strong"),
        ((1, 6), "strong"),
    ]


def test_the_paper_width_restarts_on_the_round_after_the_strong_region():
    """The eaa316d reproduction at re-read width 0, two weak units.

    The absorbed W5 and the restart W6 land in unit memory before W3's
    verdict. With no re-read W6 begins at round 19, the round after the
    strong region, and owns the faults of the rounds it reads (Toshio
    2510.25222 Sec. III C, Fig. 12); the run completes as it does with
    one buffer region of re-read, and W9 keeps its weak result.
    """
    machine = _gate_forward_window_machine(3, 3, 40.0, 5.0, 2, 0)
    result = machine.run()
    assert _run_statuses(result) == [(1, "logical_observables")]
    resliced = fabric.log_lines_containing(machine, "re-sliced")
    assert (
        "restart window (1, 6) re-sliced across strong window edge 18 "
        "(reads rounds 19-24; crossing faults owned by restart_window)"
    ) in resliced[0]
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 1), "weak"),
        ((1, 2), "weak"),
        ((1, 3), "strong"),
        ((1, 9), "weak"),
        ((1, 6), "strong"),
    ]


def test_the_forward_window_lands_in_the_declared_backlog_regime():
    """1 us rounds against a 10 us weak decode: W1 escalates with W2 landed.

    The stabilization suite pinned this run as a refusal (finding R3);
    the refusal was the bug.
    """
    machine = fabric.switching_machine(
        rounds=15, escalated_windows={1}, strong_window="forward"
    )
    machine.run()
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 4), "weak"),
        ((1, 1), "strong"),
    ]
    resliced = fabric.log_lines_containing(machine, "re-sliced")
    assert (
        "restart window (1, 4) re-sliced across strong window edge 12"
        in (resliced[0])
    )
    assert not machine.window_manager.strong_redecode.has_pending()


# ---- a shape row added from outside decsim


class RecordingContextWindow:
    """A shape row a study adds: the context window, with its plans noted.

    It fills the StrongWindowShape port by holding a ContextWindow and
    passing every call to it, which is what a new row does: one class,
    one table entry, one yaml name, and nothing else changes.
    """

    absorbs_weak_windows = False
    window_absorbed = trace_source.SILENT

    def __init__(self, collaborators) -> None:
        self.inner = strong_window_shapes.ContextWindow(collaborators)
        self.planned_windows = []
        self.assignments = []

    def plan(self, weak_job):
        """Note the window, then plan it as the context window does."""
        self.planned_windows.append(weak_job.window_id)
        assignment = self.inner.plan(weak_job)
        self.assignments.append(assignment)
        return assignment

    def release_conditions(self, assignment):
        """What releases a held job, as the context window declares it."""
        return self.inner.release_conditions(assignment)

    def held_job(self, assignment):
        """The held job, once its rounds are there."""
        return self.inner.held_job(assignment)


def test_a_shape_row_added_from_outside_runs_by_its_yaml_name():
    """A new strong window shape is one class and one table row.

    sinter's BUILT_IN_DECODERS is the shape: a name in the config
    resolves to a class in the table, and the machine builds it with the
    components the port needs (_decoding_all_built_in_decoders.py).
    """
    table = escalation_settings.STRONG_WINDOW_SHAPES
    table["recording_context"] = RecordingContextWindow
    try:
        machine = fabric.switching_machine(
            rounds=9,
            escalated_windows={2},
            strong_window="recording_context",
        )
        machine.run()
    finally:
        del table["recording_context"]
    shape = machine.window_manager.strong_redecode.shape
    assert type(shape) is RecordingContextWindow
    assert shape.planned_windows == [2]
    # the row reads raw rounds on both faces: it folds no neighbour
    # boundary into the strong job's input
    assert shape.assignments[0].folded_boundaries == ()
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 1), "weak"),
        ((1, 2), "strong"),
    ]


class RecordingForwardWindow:
    """A shape row a study adds that absorbs the windows it covers.

    It takes the same one collaborator record the shipped rows take,
    although its layout reads the planner, the requester and the ledger
    that the context window never touches.
    """

    absorbs_weak_windows = True

    def __init__(self, collaborators) -> None:
        self.inner = strong_window_shapes.ForwardWindow(collaborators)
        self.window_absorbed = self.inner.window_absorbed
        self.planned_windows = []

    def plan(self, weak_job):
        """Note the window, then plan it as the forward window does."""
        self.planned_windows.append(weak_job.window_id)
        return self.inner.plan(weak_job)

    def release_conditions(self, assignment):
        """What releases a held job, as the forward window declares it."""
        return self.inner.release_conditions(assignment)

    def held_job(self, assignment):
        """The held job, once its rounds are there."""
        return self.inner.held_job(assignment)


def test_an_absorbing_row_added_from_outside_builds_through_the_same_call():
    """One constructor signature, whatever the row's geometry.

    The two shipped rows read different components: the context window
    reads the regions, the retention and the builder; the forward window
    also re-slices on the planner, withdraws on the requester and
    rewrites the ledger. Both take one StrongWindowCollaborators record,
    so the root builds a row without branching on its geometry.
    """
    table = escalation_settings.STRONG_WINDOW_SHAPES
    table["recording_forward"] = RecordingForwardWindow
    try:
        machine = fabric.switching_machine(
            rounds=9,
            escalated_windows={0},
            strong_window="recording_forward",
        )
        machine.run()
    finally:
        del table["recording_forward"]
    shape = machine.window_manager.strong_redecode.shape
    assert type(shape) is RecordingForwardWindow
    assert shape.planned_windows == [0]
    assert fabric.frame_tiers(machine) == [((1, 0), "strong")]


def test_the_shipped_collaborators_fill_the_six_window_side_ports():
    """A row written outside decsim programs against the ports, not classes.

    StrongWindowCollaborators types its six window-side fields as
    Protocols in decsim/ports.py, so the promise only means something if
    the classes the root puts there answer the whole port.
    """
    machine = fabric.switching_machine(rounds=9, escalated_windows=set())
    collaborators = machine.window_manager.strong_redecode.shape.collaborators
    assert isinstance(collaborators.planner, ports.WindowPlan)
    assert isinstance(collaborators.retention, ports.WindowRetention)
    assert isinstance(collaborators.builder, ports.WindowJobBuilder)
    assert isinstance(collaborators.requester, ports.WindowRequests)
    assert isinstance(collaborators.ledger, ports.LogicalLedger)
    assert isinstance(collaborators.courier, ports.BoundaryCourier)


def test_a_courier_that_only_pins_a_face_fills_the_courier_port():
    """The port lists the one method the escalation side calls.

    strong_job_payloads calls pin_strong_face and nothing else
    (strong_window_shapes.py, the walk over folded_boundaries), so a
    courier supplied from outside decsim answers the whole port with
    that one method. STYLE.md rule 7: a port carries the methods one
    component needs from another.
    """
    courier = _RecordingCourier()
    assert isinstance(courier, ports.BoundaryCourier)
    assert not hasattr(courier, "committed")


def _gate_machine(strong_window: str) -> machine_module.Machine:
    """The gate's own switching point, on the named strong window row.

    p 0.008, d 3, 1 us rounds, seed 0: the point where the weak tier
    commits a correction that flips a seam detector of a window that
    later escalates, so the two-solve case a raw-read row must not fold
    is on the run, and so is the seam a pinned row must carry.
    """
    sections = copy.deepcopy(GATE_SWITCHING_CARD)
    sections["escalation"]["strong_window"] = strong_window
    base_directory = pathlib.Path(".")
    settings = machine_settings.MachineSettings.from_mapping(
        sections, name="switching_validation", base_directory=base_directory
    )
    qpu = dataclasses.replace(
        settings.qpu, distance=3, round_period_microseconds=1.0
    )
    workload = dataclasses.replace(
        settings.workload, physical_error_probability=0.008
    )
    settings = dataclasses.replace(settings, qpu=qpu, workload=workload)
    return machine_module.Machine.build(settings, 0)


def _masked_jobs(monkeypatch) -> list:
    """Records (kind, window key, whether the gate changed the input)."""
    original = decode_requests.WindowInputGate.mask_input
    masked = []

    def watch(self, job):
        before = _input_bits(job.decoder_input)
        original(self, job)
        after = _input_bits(job.decoder_input)
        key = None
        if job.window is not None:
            key = job.window.key
        changed = before != after
        masked.append((job.kind, key, changed))

    monkeypatch.setattr(decode_requests.WindowInputGate, "mask_input", watch)
    return masked


def _input_bits(decoder_input) -> tuple:
    """Every fragment's bits of a job's input, in read order."""
    if decoder_input is None:
        return ()
    bits = []
    for round_input in decoder_input.rounds:
        for fragment in round_input.fragments:
            bits.append(fragment.bits)
    return tuple(bits)


def test_a_context_row_strong_job_reads_the_raw_rounds_of_its_span(
    monkeypatch,
):
    """The row folds no boundary, so nothing masks its input.

    The context window reads one buffer of raw context on each side, so
    a mask on its commit_lo layer would flip a seam whose rounds are
    already in the input as raw defects: the double count Bombin et al.
    2303.04846 lines 775-788 rule out. The weak job of the same window
    is masked on this run, which is what makes the check meaningful.
    """
    masked = _masked_jobs(monkeypatch)
    machine = _gate_machine("two_sided_context")
    machine.run()
    weak_masked = _keys_of(masked, strong=False, changed_only=True)
    strong_keys = _keys_of(masked, strong=True, changed_only=False)
    strong_masked = _keys_of(masked, strong=True, changed_only=True)
    assert strong_keys
    assert weak_masked & strong_keys
    assert strong_masked == set()


def _keys_of(masked: list, *, strong: bool, changed_only: bool) -> set:
    """The window keys of the recorded jobs of one tier."""
    strong_redecode = decoding_records.DecodeJobKind.STRONG_REDECODE
    keys = set()
    for kind, key, changed in masked:
        is_strong = kind is strong_redecode
        if is_strong is not strong:
            continue
        if changed_only and not changed:
            continue
        keys.add(key)
    return keys


def test_a_shape_name_off_the_table_is_refused_naming_the_rows():
    """both_faces_pinned is the shape decsim refuses to build.

    A strong window that pins both faces and absorbs nothing waits for
    the window after it, which waits for the strong result: the serial
    sliding chain deadlocks on it (design audit note 21). It is not a
    row, and a name that is not a row is refused naming the rows.
    """
    with pytest.raises(ValueError) as refusal:
        fabric.switching_machine(
            rounds=9,
            escalated_windows=set(),
            strong_window="both_faces_pinned",
        )
    assert "escalation.strong_window 'both_faces_pinned' is not a row" in str(
        refusal.value
    )


def test_the_near_seam_row_reads_its_commit_region_and_one_buffer():
    """Bombin 2303.04846 lines 1456-1458: a pinned face needs no buffer.

    W1 of a d=3 run commits rounds 4-6. The two-sided context row reads
    1-9; pinning the past face on W0's committed correction drops the
    leading buffer, so this row reads 4-9 and commits the same rounds.
    """
    machine = fabric.switching_machine(
        rounds=12,
        escalated_windows={1},
        strong_window="near_seam_pinned",
        record=True,
    )
    machine.run()
    strong = _strong_request_record(machine, 1)
    assert (strong.input_round_lo, strong.input_round_hi) == (4, 9)
    assert strong.input_round_count == 6
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 1), "strong"),
        ((1, 2), "weak"),
        ((1, 3), "weak"),
    ]


def test_the_near_seam_row_pins_nothing_at_the_operations_first_window():
    """W0 has no earlier neighbour, so its past face is the readout's.

    The first window's oldest layer is the operation's own first round
    layer, closed by the initialisation, so the row declares no face
    and the job reads its commit region and one buffer.
    """
    machine = fabric.switching_machine(
        rounds=12,
        escalated_windows={0},
        strong_window="near_seam_pinned",
        record=True,
    )
    machine.run()
    strong = _strong_request_record(machine, 0)
    assert (strong.input_round_lo, strong.input_round_hi) == (1, 6)
    assert strong.input_round_count == 6


def _pinned_boundary_transfers(result) -> list:
    """(window index, payload bits) of every pinned face on the wire.

    A weak delivery is attributed to the window that produced it; a
    pinned face is attributed to the strong window that reads it, so a
    transfer whose attribution names a window other than the producing
    request's is a pin.
    """
    pinned = []
    for transfer in result.link_traffic["transfers"]:
        if transfer["path"] != "decoder_to_decoder":
            continue
        attribution = transfer["attribution"]
        source_window_id = attribution["relation"]["request_key"]["window_id"]
        if attribution["window_id"] == source_window_id:
            continue
        pinned.append((attribution["window_id"], transfer["payload_bits"]))
    return pinned


def test_a_yaml_names_the_near_seam_row_and_its_pin_crosses_the_wire():
    """The row is one class, one table row and one yaml name.

    The gate's own switching card with escalation.strong_window
    near_seam_pinned builds this row, and every face it pins is a
    message on decoder_to_decoder: Skoric 2209.08552 lines 1038-1040
    sends the artificial defects block to block, and Bombin's Fig. 14
    (lines 2256-2259) routes them through the boundary condition data
    store between decoder modules. A pinned face that crossed nothing
    would be free in the model.
    """
    machine = _gate_machine("near_seam_pinned")
    result = machine.run()
    shape = machine.window_manager.strong_redecode.shape
    assert type(shape) is strong_window_shapes.NearSeamWindow
    strong_windows = _strong_window_keys(machine)
    pinned = _pinned_boundary_transfers(result)
    pinned_windows = set()
    pinned_bits = set()
    for window_id, payload_bits in pinned:
        pinned_windows.add((1, window_id))
        pinned_bits.add(payload_bits)
    assert pinned_windows == strong_windows
    assert len(pinned) == len(strong_windows)
    # a d=3 bulk layer carries d*d-1 = 8 detectors, one bit each under
    # the dense row
    assert pinned_bits == {8}


def test_the_forward_seam_row_reads_exactly_the_rounds_it_commits():
    """Toshio 2510.25222 lines 1248-1250, as the paper states it.

    W1 of a d=3 run commits 4-6, so the forward strong region commits
    r_com + 2 r_buf = 9 rounds, 4-12. The shipped forward row reads one
    buffer of raw context on each side of that, 1-15; pinning both faces
    drops both buffers, so this row reads the 9 rounds it commits.
    """
    machine = fabric.switching_machine(
        rounds=15,
        escalated_windows={1},
        strong_window="forward_seam_pinned",
        round_microseconds=4.0,
        record=True,
    )
    machine.run()
    strong = _strong_request_record(machine, 1)
    assert (strong.input_round_lo, strong.input_round_hi) == (4, 12)
    assert strong.input_round_count == 9
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 4), "weak"),
        ((1, 1), "strong"),
    ]


def test_the_forward_seam_row_pins_its_near_and_its_far_face():
    """One message per pinned face, both on decoder_to_decoder.

    The near face is the window before the strong region, the far face
    the window that restarts the weak chain after it, which is the
    boundary the shipped forward row already waits for (Toshio
    2510.25222 lines 1253-1259). Skoric 2209.08552 lines 1038-1040 sends
    each face's defects block to block, so two faces are two messages.
    """
    machine = fabric.switching_machine(
        rounds=15,
        escalated_windows={1},
        strong_window="forward_seam_pinned",
        round_microseconds=4.0,
    )
    result = machine.run()
    pinned = _pinned_boundary_transfers(result)
    assert len(pinned) == 2
    for window_id, _payload_bits in pinned:
        assert window_id == 1
    sources = _pin_sources(result)
    # W1's own dependency, and the window that restarts the chain past
    # the strong region 4-12
    assert sources == [0, 4]


def _pin_sources(result) -> list:
    """The window that produced each pinned face's boundary, in send order."""
    sources = []
    for transfer in result.link_traffic["transfers"]:
        if transfer["path"] != "decoder_to_decoder":
            continue
        attribution = transfer["attribution"]
        source_window_id = attribution["relation"]["request_key"]["window_id"]
        if attribution["window_id"] == source_window_id:
            continue
        sources.append(source_window_id)
    return sources


def test_the_forward_seam_row_at_the_operations_end_has_no_far_pin():
    """Tan 2209.09219 lines 953-955: the last window's faces are closed.

    A terminal strong region has no later window to pin on, so it waits
    for the terminal data the way the shipped forward row does and reads
    to its own last committed round.
    """
    machine = fabric.switching_machine(
        rounds=9,
        escalated_windows={2},
        strong_window="forward_seam_pinned",
        record=True,
    )
    result = machine.run()
    submitted = fabric.log_lines_containing(
        machine, "terminal data complete -> strong window submitted"
    )
    assert len(submitted) == 1
    strong = _strong_request_record(machine, 2)
    assert (strong.input_round_lo, strong.input_round_hi) == (7, 9)
    sources = _pin_sources(result)
    # only the near face: W2's own dependency
    assert sources == [1]


def test_the_forward_seam_row_reads_an_unpinned_near_face_raw():
    """A window whose dependency was absorbed has nothing to pin on.

    W1's strong region commits 4-12 and absorbs W2 and W3, so the
    restart window W4 keeps no dependency: no committed correction
    closes its past face. When W4 escalates in turn, its strong region
    commits 13-21 and reads one buffer region of raw context behind it,
    10-21, since an open face keeps its buffer (Bombin 2303.04846 lines
    850-852) rather than reading nothing at all. Its far face is the
    window that restarts the chain after 21, W7, and that one is pinned.
    """
    machine = fabric.switching_machine(
        rounds=30,
        escalated_windows={1, 4},
        strong_window="forward_seam_pinned",
        round_microseconds=4.0,
        record=True,
    )
    result = machine.run()
    pinned_faces = _strong_request_record(machine, 1)
    assert (pinned_faces.input_round_lo, pinned_faces.input_round_hi) == (4, 12)
    open_face = _strong_request_record(machine, 4)
    assert (open_face.input_round_lo, open_face.input_round_hi) == (10, 21)
    # W1 pins both faces (W0 and W4), W4 only its far face (W7)
    assert _pin_sources(result) == [0, 4, 7]


def test_a_yaml_names_the_forward_seam_row_and_it_runs():
    """The row is one class, one table row and one yaml name.

    The gate's own switching card with escalation.strong_window
    forward_seam_pinned builds this row and runs the point through, with
    the absorbing apparatus of the shipped forward row underneath it.
    """
    machine = _gate_machine("forward_seam_pinned")
    result = machine.run()
    shape = machine.window_manager.strong_redecode.shape
    assert type(shape) is strong_window_shapes.ForwardSeamWindow
    statuses = _run_statuses(result)
    assert statuses == [(1, "logical_observables")]
    assert not machine.window_manager.strong_redecode.has_pending()


def test_a_pinned_far_face_refuses_a_re_reading_restart_window():
    """Q5 of design audit note 20, settled as a refusal.

    With escalation.restart_reread_buffer_regions 1 the restart window
    commits rounds inside the strong region, so pinning the far face on
    its correction would carry an explanation of rounds the input holds
    raw: the double count Bombin 2303.04846 lines 775-788 rule out. 0 is
    the paper's value (Toshio 2510.25222 Sec. III C, Fig. 12), and the
    section refuses the pairing at load rather than reconciling it.
    """
    section = {
        "kind": "switching",
        "gap_threshold_db": 20.0,
        "strong_window": "forward_seam_pinned",
        "restart_reread_buffer_regions": 1,
    }
    with pytest.raises(ValueError) as refusal:
        escalation_settings.EscalationSettings.from_yaml(section)
    assert "restart_reread_buffer_regions must be 0" in str(refusal.value)
    section["strong_window"] = "forward"
    kept = escalation_settings.EscalationSettings.from_yaml(section)
    assert kept.restart_reread_buffer_regions == 1


def _strong_window_keys(machine) -> set:
    """The windows that escalated and pinned a face on their neighbour.

    The operation's first window pins nothing, so it is not among them.
    """
    keys = set()
    for window_key, tier in fabric.frame_tiers(machine):
        if tier != "strong":
            continue
        if window_key[1] == 0:
            continue
        keys.add(window_key)
    return keys


class _StoredRounds:
    """A retention that answers with the rounds the window reads."""

    def strong_window_input(self, builder, window) -> tuple:
        """The stored rounds, whatever the builder and the window are."""
        del builder
        del window
        return ("the stored rounds",)


class _RecordingCourier:
    """A courier that records the faces a row asked it to pin."""

    def __init__(self) -> None:
        self.pinned = []

    def pin_strong_face(
        self, source_key, destination, model, operation, request_key
    ):
        """Note the face the row pinned, with what the fold reads."""
        self.pinned.append(
            (source_key, destination, model, operation, request_key)
        )


def _folding_collaborators(retention, courier):
    """The one collaborator record, with only the fold's two components."""
    return strong_window_shapes.StrongWindowCollaborators(
        engine=None,
        regions=None,
        planner=None,
        retention=retention,
        builder=None,
        requester=None,
        ledger=None,
        courier=courier,
    )


def test_a_row_that_pins_a_face_folds_the_boundary_it_declares():
    """The input adaptation is asked for once per declared face.

    A row that pins a face carries its neighbour's committed correction
    into the strong job's input, which is Bombin et al. 2303.04846's
    input adaptation (lines 775-788): the courier ships the committed
    boundary to this window and folds it into the window's boundary
    state, against the strong window's own model. A row that folds none
    reads the stored rounds and asks the courier for nothing.
    """
    retention = _StoredRounds()
    courier = _RecordingCourier()
    collaborators = _folding_collaborators(retention, courier)
    window = object()
    model = object()
    operation = object()
    request_key = object()
    payloads = strong_window_shapes.strong_job_payloads(
        collaborators,
        window,
        model,
        operation,
        request_key,
        strong_window_shapes.FOLDS_NO_BOUNDARY,
    )
    assert payloads == ("the stored rounds",)
    assert courier.pinned == []
    payloads = strong_window_shapes.strong_job_payloads(
        collaborators, window, model, operation, request_key, ((1, 1),)
    )
    assert payloads == ("the stored rounds",)
    assert courier.pinned == [((1, 1), window, model, operation, request_key)]

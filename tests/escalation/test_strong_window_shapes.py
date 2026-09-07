"""The two strong-window shapes plan the region the paper gives.

Toshio et al. 2510.25222: the context window is the commit region with
one buffer of context on each side, r_strong = r_com + 2 r_buf, clipped
at the operation's edge (Sec. III A; text lines 1250-1252 of
tmp/papers/txt); the forward window starts at the escalated commit,
absorbs the windows it covers, and is decoded once both of its
boundaries are weak-determined: the restart window's commit, or the
terminal data (Sec. III C, Fig. 12). A d=3 sliding window commits 3
rounds and buffers 3, so r_strong is 9 rounds.

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

import decsim.machine as machine_module
import decsim.observe.run_views as run_views
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records
import tests.escalation.declared_fabric as fabric

# The gate's switching card (validation/responsibility_audit_2026_08_30/
# frozen_suite/switching_validation.yaml over weak_decoder_baseline.yaml),
# section by section as the yaml reads.
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
        double_window=True,
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
        double_window=True,
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
        rounds=9, escalated_windows={2}, double_window=True
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
        rounds=9, escalated_windows={2}, double_window=True
    )
    machine.run()
    shape = machine.window_manager.strong_redecode.shape
    again = decoding_records.DecodeJob(
        op_id=1, window_id=2, n_rounds=3, strong_label="strong(mem1 W2)"
    )
    with pytest.raises(RuntimeError, match="duplicate strong escalation"):
        shape.plan(again)


# ---- the restart window's Buffer 0 claim under a backlog


def _gate_double_window_machine(
    commit_rounds: int,
    buffer_rounds: int,
    weak_microseconds: float,
    strong_microseconds: float,
    weak_units: int,
    reread_buffer_regions: int,
) -> machine_module.Machine:
    """The gate's switching card with double_window on, both tiers priced.

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
    sections["escalation"]["double_window"] = True
    sections["escalation"]["restart_reread_buffer_regions"] = (
        reread_buffer_regions
    )
    base_directory = pathlib.Path(".")
    settings = machine_module.MachineSettings.from_mapping(
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


def test_a_double_window_plan_claims_one_buffer_before_each_bounded_window():
    """The plan's claims: a window's reads and one buffer before them.

    The first window, and every window of an ordinary run, claims
    nothing.
    """
    doubled = fabric.switching_machine(
        rounds=15, escalated_windows=set(), double_window=True
    )
    # W1 commits 4-6 and reads to 9: one buffer before is 1-3
    assert _claim(doubled, 1) == tuple((1, index) for index in range(1, 10))
    # W4 commits 13-15 and reads to 18, clipped at the operation's end
    assert _claim(doubled, 4) == tuple((1, index) for index in range(10, 16))
    assert _claim(doubled, 0) is None
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
    machine = _gate_double_window_machine(3, 3, 40.0, 5.0, 1, 1)
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
    machine = _gate_double_window_machine(4, 4, 40.0, 5.0, 1, 1)
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
    """Two weak units: the absorbed W5 and the restart W6 land first.

    Both inputs are in unit memory before W3's verdict. A landed
    input's Buffer 0 request hold ends at the landing, so with one
    holder the re-read rounds 16-18 and W6's own 19-21 would be gone
    before the plan runs; W6's claim keeps them past both landings.
    """
    machine = _gate_double_window_machine(3, 3, 40.0, 5.0, 2, 1)
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
    machine = _gate_double_window_machine(3, 3, 40.0, 5.0, 2, 0)
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
        rounds=15, escalated_windows={1}, double_window=True
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

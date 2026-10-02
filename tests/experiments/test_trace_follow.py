"""`decsim trace follow` on gate point 1's own trace.

The reader prints one round's and one window's path from gate point 1's
trace (weak_decoder_baseline d 3 p 0.003 seed 0). The controller's
intake copy is at 1.000, the send tick, and the weak syndrome buffer
holds detection events, so round 1's residence there carries 4 bits and
not the link's 8.
"""

import pytest

import decsim.experiments.trace_file as trace_file
import decsim.experiments.trace_follow as trace_follow
import decsim.machine as machine_module
import decsim.observe.settings as observe_settings
import decsim.windows.settings as window_settings
import tests.declared_run as declared_run
import tests.escalation.declared_fabric as declared_fabric
import tests.experiments.test_measure as measure_tests
import tests.observe.gate_point as gate_point
import tests.observe.test_trace_writer as trace_writer_tests


@pytest.fixture(scope="module")
def trace_path(tmp_path_factory):
    """Gate point 1, run once with its trace written to a file."""
    directory = tmp_path_factory.mktemp("follow")
    path = directory / "point1.trace.json"
    point = gate_point.settings(trace=str(path))
    machine = machine_module.Machine.build(point, gate_point.SEED)
    machine.run()
    machine.observation.trace_writer.write(str(path))
    return path


@pytest.fixture(scope="module")
def traced(trace_path):
    """That file, read back and indexed."""
    return trace_file.load(trace_path)


def _hops_of(path) -> list:
    """Where and what, one pair per hop of a followed path."""
    return [(hop.where, hop.what) for hop in path.hops]


def _row_of(path, where, what):
    for hop in path.hops:
        if hop.where != where:
            continue
        if what not in hop.what:
            continue
        return hop
    raise AssertionError(f"no {where} hop saying {what}")


def test_round_ones_first_hops_are_its_emission_move_and_intake(traced):
    followed = trace_follow.follow(traced, "round", "1:1")

    emitted = _row_of(followed, "QPU", "emitted round 1")
    assert emitted.tick == 1_000_000
    assert emitted.bits == 8
    move = _row_of(followed, "qpu_to_controller", "move")
    assert move.tick == 1_000_000
    assert move.duration_ticks == 4_000
    assert move.transfer == "move"
    assert move.bits == 8
    intake = _row_of(followed, "Controller", "controller intake copy")
    assert intake.tick == 1_000_000
    assert intake.transfer == "copy"
    assert intake.bits == 8


def test_round_one_sits_in_buffer_zero_as_detection_events(traced):
    followed = trace_follow.follow(traced, "round", "1:1")

    residence = _row_of(followed, "weak syndrome buffer", "residence")
    # the slot is taken where the bits are, at the landing of
    # controller_to_weak_buffer, which is also when the round is readable
    assert residence.tick == 1_008_000
    assert residence.duration_ticks == 5_004_000
    assert residence.transfer == "copy"
    assert residence.bits == 4
    assert "data ready 1.008" in residence.what


def test_round_one_leaves_on_window_zeros_input_move(traced):
    followed = trace_follow.follow(traced, "round", "1:1")

    move = _row_of(followed, "weak_buffer_to_weak_decoder", "move")
    assert move.tick == 6_008_000
    assert move.duration_ticks == 4_000
    assert move.bits == 44
    assert "with W0 rounds 1:1..6" in move.what
    memory = _row_of(followed, "Decoder unit default#0", "residence")
    assert memory.tick == 6_012_000
    assert memory.duration_ticks == 92_000
    assert memory.transfer == "copy"
    assert memory.bits == 44


def test_a_bounded_unit_memory_residence_states_its_room_in_bits():
    landed = {
        "ph": "X",
        "cat": "window,residence",
        "name": "W0 input in memory",
        "thread": "Decoder unit default#0",
        "tid": 7,
        "dur": 0.092,
        "args": {"tick": 6_012_000, "window": "1:0", "capacity_bits": 96},
    }
    document = trace_file.TraceDocument([landed], "one hand-made residence")

    followed = trace_follow.follow(document, "window", "1:0")

    (hop,) = followed.hops
    assert hop.what == "residence, of 96 bits"


def test_round_ones_counts_name_four_copies_and_three_moves(traced):
    followed = trace_follow.follow(traced, "round", "1:1")

    counts = followed.counts
    assert counts.copies == 4
    assert counts.job_references == 1
    assert counts.holds == 1
    assert counts.moves == 3
    assert counts.longest_residence.where == "weak syndrome buffer"
    assert counts.longest_residence.duration_ticks == 5_004_000


def test_a_rounds_path_ends_where_its_bits_land_in_the_unit(traced):
    """data_path.md hop 5: what moves after the decode is the correction."""
    followed = trace_follow.follow(traced, "round", "1:1")

    last = followed.hops[-1]
    assert last.where == "Decoder unit default#0"
    assert last.tick == 6_012_000


def _what_happened_at(hops, where: str) -> list:
    """What each hop at one place did, in hop order."""
    happened = []
    for hop in hops:
        if hop.where == where:
            happened.append(hop.what)
    return happened


def test_a_round_two_windows_read_shows_both_of_them(traced):
    followed = trace_follow.follow(traced, "round", "1:4")

    moves = _what_happened_at(followed.hops, "weak_buffer_to_weak_decoder")

    assert "with W0 rounds 1:1..6" in moves[0]
    assert "with W1 rounds 1:4..9" in moves[1]
    assert followed.counts.job_references == 2


def test_window_zeros_path_runs_from_its_queue_to_the_frame(traced):
    followed = trace_follow.follow(traced, "window", "1:0")

    queued = _row_of(followed, "Window planner", "queued")
    assert queued.tick == 6_008_000
    assert "dispatched to default#0" in queued.what
    service = _row_of(followed, "Decoder unit default#0", "decode service")
    assert service.tick == 6_012_000
    assert service.duration_ticks == 92_000
    correction = _row_of(followed, "Frame", "residence")
    assert correction.tick == 6_108_000
    assert "committed 6.112" in correction.what


def test_a_withdrawn_request_reads_as_withdrawn_when_it_left(tmp_path):
    """double_window takes back 1:3:weak:6 1.132 us after it queued."""
    path = tmp_path / "withdrawn.trace.json"
    observation = {"trace": str(path)}
    sections = {
        **trace_writer_tests.RE_SLICED_WINDOWS,
        "observation": observation,
    }
    shot = measure_tests.switching_run(tmp_path, 20.0, sections=sections)
    shot.machine.observation.trace_writer.write(str(path))
    traced = trace_file.load(path)

    followed = trace_follow.follow(traced, "window", "1:3")

    queued = _row_of(followed, "Window planner", "queued, withdrawn")
    assert queued.tick == 15_008_000
    assert queued.duration_ticks == 1_132_000


def test_window_zeros_stages_are_the_units_own_lanes(traced):
    followed = trace_follow.follow(traced, "window", "1:0")

    fetch = _row_of(followed, "Decoder unit default#0", "stage fetch")
    algorithm = _row_of(followed, "Decoder unit default#0", "stage algorithm")
    release = _row_of(followed, "Decoder unit default#0", "stage release")

    assert fetch.duration_ticks == 24_000
    assert algorithm.duration_ticks == 28_000
    assert release.duration_ticks == 40_000


def test_window_zeros_counts_name_its_one_job_and_its_three_moves(traced):
    followed = trace_follow.follow(traced, "window", "1:0")

    counts = followed.counts
    assert counts.job_references == 1
    assert counts.copies == 1
    assert counts.moves == 3
    assert counts.longest_queue_wait.where == "Window planner"


def test_every_hop_is_ordered_by_its_tick(traced):
    followed = trace_follow.follow(traced, "round", "1:4")

    ticks = [hop.tick for hop in followed.hops]

    assert ticks == sorted(ticks)


def test_the_table_prints_one_line_per_hop_under_a_header(traced):
    followed = trace_follow.follow(traced, "round", "1:1")

    lines = trace_follow.table_lines(followed.hops)

    assert lines[0].startswith("tick (us)")
    assert len(lines) == len(followed.hops) + 1


def test_the_counts_read_as_one_sentence(traced):
    followed = trace_follow.follow(traced, "round", "1:1")

    lines = trace_follow.count_lines(followed.counts)

    assert lines[0] == "copies 4, references 1 job and 1 hold, moves 3"


def test_the_page_carries_every_row_of_the_table(traced):
    followed = trace_follow.follow(traced, "round", "1:1")

    written = trace_follow.page(followed)

    cells = [f"<td>{hop.what}</td>" for hop in followed.hops]
    assert all(cell in written for cell in cells)
    assert written.count("<tr>") == len(followed.hops) + 1


def test_the_page_needs_no_second_file(traced):
    """One self-contained page: no script, no stylesheet, no image."""
    followed = trace_follow.follow(traced, "window", "1:0")

    written = trace_follow.page(followed)

    assert "<script" not in written
    assert "src=" not in written
    assert "http" not in written


def test_the_page_draws_one_lane_per_component(traced):
    followed = trace_follow.follow(traced, "window", "1:0")

    written = trace_follow.page(followed)

    lanes = {hop.where for hop in followed.hops}
    assert written.count("<div class='lane'>") == len(lanes)
    assert "Decoder unit default#0" in written


def test_the_command_prints_the_table_and_writes_the_page(trace_path, tmp_path):
    import decsim.experiments.command as command

    page_path = tmp_path / "one.html"

    command.main(
        [
            "trace",
            "follow",
            str(trace_path),
            "--round",
            "1:1",
            "--html",
            str(page_path),
        ]
    )

    written = page_path.read_text()
    assert "<table>" in written
    assert "d3 seed0" in written


def test_a_command_line_naming_neither_a_round_nor_a_window_is_refused():
    with pytest.raises(ValueError) as refusal:
        trace_follow.main(["follow", "somewhere.json"])

    assert "--round k:n or --window k:n" in str(refusal.value)


def test_a_key_that_is_not_an_operation_and_an_index_is_refused():
    with pytest.raises(ValueError) as refusal:
        trace_follow.main(["follow", "somewhere.json", "--round", "seven"])

    assert "is not a round key" in str(refusal.value)


def test_a_round_is_followed_through_a_strong_window_hold(tmp_path):
    """The hold names its round count apart from the rounds it reads.

    Weak-primary switching with both tiers at once holds window 0's
    strong job until its context lands, and round 1 still reaches the
    strong unit's memory at 24 us on the declared fabric.
    """
    trace_path = tmp_path / "held.trace.json"
    machine = declared_fabric.switching_machine(
        rounds=9,
        escalated_windows=set(),
        run_both_at_once=True,
        trace_path=trace_path,
    )
    machine.run()
    machine.observation.trace_writer.write(str(trace_path))
    document = trace_file.load(trace_path)

    followed = trace_follow.follow(document, "round", "1:1")

    landed = _row_of(followed, "Decoder unit strong#0", "memory copy")
    assert landed.tick == 24_000_000


def test_a_rounds_path_keeps_to_its_own_operation(tmp_path):
    """Two memory operations each read their rounds 1..6 at once.

    Round 2:1 is held by operation 2's window alone and lands in one
    unit's memory, however many operations read a round 1.
    """
    trace_path = tmp_path / "two_operations.trace.json"
    operations = [
        declared_run.memory_operation(1),
        declared_run.memory_operation(2),
    ]
    observation = observe_settings.ObservationSettings(trace=str(trace_path))
    machine = declared_run.weak_only_run(
        rounds=6, operations=operations, observation=observation
    )
    machine.observation.trace_writer.write(str(trace_path))
    document = trace_file.load(trace_path)

    followed = trace_follow.follow(document, "round", "2:1")

    assert followed.counts.holds == 1
    assert followed.counts.job_references == 1


def test_a_window_across_two_operations_is_followed_from_each_ones_rounds(
    tmp_path,
):
    """Window 1:1 reads rounds 4..6 of operation 1 and 1..3 of operation 2.

    Operation 2 names operation 1 its boundary predecessor, so the
    lookahead window reads on into it. Its hold, its readiness and its
    input move each name both operations' rounds, so rounds 1:4 and 2:1
    follow all three, and round 1:1, of the same index as 2:1, none.
    """
    trace_path = tmp_path / "across.trace.json"
    operations = [
        declared_run.memory_operation(1),
        declared_run.memory_operation(2, decoder_boundary_predecessors=(1,)),
    ]
    windows = window_settings.WindowSettings(terminal_policy="lookahead")
    observation = observe_settings.ObservationSettings(trace=str(trace_path))
    machine = declared_run.weak_only_run(
        rounds=6,
        operations=operations,
        windows=windows,
        observation=observation,
    )
    machine.observation.trace_writer.write(str(trace_path))
    document = trace_file.load(trace_path)

    from_the_first = trace_follow.follow(document, "round", "1:4")
    from_the_next = trace_follow.follow(document, "round", "2:1")
    from_the_same_index = trace_follow.follow(document, "round", "1:1")

    ready = ("Window planner", "W1 ready")
    moved = (
        "weak_buffer_to_weak_decoder",
        "move, with W1 rounds 1:4..6 and 2:1..3",
    )
    first_hops = _hops_of(from_the_first)
    next_hops = _hops_of(from_the_next)
    same_index_hops = _hops_of(from_the_same_index)
    assert ready in first_hops
    assert moved in first_hops
    assert ready in next_hops
    assert moved in next_hops
    assert ready not in same_index_hops
    assert moved not in same_index_hops
    # the next operation's own window 0 holds it too
    assert from_the_next.counts.holds == 2


def test_trace_does_only_what_it_says_it_does():
    with pytest.raises(ValueError) as refusal:
        trace_follow.main(["summarise", "somewhere.json"])

    assert "decsim trace has no action summarise" in str(refusal.value)


def test_a_round_the_trace_does_not_carry_is_refused(trace_path):
    with pytest.raises(ValueError) as refused:
        trace_follow.main(["follow", str(trace_path), "--round", "1:999"])

    assert "carries round 1:999" in str(refused.value)
    assert "its round keys are 1:1 to 1:30, 30 in all" in str(refused.value)


def test_a_window_the_trace_does_not_carry_is_refused(trace_path):
    with pytest.raises(ValueError) as refused:
        trace_follow.main(["follow", str(trace_path), "--window", "1:99"])

    assert "carries window 1:99" in str(refused.value)
    assert (
        "its window keys are 1:0, 1:1, 1:2, 1:3, 1:4, 1:5, 1:6, 1:7, 1:8"
        in str(refused.value)
    )

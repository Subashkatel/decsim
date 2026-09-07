"""`decsim trace follow` against the worked example of the trace note.

docs/rewrite/notes/trace_and_viewer.md section 4 prints one round's and
one window's path from gate point 1's own trace (weak_decoder_baseline
d 3 p 0.003 seed 0), and its corrections fix two of the example's
statements against the code: the controller's intake copy is at 1.000,
the send tick, and Buffer 0 holds detection events, so round 1's
residence there carries 4 bits and not the link's 8. The rows below are
the note's, at the hops that exist today.
"""

import pytest

import decsim.front.trace_file as trace_file
import decsim.front.trace_follow as trace_follow
import decsim.machine as machine_module
import tests.observe.gate_point as gate_point

pytestmark = gate_point.needs_the_frozen_suite


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


def _row_of(path, where, what):
    for hop in path.hops:
        if hop.where != where:
            continue
        if what not in hop.what:
            continue
        return hop
    raise AssertionError(f"no {where} hop saying {what}")


def test_round_ones_hops_are_the_notes_table(traced):
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

    residence = _row_of(followed, "Buffer 0", "residence")
    assert residence.tick == 1_004_000
    assert residence.duration_ticks == 5_008_000
    assert residence.transfer == "copy"
    assert residence.bits == 4
    assert "data ready 1.008" in residence.what


def test_round_one_leaves_on_window_zeros_input_move(traced):
    followed = trace_follow.follow(traced, "round", "1:1")

    move = _row_of(followed, "weak_buffer_to_weak_decoder", "move")
    assert move.tick == 6_008_000
    assert move.duration_ticks == 4_000
    assert move.bits == 44
    assert "with W0 rounds 1..6" in move.what
    memory = _row_of(followed, "Decoder unit default#0", "residence")
    assert memory.tick == 6_012_000
    assert memory.duration_ticks == 92_000
    assert memory.transfer == "copy"
    assert memory.bits == 44


def test_round_ones_counts_are_the_notes_counts(traced):
    followed = trace_follow.follow(traced, "round", "1:1")

    counts = followed.counts
    assert counts.copies == 4
    assert counts.job_references == 1
    assert counts.holds == 1
    assert counts.moves == 3
    assert counts.longest_residence.where == "Buffer 0"
    assert counts.longest_residence.duration_ticks == 5_008_000


def test_a_rounds_path_ends_where_its_bits_land_in_the_unit(traced):
    """data_path.md hop 5: what moves after the decode is the correction."""
    followed = trace_follow.follow(traced, "round", "1:1")

    last = followed.hops[-1]
    assert last.where == "Decoder unit default#0"
    assert last.tick == 6_012_000


def test_a_round_two_windows_read_shows_both_of_them(traced):
    followed = trace_follow.follow(traced, "round", "1:4")

    moves = []
    for hop in followed.hops:
        if hop.where == "weak_buffer_to_weak_decoder":
            moves.append(hop.what)

    assert "with W0 rounds 1..6" in moves[0]
    assert "with W1 rounds 4..9" in moves[1]
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

    ticks = []
    for hop in followed.hops:
        ticks.append(hop.tick)

    assert ticks == sorted(ticks)


def test_the_table_prints_one_line_per_hop_under_a_header(traced):
    followed = trace_follow.follow(traced, "round", "1:1")

    lines = trace_follow.table_lines(followed.hops)

    assert lines[0].startswith("tick (us)")
    assert len(lines) == len(followed.hops) + 1


def test_the_counts_read_as_the_notes_sentence(traced):
    followed = trace_follow.follow(traced, "round", "1:1")

    lines = trace_follow.count_lines(followed.counts)

    assert lines[0] == "copies 4, references 1 job and 1 hold, moves 3"


def test_the_page_carries_every_row_of_the_table(traced):
    followed = trace_follow.follow(traced, "round", "1:1")

    written = trace_follow.page(followed)

    for hop in followed.hops:
        assert f"<td>{hop.what}</td>" in written
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

    lanes = []
    for hop in followed.hops:
        if hop.where in lanes:
            continue
        lanes.append(hop.where)
    assert written.count("<div class='lane'>") == len(lanes)
    assert "Decoder unit default#0" in written


def test_the_command_prints_the_table_and_writes_the_page(trace_path, tmp_path):
    import decsim.front.command as command

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
    assert "d3 p0.003 seed0" in written


def test_a_command_line_naming_neither_a_round_nor_a_window_is_refused():
    with pytest.raises(ValueError) as refusal:
        trace_follow.main(["follow", "somewhere.json"])

    assert "--round k:n or --window k:n" in str(refusal.value)


def test_a_key_that_is_not_an_operation_and_an_index_is_refused():
    with pytest.raises(ValueError) as refusal:
        trace_follow.main(["follow", "somewhere.json", "--round", "seven"])

    assert "is not a round key" in str(refusal.value)


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

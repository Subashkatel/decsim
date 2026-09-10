"""The residence table: how long the data sat, how long a move waited.

The contract, on a trace file written by hand so the answers are known
before the reader runs: a structure's row is the residences the trace
records on that structure's lane, in microseconds off `args.tick` and
the event's own duration, and a link path's row is the queue waits its
moves carry. The file is the Chrome Trace Event Format the writer emits
(docs/how-to/read_a_trace.md), read through front/trace_file.py, the reader
`decsim trace follow` uses.
"""

import json

import decsim.config as config
import decsim.front.report as report
import decsim.front.residence as residence
import decsim.front.trace_file as trace_file

STORE_LANE = 1
UNIT_LANE = 2
LINK_LANE = 3
LANE_NAMES = {
    STORE_LANE: "Buffer 0",
    UNIT_LANE: "Decoder unit default#0",
    LINK_LANE: "controller_to_weak_buffer",
}


def complete_event(lane, name, category, start_ticks, span_ticks, args):
    """One Chrome complete event, as the trace writer records one."""
    every_arg = dict(args)
    every_arg["tick"] = start_ticks
    return {
        "ph": "X",
        "name": name,
        "cat": category,
        "ts": start_ticks / config.TICKS_PER_MICROSECOND,
        "dur": span_ticks / config.TICKS_PER_MICROSECOND,
        "pid": 1,
        "tid": lane,
        "args": every_arg,
    }


def instant_event(lane, name, category, tick):
    """One Chrome instant event, which has no length and is no residence."""
    return {
        "ph": "i",
        "name": name,
        "cat": category,
        "ts": tick / config.TICKS_PER_MICROSECOND,
        "pid": 1,
        "tid": lane,
        "args": {"tick": tick},
    }


def one_shots_trace(tmp_path):
    """A shot with two rounds in a store, one job in a unit, two moves."""
    rows = [
        {
            "ph": "M",
            "name": "process_name",
            "pid": 1,
            "tid": 0,
            "args": {"name": "one shot"},
        }
    ]
    for lane in sorted(LANE_NAMES):
        named = {
            "ph": "M",
            "name": "thread_name",
            "pid": 1,
            "tid": lane,
            "args": {"name": LANE_NAMES[lane]},
        }
        rows.append(named)
    first_round = complete_event(
        STORE_LANE,
        "round 1",
        "round,residence",
        1_000_000,
        3_000_000,
        {"round": "1:1"},
    )
    rows.append(first_round)
    second_round = complete_event(
        STORE_LANE,
        "round 2",
        "round,residence",
        2_000_000,
        5_000_000,
        {"round": "1:2"},
    )
    rows.append(second_round)
    unit_input = complete_event(
        UNIT_LANE,
        "W0 input in memory",
        "window,residence",
        6_000_000,
        2_000_000,
        {"window": "1:0"},
    )
    rows.append(unit_input)
    copy = instant_event(STORE_LANE, "Buffer 0 copy", "copy", 1_000_000)
    rows.append(copy)
    first_move = complete_event(
        LINK_LANE,
        "round 1 move",
        "round,link",
        500_000,
        400_000,
        {"queue_wait_ticks": 100_000},
    )
    rows.append(first_move)
    second_move = complete_event(
        LINK_LANE,
        "round 2 move",
        "round,link",
        900_000,
        400_000,
        {"queue_wait_ticks": 300_000},
    )
    rows.append(second_move)
    path = tmp_path / "one.trace.json"
    written = json.dumps(rows)
    path.write_text(written)
    return path


class _TracedShot:
    """A measurement's fields the residence table reads, and nothing else."""

    def __init__(self, trace_path):
        self.trace_path = trace_path
        self.seed = 0
        self.distance = 3
        self.physical_error_probability = 0.001
        self.algorithm = 1.0
        self.round_period_us = 1.0


def test_a_structures_residences_are_its_lanes_complete_events(tmp_path):
    path = one_shots_trace(tmp_path)
    document = trace_file.load(path)

    held = residence.residence_ticks_by_structure(document)

    assert held["Buffer 0"] == [3_000_000, 5_000_000]
    assert held["Decoder unit default#0"] == [2_000_000]


def test_an_instant_is_not_a_residence_because_it_has_no_length(tmp_path):
    path = one_shots_trace(tmp_path)
    document = trace_file.load(path)

    held = residence.residence_ticks_by_structure(document)

    assert len(held["Buffer 0"]) == 2


def test_a_link_paths_waits_are_the_queue_wait_its_moves_carry(tmp_path):
    path = one_shots_trace(tmp_path)
    document = trace_file.load(path)

    waited = residence.queue_wait_ticks_by_link_path(document)

    assert waited["controller_to_weak_buffer"] == [100_000, 300_000]


def test_a_rows_mean_and_longest_are_the_samples_in_microseconds(tmp_path):
    path = one_shots_trace(tmp_path)
    measurement = _TracedShot(str(path))

    rows = residence.rows_of([measurement])
    by_name = {}
    for row in rows:
        by_name[row["name"]] = row

    store = by_name["Buffer 0"]
    assert store["counting"] == "residence"
    assert store["samples"] == 2
    assert store["mean_us"] == 4.0
    assert store["longest_us"] == 5.0
    wire = by_name["controller_to_weak_buffer"]
    assert wire["counting"] == "link_path"
    assert wire["samples"] == 2
    assert wire["mean_us"] == 0.2
    assert wire["longest_us"] == 0.3


def test_a_row_names_the_sweep_point_its_shot_ran_at(tmp_path):
    path = one_shots_trace(tmp_path)
    measurement = _TracedShot(str(path))

    rows = residence.rows_of([measurement])

    for row in rows:
        assert row["distance"] == 3
        assert row["physical_error_probability"] == 0.001
        assert row["round_period_us"] == 1.0
        assert row["seed"] == 0


def test_a_shot_that_was_not_traced_writes_no_row(tmp_path):
    """The trace is per shot by design, so an untraced point has no table."""
    measurement = _TracedShot(None)

    rows = residence.rows_of([measurement])
    residence.write_residence(rows, tmp_path)
    written = tmp_path / "residence.csv"

    assert rows == []
    assert not written.exists()


def test_a_traced_run_writes_the_table_beside_its_rows(tmp_path, monkeypatch):
    """The run folder's own file, off the trace the run already wrote."""
    from decsim.front.collect_command import run_experiment
    from tests.front.yaml_configs import write_config

    monkeypatch.chdir(tmp_path)
    config_path = write_config(
        tmp_path,
        {
            "observation": {"trace": "chrome", "trace_shots": [0]},
            "sweep": [
                {
                    "physical_error_probability": [0.001],
                    "distance": [3],
                    "round_period_us": [1.0],
                    "shots": 2,
                }
            ],
        },
    )
    run_dir, _rows = run_experiment(config_path)
    written = run_dir / "residence.csv"
    rows = report.read_rows(written)
    seeds = set()
    names = set()
    for row in rows:
        seeds.add(row["seed"])
        names.add(row["name"])

    assert written.exists()
    assert seeds == {0}
    assert "Buffer 0" in names
    assert "controller_to_weak_buffer" in names

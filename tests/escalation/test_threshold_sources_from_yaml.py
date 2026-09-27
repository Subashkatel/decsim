"""The switching threshold's three sources: fixed, table, online.

fixed uses the card's gap_threshold_db as given (the paper's constant
gth). table computes nothing at run time: it looks the sweep point up
in an offline calibration csv, its key columns headed by the yaml paths
they match in the point's resolved sections, and refuses a point the
table does not certify. online starts at
gap_threshold_db and adapts it across the point's shots with the
two-loop controller (rate tracker + audit lane); one calibrator per
point, shared by every shot, so the controller learns over the point's
whole window stream.
"""

import csv
import math
import random
import re

import pytest
import yaml

import decsim.build.escalation as escalation_build
import decsim.escalation.settings as escalation_settings
import decsim.experiments.collect_command as collect_command
import tests.experiments.yaml_configs as yaml_configs
from decsim.experiments.collect_command import run_sweep
from decsim.experiments.experiment import load_experiment
from tests.escalation.test_switching_mode import (
    NEAR_THRESHOLD_P,
    switching_config,
)

NATS_TO_DB = 10.0 / math.log(10.0)


def point_task(config, *, physical_error_probability, distance):
    values = {
        yaml_configs.ERROR_RATE_PATH: physical_error_probability,
        "qpu.distance": distance,
        "qpu.round_period_microseconds": 1.0,
    }
    return config.point_task(values, 1)


def resolve_gap_threshold_nats(config, *, physical_error_probability, distance):
    task = point_task(
        config,
        physical_error_probability=physical_error_probability,
        distance=distance,
    )
    return task.settings.escalation.gap_threshold_nats


def online_threshold_calibrator(
    config, *, physical_error_probability, distance
):
    task = point_task(
        config,
        physical_error_probability=physical_error_probability,
        distance=distance,
    )
    return task.online_threshold


def first_escalation(config):
    task = config.first_point_task()
    return task.settings.escalation


def source_config(tmp_path, switching_card: dict, shots: int = 1):
    config_path = switching_config(tmp_path, 20.0)
    config_text = config_path.read_text()
    raw = yaml.safe_load(config_text)
    raw["escalation"] = {"kind": "switching", **switching_card}
    raw["sweep"][0]["shots"] = shots
    edited_text = yaml.safe_dump(raw)
    config_path.write_text(edited_text)
    return config_path


def calibration_table(tmp_path) -> str:
    table_path = tmp_path / "calibration_table.csv"
    table_path.write_text(
        f"qpu.distance,{yaml_configs.ERROR_RATE_PATH},gth_brute_force,gth_eq4,"
        "gth_eq4_wilson\n"
        f"3,{NEAR_THRESHOLD_P},2.5,18.0,19.5\n"
        "5,0.005,10.0,20.1,21.4\n"
        f"7,{NEAR_THRESHOLD_P},,20.0,\n"
    )
    return table_path.name


def test_fixed_is_the_default_source(tmp_path):
    config_path = switching_config(tmp_path, 20.0)
    config = load_experiment(config_path)
    escalation = first_escalation(config)
    assert escalation.threshold_source == "fixed"
    resolved = resolve_gap_threshold_nats(
        config, physical_error_probability=NEAR_THRESHOLD_P, distance=3
    )
    assert resolved == escalation.gap_threshold_nats
    assert (
        online_threshold_calibrator(
            config, physical_error_probability=NEAR_THRESHOLD_P, distance=3
        )
        is None
    )


def test_table_source_resolves_the_sweep_point_and_refuses_others(tmp_path):
    table = calibration_table(tmp_path)
    wilson_card = {"threshold_source": "table", "threshold_table": table}
    wilson_path = source_config(tmp_path, wilson_card)
    config = load_experiment(wilson_path)
    escalation = first_escalation(config)
    assert escalation.gap_threshold_db is None
    assert escalation.threshold_column == "gth_eq4_wilson"

    resolved = resolve_gap_threshold_nats(
        config, physical_error_probability=NEAR_THRESHOLD_P, distance=3
    )
    resolved_decibels = resolved * NATS_TO_DB
    assert math.isclose(resolved_decibels, 19.5)

    off_the_table = {"qpu.distance": 3, yaml_configs.ERROR_RATE_PATH: 0.002}
    sentence = re.escape(f"has no row for {off_the_table}")
    with pytest.raises(ValueError, match=sentence):
        resolve_gap_threshold_nats(
            config, physical_error_probability=0.002, distance=3
        )
    with pytest.raises(ValueError, match="entry is empty"):
        resolve_gap_threshold_nats(
            config, physical_error_probability=NEAR_THRESHOLD_P, distance=7
        )

    brute_card = {
        "threshold_source": "table",
        "threshold_table": table,
        "threshold_column": "gth_brute_force",
    }
    brute_path = source_config(tmp_path, brute_card)
    brute = load_experiment(brute_path)
    resolved = resolve_gap_threshold_nats(
        brute, physical_error_probability=NEAR_THRESHOLD_P, distance=3
    )
    resolved_decibels = resolved * NATS_TO_DB
    assert math.isclose(resolved_decibels, 2.5)


@pytest.mark.parametrize(
    "cell, sentence",
    [("-5.0", "must be finite and not negative"), ("abc", "number of dec")],
)
def test_a_table_entry_that_is_no_nonnegative_decibel_count_is_refused(
    tmp_path, cell, sentence
):
    table_path = tmp_path / "negative_table.csv"
    header = f"qpu.distance,{yaml_configs.ERROR_RATE_PATH},gth_eq4_wilson"
    table_path.write_text(f"{header}\n3,0.002,{cell}\n")
    card = {"threshold_source": "table", "threshold_table": table_path.name}
    config_path = source_config(tmp_path, card)
    config = load_experiment(config_path)
    with pytest.raises(ValueError, match=sentence):
        resolve_gap_threshold_nats(
            config, physical_error_probability=0.002, distance=3
        )


def test_a_table_with_no_key_column_is_refused(tmp_path):
    """Headers that name no yaml path would match every point to row one."""
    table_path = tmp_path / "unkeyed_table.csv"
    table_path.write_text(
        f"distance,p,gth_eq4_wilson\n3,{NEAR_THRESHOLD_P},19.5\n"
    )
    card = {"threshold_source": "table", "threshold_table": table_path.name}
    config_path = source_config(tmp_path, card)
    config = load_experiment(config_path)

    with pytest.raises(ValueError, match="has no key column"):
        config.first_point_task()


def test_a_window_only_sweep_finds_its_table_row_by_path(tmp_path):
    """The distance and the error rate are written once and not swept.

    The table's key columns read them from each point's resolved
    sections, so a sweep over the window alone finds its row.
    """
    table_path = tmp_path / "window_table.csv"
    table_path.write_text(
        f"qpu.distance,{yaml_configs.ERROR_RATE_PATH},windows.commit_rounds,"
        "gth_eq4_wilson\n"
        f"3,{NEAR_THRESHOLD_P},2,12.0\n"
        f"3,{NEAR_THRESHOLD_P},3,13.0\n"
    )
    card = {"threshold_source": "table", "threshold_table": table_path.name}
    config_path = source_config(tmp_path, card)
    config_text = config_path.read_text()
    raw = yaml.safe_load(config_text)
    raw["qpu"]["distance"] = 3
    raw["qpu"]["round_period_microseconds"] = 1.0
    raw["workload"]["arguments"]["physical_error_probability"] = (
        NEAR_THRESHOLD_P
    )
    raw["sweep"] = [{"axes": {"windows.commit_rounds": [2, 3]}, "shots": 1}]
    edited_text = yaml.safe_dump(raw)
    config_path.write_text(edited_text)
    config = load_experiment(config_path)

    two, three = config.tasks()

    two_nats = two.settings.escalation.gap_threshold_nats
    three_nats = three.settings.escalation.gap_threshold_nats
    two_decibels = two_nats * NATS_TO_DB
    three_decibels = three_nats * NATS_TO_DB
    assert math.isclose(two_decibels, 12.0)
    assert math.isclose(three_decibels, 13.0)
    assert two.metadata == {"windows.commit_rounds": 2}


def test_an_integer_key_matches_its_row_exactly(tmp_path):
    """Only a float key reads back within a relative 1e-9 of its text.

    A whole number is written exactly, so 1000000001 finds no row keyed
    1000000000 although the two lie within 1e-9 of each other.
    """
    table_path = tmp_path / "window_table.csv"
    table_path.write_text(
        "windows.commit_rounds,gth_eq4_wilson\n1000000000,12.0\n"
    )
    card = {"threshold_source": "table", "threshold_table": table_path.name}
    config_path = source_config(tmp_path, card)
    config_text = config_path.read_text()
    raw = yaml.safe_load(config_text)
    raw["qpu"]["distance"] = 3
    raw["workload"]["arguments"]["physical_error_probability"] = (
        NEAR_THRESHOLD_P
    )
    raw["sweep"] = [
        {"axes": {"windows.commit_rounds": [1000000001]}, "shots": 1}
    ]
    edited_text = yaml.safe_dump(raw)
    config_path.write_text(edited_text)
    config = load_experiment(config_path)
    sentence = (
        "has no row for {'windows.commit_rounds': 1000000001}; its rows "
        "are [{'windows.commit_rounds': '1000000000'}]"
    )
    pattern = re.escape(sentence)

    with pytest.raises(ValueError, match=pattern):
        config.tasks()


def test_table_source_key_guards(tmp_path):
    table = calibration_table(tmp_path)
    both_sources_card = {
        "threshold_source": "table",
        "threshold_table": table,
        "gap_threshold_db": 20.0,
    }
    both_sources_path = source_config(tmp_path, both_sources_card)
    with pytest.raises(ValueError, match="drop gap_threshold_db"):
        load_experiment(both_sources_path)
    no_table_card = {"threshold_source": "table"}
    no_table_path = source_config(tmp_path, no_table_card)
    with pytest.raises(ValueError, match="needs threshold_table"):
        load_experiment(no_table_path)
    fixed_with_table_card = {
        "gap_threshold_db": 20.0,
        "threshold_table": table,
    }
    fixed_with_table_path = source_config(tmp_path, fixed_with_table_card)
    with pytest.raises(ValueError, match="belong to"):
        load_experiment(fixed_with_table_path)
    missing_table_card = {
        "threshold_source": "table",
        "threshold_table": "missing.csv",
    }
    missing_table_path = source_config(tmp_path, missing_table_card)
    missing_table = load_experiment(missing_table_path)
    with pytest.raises(ValueError, match="does not exist"):
        resolve_gap_threshold_nats(
            missing_table,
            physical_error_probability=NEAR_THRESHOLD_P,
            distance=3,
        )


def test_online_card_guards(tmp_path):
    forward_window_card = {
        "threshold_source": "online",
        "gap_threshold_db": 20.0,
        "strong_window": "forward_seam_pinned",
    }
    forward_window_path = source_config(tmp_path, forward_window_card)
    with pytest.raises(ValueError, match="serial-only"):
        load_experiment(forward_window_path)
    fixed_with_online_card = {
        "gap_threshold_db": 20.0,
        "online": {"audit_rate": 0.1},
    }
    fixed_with_online_path = source_config(tmp_path, fixed_with_online_card)
    with pytest.raises(ValueError, match="online card belongs"):
        load_experiment(fixed_with_online_path)
    unknown_key_card = {
        "threshold_source": "online",
        "gap_threshold_db": 20.0,
        "online": {"audit_probability": 0.1},
    }
    unknown_key_path = source_config(tmp_path, unknown_key_card)
    with pytest.raises(ValueError, match="does not know"):
        load_experiment(unknown_key_path)
    high_audit_rate_card = {
        "threshold_source": "online",
        "gap_threshold_db": 20.0,
        "online": {"audit_rate": 1.5},
    }
    high_audit_rate_path = source_config(tmp_path, high_audit_rate_card)
    with pytest.raises(ValueError, match="audit_rate"):
        load_experiment(high_audit_rate_path)


def test_a_target_that_leaves_no_room_for_the_audits_is_refused(tmp_path):
    """The audits reach the strong tier beside the target.

    Toshio 2510.25222 lines 1333-1340 count every strong decode in the
    backlog, so a target of 0.30 under a 0.30 cap with audit_rate 0.01
    would put the strong duty past the cap.
    """
    card = {
        "threshold_source": "online",
        "gap_threshold_db": 20.0,
        "online": {
            "target_escalation_rate": 0.30,
            "audit_rate": 0.01,
            "max_escalation_rate": 0.30,
        },
    }
    config_path = source_config(tmp_path, card)
    with pytest.raises(ValueError, match="max_escalation_rate - audit_rate"):
        load_experiment(config_path)


@pytest.mark.parametrize(
    "online, sentence",
    [
        ({"step_db": True}, "online.step_db must be a finite number"),
        ({"step_db": "abc"}, "online.step_db must be a finite number"),
        ({"adjust_factor": math.inf}, "online.adjust_factor must be a finite"),
        ({"step_db": math.nan}, "online.step_db must be a finite number"),
        (5, "escalation.online must be a mapping"),
    ],
)
def test_an_online_knob_that_is_no_finite_number_is_refused(
    tmp_path, online, sentence
):
    card = {
        "threshold_source": "online",
        "gap_threshold_db": 20.0,
        "online": online,
    }
    config_path = source_config(tmp_path, card)
    with pytest.raises(ValueError, match=sentence):
        load_experiment(config_path)


def test_an_online_target_written_in_exponent_form_loads(tmp_path):
    """YAML 1.1 reads 1e-3 as text; the card reads it as the number."""
    card = {
        "threshold_source": "online",
        "gap_threshold_db": 20.0,
        "online": {"target_escalation_rate": "1e-3"},
    }
    config_path = source_config(tmp_path, card)
    config = load_experiment(config_path)
    escalation = first_escalation(config)
    assert escalation.online.target_escalation_rate == 0.001


def test_the_online_seed_reads_its_distance_and_error_rate_by_path(tmp_path):
    """The seed text is the one the frozen gate's online point ran with.

    Its two numbers are read from the point's resolved sections.
    """
    card = {"threshold_source": "online", "gap_threshold_db": 20.0}
    config_path = source_config(tmp_path, card)
    config = load_experiment(config_path)

    task = config.first_point_task()

    expected = random.Random(f"online-threshold d=3 p={NEAR_THRESHOLD_P}")
    generator = task.online_threshold.random_generator
    assert generator.getstate() == expected.getstate()


def test_online_source_learns_across_a_point_and_records_the_path(tmp_path):
    """One calibrator serves every shot of the point.

    Its window count spans all shots, every audit resolves, and the
    trajectory csv lands in the run dir, its rows named by the point's
    id, swept values and algorithm as every other csv's are.
    """
    online_card = {
        "threshold_source": "online",
        "gap_threshold_db": 15.0,
        "online": {
            "audit_rate": 0.3,
            "target_escalation_rate": 0.2,
            "max_escalation_rate": 0.5,
        },
    }
    config_path = source_config(tmp_path, online_card, shots=2)
    config = load_experiment(config_path)
    out_dir = tmp_path / "results"

    run_dir, _rows = collect_command.run_experiment(config_path, out_dir)

    shots_path = run_dir / "shots.csv"
    shots = _csv_rows(shots_path)
    windows_per_shot = int(shots[0]["decoded_windows"])
    calibrator = online_threshold_calibrator(
        config, physical_error_probability=NEAR_THRESHOLD_P, distance=3
    )
    summary = calibrator.summary()
    assert len(shots) == 2
    assert summary["windows"] == 0  # a fresh one is fresh
    trajectory_paths = run_dir.glob("online_threshold_*.csv")
    (trajectory_path,) = list(trajectory_paths)
    trajectory = _csv_rows(trajectory_path)
    first_row = trajectory[0]
    last_row = trajectory[-1]
    assert list(first_row) == [
        "point_id",
        "qpu.distance",
        "qpu.round_period_microseconds",
        yaml_configs.ERROR_RATE_PATH,
        "algorithm",
        "window_count",
        "threshold_db",
        "event",
    ]
    assert first_row["point_id"] == shots[0]["point_id"]
    assert first_row["qpu.distance"] == "3"
    assert first_row["algorithm"] == shots[0]["algorithm"]
    assert (first_row["window_count"], first_row["threshold_db"]) == (
        "0",
        "15.0",
    )
    assert first_row["event"] == "start"
    assert int(last_row["window_count"]) > windows_per_shot


def _csv_rows(path) -> list:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def test_online_source_reproduces_its_decisions(tmp_path):
    """The calibrator's random stream is seeded by the point identity.

    Rerunning the point reruns the same audits.
    """
    online_card = {
        "threshold_source": "online",
        "gap_threshold_db": 15.0,
        "online": {
            "audit_rate": 0.3,
            "target_escalation_rate": 0.2,
            "max_escalation_rate": 0.5,
        },
    }
    config_path = source_config(tmp_path, online_card, shots=2)
    config = load_experiment(config_path)

    first_tasks = config.tasks()
    second_tasks = config.tasks()
    first = run_sweep(first_tasks, None)
    second = run_sweep(second_tasks, None)

    first_links = [
        measurement.link_totals["weak_decoder_to_strong_decoder"]["transfers"]
        for measurement in first
    ]
    second_links = [
        measurement.link_totals["weak_decoder_to_strong_decoder"]["transfers"]
        for measurement in second
    ]
    assert first_links == second_links
    assert [m.logical_failure for m in first] == [
        m.logical_failure for m in second
    ]


# ---- a threshold row written outside decsim


class _OutsideCalibratedThreshold:
    """A threshold row written outside decsim whose number comes from a csv.

    It subclasses no shipped row: the three facts the yaml boundary reads
    are declared here and the decision is a constant, exactly as the
    shipped rows behave once the experiments layer has resolved the
    point.
    """

    audits_by_escalating = False
    reads_a_calibration_table = True
    built_per_sweep_point = False

    def __init__(self, threshold_nats: float) -> None:
        self.threshold_nats = threshold_nats

    def decide_keep(self, job, result) -> bool:
        """The gap against the threshold, both in nats."""
        del job
        return result.soft_output.gap >= self.threshold_nats

    def learn_from_strong_result(self, window_key, result) -> None:
        """A constant learns nothing."""
        del window_key
        del result


class _OutsideLearningThreshold(_OutsideCalibratedThreshold):
    """A threshold row from outside that builds its own per-point source.

    for_sweep_point is what built_per_sweep_point promises. This row
    starts at the point's threshold in nats and learns nothing, which is
    all the law needs: what the experiments layer installs is an
    instance of the row the table names.
    """

    reads_a_calibration_table = False
    built_per_sweep_point = True

    @classmethod
    def for_sweep_point(
        cls, online, threshold_nats: float, resolved
    ) -> "_OutsideLearningThreshold":
        """One instance of this row for the point."""
        del online
        del resolved
        return cls(threshold_nats)


def test_an_outside_row_that_reads_a_table_gets_the_column_and_no_card(
    tmp_path, monkeypatch
):
    """The table keys follow reads_a_calibration_table, not the row's name."""
    monkeypatch.setitem(
        escalation_settings.THRESHOLD_SOURCES,
        "outside_calibrated",
        _OutsideCalibratedThreshold,
    )
    table = calibration_table(tmp_path)
    table_card = {
        "threshold_source": "outside_calibrated",
        "threshold_table": table,
        "threshold_column": "gth_eq4",
    }
    table_path = source_config(tmp_path, table_card)
    config = load_experiment(table_path)
    escalation = first_escalation(config)
    assert escalation.threshold_column == "gth_eq4"
    with_card = {
        "threshold_source": "outside_calibrated",
        "threshold_table": table,
        "gap_threshold_db": 20.0,
    }
    with_card_path = source_config(tmp_path, with_card)

    with pytest.raises(ValueError, match="drop gap_threshold_db"):
        load_experiment(with_card_path)


def test_an_outside_row_built_per_point_gets_the_online_card(
    tmp_path, monkeypatch
):
    """The online card follows built_per_sweep_point, not the row's name."""
    monkeypatch.setitem(
        escalation_settings.THRESHOLD_SOURCES,
        "outside_learning",
        _OutsideLearningThreshold,
    )
    learning_card = {
        "threshold_source": "outside_learning",
        "gap_threshold_db": 20.0,
        "online": {"audit_rate": 0.2},
    }
    config_path = source_config(tmp_path, learning_card)

    config = load_experiment(config_path)

    escalation = first_escalation(config)
    assert escalation.online.audit_rate == 0.2


def test_an_outside_row_built_per_point_is_the_installed_source(
    tmp_path, monkeypatch
):
    """The row builds its own per-point source, and that is what runs.

    A row that declares built_per_sweep_point has its card read and
    builds the source itself; the instance the experiments layer puts on
    the point's task is the row's own, and it reaches the policy the
    root builds for the shot (build/escalation.py _threshold_source,
    which reads the same declaration).
    """
    monkeypatch.setitem(
        escalation_settings.THRESHOLD_SOURCES,
        "outside_learning",
        _OutsideLearningThreshold,
    )
    learning_card = {
        "threshold_source": "outside_learning",
        "gap_threshold_db": 20.0,
        "online": {"audit_rate": 0.2},
    }
    config_path = source_config(tmp_path, learning_card)
    config = load_experiment(config_path)
    task = point_task(
        config, physical_error_probability=NEAR_THRESHOLD_P, distance=3
    )

    installed = task.online_threshold

    assert type(installed) is _OutsideLearningThreshold
    expected_nats = task.settings.escalation.gap_threshold_nats
    assert installed.threshold_nats == expected_nats
    shot_settings = task.shot_settings()
    policy = escalation_build.build_escalation_policy(
        shot_settings.escalation, shot_settings.weak_decoder
    )
    assert policy.threshold is installed

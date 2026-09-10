"""The data-movement figure: bits per shot by memory class, against d.

The figure's contract: one panel per run folder, one line per memory
class and per quantity that has bits, read from that folder's
data_movement.csv memory-class rows. The classes are kept apart because
the classical sources make the class the cost (Horowitz, ISSCC 2014
lines 232-247; Dally, CACM 2020 lines 231-234), and a reference has no
line at all, because a hold books the rounds it keeps where they are and
copies no bits (observe/data_movement.py).
"""

import pytest

import decsim.front.plots as plots
import decsim.front.refusal as refusal
from decsim.front.collect_command import run_experiment
from tests.front.yaml_configs import write_config

COUNTING_SWEEP = {
    "observation": {"data_movement": True},
    "sweep": [
        {
            "physical_error_probability": [0.001],
            "distance": [3, 5],
            "round_period_us": [1.0],
            "shots": 1,
        }
    ],
}


def counting_run(tmp_path, out_name):
    """One small counting run, in a folder of its own."""
    config_path = write_config(tmp_path, COUNTING_SWEEP)
    out_dir = tmp_path / out_name
    run_dir, _rows = run_experiment(config_path, out_dir)
    return run_dir


def test_a_class_series_is_that_classs_bits_at_each_swept_distance(tmp_path):
    run_dir = counting_run(tmp_path, "one")

    series = plots.memory_class_series(run_dir)

    assert series["on_chip"]["copied"][0] == [3, 5]
    assert series["on_board"]["copied"][0] == [3, 5]
    for _distances, bits in series["on_chip"].values():
        for value in bits:
            assert value > 0


def test_an_off_board_hop_of_this_machine_moves_and_never_copies(tmp_path):
    """Its copied series is empty, so no zero is drawn on the log axis."""
    run_dir = counting_run(tmp_path, "one")

    series = plots.memory_class_series(run_dir)

    assert series["off_board"]["copied"] == ([], [])
    assert series["off_board"]["moved"][0] == [3, 5]


def test_the_classes_are_listed_cheapest_first(tmp_path):
    run_dir = counting_run(tmp_path, "one")

    series = plots.memory_class_series(run_dir)

    assert list(series) == ["on_chip", "on_board", "off_board"]


def test_the_figure_is_drawn_from_one_run_folder(tmp_path):
    run_dir = counting_run(tmp_path, "one")
    figure_path = tmp_path / "data_movement.png"

    plots.figure("data_movement", [run_dir], figure_path)

    assert figure_path.exists()


def test_the_figure_is_drawn_from_several_study_folders(tmp_path):
    first_dir = counting_run(tmp_path, "first")
    second_dir = counting_run(tmp_path, "second")
    figure_path = tmp_path / "data_movement.png"

    plots.figure("data_movement", [first_dir, second_dir], figure_path)

    assert figure_path.exists()


def test_a_run_that_counted_no_movement_is_refused(tmp_path):
    """A run with observation.data_movement off wrote no rows to draw."""
    silent_path = write_config(
        tmp_path,
        {
            "sweep": [
                {
                    "physical_error_probability": [0.001],
                    "distance": [3],
                    "round_period_us": [1.0],
                    "shots": 1,
                }
            ]
        },
    )
    out_dir = tmp_path / "silent"
    run_dir, _rows = run_experiment(silent_path, out_dir)
    figure_path = tmp_path / "data_movement.png"

    with pytest.raises(refusal.RefusalError, match="data_movement.csv"):
        plots.figure("data_movement", [run_dir], figure_path)

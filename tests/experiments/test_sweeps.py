"""Sweeps: an axis is a machine's argument, a path of it, or a constant.

The cells run in the grid's order, the last axis fastest; a cell's
paths are assigned before the command line's, and an assignment in
force to a swept value is refused, so no cell runs a value its name
does not say.
"""

import pathlib

import pytest

import decsim.experiments.collect as collect
import decsim.experiments.collection as collection
import decsim.experiments.sweeps as sweeps
import decsim.parameters.fields as fields
import decsim.parameters.in_force as in_force
import decsim.parameters.tree as tree
import decsim.presets.machines as machines
import decsim.topology.constants as constants_module

STOP = collection.CollectionSettings(max_shots=10)
CONSTANTS = """
memory:
  probability: 0.002  # a chosen rate
  spare: 3
"""


def test_an_argument_axis_makes_each_cells_machine():
    sweep = _sweep(grid={"distance": [3, 5]}, task_name="d{distance}")

    tasks = sweep.tasks()

    names = [task.name for task in tasks]
    distances = [task.machine.qpu.distance for task in tasks]
    assert names == ["d3", "d5"]
    assert distances == [3, 5]


def test_a_path_axis_is_assigned_to_each_cells_machine():
    sweep = _sweep(
        grid={"distance": [3], "weak.unit_count": [1, 2]},
        task_name="d{distance}_units{weak.unit_count}",
    )

    tasks = sweep.tasks()

    names = [task.name for task in tasks]
    units = [task.machine.weak_decoder.unit_count for task in tasks]
    assert names == ["d3_units1", "d3_units2"]
    assert units == [1, 2]


def test_a_task_of_a_cell_is_the_task_its_machine_makes():
    sweep = _sweep(grid={"distance": [5]}, task_name="d{distance}")

    (task,) = sweep.tasks()

    machine = _machine(5)
    built = collect.Task("d5", machine, {"distance": 5}, STOP)
    assert task.strong_id() == built.strong_id()
    assert task.collection == STOP


def test_a_constant_axis_sets_the_constant_for_its_cell(tmp_path):
    constants = _constants(tmp_path)
    machine = _machine_reading(constants)
    sweep = sweeps.Sweep(
        "rates",
        machine,
        {"constants.memory.probability": [0.001, 0.004]},
        "p{constants.memory.probability}",
        stop=STOP,
        constants=constants,
        allow_unread=["memory.spare"],
    )

    tasks = sweep.tasks()

    names = [task.name for task in tasks]
    rates = [_probability(task) for task in tasks]
    assert names == ["p0.001", "p0.004"]
    assert rates == [0.001, 0.004]


def test_a_constant_no_cell_read_is_refused(tmp_path):
    constants = _constants(tmp_path)
    machine = _machine_reading(constants)
    sweep = sweeps.Sweep(
        "rates",
        machine,
        {"distance": [3]},
        "d{distance}",
        stop=STOP,
        constants=constants,
    )

    with pytest.raises(fields.ParameterError, match="memory.spare"):
        sweep.tasks()


def test_metadata_is_the_cells_values_or_what_its_function_writes():
    plain = _sweep(grid={"distance": [3]}, task_name="d{distance}")
    written = _sweep(
        grid={"distance": [3]},
        task_name="d{distance}",
        metadata=_qpu_metadata,
    )

    (plain_task,) = plain.tasks()
    (written_task,) = written.tasks()

    assert plain_task.metadata == {"distance": 3}
    assert written_task.metadata == {"qpu.distance": 3, "rate": 0.002}


def test_a_stop_may_be_a_function_of_the_cell():
    sweep = _sweep(
        grid={"distance": [3, 5]},
        task_name="d{distance}",
        stop=_stop_by_distance,
    )

    tasks = sweep.tasks()

    shots = [task.collection.max_shots for task in tasks]
    assert shots == [30, 50]


def test_an_assignment_in_force_to_a_swept_path_is_refused():
    sweep = _sweep(
        grid={"distance": [3], "weak.unit_count": [1, 2]},
        task_name="units{weak.unit_count}",
    )
    units = tree.Assignment("weak.unit_count", 4, "--set")

    with pytest.raises(fields.ParameterError, match="sweeps"):
        with in_force.in_force([units]):
            sweep.tasks()


def test_an_assignment_in_force_that_makes_two_cells_one_is_refused():
    sweep = _sweep(grid={"distance": [3, 5]}, task_name="d{distance}")
    distance = tree.Assignment("qpu.distance", 7, "--set")

    with pytest.raises(fields.ParameterError, match="d3 and d5"):
        with in_force.in_force([distance]):
            sweep.tasks()


def test_an_assignment_in_force_to_a_value_no_axis_names_is_kept():
    sweep = _sweep(grid={"distance": [3, 5]}, task_name="d{distance}")
    units = tree.Assignment("weak.unit_count", 2, "--set")

    with in_force.in_force([units]):
        tasks = sweep.tasks()

    counts = [task.machine.weak_decoder.unit_count for task in tasks]
    assert counts == [2, 2]


def test_an_axis_that_names_nothing_is_refused_with_the_near_names():
    unknown_argument = _sweep(grid={"distnce": [3]}, task_name="t")
    unknown_path = _sweep(
        grid={"distance": [3], "weak.unit_cont": [1]}, task_name="t"
    )

    with pytest.raises(fields.ParameterError, match="'distance'"):
        unknown_argument.tasks()
    with pytest.raises(fields.ParameterError, match="weak.unit_count"):
        unknown_path.tasks()


def test_a_python_file_beside_the_run_file_is_loaded_as_a_module(tmp_path):
    run_file = tmp_path / "run.py"
    config = tmp_path / "config.py"
    config.write_text("DISTANCE = 7\n")

    module = sweeps.load_beside(run_file, "config.py")

    assert module.DISTANCE == 7


def _sweep(grid, task_name, metadata=None, stop=STOP) -> sweeps.Sweep:
    return sweeps.Sweep(
        "memory", _machine, grid, task_name, stop=stop, metadata=metadata
    )


def _machine(distance):
    return machines.weak_baseline(distance, 0.002, 1.0)


def _machine_reading(constants):
    """A machine function whose rate is a constant."""

    def machine(distance=3):
        probability = constants.memory.probability
        return machines.weak_baseline(distance, probability, 1.0)

    return machine


def _constants(tmp_path: pathlib.Path) -> constants_module.Section:
    path = tmp_path / "constants.yaml"
    path.write_text(CONSTANTS)
    return constants_module.load(path)


def _probability(task: collect.Task) -> float:
    record = task.machine.workload.workload_record
    return record.physical_error_probability


def _qpu_metadata(cell, settings) -> dict:
    del cell
    record = settings.workload.workload_record
    rate = record.physical_error_probability
    return {"qpu.distance": settings.qpu.distance, "rate": rate}


def _stop_by_distance(cell) -> collection.CollectionSettings:
    shots = 10 * cell["distance"]
    return collection.CollectionSettings(max_shots=shots)

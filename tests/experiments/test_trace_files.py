"""One traced shot, one file: the shot's point id and seed name it.

observation.trace names either a word (chrome, and the experiments
layer names the file by the shot's label) or a path of the study's own.
A sweep traces the shots trace_shots names at every point, so the label,
the point id and the seed, goes into the path before its suffixes, the
way gem5's multisim names each simulation's output by its id
(src/python/gem5/utils/multisim/multisim.py).
"""

import decsim.experiments.collect_command as collect_command
import decsim.experiments.measure as measure
import tests.experiments.yaml_configs as yaml_configs


def test_a_shots_label_is_its_point_id_and_its_seed():
    assert measure.shot_label("0123abcd", 3) == "0123abcd_seed3"


def test_the_label_goes_before_every_suffix_so_gzip_still_opens():
    path = measure.trace_path_for_shot(
        "/tmp/run.trace.json.gz", "0123abcd_seed7"
    )

    assert path == "/tmp/run_0123abcd_seed7.trace.json.gz"


def test_a_path_with_no_suffix_takes_the_label_at_its_end():
    path = measure.trace_path_for_shot("/tmp/run", "0123abcd_seed7")

    assert path == "/tmp/run_0123abcd_seed7"


def test_two_points_at_one_seed_write_two_files():
    first = measure.trace_path_for_shot("/tmp/run.trace.json", "0123abcd_seed0")
    second = measure.trace_path_for_shot(
        "/tmp/run.trace.json", "4567ef01_seed0"
    )

    assert first == "/tmp/run_0123abcd_seed0.trace.json"
    assert second == "/tmp/run_4567ef01_seed0.trace.json"


def test_points_apart_only_in_basis_write_their_own_log_and_trace(tmp_path):
    """Two points whose one difference is the memory's basis.

    Each file is named by its point's id, which the basis moves, so the
    two points write two logs and two traces.
    """
    workload = yaml_configs.memory_workload(6)
    workload["arguments"]["physical_error_probability"] = 0.001
    card = {
        "qpu": {"kind": "stim_device", "distance": 3},
        "workload": workload,
        "observation": {"log": "file", "trace": "chrome"},
        "sweep": [
            {
                "axes": {
                    "workload.arguments.code_task": [
                        "surface_code:rotated_memory_x",
                        "surface_code:rotated_memory_z",
                    ]
                },
                "shots": 1,
            }
        ],
    }
    config_path = yaml_configs.write_config(tmp_path, card)
    run_dir = tmp_path / "run"

    collect_command.run_experiment(config_path, run_dir)

    log_folder = run_dir / "log"
    trace_folder = run_dir / "trace"
    log_files = log_folder.iterdir()
    trace_files = trace_folder.iterdir()
    assert len(list(log_files)) == 2
    assert len(list(trace_files)) == 2

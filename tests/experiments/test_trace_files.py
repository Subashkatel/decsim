"""One traced shot, one file: the shot's point id and seed name it.

observation.trace names either a word (chrome, and the experiments
layer names the file by the shot's label) or a path of the study's own.
A sweep traces the shots trace_shots names at every point, so the label,
the point id and the seed, goes into the path before its suffixes, the
way gem5's multisim names each simulation's output by its id
(src/python/gem5/utils/multisim/multisim.py).
"""

import decsim.experiments.measure as measure


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

"""One traced shot, one file: the seed joins the path when several trace.

observation.trace names either a word (chrome, and the front names the
file after the point) or a path of the study's own. A path plus
trace_shots of more than one seed would have every traced shot write the
same file, so the seed goes into the name before its suffixes
(docs/rewrite/notes/trace_and_viewer.md section 10, ruling 1: one file
per traced shot).
"""

from decsim.front.measure import trace_path_for_shot


def test_one_traced_shot_keeps_the_path_the_yaml_gave():
    assert (
        trace_path_for_shot("/tmp/run.trace.json", 0, (0,))
        == "/tmp/run.trace.json"
    )


def test_several_traced_shots_each_take_their_seed():
    first = trace_path_for_shot("/tmp/run.trace.json", 0, (0, 3))
    second = trace_path_for_shot("/tmp/run.trace.json", 3, (0, 3))

    assert first == "/tmp/run_seed0.trace.json"
    assert second == "/tmp/run_seed3.trace.json"
    assert first != second


def test_the_seed_goes_before_every_suffix_so_gzip_still_opens():
    path = trace_path_for_shot("/tmp/run.trace.json.gz", 7, (0, 7))

    assert path == "/tmp/run_seed7.trace.json.gz"


def test_a_path_with_no_suffix_takes_the_seed_at_its_end():
    path = trace_path_for_shot("/tmp/run", 7, (0, 7))

    assert path == "/tmp/run_seed7"

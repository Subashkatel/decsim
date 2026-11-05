"""A recorded run's rows of one seed, from one file or from its pieces.

A replay of a long recording reads only the piece that ran the seed, so
the two forms must give the same rows, and a seed no piece ran must be
told apart from a seed that wrote no row.
"""

import decsim.records.recorded_rows as recorded_rows

HEADER = "seed,window_index,is_escalated\n"


def _write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    ended_lines = [line + "\n" for line in lines]
    body = "".join(ended_lines)
    contents = HEADER + body
    path.write_text(contents)


def test_a_single_file_gives_each_seeds_rows_in_order(tmp_path):
    path = tmp_path / "window_outcomes.csv"
    _write(path, ["0,0,False", "1,0,True", "0,1,True"])

    rows = recorded_rows.rows_of_seed(str(path), "window_outcomes.csv", 0)

    assert [row["window_index"] for row in rows] == ["0", "1"]
    assert (
        recorded_rows.rows_of_seed(str(path), "window_outcomes.csv", 7) is None
    )


def test_a_pieces_folder_reads_the_piece_that_ran_the_seed(tmp_path):
    pieces = tmp_path / "pieces" / "task"
    first_piece = pieces / "0-9" / "window_outcomes.csv"
    second_piece = pieces / "10-19" / "window_outcomes.csv"
    _write(first_piece, ["3,0,False"])
    _write(second_piece, ["12,0,True", "12,1,False"])

    rows = recorded_rows.rows_of_seed(str(pieces), "window_outcomes.csv", 12)

    assert [(row["seed"], row["is_escalated"]) for row in rows] == [
        ("12", "True"),
        ("12", "False"),
    ]


def test_a_seed_a_piece_ran_without_rows_is_empty_and_an_unrun_seed_is_none(
    tmp_path,
):
    pieces = tmp_path / "pieces" / "task"
    first_piece = pieces / "0-9" / "window_outcomes.csv"
    _write(first_piece, ["3,0,False"])

    assert (
        recorded_rows.rows_of_seed(str(pieces), "window_outcomes.csv", 4) == []
    )
    assert (
        recorded_rows.rows_of_seed(str(pieces), "window_outcomes.csv", 10)
        is None
    )

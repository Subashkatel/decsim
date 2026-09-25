"""The tutorials' printed results are what a fresh run prints.

Each tutorial page shows the output of the commands it asks the reader
to type. The lines of it that are a function of the config and the seed
alone (the logical failure counts, the correctness check, the QPU's
finishing tick, the decoded observable, and the csv rows) are compared
here with a run of the same commands, so a change to the machine that
moves them fails this check instead of leaving the page wrong. Lines
charged a decoder's wall clock are the host's and are not compared.

Run it from the repo root in an environment with the run extra, after
the README's install: `python tools/check_tutorial_runs.py`.
"""

import csv
import dataclasses
import pathlib
import re
import subprocess
import sys
import tempfile

TOOL_FILE = pathlib.Path(__file__)
TOOL_PATH = TOOL_FILE.resolve()
CHECKOUT = TOOL_PATH.parent.parent
RUN_LINE = re.compile(
    r"^(?:terminal status|execution done|operation 1): .*$", re.MULTILINE
)
COLLECT_LINE = re.compile(
    r"^(?:logical failures|mismatches vs direct PyMatching): .*$",
    re.MULTILINE,
)
BLOCK_FLAGS = re.MULTILINE | re.DOTALL
FENCED_BLOCK = re.compile(r"^```\n(.*?)^```$", BLOCK_FLAGS)
CSV_HEADER = "distance,"


@dataclasses.dataclass(frozen=True)
class Tutorial:
    """One page, the `decsim run` and `decsim collect` it shows."""

    page: str
    run_arguments: tuple
    collect_arguments: tuple


TUTORIALS = (
    Tutorial(
        page="docs/tutorials/first_run.md",
        run_arguments=("configs/reference.yaml", "--seed", "0", "--trace"),
        collect_arguments=("configs/reference.yaml",),
    ),
    Tutorial(
        page="docs/tutorials/first_sweep.md",
        run_arguments=(),
        collect_arguments=("configs/my_first_sweep.yaml", "--processes", "4"),
    ),
    Tutorial(
        page="docs/tutorials/two_tiers.md",
        run_arguments=("configs/two_tiers.yaml", "--seed", "1", "--trace"),
        collect_arguments=("configs/two_tiers.yaml",),
    ),
)


def main() -> int:
    """Run every tutorial's commands; 1 when any page differs from them."""
    differences = []
    for tutorial in TUTORIALS:
        found = tutorial_differences(tutorial)
        differences.extend(found)
    for difference in differences:
        print(difference)
    if differences:
        return 1
    print(f"{len(TUTORIALS)} tutorials print what a fresh run prints")
    return 0


def tutorial_differences(tutorial: Tutorial) -> list:
    """Every shown line of one page that its commands no longer print."""
    path = CHECKOUT / tutorial.page
    text = path.read_text()
    with tempfile.TemporaryDirectory() as scratch:
        run_folder = pathlib.Path(scratch) / "run"
        collect_folder = pathlib.Path(scratch) / "collect"
        run_output = decsim_output("run", tutorial.run_arguments, run_folder)
        collect_output = decsim_output(
            "collect", tutorial.collect_arguments, collect_folder
        )
        sweep_path = collect_folder / "sweep.csv"
        sweep_rows = sweep_table(sweep_path)
    differences = []
    shown_run = RUN_LINE.findall(text)
    printed_run = RUN_LINE.findall(run_output)
    differences += line_differences(tutorial.page, shown_run, printed_run)
    shown_counts = COLLECT_LINE.findall(text)
    printed_counts = COLLECT_LINE.findall(collect_output)
    differences += line_differences(tutorial.page, shown_counts, printed_counts)
    for block in csv_blocks(text):
        differences += block_differences(tutorial.page, block, sweep_rows)
    return differences


def decsim_output(verb: str, arguments: tuple, out: pathlib.Path) -> str:
    """What `decsim <verb>` prints, its folder written under out."""
    if not arguments:
        return ""
    command = ["decsim", verb, *arguments, "--out", str(out)]
    completed = subprocess.run(
        command, cwd=CHECKOUT, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        typed = " ".join(command)
        raise SystemExit(f"{typed} failed:\n{completed.stderr}")
    return completed.stdout


def sweep_table(path: pathlib.Path) -> list:
    """The run's sweep.csv as rows of column name to value."""
    with path.open(newline="") as sweep_file:
        reader = csv.DictReader(sweep_file)
        return list(reader)


def csv_blocks(text: str) -> list:
    """Every plain fenced block of a page that is sweep.csv columns."""
    blocks = []
    for body in FENCED_BLOCK.findall(text):
        if body.startswith(CSV_HEADER):
            blocks.append(body)
    return blocks


def line_differences(page: str, shown: list, printed: list) -> list:
    """The page's lines against the run's, as one sentence when they differ."""
    if shown == printed:
        return []
    return [f"{page} shows {shown}, and a run prints {printed}"]


def block_differences(page: str, block: str, sweep_rows: list) -> list:
    """One csv block against the same columns of the run's sweep.csv."""
    lines = block.splitlines()
    header = lines[0]
    columns = header.split(",")
    printed = [header]
    for row in sweep_rows:
        values = [row[column] for column in columns]
        row_text = ",".join(values)
        printed.append(row_text)
    return line_differences(page, lines, printed)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

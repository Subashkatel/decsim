"""The tutorials print what a fresh run of their own commands prints.

Each tutorial page shows commands and, under them, what they print.
This check runs the page's commands (every `decsim`, `cut` and `ls` line
of its bash blocks, in page order) in a scratch folder, and holds each
plain block of the page to the output it was copied from, found by the
block's first line.

On a page whose decoders are priced by cards, every tick is a function
of the config and the seed, so each block is compared whole: the
timings, the trace and the counts alike. On a page that names a decoder,
a decode is charged the wall clock it took (decsim/decoders/decoder.py,
decode_timed), so only the lines no clock moves are compared there: the
logical failure counts, the correctness check, the QPU's finishing
tick, the decoded observable, and everything `cut` and `ls` print.

Run it from the repo root in an environment with the run extra:
`python tools/check_tutorial_runs.py`.
"""

import dataclasses
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Optional

TOOL_FILE = pathlib.Path(__file__)
TOOL_PATH = TOOL_FILE.resolve()
CHECKOUT = TOOL_PATH.parent.parent
CONFIGS = CHECKOUT / "configs"
FENCE = "```"
COMMAND_WORDS = ("decsim", "cut", "ls")
CLOCK_FREE_LINE = re.compile(
    r"^(?:terminal status|execution done|operation 1|logical failures"
    r"|mismatches vs direct PyMatching): "
)


@dataclasses.dataclass(frozen=True)
class Tutorial:
    """One page, and whether its decoders are priced by cards."""

    page: str
    is_priced: bool


@dataclasses.dataclass(frozen=True)
class FencedBlock:
    """One fenced block of a page: its info word (bash, yaml or none)."""

    info: str
    lines: list


@dataclasses.dataclass(frozen=True)
class Printed:
    """What one command printed, one entry per line."""

    command: str
    lines: tuple


TUTORIALS = (
    Tutorial(page="docs/tutorials/first_run.md", is_priced=False),
    Tutorial(page="docs/tutorials/first_sweep.md", is_priced=False),
    Tutorial(page="docs/tutorials/two_tiers.md", is_priced=True),
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
    """Every block of one page that its commands no longer print."""
    path = CHECKOUT / tutorial.page
    text = path.read_text()
    commands = page_commands(text)
    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = pathlib.Path(scratch_name)
        outputs = run_commands(commands, scratch)
    return page_differences(tutorial, text, outputs)


def page_commands(text: str) -> list:
    """The page's decsim, cut and ls lines, continuations joined."""
    commands = []
    for block in fenced_blocks(text):
        if block.info != "bash":
            continue
        found = block_commands(block.lines)
        commands.extend(found)
    return commands


def block_commands(lines: list) -> list:
    """The decsim, cut and ls lines of one bash block."""
    text = "\n".join(lines)
    joined = text.replace("\\\n", " ")
    commands = []
    for line in joined.splitlines():
        words = line.split()
        if words and words[0] in COMMAND_WORDS:
            commands.append(line)
    return commands


def fenced_blocks(text: str) -> list:
    """Every fenced block of a page, each fence paired with its close."""
    blocks = []
    current = None
    for line in text.splitlines():
        is_fence = line.startswith(FENCE)
        if is_fence and current is None:
            info = line[len(FENCE) :]
            current = FencedBlock(info=info, lines=[])
            continue
        if is_fence:
            blocks.append(current)
            current = None
            continue
        if current is not None:
            current.lines.append(line)
    return blocks


def run_commands(commands: list, scratch: pathlib.Path) -> list:
    """Each command's output, run in order in a folder that sees configs/."""
    linked_configs = scratch / "configs"
    linked_configs.symlink_to(CONFIGS)
    outputs = []
    for command in commands:
        printed = command_output(command, scratch)
        outputs.append(printed)
    return outputs


def command_output(command: str, scratch: pathlib.Path) -> Printed:
    """What one command prints to the terminal; a failure stops the check.

    The progress lines go to stderr and the summary to stdout, so the two
    are read as one stream, unbuffered, in the order a terminal shows.
    """
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(
        command,
        shell=True,
        cwd=scratch,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"{command} failed:\n{completed.stdout}")
    lines = completed.stdout.splitlines()
    return Printed(command=command, lines=tuple(lines))


def page_differences(tutorial: Tutorial, text: str, outputs: list) -> list:
    """Every plain block of the page against the output it came from."""
    differences = []
    for block in fenced_blocks(text):
        if block.info:
            continue
        found = block_differences(tutorial, block.lines, outputs)
        differences.extend(found)
    return differences


def block_differences(tutorial: Tutorial, shown: list, outputs: list) -> list:
    """One block against the output lines from its first line on."""
    first_line = shown[0]
    printed = printed_from(first_line, outputs)
    if printed is None:
        return [
            f"{tutorial.page} shows a block no command prints: {first_line}"
        ]
    shown_lines, printed_lines = compared_lines(tutorial, shown, printed)
    if shown_lines == printed_lines:
        return []
    return [
        f"{tutorial.page} shows {shown_lines}, and a run prints {printed_lines}"
    ]


def printed_from(first_line: str, outputs: list) -> Optional[Printed]:
    """The first output holding the line, cut to start there."""
    for printed in outputs:
        if first_line not in printed.lines:
            continue
        start = printed.lines.index(first_line)
        rest = printed.lines[start:]
        return Printed(command=printed.command, lines=rest)
    return None


def compared_lines(tutorial: Tutorial, shown: list, printed: Printed) -> tuple:
    """The lines of each side the page's pricing lets the check compare."""
    words = printed.command.split()
    is_clock_free = words[0] != "decsim"
    if tutorial.is_priced or is_clock_free:
        count = len(shown)
        return shown, list(printed.lines[:count])
    shown_lines = _clock_free_lines(shown)
    printed_lines = _clock_free_lines(printed.lines)
    count = len(shown_lines)
    return shown_lines, printed_lines[:count]


def _clock_free_lines(lines) -> list:
    """The lines of an output that no decoder's wall clock moves."""
    kept = []
    for line in lines:
        if CLOCK_FREE_LINE.match(line):
            kept.append(line)
    return kept


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

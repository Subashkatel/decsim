"""The tutorials print what a fresh run of their own code prints.

Each tutorial page shows commands and Python and, under them, what they
print. This check runs a page's blocks in page order, in a scratch
folder that sees examples/: every `decsim`, `cut`, `ls` and `rm` line of
its bash blocks, and its Python blocks in one session, an exception
printing its name and its message. The output block right under a
Python block is held to what that block printed, and a Python block
with none under it must print nothing. Every other output block is held
to the command output it was copied from, found by the block's first
line. A Python block whose fence names a file (```python
examples/two_tiers.py) shows lines of that file, which tests/test_docs.py
holds to the file, and is not run.

On a page whose decoders are priced by cards, every tick is a function
of the machine and the seed, so each block is compared whole: the
timings, the trace and the counts alike. On a page that names a decoder,
a decode is charged the wall clock it took (decsim/decoders/decoder.py,
decode_timed), so only the lines no clock moves are compared there: the
correctness check, the QPU's finishing tick, the decoded observable,
and everything `cut` and `ls` print.

Run it from the repo root in an environment with the run extra:
`python tools/check_tutorial_runs.py`.
"""

import contextlib
import dataclasses
import io
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
EXAMPLES = CHECKOUT / "examples"
FENCE = "```"
# the info words of a block that shows what the code above it printed
OUTPUT_INFOS = ("", "text")
COMMAND_WORDS = ("decsim", "cut", "ls", "rm")
CLOCK_FREE_LINE = re.compile(
    r"^(?:terminal status|execution done|operation 1): "
)


@dataclasses.dataclass(frozen=True)
class Tutorial:
    """One tutorial page.

    is_priced says whether its decoders are priced by cards.
    """

    page: str
    is_priced: bool


@dataclasses.dataclass(frozen=True)
class FencedBlock:
    """One fenced block of a page: its info words (bash, python or none)."""

    info: str
    lines: list


@dataclasses.dataclass(frozen=True)
class Printed:
    """What one command printed, one entry per line."""

    command: str
    lines: tuple


TUTORIALS = (
    Tutorial(page="docs/tutorials/first_run.md", is_priced=True),
    Tutorial(page="docs/tutorials/first_sweep.md", is_priced=False),
    Tutorial(page="docs/tutorials/two_tiers.md", is_priced=True),
    Tutorial(page="docs/tutorials/build_a_machine.md", is_priced=True),
)


def main() -> int:
    """Run every tutorial's code; 1 when any page differs from it."""
    sys.path.insert(0, str(CHECKOUT))
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
    """Every block of one page that its code no longer prints."""
    path = CHECKOUT / tutorial.page
    text = path.read_text()
    blocks = fenced_blocks(text)
    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = pathlib.Path(scratch_name)
        linked_examples = scratch / "examples"
        linked_examples.symlink_to(EXAMPLES)
        outputs, differences = run_page(tutorial, blocks, scratch)
    command_differences = page_differences(tutorial, blocks, outputs)
    return differences + command_differences


def run_page(tutorial: Tutorial, blocks: list, scratch: pathlib.Path) -> tuple:
    """Run every block in page order; the command outputs and differences.

    The Python blocks are compared as they run, and the commands' outputs
    once every command has run.
    """
    namespace = {"__name__": "__main__"}
    outputs = []
    differences = []
    for index, block in enumerate(blocks):
        if block.info == "bash":
            commands = block_commands(block.lines)
            printed = run_commands(commands, scratch)
            outputs.extend(printed)
            continue
        if block.info != "python":
            continue
        printed_lines = run_python(block.lines, namespace, scratch)
        shown = output_after(blocks, index)
        found = python_differences(tutorial, shown, printed_lines)
        differences.extend(found)
    return outputs, differences


def run_python(lines: list, namespace: dict, scratch: pathlib.Path) -> list:
    """What one Python block prints, run in the page's session."""
    source = "\n".join(lines)
    code = compile(source, "<tutorial>", "exec")
    printed = io.StringIO()
    working_dir = os.getcwd()
    os.chdir(scratch)
    try:
        with contextlib.redirect_stdout(printed):
            _run_code(code, namespace)
    finally:
        os.chdir(working_dir)
    text = printed.getvalue()
    return text.splitlines()


def _run_code(code, namespace: dict) -> None:
    """Run the code; an exception prints as a session shows its last line."""
    try:
        exec(code, namespace)
    except Exception as error:
        error_type = type(error)
        print(f"{error_type.__name__}: {error}")


def output_after(blocks: list, index: int) -> list:
    """The output block right under a block, or none: no lines."""
    following = blocks[index + 1 : index + 2]
    if not following or following[0].info not in OUTPUT_INFOS:
        return []
    return following[0].lines


def python_differences(
    tutorial: Tutorial, shown: list, printed_lines: list
) -> list:
    """A Python block's output against the block the page shows under it."""
    if shown == printed_lines:
        return []
    return [
        f"{tutorial.page} shows {shown}, and the Python prints {printed_lines}"
    ]


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
    """Each command's output, run in order in the page's scratch folder."""
    outputs = []
    for command in commands:
        printed = command_output(command, scratch)
        outputs.append(printed)
    return outputs


def command_output(command: str, scratch: pathlib.Path) -> Printed:
    """What one command prints to the terminal; a failure stops the check.

    The progress lines go to stderr and a shot's narration to stdout, so
    the two are read as one stream, unbuffered, in the order a terminal
    shows.
    """
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = str(CHECKOUT)
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


def page_differences(tutorial: Tutorial, blocks: list, outputs: list) -> list:
    """Every output block a command printed against that output."""
    differences = []
    for index, block in enumerate(blocks):
        if block.info not in OUTPUT_INFOS:
            continue
        is_under_python = index > 0 and blocks[index - 1].info == "python"
        if is_under_python:
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

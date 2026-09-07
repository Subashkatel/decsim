"""`decsim <verb>`: the command set, dispatched on the first word.

sinter's shape (sinter/_command/_main.py:1-40): one command, one
subcommand per word, and each verb's module imported only when that verb
runs, so `decsim show` never loads matplotlib and `decsim plot` never
loads Stim. The console script and `python -m decsim` both land here.

    decsim run <yaml> [--seed N] [--out DIR] [--log ...] [--trace]
    decsim collect <yaml> [--out DIR] [--processes N] [--shard i/n]
    decsim combine <run_dir>... [--out DIR]
    decsim show <yaml>
    decsim plot <run_dir>... [--figure NAME] [--out PATH] [--probability P]
    decsim trace follow <file> --round k:n | --window k:n [--html PATH]

What the front refuses reaches the user as one sentence and exit 1
(decsim/front/refusal.py); anything else keeps its traceback.
"""

import sys
from typing import Optional

import decsim.front.refusal as refusal

VERBS = ("run", "collect", "combine", "show", "plot", "trace")
HELP_WORDS = ("help", "-h", "--help")


def main(argv: Optional[list] = None) -> None:
    """Run the verb the first word names; a refusal is one line and exit 1."""
    if argv is None:
        argv = sys.argv[1:]
    verb = None
    if argv:
        verb = argv[0]
    rest = argv[1:]
    try:
        _verb(verb, rest)
    except refusal.RefusalError as refused:
        _report_the_refusal(refused)


def usage() -> str:
    """The verbs, one per line, as the command prints them."""
    lines = ["the decsim commands are:"]
    for verb in VERBS:
        lines.append(f"    decsim {verb}")
    return "\n".join(lines)


def _verb(verb: Optional[str], rest: list) -> None:
    """The one verb the first word names, its module imported here."""
    if verb == "run":
        return _run(rest)
    if verb == "collect":
        return _collect(rest)
    if verb == "combine":
        return _combine(rest)
    if verb == "show":
        return _show(rest)
    if verb == "plot":
        return _plot(rest)
    if verb == "trace":
        return _trace(rest)
    _report_no_verb(verb)


def _run(argv: list) -> None:
    """One seeded shot of one yaml."""
    import decsim.front.run_command as run_command

    run_command.main(argv)


def _collect(argv: list) -> None:
    """The whole sweep of one yaml into a run folder."""
    import argparse

    import decsim.front.collect_command as collect_command
    import decsim.front.report as report

    parser = argparse.ArgumentParser(prog="decsim collect")
    parser.add_argument("config", help="the experiment yaml to sweep")
    parser.add_argument("--out", default=None, help="the run folder to write")
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="worker processes, one task each (shots stay serial)",
    )
    parser.add_argument(
        "--shard",
        default=None,
        help="i/n: run the tasks whose index modulo n is i",
    )
    parsed = parser.parse_args(argv)
    shard = _shard_of(parsed.shard)
    run_dir, rows = collect_command.run_experiment(
        parsed.config,
        parsed.out,
        processes=parsed.processes,
        shard=shard,
    )
    lines = report.terminal_lines(rows)
    text = "\n".join(lines)
    print(text)
    print(f"\nevery column: {run_dir}/sweep.csv")


def _combine(argv: list) -> None:
    """Several run folders' rows folded into one report."""
    import argparse

    import decsim.front.report as report
    import decsim.front.run_folder as run_folder

    parser = argparse.ArgumentParser(prog="decsim combine")
    parser.add_argument("run_dirs", nargs="+", help="the folders to fold")
    parser.add_argument(
        "--out", default=None, help="the folder the combined rows go in"
    )
    parsed = parser.parse_args(argv)
    out_dir = run_folder.combined_run_dir(parsed.out)
    rows = report.combine(parsed.run_dirs, out_dir)
    lines = report.terminal_lines(rows)
    text = "\n".join(lines)
    print(text)
    print(f"\nevery column: {out_dir}/sweep.csv")


def _show(argv: list) -> None:
    """What one yaml resolves to, before anything runs."""
    import argparse

    import decsim.front.experiment as experiment

    parser = argparse.ArgumentParser(prog="decsim show")
    parser.add_argument("config", help="the experiment yaml to resolve")
    parsed = parser.parse_args(argv)
    config = experiment.load_experiment(parsed.config)
    lines = experiment.resolved_description(config)
    text = "\n".join(lines)
    print(text)


def _plot(argv: list) -> None:
    """One figure, drawn from run folders' csv and trace files."""
    import argparse

    import decsim.front.plots as plots

    parser = argparse.ArgumentParser(prog="decsim plot")
    parser.add_argument("run_dirs", nargs="+", help="the folders to read")
    parser.add_argument(
        "--figure", default="timeline", help="which figure to draw"
    )
    parser.add_argument("--out", default=None, help="where the figure goes")
    parser.add_argument(
        "--probability",
        type=float,
        default=None,
        help="the physical error rate the ler_vs_d figure is drawn at",
    )
    parsed = parser.parse_args(argv)
    out_path = plots.figure(
        parsed.figure, parsed.run_dirs, parsed.out, parsed.probability
    )
    print(out_path)


def _trace(argv: list) -> None:
    """One round's or one window's path through one shot's trace file."""
    import decsim.front.trace_follow as trace_follow

    trace_follow.main(argv)


def _shard_of(text: Optional[str]) -> Optional[tuple]:
    """The --shard argument as (index, count); None when it was not given.

    The whole argument is checked here, where it is read, so a shard
    outside its count is refused before a run folder exists.
    """
    if text is None:
        return None
    words = text.split("/")
    if len(words) != 2:
        _refuse_the_shard(text)
    index = _shard_number(words[0], text)
    count = _shard_number(words[1], text)
    if count < 1:
        _refuse_the_shard(text)
    if not 0 <= index < count:
        _refuse_the_shard(text)
    return (index, count)


def _shard_number(word: str, text: str) -> int:
    """One side of i/n as a whole number; anything else is not a shard."""
    if not word.isdigit():
        _refuse_the_shard(text)
    return int(word)


def _refuse_the_shard(text: str) -> None:
    """What a shard is, in the sentence the user reads."""
    raise refusal.RefusalError(
        f"--shard {text} is not a shard; write it as i/n with n at least 1 "
        "and i between 0 and n - 1, so --shard 0/4 is the first of four"
    )


def _report_the_refusal(refused: refusal.RefusalError) -> None:
    """One sentence and exit 1, gem5's fatal (src/base/logging.hh)."""
    print(f"decsim: {refused}", file=sys.stderr)
    raise SystemExit(1)


def _report_no_verb(verb: Optional[str]) -> None:
    """Print the verbs, and fail unless the user asked for help."""
    wants_help = verb in HELP_WORDS
    if not wants_help:
        if verb is None:
            print("decsim needs a command.\n", file=sys.stderr)
        else:
            print(f"decsim has no command {verb}.\n", file=sys.stderr)
    verbs = usage()
    print(verbs, file=sys.stderr)
    if not wants_help:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

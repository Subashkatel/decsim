"""`decsim <verb>`: the command set, dispatched on the first word.

sinter's shape (sinter/_command/_main.py:1-40): one command, one
subcommand per word, and each verb's module imported only when that verb
runs, so `decsim show` never loads matplotlib and `decsim plot` never
loads Stim. The console script and `python -m decsim` both land here.

    decsim run <run file> [--out DIR] [--processes N] [--only NAME]
        [--shots N]
    decsim run <run file> --list
    decsim run <run file> --seed S [--only NAME] [--out DIR] [--log ...]
        [--trace]
    decsim run --plan <round>/plan.csv --task K [--processes N]
    decsim plan <run file> --out DIR --tasks N [--cores C] [--hours H]
    decsim status <results folder>
    decsim show <yaml>
    decsim diff <run_dir> <run_dir>
    decsim plot <run_dir> [--figure timeline|stage_breakdown] [--out PATH]
    decsim trace follow <file> --round k:n | --window k:n [--html PATH]

What the experiments layer refuses reaches the user as one sentence and
exit 1 (decsim/experiments/refusal.py); anything else keeps its
traceback.
"""

import pathlib
import sys
from typing import Optional

import decsim.experiments.refusal as refusal

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
    for verb in _RUN_BY_VERB:
        lines.append(f"    decsim {verb}")
    return "\n".join(lines)


def _verb(verb: Optional[str], rest: list) -> None:
    """The one verb the first word names, its module imported there."""
    run_verb = _RUN_BY_VERB.get(verb)
    if run_verb is None:
        _report_no_verb(verb)
        return
    run_verb(rest)


def _run(argv: list) -> None:
    """An experiment's points, one point, one narrated shot, or a task."""
    import decsim.experiments.collect_command as collect_command

    parser = _run_parser()
    parsed = parser.parse_args(argv)
    _check_the_run_arguments(parser, parsed)
    if parsed.plan is not None:
        plan_path = pathlib.Path(parsed.plan)
        collect_command.run_planned(
            plan_path, parsed.task, processes=parsed.processes
        )
        return
    if parsed.list:
        _list_the_points(parsed.run_file)
        return
    if parsed.seed is not None:
        _run_one_shot(parsed)
        return
    _collect(parsed)


def _collect(parsed) -> None:
    """Every chosen point collected until it stops, then the summary."""
    import decsim.experiments.collect_command as collect_command
    import decsim.experiments.report as report

    run_dir, rows = collect_command.run_experiment(
        parsed.run_file,
        parsed.out,
        processes=parsed.processes,
        only=parsed.only,
        shot_count=parsed.shots,
    )
    if not rows:
        return
    lines = report.terminal_lines(rows, run_dir)
    text = "\n".join(lines)
    print(text)
    print(f"\nevery column: {run_dir}/sweep.csv")


def _run_one_shot(parsed) -> None:
    """One seeded shot of the chosen point, narrated."""
    import decsim.experiments.collect_command as collect_command
    import decsim.experiments.experiment as experiment

    run_path = pathlib.Path(parsed.run_file)
    study = experiment.load_one_point(run_path, parsed.only)
    lines = collect_command.run_one_shot(
        study,
        run_path,
        parsed.seed,
        parsed.out,
        log=parsed.log,
        trace=parsed.trace,
    )
    text = "\n".join(lines)
    print(text)


def _list_the_points(run_file: str) -> None:
    """The experiment's point names, one a line, MultiSim's --list."""
    import decsim.experiments.experiment as experiment

    study = experiment.load(run_file)
    for point in study.points:
        print(point.name)


def _run_parser():
    """The run command's arguments."""
    import argparse

    parser = argparse.ArgumentParser(prog="decsim run")
    parser.add_argument(
        "run_file",
        nargs="?",
        default=None,
        help="the experiment's run file, Python or yaml",
    )
    parser.add_argument(
        "--out", default=None, help="the results folder to write"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="worker processes, one piece each (shots stay serial)",
    )
    _add_the_choice_arguments(parser)
    parser.add_argument(
        "--plan", default=None, help="a round's plan.csv, from decsim plan"
    )
    parser.add_argument(
        "--task", type=int, default=None, help="the plan's task to run"
    )
    return parser


def _add_the_choice_arguments(parser) -> None:
    """Which points run, and whether as a collection or one narrated shot."""
    parser.add_argument(
        "--list", action="store_true", help="print the point names and stop"
    )
    parser.add_argument(
        "--only", default=None, help="the one point to run, by its name"
    )
    parser.add_argument(
        "--shots",
        type=_shot_count,
        default=None,
        help="stop every point at this many shots, its first seeds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="run one narrated shot of this seed, of the first point or "
        "the --only one",
    )
    parser.add_argument(
        "--log",
        default=None,
        choices=("off", "print", "file", "both"),
        help="the narrated shot's engine log, over the point's",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="write the narrated shot's Chrome trace of the data path",
    )


def _shot_count(text: str) -> int:
    """A shot count: a whole number of at least one."""
    import argparse

    count = int(text)
    if count < 1:
        raise argparse.ArgumentTypeError(
            f"a shot count is at least 1, got {count}"
        )
    return count


def _check_the_run_arguments(parser, parsed) -> None:
    """The run's arguments name one thing to do."""
    if parsed.plan is not None:
        _check_the_plan_arguments(parser, parsed)
        return
    if parsed.run_file is None:
        parser.error("name the run file, or --plan and --task")
    if parsed.seed is not None and parsed.shots is not None:
        parser.error(
            "--seed runs one narrated shot and --shots a collection; give one"
        )
    narrates = parsed.log is not None or parsed.trace
    if narrates and parsed.seed is None:
        parser.error(
            "--log and --trace narrate the one shot --seed runs; a "
            "collection traces the shots its observation names"
        )


def _check_the_plan_arguments(parser, parsed) -> None:
    """A planned task takes its run file and folder from the plan."""
    if parsed.task is None:
        parser.error("--plan runs one task of the plan; name it with --task")
    if parsed.run_file is not None or parsed.out is not None:
        parser.error(
            "--plan reads the run file and the folder from the plan; name "
            "neither"
        )


def _plan(argv: list) -> None:
    """The next round of an experiment's pieces, dealt to tasks."""
    import decsim.experiments.plan_command as plan_command

    plan_command.main(argv)


def _status(argv: list) -> None:
    """An experiment's pieces folded, and where each point stands."""
    import decsim.experiments.status_command as status_command

    status_command.main(argv)


def _show(argv: list) -> None:
    """What one yaml resolves to, before anything runs.

    The first point's machine is built and not run, so show refuses
    whatever `decsim run` would refuse.
    """
    import argparse

    import decsim.experiments.experiment as experiment

    parser = argparse.ArgumentParser(prog="decsim show")
    parser.add_argument("config", help="the experiment yaml to resolve")
    parsed = parser.parse_args(argv)
    config = experiment.load_experiment(parsed.config)
    first_point = config.first_point_task()
    shot_settings = first_point.shot_settings()
    config.built_machine(shot_settings, 0)
    settings = first_point.settings
    lines = experiment.resolved_description(config, settings)
    lines.append("values:")
    value_lines = experiment.value_lines(config, settings)
    lines.extend(value_lines)
    text = "\n".join(lines)
    print(text)


def _diff(argv: list) -> None:
    """How two run folders differ: settings, inputs, then results."""
    import decsim.experiments.diff_command as diff_command

    diff_command.main(argv)


def _plot(argv: list) -> None:
    """One figure of decsim's own records, drawn from a run folder."""
    import argparse

    import decsim.experiments.plots as plots

    parser = argparse.ArgumentParser(prog="decsim plot")
    parser.add_argument(
        "run_dir", help="the folder to read, or a trace file to draw"
    )
    parser.add_argument(
        "--figure", default="timeline", help="which figure to draw"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="where the figure goes; beside its source if unset",
    )
    parsed = parser.parse_args(argv)
    out_path = plots.figure(parsed.figure, parsed.run_dir, parsed.out)
    print(out_path)


def _trace(argv: list) -> None:
    """One round's or one window's path through one shot's trace file."""
    import decsim.experiments.trace_follow as trace_follow

    trace_follow.main(argv)


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


# Each verb and the function that runs it, in the order usage lists them.
_RUN_BY_VERB = {
    "run": _run,
    "plan": _plan,
    "status": _status,
    "show": _show,
    "diff": _diff,
    "plot": _plot,
    "trace": _trace,
}

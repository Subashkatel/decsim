"""`decsim <verb>`: the command set, dispatched on the first word.

sinter's shape (sinter/_command/_main.py:1-40): one command, one
subcommand per word, and each verb's module imported only when that verb
runs, so `decsim show` and `decsim trace` never load Stim. The console
script and `python -m decsim` both land here.

    decsim run <run file> [--out DIR] [--processes N] [--only NAME]
        [--shots N]
    decsim run <run file> --list
    decsim run <run file> --seed S [--only NAME] [--out DIR] [--log ...]
        [--trace]
    decsim run <run file> --slurm --tasks N [--cores C] [--hours H]
        [--memory-mb M] [--out DIR] [--dry-run]
    decsim status <results folder>
    decsim show <yaml>
    decsim trace follow <file> --round k:n | --window k:n

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
    """An experiment's points, one point, one shot, Slurm, or a task."""
    parser = _run_parser()
    parsed = parser.parse_args(argv)
    _check_the_run_arguments(parser, parsed)
    if parsed.batch is not None:
        _run_a_batch_task(parsed)
        return
    if parsed.list:
        _list_the_points(parsed.run_file)
        return
    if parsed.slurm:
        _launch_on_slurm(parsed)
        return
    if parsed.seed is not None:
        _run_one_shot(parsed)
        return
    _collect(parsed)


def _launch_on_slurm(parsed) -> None:
    """One step of the batch loop: the next batch planned and submitted."""
    import decsim.experiments.plan_command as plan_command

    job = plan_command.JobShape(parsed.cores, parsed.hours, parsed.memory_mb)
    plan_command.launch(
        parsed.run_file,
        parsed.out,
        parsed.tasks,
        job,
        dry_run=parsed.dry_run,
    )


def _run_a_batch_task(parsed) -> None:
    """One array task: its share of a batch's plan, on a tree git names."""
    import decsim.experiments.collect_command as collect_command
    import decsim.experiments.plan_command as plan_command

    plan_command.refuse_an_unnamed_tree()
    results_dir = pathlib.Path(parsed.out)
    collect_command.run_planned(
        results_dir, parsed.batch, parsed.task, processes=parsed.processes
    )


def _collect(parsed) -> None:
    """Every chosen point collected until it stops."""
    import decsim.experiments.collect_command as collect_command

    collect_command.run_experiment(
        parsed.run_file,
        parsed.out,
        processes=parsed.processes,
        only=parsed.only,
        shot_count=parsed.shots,
    )


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
    """The run command's arguments, all here, where the command page reads them.

    The Slurm defaults are the ones the batches have been run with: four
    cores, a day of walltime, 4096 MB a piece before one is measured.
    --batch and --task are an array task's entry, which the loop writes.
    """
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
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="run the experiment as batches of Slurm arrays",
    )
    parser.add_argument(
        "--tasks", type=int, default=None, help="the most tasks a batch has"
    )
    parser.add_argument(
        "--cores", type=int, default=4, help="pieces a task runs at once"
    )
    parser.add_argument(
        "--hours", type=int, default=24, help="a task's walltime"
    )
    parser.add_argument(
        "--memory-mb",
        type=int,
        default=4096,
        help="one piece's memory before its point has a measured peak",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the sbatch lines and submit nothing",
    )
    parser.add_argument(
        "--batch", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--task", type=int, default=None, help=argparse.SUPPRESS
    )
    return parser


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
    if parsed.batch is not None:
        _check_the_task_arguments(parser, parsed)
        return
    if parsed.run_file is None:
        parser.error("name the run file")
    if parsed.slurm:
        _check_the_slurm_arguments(parser, parsed)
        return
    if parsed.tasks is not None or parsed.dry_run:
        parser.error("--tasks and --dry-run go with --slurm")
    _check_the_local_arguments(parser, parsed)


def _check_the_local_arguments(parser, parsed) -> None:
    """A local run collects, or narrates one shot, not both."""
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


def _check_the_slurm_arguments(parser, parsed) -> None:
    """A Slurm run plans every point to its stop, in tasks of a shape."""
    if parsed.tasks is None:
        parser.error("--slurm deals each batch to at most --tasks tasks")
    chosen = (parsed.only, parsed.shots, parsed.seed)
    if any(option is not None for option in chosen):
        parser.error(
            "--slurm runs every point to its stop; --only, --shots and "
            "--seed run locally"
        )


def _check_the_task_arguments(parser, parsed) -> None:
    """An array task reads its run file from the folder its batch is in."""
    if parsed.task is None or parsed.out is None:
        parser.error("--batch runs one task of a batch; name --task and --out")
    if parsed.run_file is not None:
        parser.error("--batch reads the run file from the folder; name none")


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
    config.built_machine(first_point, 0)
    settings = first_point.settings
    lines = experiment.resolved_description(config, settings)
    lines.append("values:")
    value_lines = experiment.value_lines(config, settings)
    lines.extend(value_lines)
    text = "\n".join(lines)
    print(text)


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
    "status": _status,
    "show": _show,
    "trace": _trace,
}

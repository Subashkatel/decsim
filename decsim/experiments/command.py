"""`decsim <verb>`: the command set, dispatched on the first word.

sinter's shape (sinter/_command/_main.py:1-40): one command, one
subcommand per word, and each verb's module imported only when that verb
runs, so `decsim trace` never loads Stim. The console script and
`python -m decsim` both land here.

    decsim run <run file> [--out DIR] [--processes N] [--only NAME]
        [--shots N]
    decsim run <run file> --list
    decsim run <run file> --seed S [--only NAME] [--out DIR] [--log ...]
        [--trace]
    decsim run <run file> --slurm [--cores C] [--hours H] [--memory-mb M]
        [--out DIR] [--dry-run]
    decsim run --fold --out DIR
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
    """An experiment's tasks, one task, one shot, Slurm, a job, a fold."""
    parser = _run_parser()
    parsed = parser.parse_args(argv)
    _check_the_run_arguments(parser, parsed)
    if parsed.fold:
        _fold(parsed)
        return
    if parsed.job is not None:
        _run_a_job(parsed)
        return
    if parsed.list:
        _list_the_tasks(parsed.run_file)
        return
    if parsed.slurm:
        _launch_on_slurm(parsed)
        return
    if parsed.seed is not None:
        _run_one_shot(parsed)
        return
    _collect(parsed)


def _launch_on_slurm(parsed) -> None:
    """Every task recorded, its pieces packed into Slurm jobs, then a fold."""
    import decsim.experiments.plan_command as plan_command

    job = plan_command.JobShape(parsed.cores, parsed.hours, parsed.memory_mb)
    plan_command.launch(
        parsed.run_file, parsed.out, job, dry_run=parsed.dry_run
    )


def _run_a_job(parsed) -> None:
    """One Slurm job: the pieces packed into it, on a tree git names."""
    import decsim.experiments.plan_command as plan_command

    plan_command.refuse_an_unnamed_tree()
    run_path = pathlib.Path(parsed.run_file)
    run_dir = pathlib.Path(parsed.out)
    plan_command.run_job(
        run_path, run_dir, parsed.job, processes=parsed.processes
    )


def _fold(parsed) -> None:
    """A results folder's saved pieces folded; nothing run."""
    import decsim.experiments.collect_command as collect_command

    run_dir = pathlib.Path(parsed.out)
    collect_command.fold_the_run(run_dir)


def _collect(parsed) -> None:
    """Every chosen task collected until it stops."""
    import decsim.experiments.collect_command as collect_command

    collect_command.run_experiment(
        parsed.run_file,
        parsed.out,
        processes=parsed.processes,
        only=parsed.only,
        shot_count=parsed.shots,
    )


def _run_one_shot(parsed) -> None:
    """One seeded shot of the chosen task, narrated."""
    import decsim.experiments.collect_command as collect_command
    import decsim.experiments.experiment as experiment

    run_path = pathlib.Path(parsed.run_file)
    study = experiment.load_one_task(run_path, parsed.only)
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


def _list_the_tasks(run_file: str) -> None:
    """The experiment's task names, one a line, MultiSim's --list."""
    import decsim.experiments.experiment as experiment

    study = experiment.load(run_file)
    for task in study.tasks:
        print(task.name)


def _run_parser():
    """The run command's arguments, all here, where the command page reads them.

    The Slurm defaults are three cores a job (decsim/experiments/
    plan_command.py says why), a walltime limit of a day and 16384 MB a
    job. --job is a Slurm job's entry, which run.sbatch writes.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="decsim run")
    parser.add_argument(
        "run_file",
        nargs="?",
        default=None,
        help="the experiment's run file",
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
        "--list", action="store_true", help="print the task names and stop"
    )
    parser.add_argument(
        "--only", default=None, help="the one task to run, by its name"
    )
    parser.add_argument(
        "--shots",
        type=_shot_count,
        default=None,
        help="stop every task at this many shots, its first seeds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="run one narrated shot of this seed, of the first task or "
        "the --only one",
    )
    parser.add_argument(
        "--log",
        default=None,
        choices=("off", "print", "file", "both"),
        help="the narrated shot's engine log, over the task's",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="write the narrated shot's Chrome trace of the data path",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="pack the tasks' pieces into Slurm job arrays, then fold",
    )
    parser.add_argument(
        "--cores", type=int, default=3, help="pieces a job runs at once"
    )
    parser.add_argument(
        "--hours", type=int, default=24, help="a job's longest walltime"
    )
    parser.add_argument(
        "--memory-mb", type=int, default=16384, help="a job's memory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write the sbatch files and submit nothing",
    )
    parser.add_argument(
        "--fold",
        action="store_true",
        help="fold the --out folder's saved pieces and run nothing",
    )
    parser.add_argument("--job", type=int, default=None, help=argparse.SUPPRESS)
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
    if parsed.fold:
        _check_the_fold_arguments(parser, parsed)
        return
    if parsed.run_file is None:
        parser.error("name the run file")
    if parsed.job is not None and parsed.out is None:
        parser.error("--job runs one task into a folder; name it with --out")
    if parsed.slurm:
        _check_the_slurm_arguments(parser, parsed)
        return
    _check_the_local_arguments(parser, parsed)


def _check_the_fold_arguments(parser, parsed) -> None:
    """A fold names the results folder it folds."""
    if parsed.out is None:
        parser.error("--fold folds a results folder; name it with --out")


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
    """A Slurm run runs every task to its max_shots."""
    chosen = (parsed.only, parsed.shots, parsed.seed)
    if any(option is not None for option in chosen):
        parser.error(
            "--slurm runs every task to its max_shots; --only, --shots and "
            "--seed run locally"
        )


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
    "trace": _trace,
}

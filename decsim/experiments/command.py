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
    decsim run <run file> --record --out DIR
    decsim run <run file> --seeds FIRST:END --out DIR [--only NAME]
        [--processes N]
    decsim run --fold --out DIR
    decsim trace follow <file> --round k:n | --window k:n
    decsim params <run file> [--only NAME] [--match PATTERN]
    decsim info <record name>
    decsim describe <run file> [--only NAME]
    decsim check <run file> [--only NAME]
    decsim stream <run file> --rounds N [--only NAME] [--warmup W]
        [--seed S] [--out FILE]

Every run form takes --set PATH=VALUE, repeated, to set any parameter
of every task's machine (decsim/parameters; `decsim params` lists the
paths); DECSIM_SET holds more, `;` between them, the flags winning.

decsim writes no job script. A run split over jobs, on Slurm or any
other scheduler, is --record once, --seeds in each job, then --fold
(docs/how-to/run_on_a_slurm_cluster.md).

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
    """An experiment's tasks, one task, one shot, a record, seeds, a fold."""
    import decsim.experiments.parameters_command as parameters_command

    parser = _run_parser()
    parsed = parser.parse_args(argv)
    _check_the_run_arguments(parser, parsed)
    parameters_command.put_in_force(parsed.set)
    if parsed.fold:
        _fold(parsed)
        return
    if parsed.list:
        _list_the_tasks(parsed.run_file)
        return
    if parsed.record:
        _record(parsed)
        return
    if parsed.seeds is not None:
        _run_seeds(parsed)
        return
    if parsed.seed is not None:
        _run_one_shot(parsed)
        return
    _collect(parsed)


def _record(parsed) -> None:
    """Every task recorded into the folder jobs will share; no shot run."""
    import decsim.experiments.collect_command as collect_command

    run_dir = pathlib.Path(parsed.out)
    names = collect_command.record_the_folder(parsed.run_file, run_dir)
    print(f"recorded {len(names)} tasks in {run_dir}")


def _run_seeds(parsed) -> None:
    """One job's range of seeds, run into a recorded folder; no fold."""
    import decsim.experiments.collect_command as collect_command

    run_dir = pathlib.Path(parsed.out)
    collect_command.run_seeds(
        parsed.run_file,
        run_dir,
        parsed.seeds,
        only=parsed.only,
        processes=parsed.processes,
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
    """The run command's arguments, all here, where the page reads them."""
    import argparse

    import decsim.experiments.parameters_command as parameters_command

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
        "--record",
        action="store_true",
        help="record every task into the --out folder that jobs will "
        "share, and run no shot",
    )
    parser.add_argument(
        "--seeds",
        type=_seed_range,
        default=None,
        help="FIRST:END, run seeds FIRST to END - 1 of every task, or the "
        "--only one, into the recorded --out folder, skipping saved "
        "seeds, and fold nothing",
    )
    parser.add_argument(
        "--fold",
        action="store_true",
        help="fold the --out folder's saved pieces and run nothing",
    )
    parameters_command.add_set_argument(parser)
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


def _seed_range(text: str) -> range:
    """FIRST:END, seeds FIRST to END - 1, with 0 <= FIRST < END."""
    import argparse

    first_text, _colon, end_text = text.partition(":")
    try:
        first_seed = int(first_text)
        end_seed = int(end_text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"a seed range is FIRST:END, two whole numbers, got {text}"
        ) from None
    if not 0 <= first_seed < end_seed:
        raise argparse.ArgumentTypeError(
            f"a seed range FIRST:END has 0 <= FIRST < END, got {text}"
        )
    return range(first_seed, end_seed)


def _check_the_run_arguments(parser, parsed) -> None:
    """The run's arguments name one thing to do."""
    if parsed.fold:
        _check_the_fold_arguments(parser, parsed)
        return
    if parsed.run_file is None:
        parser.error("name the run file")
    if parsed.record or parsed.seeds is not None:
        _check_the_shared_folder_arguments(parser, parsed)
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


def _check_the_shared_folder_arguments(parser, parsed) -> None:
    """A record or a range of seeds names the folder jobs share.

    A folder of a new date for each job would scatter one run's pieces.
    """
    if parsed.out is None:
        parser.error(
            "--record and --seeds work in the folder jobs share; name it "
            "with --out"
        )
    local_options = (parsed.shots, parsed.seed)
    if any(option is not None for option in local_options):
        parser.error(
            "--record and --seeds take every task to its max_shots, or "
            "the given seeds; --shots and --seed run locally"
        )
    if parsed.record:
        _check_the_record_arguments(parser, parsed)


def _check_the_record_arguments(parser, parsed) -> None:
    """A record runs no shot, and records every task the fold folds."""
    if parsed.seeds is not None:
        parser.error("--record runs no shot and --seeds runs some; give one")
    if parsed.only is not None:
        parser.error(
            "--record records every task, which the fold folds; --only "
            "chooses the tasks of --seeds"
        )


def _trace(argv: list) -> None:
    """One round's or one window's path through one shot's trace file."""
    import decsim.experiments.trace_follow as trace_follow

    trace_follow.main(argv)


def _params(argv: list) -> None:
    """One task's settable paths, their values, defaults and setters."""
    import decsim.experiments.parameters_command as parameters_command

    parameters_command.params(argv)


def _info(argv: list) -> None:
    """One record's parameters and the records each of its parts takes."""
    import decsim.experiments.parameters_command as parameters_command

    parameters_command.info(argv)


def _describe(argv: list) -> None:
    """One task's machine as a graph, and its priced paths."""
    import decsim.experiments.describe_command as describe_command

    describe_command.describe(argv)


def _check(argv: list) -> None:
    """Every task's machine built once, before any job runs it."""
    import decsim.experiments.check_command as check_command

    check_command.check(argv)


def _stream(argv: list) -> None:
    """One memory task run as an unbounded stream, measured as it runs."""
    import decsim.experiments.stream_command as stream_command

    stream_command.stream(argv)


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
    "params": _params,
    "info": _info,
    "describe": _describe,
    "check": _check,
    "stream": _stream,
}

"""`decsim check`: every task's machine built once, as a run builds it.

    decsim check <run file> [--only NAME] [--set PATH=VALUE ...]

gem5's m5.instantiate() resolves every object's parameters and makes
every object before a tick runs (src/python/m5/simulate.py:220, 88-89,
162-163), so a setting no part can take fails then, not hours into a
job; qecboard's check built each task the same way. Loading the run
file makes each task (collect.Task), which lowers its workload and
checks across its records (MachineSettings.at_task); building runs
every part's own checks. No shot runs. A task its build refuses is
printed with the reason and the others are still built, so one check
lists every task to fix.
"""

import argparse
import time

import decsim.experiments.collect as collect
import decsim.experiments.experiment as experiment
import decsim.experiments.parameters_command as parameters_command
import decsim.experiments.refusal as refusal
import decsim.machine as machine_module

_FIRST_SEED = 0


def check(argv: list) -> None:
    """Build each task's machine; refuse the run if any build refused."""
    parser = _parser()
    parsed = parser.parse_args(argv)
    parameters_command.put_in_force(parsed.set)
    study = experiment.load(parsed.run_file)
    if parsed.only is not None:
        study = study.only(parsed.only)
    refused = []
    for task in study.tasks:
        reason = _refusal_of(task)
        if reason is not None:
            refused.append(task.name)
    if refused:
        raise refusal.RefusalError(
            f"{len(refused)} of {len(study.tasks)} tasks were refused: "
            f"{refused}"
        )


def built(task: collect.Task) -> machine_module.Machine:
    """The task's machine for its first seed, built as run_shot builds it."""
    return machine_module.Machine.build(
        task.machine,
        _FIRST_SEED,
        None,
        task.online_threshold,
        layout=task.layout,
    )


def _refusal_of(task: collect.Task) -> object:
    """The reason the task's build refused, printed; None when it built."""
    start = time.perf_counter()
    try:
        built(task)
    except ValueError as reason:
        print(f"{task.name}: refused: {reason}")
        return reason
    seconds = time.perf_counter() - start
    print(f"{task.name}: built ({seconds:.1f}s)")
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="decsim check")
    parser.add_argument("run_file", help="the experiment's run file")
    parser.add_argument(
        "--only", default=None, help="the task to check, by its name"
    )
    parameters_command.add_set_argument(parser)
    return parser

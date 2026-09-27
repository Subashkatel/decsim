"""`decsim run`: one seeded shot of one yaml, narrated.

The Python it wraps is `experiment.load_experiment(path)`, the first
point's task, and `Machine.build(task.shot_settings(), seed).run()`,
with the observation knobs on the command line instead of in the file,
the way gem5's --debug-flags and --debug-file set what the config
script did not (src/python/m5/main.py:280, 299). The shot is the first
point of the first sweep block, so one yaml runs without naming a
point.
"""

import argparse
import dataclasses
import pathlib
from typing import Optional

import decsim.collect as collect
import decsim.experiments.experiment as experiment
import decsim.experiments.measure as measure
import decsim.experiments.run_folder as run_folder
import decsim.machine as machine_module
import decsim.records.results as result_records
import decsim.settings as machine_settings


def run_one_shot(
    config_path,
    seed: int = 0,
    out_dir: Optional[pathlib.Path] = None,
    *,
    log: Optional[str] = None,
    trace: bool = False,
) -> list:
    """Build the first sweep point at one seed, run it, and say what it did.

    Returns the lines the command prints. The shot writes a run folder as
    a collect does (run_folder.py): the manifest, the config, the point's
    every value and its workload, the result, the QPU's commands, and the
    log and the trace when this run asked for them, the finished flag last.
    """
    config = experiment.load_experiment(config_path)
    task = config.first_point_task()
    settings = _with_observation(task.settings, log, trace)
    task = dataclasses.replace(task, settings=settings)
    shot_settings = task.shot_settings()
    machine = config.built_machine(shot_settings, seed)
    run_dir = run_folder.run_dir_for(config, out_dir)
    point_id = task.strong_id()
    started_utc = run_folder.start_run(config, run_dir, [point_id])
    run_folder.write_producer(run_dir, settings.workload)
    run_folder.record_point(run_dir, task, [(seed, 1)])
    result = machine.run()
    label = measure.shot_label(point_id, seed)
    write_shot(machine, settings, run_dir, label, result)
    run_folder.finish_run(config, run_dir, [point_id], started_utc)
    return _result_lines(config, task, seed, result, run_dir)


def main(argv: list) -> None:
    """The command line: one yaml, and the knobs for this one run."""
    arguments = _parsed(argv)
    lines = run_one_shot(
        arguments.config,
        arguments.seed,
        arguments.out,
        log=arguments.log,
        trace=arguments.trace,
    )
    text = "\n".join(lines)
    print(text)


def write_shot(
    machine: machine_module.Machine,
    settings: machine_settings.MachineSettings,
    run_dir: pathlib.Path,
    label: str,
    result: result_records.RunResult,
) -> None:
    """A shot's files in its run folder, the log and the trace named label.

    result.json and commands.json always, and the log and the trace when
    the observation asks for them. tools/deltakit_example.py and
    tools/live_memory_example.py write their shot through it too.
    """
    _write_files(machine, settings, run_dir, label)
    _write_result(result, run_dir)
    _write_commands(machine, run_dir)


@dataclasses.dataclass(frozen=True)
class _Arguments:
    """What `decsim run` was asked for."""

    config: str
    seed: int
    out: Optional[str]
    log: Optional[str]
    trace: bool


def _parsed(argv: list) -> _Arguments:
    """Parse `run`'s arguments; a bad one raises with the usage."""
    parser = argparse.ArgumentParser(prog="decsim run")
    parser.add_argument("config", help="the experiment yaml to run")
    parser.add_argument(
        "--seed", type=int, default=0, help="the shot's seed (default 0)"
    )
    parser.add_argument(
        "--out", default=None, help="the run folder the shot writes"
    )
    parser.add_argument(
        "--log",
        default=None,
        choices=("off", "print", "file", "both"),
        help="the engine narrator, overriding the yaml for this run",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="write this shot's Chrome trace of the data path",
    )
    parsed = parser.parse_args(argv)
    return _Arguments(
        config=parsed.config,
        seed=parsed.seed,
        out=parsed.out,
        log=parsed.log,
        trace=parsed.trace,
    )


def _with_observation(
    settings: machine_settings.MachineSettings,
    log: Optional[str],
    trace: bool,
) -> machine_settings.MachineSettings:
    """The flags this run gave, over the yaml's observation section."""
    changes = {}
    if log is not None:
        changes["log"] = log
    if trace:
        changes["trace"] = "chrome"
    if not changes:
        return settings
    observation = dataclasses.replace(settings.observation, **changes)
    return dataclasses.replace(settings, observation=observation)


def _write_result(
    result: result_records.RunResult, run_dir: pathlib.Path
) -> None:
    """result.json: every field of the shot's result record."""
    value = collect.json_value(result)
    result_path = run_dir / "result.json"
    run_folder.write_json(result_path, value)


def _write_commands(
    machine: machine_module.Machine, run_dir: pathlib.Path
) -> None:
    """commands.json: when each QPU command arrived and when it started."""
    values = []
    for event in machine.observation.command_events.events:
        value = {
            "kind": event.kind,
            "tick": event.tick,
            "operation_id": event.command.operation.id,
        }
        values.append(value)
    commands_path = run_dir / "commands.json"
    run_folder.write_json(commands_path, values)


def _write_files(
    machine: machine_module.Machine,
    settings: machine_settings.MachineSettings,
    run_dir: pathlib.Path,
    label: str,
) -> None:
    """The shot's log and trace, each where its knob says."""
    observation = settings.observation
    if observation.writes_log:
        log_dir = run_dir / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        text = "\n".join(machine.observation.log.lines)
        log_path = log_dir / f"{label}.log"
        contents = text + "\n"
        log_path.write_text(contents)
    if not observation.writes_trace:
        return
    trace_path = _trace_path(observation, run_dir, label)
    machine.observation.trace_writer.write(str(trace_path))


def _trace_path(observation, run_dir: pathlib.Path, label: str) -> pathlib.Path:
    """Where this shot's trace goes: the named path, or trace/ in the folder."""
    named = observation.trace_path
    if named is not None:
        return pathlib.Path(named)
    trace_dir = run_dir / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir / f"{label}.trace.json"


def _result_lines(
    config,
    task: collect.Task,
    seed: int,
    result: result_records.RunResult,
    run_dir: pathlib.Path,
) -> list:
    """The point, the terminal status, the ticks and every result."""
    metadata = collect.metadata_text(task.metadata)
    lines = [
        f"config: {config.name}",
        f"point: {metadata} seed {seed}",
        f"terminal status: {result.terminal_status}",
        f"execution done: {result.execution_done_ticks} ticks",
        f"fully done: {result.fully_done_ticks} ticks",
    ]
    for row in result.operation_results:
        operation_line = _operation_line(row)
        lines.append(operation_line)
    lines.append(f"run dir: {run_dir}")
    return lines


def _operation_line(row) -> str:
    """One operation's status, prediction and truth, as the gate reads it."""
    return (
        f"operation {row.operation_id}: {row.result_status}, "
        f"observables {row.logical_observables}, "
        f"truth {row.observable_truth}"
    )

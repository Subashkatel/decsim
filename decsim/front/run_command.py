"""`decsim run`: one seeded shot of one yaml, narrated.

The two-line Python front is `MachineSettings.from_yaml` and
`Machine.build(settings, seed).run()`; this is that, with the
observation knobs on the command line instead of in the file, the way
gem5's --debug-flags and --debug-file set what the config script did not
(src/python/m5/main.py:280, 299). The shot is the first point of the
first sweep block, so one yaml runs without naming a point.
"""

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Optional

import decsim.front.experiment as experiment
import decsim.front.measure as measure
import decsim.front.run_folder as run_folder
import decsim.machine as machine_module
import decsim.records.results as result_records
import decsim.settings as machine_settings


def run_one_shot(
    config_path,
    seed: int = 0,
    out_dir: Optional[Path] = None,
    *,
    log: Optional[str] = None,
    trace: bool = False,
) -> list:
    """Build the first sweep point at one seed, run it, and say what it did.

    Returns the lines the command prints. The log and the trace are
    written into the run folder when this run asked for them; a run that
    asks for neither writes no folder at all.
    """
    config = experiment.load_experiment(config_path)
    settings = _first_point_settings(config)
    settings = _with_observation(settings, log, trace)
    writes_files = settings.observation.writes_log
    if settings.observation.writes_trace:
        writes_files = True
    run_dir = None
    if writes_files:
        run_dir = run_folder.run_dir_for(config, out_dir)
    machine = machine_module.Machine.build(settings, seed)
    result = machine.run()
    _write_files(machine, settings, run_dir, seed)
    return _result_lines(config, settings, seed, result, run_dir)


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
        "--out", default=None, help="the folder the log and trace go in"
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


def _first_point_settings(config) -> machine_settings.MachineSettings:
    """The machine at the first point of the first sweep block."""
    block = config.sweep[0]
    return config.point_settings(
        physical_error_probability=block.physical_error_probabilities[0],
        distance=block.distances[0],
        round_period_us=block.round_periods_microseconds[0],
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


def _write_files(
    machine: machine_module.Machine,
    settings: machine_settings.MachineSettings,
    run_dir: Optional[Path],
    seed: int,
) -> None:
    """The shot's log and trace, each where its knob says."""
    if run_dir is None:
        return
    label = measure.shot_label(settings, seed)
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


def _trace_path(observation, run_dir: Path, label: str) -> Path:
    """Where this shot's trace goes: the named path, or trace/ in the folder."""
    named = observation.trace_path
    if named is not None:
        return Path(named)
    trace_dir = run_dir / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir / f"{label}.trace.json"


def _result_lines(
    config,
    settings: machine_settings.MachineSettings,
    seed: int,
    result: result_records.RunResult,
    run_dir: Optional[Path],
) -> list:
    """The point, the terminal status, the ticks and every result."""
    lines = [
        f"config: {config.name}",
        f"point: p{settings.workload.physical_error_probability:g} "
        f"d{settings.qpu.distance} "
        f"round period {settings.qpu.round_period_microseconds:g} us "
        f"seed {seed}",
        f"terminal status: {result.terminal_status}",
        f"execution done: {result.execution_done_ticks} ticks",
        f"fully done: {result.fully_done_ticks} ticks",
    ]
    for row in result.operation_results:
        operation_line = _operation_line(row)
        lines.append(operation_line)
    if run_dir is not None:
        lines.append(f"run dir: {run_dir}")
    return lines


def _operation_line(row) -> str:
    """One operation's status, prediction and truth, as the gate reads it."""
    return (
        f"operation {row.operation_id}: {row.result_status}, "
        f"observables {row.logical_observables}, "
        f"truth {row.observable_truth}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])

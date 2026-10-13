"""`decsim describe`: one task's machine as a graph, and its priced paths.

    decsim describe <run file> [--only NAME] [--set PATH=VALUE ...] [--json]

It prints the task's machine as YAML (topology/machine_file.py), then
each path's latency and rate as the run prices them. gem5 writes the
instantiated system as config.ini and config.json for the same reason:
what ran is read, not inferred from the script
(src/python/m5/simulate.py). --json writes the same graph and paths as
one json object, as config.json holds what config.ini does.
"""

import argparse
import json

import decsim.experiments.experiment as experiment
import decsim.experiments.parameters_command as parameters_command
import decsim.links.settings as link_settings
import decsim.parameters.fields as fields
import decsim.topology.machine_file as machine_file

UNBOUNDED = "unbounded"
# the channel's parameters, which write a latency and a rate as --set does
_CHANNEL = fields.parameters_of(link_settings.ChannelSettings)
_CAPACITY = fields.parameters_of(link_settings.CapacitySettings)


def describe(argv: list) -> None:
    """Print one task's machine as a graph, then its priced paths."""
    parser = _parser()
    parsed = parser.parse_args(argv)
    parameters_command.put_in_force(parsed.set)
    study = experiment.load_one_task(parsed.run_file, parsed.only)
    task = study.tasks[0]
    written = task.machine
    if parsed.json:
        _print_json(task, written)
        return
    text = machine_file.dumps(written)
    print(f"# {task.name}, as a graph")
    print(text)
    print("# its paths, as the run prices them: latency, rate")
    links = task.layout.card
    for path in links.path_names():
        path_settings = links.path_settings(path)
        line = path_line(path, path_settings)
        print(line)


def path_line(name: str, path_settings: link_settings.PathSettings) -> str:
    """One path's name, latency and rate, as --set writes them."""
    latency, rate = priced(path_settings)
    return f"{name:34} {latency:>12}  {rate}"


def priced(path_settings: link_settings.PathSettings) -> tuple:
    """A path's latency and rate, as --set writes them."""
    channel = path_settings.channel
    latency_parameter = _CHANNEL["propagation_latency"]
    latency = latency_parameter.written(channel.propagation_latency_ticks)
    rate = UNBOUNDED
    if channel.capacity is not None:
        rate_parameter = _CAPACITY["input"]
        bits_per_microsecond = channel.capacity.input_bits_per_microsecond
        rate = rate_parameter.written(bits_per_microsecond)
    return latency, rate


def _print_json(task: object, written: object) -> None:
    """The task's graph and its priced paths as one json object."""
    links = task.layout.card
    paths = {}
    for path in links.path_names():
        path_settings = links.path_settings(path)
        latency, rate = priced(path_settings)
        paths[path] = {"latency": latency, "rate": rate}
    graph_document = machine_file.document_of(written)
    described = {"task": task.name, "graph": graph_document, "paths": paths}
    text = json.dumps(described, indent=2)
    print(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="decsim describe")
    parser.add_argument("run_file", help="the experiment's run file")
    parser.add_argument(
        "--only", default=None, help="the task to describe, by its name"
    )
    parser.add_argument(
        "--json", action="store_true", help="write it as one json object"
    )
    parameters_command.add_set_argument(parser)
    return parser

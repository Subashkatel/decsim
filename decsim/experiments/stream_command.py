"""`decsim stream`: one memory task run as an unbounded stream.

    decsim stream <run file> --rounds N [--only NAME] [--warmup W]
        [--seed S] [--out FILE] [--set PATH=VALUE ...]

The task's memory circuit becomes a stream of N rounds read out at the
last (frontends/memory_stream.py), sampled a round at a time by the flip
stream source (qpu/flip_stream_device.py), the task's own when it names
that source, as a drift study does, on the task's own machine. The
machine lets each window go once it is final and the stream's measures
fold it (observation.folds_windows; windows/window_release.py,
experiments/stream_measures.py), so a stream of 10^6 rounds runs in
flat memory. --warmup W begins the region of interest after W rounds,
as gem5's m5.stats.reset() begins one: the statistics restart there
(observation.warmup_rounds), and the latency samples start with the
windows that commit from the round after it
(observation.first_recorded_round, unless the task's starts later).

It writes one record, as json: the task, its id, the rounds, the warmup,
the seed and the StreamMeasurement, which holds the region of
interest's latencies, windows, queues, backlog, links and statistics
and the one logical outcome of the end readout. A logical failure
inside the stream is not counted: no source defines one inside an
unbounded stream.

A shape a stream does not build yet is refused, each "not implemented
yet" with its reason: a workload other than one decoded Stim memory
patch, a source other than Stim's, the trace, the log, the data
movement and window outcome ledgers, a strong window that absorbs weak ones
(the double window), batched or bulk strong decoding, and a graph of a
new shape.
"""

import argparse
import dataclasses
import json
import pathlib
from typing import Optional

import decsim.build.decoders as decoders_part
import decsim.build.layout as layout_module
import decsim.experiments.collect as collect
import decsim.experiments.experiment as experiment
import decsim.experiments.parameters_command as parameters_command
import decsim.experiments.refusal as refusal
import decsim.experiments.stream_measures as stream_measures
import decsim.frontends.memory_stream as memory_stream
import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.qpu.flip_stream_device as flip_stream_device
import decsim.refusals as refusals
import decsim.topology.graph as topology_graph

# the sources whose rounds the flip stream samples the same way
STIM_SOURCES = ("stim_device", "flip_stream")


def stream(argv: list) -> None:
    """Run one task's memory as a stream; print or write its record."""
    parser = _parser()
    parsed = parser.parse_args(argv)
    _check_the_arguments(parser, parsed)
    parameters_command.put_in_force(parsed.set)
    study = experiment.load_one_task(parsed.run_file, parsed.only)
    task = study.tasks[0]
    record = stream_record(task, parsed.rounds, parsed.warmup, parsed.seed)
    text = json.dumps(record, indent=2)
    if parsed.out is None:
        print(text)
        return
    path = pathlib.Path(parsed.out)
    lines = text + "\n"
    path.write_text(lines)
    print(f"decsim stream wrote {path}")


def stream_record(
    task: collect.Task, rounds: int, warmup: int, seed: int
) -> dict:
    """The task's memory streamed for the rounds, measured as one record."""
    graph = stream_machine(task, rounds, warmup)
    machine = machine_module.Machine.build(
        graph, seed, online_threshold=task.online_threshold
    )
    measures = stream_measures.StreamMeasures(machine)
    result = machine.run()
    measurement = measures.measurement(result)
    measured = dataclasses.asdict(measurement)
    task_id = task.strong_id()
    return {
        "task": task.name,
        "task_id": task_id,
        "rounds": rounds,
        "warmup_rounds": warmup,
        "seed": seed,
        "measurement": measured,
    }


def stream_machine(
    task: collect.Task, rounds: int, warmup: int
) -> topology_graph.Machine:
    """The task's machine running its memory as a stream that folds.

    A task with no Pauli frame gets the default one, whose commit is
    what a window's measures are read at.
    """
    _refuse_a_graph_of_a_new_shape(task)
    machine = task.machine
    _refuse_what_a_stream_does_not_build(machine, task.layout)
    workload = machine.held("workload")
    streamed = _stream_workload(workload, rounds)
    qpu = machine.held("qpu")
    source = qpu.source
    if source.name != flip_stream_device.FlipStreamDevice.Settings.name:
        source = flip_stream_device.FlipStreamDevice.Settings()
    flip_qpu = dataclasses.replace(qpu, source=source)
    first_after_warmup = warmup + 1
    first_recorded = max(
        machine.observation.first_recorded_round, first_after_warmup
    )
    observation = dataclasses.replace(
        machine.observation,
        folds_windows=True,
        warmup_rounds=warmup,
        first_recorded_round=first_recorded,
    )
    machine = dataclasses.replace(machine, observation=observation)
    machine = machine.with_held("workload", streamed)
    machine = machine.with_held("qpu", flip_qpu)
    if machine.held("pauli_frame") is None:
        machine = machines.with_frame(machine)
    return machine


def _stream_workload(workload, rounds: int):
    """The memory as a stream; what memory_stream refuses, one sentence."""
    try:
        return memory_stream.memory_stream(workload, rounds)
    except ValueError as refused:
        raise refusal.RefusalError(f"{refused}; {refusals.REFUSALS}") from None


def _refuse_a_graph_of_a_new_shape(task: collect.Task) -> None:
    """A stream builds the entry edge alone: at most two pools, no router."""
    layout = task.layout
    if len(layout.pools) <= 2 and layout.router is None:
        return
    raise refusal.RefusalError(
        f"not implemented yet: task {task.name} is a graph of a new shape, "
        "and a stream runs its entry edge alone; "
        f"{refusals.REFUSALS}"
    )


def _refuse_what_a_stream_does_not_build(
    machine: topology_graph.Machine,
    layout: layout_module.Layout,
) -> None:
    """The first shape a stream does not build yet, refused."""
    reasons = [check(machine) for check in _STREAM_CHECKS]
    absorbing_reason = _an_absorbing_strong_window(layout)
    bulk_reason = _bulk_strong_decoding(machine)
    batching_reason = _batching(layout)
    reasons.extend((absorbing_reason, bulk_reason, batching_reason))
    for reason in reasons:
        if reason is None:
            continue
        raise refusal.RefusalError(
            f"not implemented yet: {reason}; {refusals.REFUSALS}"
        )


def _another_source(machine) -> Optional[str]:
    qpu = machine.held("qpu")
    source_name = qpu.source.name
    if source_name in STIM_SOURCES:
        return None
    return (
        "a stream samples the memory's Stim circuit as the Stim device "
        f"does, and this task's source is {source_name}"
    )


def _a_trace(machine) -> Optional[str]:
    if machine.observation.trace == "off":
        return None
    return "a stream keeps no trace, which grows with every round"


def _a_log(machine) -> Optional[str]:
    if machine.observation.log == "off":
        return None
    return "a stream keeps no log, which grows with every round"


def _a_data_movement_ledger(machine) -> Optional[str]:
    if not machine.observation.data_movement:
        return None
    return "a stream keeps no data movement ledger, which keeps every hop"


def _a_window_outcome_ledger(machine) -> Optional[str]:
    if not machine.observation.record_window_outcomes:
        return None
    return "a stream keeps no window outcomes, which keep every window"


def _an_absorbing_strong_window(
    layout: layout_module.Layout,
) -> Optional[str]:
    switching = layout.entry_switching()
    if switching is None:
        return None
    strong_window = switching.strong_window
    if not strong_window.absorbs_weak_windows:
        return None
    return (
        f"the {strong_window.name} absorbs windows the plan lays up front, "
        "and a stream lays its windows as its rounds arrive"
    )


def _bulk_strong_decoding(machine) -> Optional[str]:
    manager = machine.held("decoder_manager")
    if not manager.bulk_strong:
        return None
    return (
        "a merged batch files its stages under its own key, which a "
        "stream's per-window measures do not read (bulk_strong)"
    )


def _batching(layout: layout_module.Layout) -> Optional[str]:
    if not decoders_part.batches_each_member(layout):
        return None
    return (
        "a merged batch files its stages under its own key, which a "
        "stream's per-window measures do not read (a pool's batching)"
    )


# each shape of the observation or the QPU a stream does not build yet,
# in the order a refusal names it; the layout's follow
_STREAM_CHECKS = (
    _another_source,
    _a_trace,
    _a_log,
    _a_data_movement_ledger,
    _a_window_outcome_ledger,
)


def _check_the_arguments(parser, parsed) -> None:
    """The rounds run past the warmup."""
    if parsed.rounds < 1:
        parser.error("--rounds is a whole number of rounds, at least 1")
    if not 0 <= parsed.warmup < parsed.rounds:
        parser.error(
            "--warmup is a whole number of rounds below --rounds, so the "
            "region of interest begins before the stream ends"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="decsim stream")
    parser.add_argument("run_file", help="the experiment's run file")
    parser.add_argument(
        "--rounds",
        type=int,
        required=True,
        help="the stream's rounds; the readout is at the last",
    )
    parser.add_argument(
        "--only", default=None, help="the task to stream; the first if none"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="rounds before the region of interest begins",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="the stream's root seed"
    )
    parser.add_argument(
        "--out", default=None, help="the json file to write; printed if none"
    )
    parameters_command.add_set_argument(parser)
    return parser

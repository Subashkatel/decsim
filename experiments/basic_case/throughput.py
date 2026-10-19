r"""The outer loop on a timing-only stream: throughput first, accuracy read off.

    python throughput.py RESULTS PROCESSES

Toshio et al. 2510.25222 study the backlog on a model with no syndrome
(Sec. III C, App. C): a window escalates at random at a rate, each
decoder takes a fixed time, and the queue does the rest; SWIPER
(2412.05115 Sec. 5.3) likewise samples its decode times rather than
decoding. This is that model at the basic case's points: windows of d
rounds committing every d us, union find taking the pilot's mean time a
kept window took and an escalated window the pilot's mean time an
escalated one took (threshold_plots.py's q* reads the same two means).
The accuracy is not simulated here: the threshold sweep measured the
error rate at each escalated share on real shots, and throughput_plots.py
reads a run's share off it.

Three studies, each at d = 5 and 7:
  - steady: p = 0.002 throughout; fixed shares of 5 to 20%, and the
    outer loop (target_steps.py) from 10%, at the approved watermarks
    (loop) and at them counted in one escalation's backlog
    (loop_scaled);
  - p_rises: p steps from 0.002 to 0.003 halfway, so the decoders take
    the 0.003 times and the fixed 20 dB threshold escalates the share
    the sweep measured at 0.003; the loop from 10%;
  - strong_slows: the strong decoder takes 1.5 times as long halfway,
    as a shared GPU would; a fixed 10% and the loop from 10%.

A window's backlog is the rounds measured and not committed when it
commits, its d buffer rounds and its reaction, as the backlog study's.
RESULTS/streams.csv holds a row a stream; RESULTS/windows.csv every
window's commit time, tier, share and backlog.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md):

```
export PYTHONPATH=$PWD/../..
sbatch -N1 -c32 --mem=64G -t 2:00:00 --wrap "python throughput.py \
  $PWD/../../results/basic_case_throughput 32"
```
"""

import concurrent.futures
import csv
import itertools
import math
import pathlib
import statistics
import sys

import decsim
import decsim.confidence.no_signal as no_signal
import decsim.config as decsim_config
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.escalation.target_steps as target_steps
import decsim.escalation.threshold_sources as threshold_sources
import decsim.escalation.verdicts as verdicts
import decsim.experiments.collect as collect
import decsim.experiments.measure as measure
import decsim.presets.clocks as clocks
import decsim.presets.link_cards as link_cards
import decsim.presets.machines as machines
import decsim.producers as producers
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.topology.fabrics as fabrics

ROUND_PERIOD_MICROSECONDS = 1.0
ROUNDS = 40_000
CHANGE_ROUND = 20_000
SEEDS = 10
DISTANCES = (5, 7)
# the pilot's mean us from data ready to commit, less the boundary wait,
# of a kept and of an escalated window (threshold_plots.py _service)
MEAN_SERVICE_MICROSECONDS = {
    (5, 0.002): (2.386, 21.769),
    (5, 0.003): (2.421, 24.889),
    (7, 0.002): (2.727, 27.699),
    (7, 0.003): (2.883, 31.973),
}
# the share the fixed 20 dB threshold escalated on the sweep's shots
FIXED_20_DB_SHARE = {
    (5, 0.002): 0.08694,
    (5, 0.003): 0.243085,
    (7, 0.002): 0.080705,
    (7, 0.003): 0.27159,
}
STEADY_SHARES = (0.05, 0.1, 0.15, 0.2)
LOOP_START = 0.1
LOOPS = ("loop", "loop_scaled")
# loop_scaled's watermarks, in one escalation's backlog: the windows
# that arrive while one escalated window is served
CALM_ESCALATIONS = 1
HIGH_ESCALATIONS = 3
STRONG_SLOWDOWN = 1.5
STREAM_FIELDS = (
    "study",
    "control",
    "d",
    "seed",
    "windows",
    "share_before",
    "share_after",
    "mean_backlog_after_rounds",
    "max_backlog_rounds",
    "final_backlog_rounds",
)
WINDOW_FIELDS = (
    "study",
    "control",
    "d",
    "seed",
    "window",
    "committed_us",
    "tier",
    "backlog_rounds",
)


def main(results: pathlib.Path, processes: int) -> None:
    """Every stream, and the two tables."""
    results.mkdir(parents=True, exist_ok=True)
    runs = _runs()
    streams = []
    windows = []
    with concurrent.futures.ProcessPoolExecutor(processes) as pool:
        for stream, window_rows in pool.map(run_stream, runs):
            streams.append(stream)
            windows.extend(window_rows)
    streams_path = results / "streams.csv"
    windows_path = results / "windows.csv"
    _write(streams_path, STREAM_FIELDS, streams)
    _write(windows_path, WINDOW_FIELDS, windows)


def _runs() -> list:
    """(study, control, d, seed) of every stream."""
    steady = [f"fixed_{share:g}" for share in STEADY_SHARES]
    controls = {
        "steady": [*steady, *LOOPS],
        "p_rises": ["fixed_20dB", *LOOPS],
        "strong_slows": ["fixed_0.1", *LOOPS],
    }
    runs = []
    for study, names in controls.items():
        grid = itertools.product(names, DISTANCES, range(SEEDS))
        runs.extend((study, name, d, seed) for name, d, seed in grid)
    return runs


def machine(study: str, control: str, distance: int):
    """The timing-only stream of one study and control at one d."""
    change_window = CHANGE_ROUND // distance
    keep, escalated = MEAN_SERVICE_MICROSECONDS[(distance, 0.002)]
    later_keep, later_escalated = keep, escalated
    if study == "p_rises":
        later = MEAN_SERVICE_MICROSECONDS[(distance, 0.003)]
        later_keep, later_escalated = later
    if study == "strong_slows":
        later_escalated = keep + STRONG_SLOWDOWN * (escalated - keep)
    weak = _pool((keep, later_keep), change_window)
    strong_latencies = (escalated - keep, later_escalated - later_keep)
    strong = _pool(strong_latencies, change_window)
    verdict = _verdict(study, control, distance, change_window)
    return _stream(distance, weak, strong, verdict)


def _pool(latencies: tuple, change_window: int):
    """A pool taking the first latency, the second from the window on."""
    first, later = latencies
    algorithm = decoders.PresetLatencyDecoder.Settings(
        first, later_latency_microseconds=later, from_window=change_window
    )
    # the pilot's means hold the fetch and release, so they cost nothing
    engine = decoder_settings.EngineSettings(
        clock=clocks.FRIDGE_CLOCK,
        fetch_cycles_per_round=0,
        fetch_cycles_per_job=0,
        release_cycles_per_job=0,
        release_cycles_per_round=0,
    )
    return decoder_settings.DecoderPoolSettings(
        algorithm=algorithm, engine=engine
    )


def _verdict(study: str, control: str, distance: int, change_window: int):
    """The loop from 10%, a fixed share, or the 20 dB threshold's shares."""
    if control in LOOPS:
        steps = _steps(control, distance)
        return verdicts.BacklogBernoulli.Settings(steps=steps)
    if control == "fixed_20dB":
        before = FIXED_20_DB_SHARE[(distance, 0.002)]
        after = FIXED_20_DB_SHARE[(distance, 0.003)]
        return verdicts.Bernoulli.Settings(
            before, later_rate=after, from_window=change_window
        )
    share_word = control.removeprefix("fixed_")
    share = float(share_word)
    return verdicts.Bernoulli.Settings(share)


def _steps(control: str, distance: int) -> target_steps.TargetSteps.Settings:
    """The approved watermarks (4 and 1 windows), or them in escalations.

    One escalated window holds the chain for its service, so about
    escalated / d windows wait behind it even when the chain keeps up;
    loop_scaled counts its watermarks in that backlog.
    """
    if control == "loop":
        return target_steps.TargetSteps.Settings(start=LOOP_START)
    _keep, escalated = MEAN_SERVICE_MICROSECONDS[(distance, 0.002)]
    windows_behind = escalated / distance
    one_escalation = math.ceil(windows_behind)
    high = HIGH_ESCALATIONS * one_escalation
    low = CALM_ESCALATIONS * one_escalation
    return target_steps.TargetSteps.Settings(
        start=LOOP_START, high_backlog_windows=high, low_backlog_windows=low
    )


def _stream(distance: int, weak, strong, verdict):
    """The weak baseline as a timing-only stream, the strong tier on it."""
    probability = 0.002
    rounds_per_distance = ROUNDS // distance
    memory = producers.MemorySettings(
        probability, rounds_per_distance=rounds_per_distance
    )
    timing_only = syndrome_devices.TimingOnlyDevice.Settings()
    base = machines.weak_baseline(
        distance, probability, ROUND_PERIOD_MICROSECONDS
    )
    stream = base.with_held("workload", memory)
    stream = stream.with_settings("qpu", source=timing_only)
    weak_tier = stream.with_record(machines.WEAK_POOL, weak)
    no_confidence = no_signal.NoSignal.Settings()
    unread = threshold_sources.FixedThreshold.Settings(threshold_decibels=0.0)
    switching = escalation_settings.SwitchingSettings(
        confidence=no_confidence, threshold=unread, verdict=verdict
    )
    # the pilot's means hold the hops, so the hops here take no time
    instant = fabrics.TimedLink(0.0, excludes_receiver_processing=True)
    weak_side = fabrics.RoleLinks(write_written=instant, base=weak_tier.fabric)
    fabric = link_cards.strong_side(weak_side, instant)
    return machines.with_strong_tier(weak_tier, strong, switching, fabric)


def run_stream(run: tuple) -> tuple:
    """One stream's row, and its windows' rows."""
    study, control, distance, seed = run
    stream_machine = machine(study, control, distance)
    task = decsim.Task(study, stream_machine, {})
    shot = collect.run_shot(task, seed)
    decoded = measure.decoded_windows(shot.machine.observation)
    rows = [_window_row(run, window, frame) for window, frame in decoded]
    return _stream_row(run, rows), rows


def _stream_row(run: tuple, rows: list) -> tuple:
    """The shares before and after the change, and the backlogs.

    rows are in window order, a row a window from window 0.
    """
    distance = run[2]
    change_window = CHANGE_ROUND // distance
    before = rows[:change_window]
    after = rows[change_window:]
    backlogs = [row[-1] for row in rows]
    after_backlogs = backlogs[change_window:]
    return (
        *run,
        len(rows),
        _share(before),
        _share(after),
        statistics.fmean(after_backlogs),
        max(backlogs),
        backlogs[-1],
    )


def _window_row(run: tuple, window, frame) -> tuple:
    """A window's commit, tier and backlog in rounds."""
    reaction_ticks = frame.committed_ticks - window.t_data_complete
    reaction_us = decsim_config.ticks_to_microseconds(reaction_ticks)
    _study, _control, distance, _seed = run
    backlog = distance + reaction_us / ROUND_PERIOD_MICROSECONDS
    committed_us = decsim_config.ticks_to_microseconds(frame.committed_ticks)
    return (*run, window.window_index, committed_us, frame.tier, backlog)


def _share(rows: list) -> float:
    tiers = [row[6] for row in rows]
    return tiers.count("strong") / len(tiers)


def _write(path: pathlib.Path, fields: tuple, rows: list) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


if __name__ == "__main__":
    folder = pathlib.Path(sys.argv[1])
    process_count = int(sys.argv[2])
    main(folder, process_count)

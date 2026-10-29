"""A stream's measures: each window read once it is final, then let go.

measure_shot reads a finished shot's ledgers whole (measure.py). A
stream of 10^6 rounds cannot keep them, so its measures are folded as
it runs, as ns-3's FlowMonitor keeps a running record per flow rather
than its packets (flow-monitor.h, FlowStats) and gem5 accumulates a
statistic as the run goes and dumps it at the end (src/base/stats).

A window is read when its record is final (WindowCommitter's
window_final): the frame has taken its one correction (a second write
raises, pauli_frame.py), and its decodes, inputs, hops and the sends of
the rounds it read have all happened. It is read by measure.window_sample,
the code measure_shot reads every window with, off the ledgers'
per-window records and a running TransferIndex. With
observation.folds_windows the window's records are then let go.

What is folded is what measure_shot computes from windows, for the
windows the region of interest records: each latency's values (count,
mean, maximum; the mean, an exact sum rounded once, is fmean's
number; and its p50, p90 and p99, each within 1% of the shot's exact
one, quantile_sketch.py), the windows by tier and by status, the
escalated windows, the reaction growth rates (an exact least-squares
fit in ticks), the deadline misses, the end-of-stream reaction and the
throughput. The per-round latencies (cwb_per_round and the stalls) are
not folded.

Past measure_shot, it samples the escalated share as the stream goes:
every hundredth committed window, the windows committed so far, the
escalated among them and the window's first committed round, so a
stream whose noise drifts shows its share before and after. With
observation.series_every_rounds it also writes a row every that many
rounds, each pool's waiting line among it (stream_series.py). With
answer checks, each checked window's row also holds the machine's own
decode of it: the pool that committed, its decode time and its reaction
(_WindowTimes).
"""

import collections
import dataclasses
import fractions
import functools
from typing import Optional

import decsim.build.decoders as decoders_part
import decsim.config as config_module
import decsim.experiments.measure as measure
import decsim.experiments.quantile_sketch as quantile_sketch
import decsim.experiments.stream_series as stream_series
import decsim.experiments.transfer_index as transfer_index
import decsim.machine as machine_module
import decsim.records.decoding as decoding_records
import decsim.records.results as result_records
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records

# A round's send is kept while a window may still read it: from the
# lowest start of the windows not yet read, less this many of the
# widest window's rounds, which covers a restart reading back.
KEPT_WINDOW_WIDTHS = 4
# the windows read between two lettings-go of the rounds no window reads
ROUNDS_LET_GO_EVERY = 64
# the committed windows between two samples of the escalated share
ESCALATION_SAMPLED_EVERY = 100
# a checked window the frame never committed (_WindowTimes)
NO_WINDOW_TIMES = {
    "committed_pool": None,
    "decode_us": None,
    "reaction_us": None,
}


@dataclasses.dataclass(frozen=True)
class StreamMeasurement:
    """What a stream measured, the region of interest's where it has one.

    values holds each latency's count, mean, maximum, p50, p90 and p99
    in us, the ones window_points_us reads; growth rates and the
    end-of-stream reaction are None when nothing gives them, as in
    ShotMeasurement.
    """

    rounds: int
    logical_failure: bool
    is_scored: bool
    unscored_reason: str
    decoded_windows: int
    recorded_windows: int
    values: dict
    window_tiers: dict
    window_statuses: dict
    provisional_no_correction_windows: int
    end_of_stream_reaction_us: Optional[float]
    weak_reaction_growth_rate: Optional[float]
    strong_reaction_growth_rate: Optional[float]
    deadline_missed_windows: Optional[int]
    first_deadline_miss_us: Optional[float]
    escalated_windows: int
    strong_decoded_rounds: int
    strong_service_sum_us: float
    throughput_windows_per_us: float
    throughput_rounds_per_us: float
    max_queued_windows: int
    pools: dict
    # each pool's own load and windows, the PoolRow a shot's
    # shot_pools.csv holds, one dict a pool in layout order
    pool_rows: list
    backlog_peak_rounds: Optional[int]
    referee_windows_checked: int
    referee_window_disagreements: int
    link_totals: dict
    statistics: list
    statistic_values: list
    escalation_series: list
    # one row a checkpoint round, when the run closes checkpoints
    # (observe/checkpoints.py Checkpoints.rows)
    checkpoints: list
    # one row a path, when the run checks each window's two answers
    # (observe/answer_checks.py AnswerChecks.rows), each window's
    # committed_pool, decode_us and reaction_us joined (_WindowTimes)
    answer_checks: list
    # one row a static threshold, when the run keeps static chains
    # (observe/answer_checks.py AnswerChecks.threshold_rows)
    threshold_checks: list
    # one row every observation.series_every_rounds rounds, when the run
    # sets it (stream_series.py SeriesRow)
    series: list


class StreamMeasures:
    """Listens to a built machine and folds each final window."""

    def __init__(self, machine: machine_module.Machine) -> None:
        self.machine = machine
        routes = machine.layout.routes()
        self.transfers = transfer_index.TransferIndex(
            routes.name_bits_by_answer_hop
        )
        self.frame_records: dict = {}
        observation_settings = machine.graph.observation
        deadline_us = observation_settings.reaction_deadline_microseconds
        self.folds = _Folds(deadline_us)
        # the weak and the strong tier's pools (Layout.tier_pools)
        self.tier_pools = machine.layout.tier_pools()
        self.series = _series_of(machine)
        self.window_times = _window_times_of(machine)
        self._connect(machine)

    def transfer_delivered(
        self, record: transfer_records.TransferRecord
    ) -> None:
        """One transfer landed: its row folded into the index."""
        traffic = self.machine.observation.traffic
        row = traffic.transfer_row(record)
        self.transfers.add(row)

    def correction_committed(
        self, record: decoding_records.PauliFrameCommitRecord
    ) -> None:
        """The frame took a window's one correction; read at window_final."""
        self.frame_records[record.window_key] = record

    def round_read_out(self, readout: round_records.QPUReadout) -> None:
        """The QPU read out one round: the generation span and the count."""
        self._readout_now()
        self.folds.count_round(readout.stream_id, readout.round_index)
        if self.series is not None:
            now = self.machine.engine.now
            self.series.round_read_out(readout.round_index, now)

    def idle_round_read_out(
        self,
        operation_id: object,
        patch: object,
        round_index: int,
    ) -> None:
        """An idle patch's round stretches the generation span too."""
        del operation_id, patch, round_index
        self._readout_now()

    def window_final(self, window: window_records.Window) -> None:
        """Read the window, then let it go when the stream asks."""
        frame_record = self.frame_records.pop(window.key, None)
        self.folds.fold_window_record(window)
        if frame_record is not None:
            self._fold_committed(window, frame_record)
        observation_settings = self.machine.graph.observation
        if observation_settings.folds_windows:
            self._let_go(window)

    def window_absorbed(self, key: tuple, owner_key: tuple) -> None:
        """A strong region took the window over; its records go.

        An absorbed window is never decoded and never final on its own:
        its strong region's window answers for its rounds
        (escalation/strong_window_shapes.py DoubleWindow).
        """
        del owner_key
        observation_settings = self.machine.graph.observation
        if not observation_settings.folds_windows:
            return
        observation = self.machine.observation
        observation.windows.forget(key)
        observation.stages.forget_window(key)
        self.transfers.forget_window(key)

    def measurement(
        self, result: result_records.RunResult
    ) -> StreamMeasurement:
        """What the stream measured, once its run has ended."""
        windows = self.machine.observation.windows.windows
        remaining = _unread_windows(windows)
        window_numbers = self._window_numbers(remaining)
        run_numbers = self._run_numbers(result, window_numbers)
        return StreamMeasurement(**window_numbers, **run_numbers)

    def _window_numbers(self, remaining: list) -> dict:
        """What the folded windows give, and the windows not read yet."""
        folds = self.folds
        statuses = folds.statuses_with(remaining)
        reasons = folds.reasons_with(remaining)
        ordered_reasons = sorted(reasons)
        provisional = folds.provisional_with(remaining)
        values = folds.value_summaries()
        tiers = dict(folds.window_tiers)
        end_reaction = folds.end_of_stream_reaction()
        weak_pools, strong_pools = self.tier_pools
        weak_growth = folds.growth_rate(weak_pools)
        strong_growth = folds.growth_rate(strong_pools)
        missed, first_miss = folds.deadline_misses()
        service_us = config_module.ticks_to_microseconds(
            folds.strong_service_ticks
        )
        windows_per_us, rounds_per_us = folds.throughput(remaining)
        return {
            "unscored_reason": ";".join(ordered_reasons),
            "decoded_windows": folds.decoded_windows,
            "recorded_windows": folds.recorded_windows,
            "values": values,
            "window_tiers": tiers,
            "window_statuses": statuses,
            "provisional_no_correction_windows": provisional,
            "end_of_stream_reaction_us": end_reaction,
            "weak_reaction_growth_rate": weak_growth,
            "strong_reaction_growth_rate": strong_growth,
            "deadline_missed_windows": missed,
            "first_deadline_miss_us": first_miss,
            "escalated_windows": folds.strong_windows,
            "strong_decoded_rounds": folds.strong_rounds,
            "strong_service_sum_us": service_us,
            "throughput_windows_per_us": windows_per_us,
            "throughput_rounds_per_us": rounds_per_us,
            "escalation_series": list(folds.counts.escalation_series),
        }

    def _run_numbers(
        self, result: result_records.RunResult, window_numbers: dict
    ) -> dict:
        """What the run's result, statistics and ledger counters give."""
        observation = self.machine.observation
        run_statistics = observation.statistics
        is_scored = window_numbers["unscored_reason"] == ""
        owners = _scored_owners(result)
        missed_truth = _logical_failure(owners)
        managers = run_statistics.groups[decoders_part.MANAGERS_PATH]
        audit = observation.referee_audit
        pool_numbers, pool_row_numbers = self._pool_numbers()
        link_totals = measure.link_totals(result.link_traffic)
        rows = run_statistics.rows()
        value_counts = run_statistics.value_counts()
        rounds = self.folds.rounds_read_out()
        backlog_peak = _backlog_peak(observation.decode_backlog)
        return {
            "rounds": rounds,
            "logical_failure": missed_truth and is_scored,
            "is_scored": is_scored,
            "max_queued_windows": managers.queued.peak,
            "pools": pool_numbers,
            "pool_rows": pool_row_numbers,
            "backlog_peak_rounds": backlog_peak,
            "referee_windows_checked": audit.windows_checked,
            "referee_window_disagreements": audit.window_disagreements,
            "link_totals": link_totals,
            "statistics": rows,
            "statistic_values": value_counts,
            "checkpoints": _checkpoint_rows(observation),
            "answer_checks": _answer_check_rows(observation, self.window_times),
            "threshold_checks": _threshold_check_rows(observation),
            "series": _series_rows(self.series),
        }

    def _connect(self, machine: machine_module.Machine) -> None:
        window_manager = machine.windows.window_manager
        sources = window_manager.window_sources()
        sources.window_final.connect(self.window_final)
        sources.window_absorbed.connect(self.window_absorbed)
        machine.links.trace.transfer_delivered.connect(self.transfer_delivered)
        run_statistics = machine.observation.statistics
        run_statistics.on_reset(self.folds.restart_pool_commits)
        frame = machine.control.pauli_frame
        if frame is not None:
            frame.trace.correction_committed.connect(self.correction_committed)
        machine.qpu.device.trace.round_emitted.connect(self.round_read_out)
        idle_rounds = machine.control.idle_rounds
        idle_rounds.trace.idle_round_emitted.connect(self.idle_round_read_out)

    def _readout_now(self) -> None:
        engine = self.machine.engine
        reset_tick = self.machine.observation.statistics.reset_tick
        self.folds.readout_at(engine.now, reset_tick)

    def _fold_committed(
        self,
        window: window_records.Window,
        frame_record: decoding_records.PauliFrameCommitRecord,
    ) -> None:
        """A window the frame committed: its decode, then its latencies."""
        stages = self.machine.observation.stages
        decode = measure.committed_decode(stages, frame_record)
        _weak_pools, strong_pools = self.tier_pools
        is_escalated = decode.tier in strong_pools
        if is_escalated:
            self.folds.fold_strong(decode)
        self.folds.note_commit(frame_record)
        self.folds.sample_escalation(window)
        if window.t_done is None:
            return
        reaction_us = measure.reaction_microseconds(window, frame_record)
        if self.series is not None:
            self.series.window_committed(reaction_us, is_escalated)
        if self.window_times is not None:
            self.window_times.add(window.key, decode, reaction_us)
        send = measure.last_required_send_ticks(decode, self.transfers.qpu_send)
        self.folds.fold_decoded(frame_record, send)
        observation_settings = self.machine.graph.observation
        if not observation_settings.records_round(window.commit_lo):
            return
        sample = self._sample(window, frame_record)
        self.folds.fold_recorded(window, frame_record, sample)

    def _sample(
        self,
        window: window_records.Window,
        frame_record: decoding_records.PauliFrameCommitRecord,
    ) -> measure.WindowSample:
        """The window's latencies off its own keys of the running index."""
        index = self.transfers
        link_delay = index.link_delays_of(window.key)
        input_hop = index.input_hops_of(window.key)
        reads = measure.TransferReads(
            link_delay, input_hop, index.qpu_send, index.answer_send
        )
        stages = self.machine.observation.stages
        layout = self.machine.layout
        input_hops = layout.input_hop_by_pool()
        escalation_hops = layout.entry_escalation_hops()
        return measure.window_sample(
            window, frame_record, stages, reads, input_hops, escalation_hops
        )

    def _let_go(self, window: window_records.Window) -> None:
        """The window's records leave every ledger that held them."""
        observation = self.machine.observation
        observation.windows.forget(window.key)
        observation.stages.forget_window(window.key)
        self.transfers.forget_window(window.key)
        self.folds.note_width(window.round_count)
        if self.folds.folded_windows % ROUNDS_LET_GO_EVERY:
            return
        self._let_go_of_rounds(window.stream_id)

    def _let_go_of_rounds(self, stream_id: object) -> None:
        """The sends of rounds below every unread window's reach."""
        windows = self.machine.observation.windows.windows
        starts = [
            each.start_round
            for each in windows.values()
            if each.stream_id == stream_id
        ]
        if not starts:
            return
        margin = KEPT_WINDOW_WIDTHS * self.folds.widest_window
        kept_from = min(starts) - margin
        let_go = self.folds.rounds_to_let_go(stream_id, kept_from)
        self.transfers.forget_rounds(stream_id, let_go)

    def _pool_numbers(self) -> tuple:
        """Each tier's load, then each pool's own row, as plain data.

        Both are over the span the QPU generated the region of interest's
        rounds over.
        """
        run_statistics = self.machine.observation.statistics
        layout = self.machine.layout
        device = self.machine.qpu.device
        round_period_ticks = device.clock.period_ticks
        generation_ticks = self.folds.generation_ticks(round_period_ticks)
        tiers = measure.pool_measures(run_statistics, layout, generation_ticks)
        committed = self.folds.counts.committed_by_pool
        rows = measure.pool_rows(
            run_statistics, layout, committed, generation_ticks
        )
        tier_numbers = dataclasses.asdict(tiers)
        row_numbers = [dataclasses.asdict(row) for row in rows]
        return tier_numbers, row_numbers


class _Folds:
    """The running values a stream's measures are read from."""

    def __init__(self, deadline_us: Optional[float]) -> None:
        self.values_by_name: dict = collections.defaultdict(_Values)
        self.window_tiers: collections.Counter = collections.Counter()
        self.statuses: collections.Counter = collections.Counter()
        self.reasons: set = set()
        self.growth_by_tier: dict = collections.defaultdict(_Growth)
        self.counts = _Counts(deadline_us=deadline_us)

    @property
    def decoded_windows(self) -> int:
        return self.counts.decoded_windows

    @property
    def recorded_windows(self) -> int:
        return self.counts.recorded_windows

    @property
    def folded_windows(self) -> int:
        return self.counts.folded_windows

    @property
    def strong_windows(self) -> int:
        return self.counts.strong_windows

    @property
    def strong_rounds(self) -> int:
        return self.counts.strong_rounds

    @property
    def strong_service_ticks(self) -> int:
        return self.counts.strong_service_ticks

    @property
    def widest_window(self) -> int:
        return self.counts.widest_window

    def fold_window_record(self, window: window_records.Window) -> None:
        """What every window gives: its status, reasons and first round."""
        counts = self.counts
        counts.folded_windows += 1
        if window.decode_status is not None:
            column = f"{window.decode_status}_windows"
            self.statuses[column] += 1
        self.reasons.add(window.no_correction_reason)
        self.reasons.add(window.provisional_no_correction_reason)
        if window.provisional_no_correction_reason is not None:
            counts.provisional_windows += 1
        counts.first_round_tick = _lower(
            counts.first_round_tick, window.t_first_round
        )

    def fold_strong(self, decode) -> None:
        """A committed decode of the strong tier: its rounds and service."""
        counts = self.counts
        counts.strong_windows += 1
        counts.strong_rounds += decode.round_count
        service = decode.done_ticks - decode.compute_start_ticks
        counts.strong_service_ticks += service

    def note_commit(
        self, frame_record: decoding_records.PauliFrameCommitRecord
    ) -> None:
        counts = self.counts
        counts.committed_windows += 1
        counts.committed_by_pool[frame_record.tier] += 1
        counts.last_commit_tick = _higher(
            counts.last_commit_tick, frame_record.committed_ticks
        )

    def restart_pool_commits(self, tick: int) -> None:
        """The statistics restart: each pool's commits count from here.

        A pool's row reads its load over the region of interest, so its
        committed windows are counted over the same region.
        """
        del tick
        self.counts.committed_by_pool.clear()

    def sample_escalation(self, window: window_records.Window) -> None:
        """Every hundredth committed window, the share escalated so far."""
        counts = self.counts
        if counts.committed_windows % ESCALATION_SAMPLED_EVERY:
            return
        sample = (counts.committed_windows, counts.strong_windows)
        self.counts.escalation_series.append((*sample, window.commit_lo))

    def fold_decoded(
        self,
        frame_record: decoding_records.PauliFrameCommitRecord,
        last_send_ticks: int,
    ) -> None:
        """A decoded window: the end-of-stream reaction's two ends."""
        counts = self.counts
        counts.decoded_windows += 1
        counts.final_send_tick = _higher(
            counts.final_send_tick, last_send_ticks
        )
        counts.decoded_commit_tick = _higher(
            counts.decoded_commit_tick, frame_record.committed_ticks
        )

    def fold_recorded(
        self,
        window: window_records.Window,
        frame_record: decoding_records.PauliFrameCommitRecord,
        sample: measure.WindowSample,
    ) -> None:
        """A recorded window: its latencies, tier, growth and deadline."""
        self.counts.recorded_windows += 1
        for name, value in sample.values.items():
            self.values_by_name[name].add(value)
        tier = sample.decode.tier
        self.window_tiers[tier.word] += 1
        frame_tier = window_records.PoolId(frame_record.tier)
        growth = self.growth_by_tier[frame_tier]
        reaction_ticks = frame_record.committed_ticks - window.t_data_complete
        growth.add(window.t_data_complete, reaction_ticks)
        reaction_us = sample.values["buffer0_ready_to_frame"]
        arrival_us = config_module.ticks_to_microseconds(window.t_data_complete)
        self.counts.count_against_deadline(reaction_us, arrival_us)

    def count_round(self, stream_id: object, round_index: int) -> None:
        latest = self.counts.latest_round_by_stream.get(stream_id, 0)
        self.counts.latest_round_by_stream[stream_id] = max(latest, round_index)

    def readout_at(self, tick: int, reset_tick: int) -> None:
        """A readout: the first since the region began, and the last."""
        counts = self.counts
        first = counts.first_readout_tick
        if first is None or first < reset_tick:
            counts.first_readout_tick = tick
        counts.last_readout_tick = tick

    def note_width(self, round_count: int) -> None:
        counts = self.counts
        counts.widest_window = max(counts.widest_window, round_count)

    def rounds_to_let_go(self, stream_id: object, kept_from: int) -> range:
        """The rounds below kept_from not let go yet."""
        counts = self.counts
        let_go_below = counts.let_go_below_by_stream.get(stream_id, 1)
        counts.let_go_below_by_stream[stream_id] = max(let_go_below, kept_from)
        return range(let_go_below, kept_from)

    def rounds_read_out(self) -> int:
        rounds = self.counts.latest_round_by_stream.values()
        return sum(rounds)

    def value_summaries(self) -> dict:
        return {
            name: values.summary()
            for name, values in self.values_by_name.items()
        }

    def statuses_with(self, remaining: list) -> dict:
        """Each status's windows, the unread windows' too."""
        statuses = collections.Counter(self.statuses)
        for window in remaining:
            if window.decode_status is not None:
                statuses[f"{window.decode_status}_windows"] += 1
        counts = dict.fromkeys(measure.WINDOW_STATUS_COLUMNS, 0)
        counts.update(statuses)
        return counts

    def reasons_with(self, remaining: list) -> set:
        """Every no-correction reason, the unread windows' too."""
        reasons = set(self.reasons)
        for window in remaining:
            reasons.add(window.no_correction_reason)
            reasons.add(window.provisional_no_correction_reason)
        reasons.discard(None)
        return reasons

    def provisional_with(self, remaining: list) -> int:
        count = self.counts.provisional_windows
        for window in remaining:
            if window.provisional_no_correction_reason is not None:
                count += 1
        return count

    def throughput(self, remaining: list) -> tuple:
        """Windows and rounds per us, first round in to the last commit."""
        counts = self.counts
        if counts.last_commit_tick is None:
            return 0.0, 0.0
        first = counts.first_round_tick
        for window in remaining:
            first = _lower(first, window.t_first_round)
        span_ticks = counts.last_commit_tick - first
        span_us = config_module.ticks_to_microseconds(span_ticks)
        rounds = self.rounds_read_out()
        windows_per_us = counts.decoded_windows / span_us
        rounds_per_us = rounds / span_us
        return windows_per_us, rounds_per_us

    def end_of_stream_reaction(self) -> Optional[float]:
        counts = self.counts
        if counts.final_send_tick is None:
            return None
        span_ticks = counts.decoded_commit_tick - counts.final_send_tick
        return config_module.ticks_to_microseconds(span_ticks)

    def growth_rate(self, tier_pools: tuple) -> Optional[float]:
        """A tier's reaction growth, fit over every window of its pools."""
        tier_growth = _Growth()
        for pool in tier_pools:
            growth = self.growth_by_tier.get(pool)
            if growth is not None:
                tier_growth.merge(growth)
        return tier_growth.slope()

    def deadline_misses(self) -> tuple:
        """The recorded windows over the deadline, the earliest's arrival."""
        counts = self.counts
        if counts.deadline_us is None:
            return None, None
        return counts.missed, counts.first_miss_us

    def generation_ticks(self, round_period_ticks: int) -> int:
        counts = self.counts
        span_ticks = counts.last_readout_tick - counts.first_readout_tick
        return span_ticks + round_period_ticks


@dataclasses.dataclass
class _Counts:
    """The running counts and ticks of the fold."""

    folded_windows: int = 0
    committed_windows: int = 0
    decoded_windows: int = 0
    recorded_windows: int = 0
    provisional_windows: int = 0
    strong_windows: int = 0
    strong_rounds: int = 0
    strong_service_ticks: int = 0
    widest_window: int = 0
    first_round_tick: Optional[int] = None
    last_commit_tick: Optional[int] = None
    final_send_tick: Optional[int] = None
    decoded_commit_tick: Optional[int] = None
    first_readout_tick: Optional[int] = None
    last_readout_tick: Optional[int] = None
    latest_round_by_stream: dict = dataclasses.field(default_factory=dict)
    let_go_below_by_stream: dict = dataclasses.field(default_factory=dict)
    # the windows the frame committed off each pool's decode, by word,
    # since the statistics' restart
    committed_by_pool: collections.Counter = dataclasses.field(
        default_factory=collections.Counter
    )
    # (committed windows, escalated windows, first committed round)
    escalation_series: list = dataclasses.field(default_factory=list)
    # the reaction budget (observation.reaction_deadline_microseconds)
    deadline_us: Optional[float] = None
    missed: int = 0
    first_miss_us: Optional[float] = None

    def count_against_deadline(
        self, reaction_us: float, arrival_us: float
    ) -> None:
        """A recorded window over the deadline, as measure.deadline_misses."""
        if self.deadline_us is None or reaction_us <= self.deadline_us:
            return
        self.missed += 1
        self.first_miss_us = _lower(self.first_miss_us, arrival_us)


class _Values:
    """One latency's values in us: count, exact sum, maximum, sketch.

    measure_shot averages by statistics.fmean, fsum over the count, and
    fsum is the exact sum correctly rounded. The floats' exact sum, kept
    as a Fraction and rounded once (float(Fraction) rounds correctly),
    is the same number in any order. The percentiles come off a sketch
    of the values (quantile_sketch.py), so a stream of any length keeps
    three numbers and a few hundred bucket counts per latency.
    """

    def __init__(self) -> None:
        self.count = 0
        self.total = fractions.Fraction(0)
        self.maximum: Optional[float] = None
        self.sketch = quantile_sketch.QuantileSketch()

    def add(self, value: float) -> None:
        self.count += 1
        exact = fractions.Fraction(value)
        self.total += exact
        self.maximum = _higher(self.maximum, value)
        self.sketch.add(value)

    def summary(self) -> dict:
        total = float(self.total)
        mean = total / self.count
        percentiles = self.sketch.percentiles()
        return {
            "count": self.count,
            "mean": mean,
            "maximum": self.maximum,
            **percentiles,
        }


class _Growth:
    """A tier's reaction against arrival, as exact sums in ticks."""

    def __init__(self) -> None:
        self.count = 0
        self.sums = [0, 0, 0, 0]
        self.first_arrival: Optional[int] = None
        self.has_two_arrivals = False

    def add(self, arrival_ticks: int, reaction_ticks: int) -> None:
        self.count += 1
        sums = self.sums
        sums[0] += arrival_ticks
        sums[1] += reaction_ticks
        sums[2] += arrival_ticks * reaction_ticks
        sums[3] += arrival_ticks * arrival_ticks
        if self.first_arrival is None:
            self.first_arrival = arrival_ticks
        if arrival_ticks != self.first_arrival:
            self.has_two_arrivals = True

    def merge(self, other: "_Growth") -> None:
        """Another pool's windows added, as if each had been added here."""
        if other.first_arrival is None:
            return
        self.count += other.count
        for index, value in enumerate(other.sums):
            self.sums[index] += value
        if self.first_arrival is None:
            self.first_arrival = other.first_arrival
        is_another_arrival = other.first_arrival != self.first_arrival
        if other.has_two_arrivals or is_another_arrival:
            self.has_two_arrivals = True

    def slope(self) -> Optional[float]:
        """The least-squares slope, None below two distinct arrivals.

        Microseconds over microseconds is ticks over ticks, so the slope
        is fit in exact integers and rounded once.
        """
        if not self.has_two_arrivals:
            return None
        arrivals, reactions, products, squares = self.sums
        count = self.count
        covariance = count * products - arrivals * reactions
        variance = count * squares - arrivals * arrivals
        slope = fractions.Fraction(covariance, variance)
        return float(slope)


class _WindowTimes:
    """Each committed window's own decode and reaction, for its checks.

    A run that checks answers keeps a row a window already
    (observe/answer_checks.py). This keeps beside it the machine's own
    decode of the window, the one the frame committed: its pool, its
    time from compute start to done (measure.committed_decode), and the
    window's reaction (measure.reaction_microseconds). So a window's
    check reads beside the time the machine took on it, and a slow
    decode can be set against a failed one. A window the frame never
    committed reads None.
    """

    def __init__(self) -> None:
        # window key -> the times a checked window's row takes
        self.by_key: dict = {}

    def add(self, window_key: tuple, decode, reaction_us: float) -> None:
        """One committed window's decode and reaction."""
        decode_ticks = decode.done_ticks - decode.compute_start_ticks
        decode_us = config_module.ticks_to_microseconds(decode_ticks)
        self.by_key[window_key] = {
            "committed_pool": decode.tier.word,
            "decode_us": decode_us,
            "reaction_us": reaction_us,
        }

    def join_into(self, rows: list, stream_ids: tuple) -> None:
        """Each checked window's times, written into its row.

        rows are the answer checks' (AnswerChecks.rows), each holding a
        list of windows for each stream in stream_ids' order.
        """
        for row in rows:
            stream_windows = row["windows"]
            streams = zip(stream_ids, stream_windows, strict=True)
            for stream_id, windows in streams:
                self._join_stream(stream_id, windows)

    def _join_stream(self, stream_id, windows: list) -> None:
        for window in windows:
            key = (stream_id, window["window_index"])
            times = self.by_key.get(key, NO_WINDOW_TIMES)
            window.update(times)


def _backlog_peak(backlog) -> Optional[int]:
    """The backlog sampler's peak, None when the run samples none."""
    if backlog is None:
        return None
    return backlog.peak


def _unread_windows(windows: dict) -> list:
    """The windows the fold has not read: none final yet."""
    return [
        window
        for window in windows.values()
        if window.published_request_key is None
    ]


def _series_of(
    machine: machine_module.Machine,
) -> Optional[stream_series.StreamSeries]:
    """The run's series, None when observation.series_every_rounds is."""
    every_rounds = machine.graph.observation.series_every_rounds
    if every_rounds is None:
        return None
    run_statistics = machine.observation.statistics
    paths = machine.layout.statistics_path_by_pool()
    queue_paths = {pool.word: f"{path}.queue" for pool, path in paths.items()}
    queue_levels = functools.partial(_queue_levels, run_statistics, queue_paths)
    backlog = machine.observation.decode_backlog
    return stream_series.StreamSeries(every_rounds, queue_levels, backlog)


def _queue_levels(run_statistics, queue_paths: dict) -> dict:
    """Each pool's jobs waiting now, by its word."""
    return {
        word: run_statistics.groups[path].queued.level
        for word, path in queue_paths.items()
    }


def _window_times_of(machine: machine_module.Machine) -> Optional[_WindowTimes]:
    """The windows' times, None when the run checks no answer."""
    if machine.observation.answer_checks is None:
        return None
    return _WindowTimes()


def _series_rows(series: Optional[stream_series.StreamSeries]) -> list:
    """The series' rows; none when the run writes no series."""
    if series is None:
        return []
    return series.row_dicts()


def _checkpoint_rows(observation) -> list:
    """The checkpoints' rows; none when the run closes no checkpoint."""
    if observation.checkpoints is None:
        return []
    return observation.checkpoints.rows()


def _answer_check_rows(
    observation, window_times: Optional[_WindowTimes]
) -> list:
    """The answer checks' rows, each window's times joined (_WindowTimes).

    None when the run checks no window.
    """
    answer_checks = observation.answer_checks
    if answer_checks is None:
        return []
    rows = answer_checks.rows()
    stream_ids = answer_checks.stream_ids()
    window_times.join_into(rows, stream_ids)
    return rows


def _threshold_check_rows(observation) -> list:
    """The static chains' rows; none when the run checks no window."""
    if observation.answer_checks is None:
        return []
    return observation.answer_checks.threshold_rows()


def _scored_owners(result: result_records.RunResult) -> tuple:
    owners = []
    for operation_result in result.operation_results:
        if operation_result.observable_truth is not None:
            owners.append(operation_result)
    return tuple(owners)


def _logical_failure(owners: tuple) -> bool:
    """Whether any owner's end readout missed the truth (measure.py)."""
    failure = False
    for owner in owners:
        truth = tuple(owner.observable_truth)
        prediction = tuple(owner.logical_observables)
        failure |= prediction != truth
    return failure


def _lower(current: Optional[float], value: Optional[float]):
    if value is None:
        return current
    if current is None:
        return value
    return min(current, value)


def _higher(current: Optional[float], value: float) -> float:
    if current is None:
        return value
    return max(current, value)

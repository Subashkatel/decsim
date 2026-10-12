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
number), the windows by tier and by status, the escalated windows, the
reaction growth rates (an exact least-squares fit in ticks), the
deadline misses, the end-of-stream reaction and the throughput. The
per-round latencies (cwb_per_round and the stalls) are not folded.
"""

import collections
import dataclasses
import fractions
from typing import Optional

import decsim.build.decoders as decoders_part
import decsim.config as config_module
import decsim.experiments.measure as measure
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


@dataclasses.dataclass(frozen=True)
class StreamMeasurement:
    """What a stream measured, the region of interest's where it has one.

    values holds each latency's count, mean and maximum in us, the ones
    window_points_us reads; growth rates and the end-of-stream reaction
    are None when nothing gives them, as in ShotMeasurement.
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
    backlog_peak_rounds: Optional[int]
    referee_windows_checked: int
    referee_window_disagreements: int
    link_totals: dict
    statistics: list
    statistic_values: list


class StreamMeasures:
    """Listens to a built machine and folds each final window."""

    def __init__(self, machine: machine_module.Machine) -> None:
        self.machine = machine
        self.lets_go = machine.settings.observation.folds_windows
        routes = machine.layout.routes()
        self.transfers = transfer_index.TransferIndex(
            routes.name_bits_by_answer_hop
        )
        self.frame_records: dict = {}
        observation_settings = machine.settings.observation
        deadline_us = observation_settings.reaction_deadline_microseconds
        self.folds = _Folds(deadline_us)
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
        self.folds.count_round(readout.operation_id, readout.round_index)

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
        if self.lets_go:
            self._let_go(window)

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
        weak_growth = folds.growth_rate(window_records.WEAK_POOL_ID)
        strong_growth = folds.growth_rate(window_records.STRONG_POOL_ID)
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
        pools = self._pools()
        pool_numbers = dataclasses.asdict(pools)
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
            "backlog_peak_rounds": backlog_peak,
            "referee_windows_checked": audit.windows_checked,
            "referee_window_disagreements": audit.window_disagreements,
            "link_totals": link_totals,
            "statistics": rows,
            "statistic_values": value_counts,
        }

    def _connect(self, machine: machine_module.Machine) -> None:
        window_manager = machine.windows.window_manager
        sources = window_manager.window_sources()
        sources.window_final.connect(self.window_final)
        machine.links.trace.transfer_delivered.connect(self.transfer_delivered)
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
        self.folds.fold_strong(decode)
        self.folds.note_commit(frame_record)
        if window.t_done is None:
            return
        send = measure.last_required_send_ticks(decode, self.transfers.qpu_send)
        self.folds.fold_decoded(frame_record, send)
        observation_settings = self.machine.settings.observation
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
        _operation_id, window_id = window.key
        index = self.transfers
        link_delay = index.link_delays_of(window.key)
        input_hop = index.input_hops_of(window_id)
        reads = measure.TransferReads(
            link_delay, input_hop, index.qpu_send, index.answer_send
        )
        stages = self.machine.observation.stages
        input_hops = self.machine.layout.input_hop_by_pool()
        return measure.window_sample(
            window, frame_record, stages, reads, input_hops
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
        self._let_go_of_rounds(window.operation_id)

    def _let_go_of_rounds(self, operation_id: object) -> None:
        """The sends of rounds below every unread window's reach."""
        windows = self.machine.observation.windows.windows
        starts = [
            each.start_round
            for each in windows.values()
            if each.operation_id == operation_id
        ]
        if not starts:
            return
        margin = KEPT_WINDOW_WIDTHS * self.folds.widest_window
        kept_from = min(starts) - margin
        let_go = self.folds.rounds_to_let_go(operation_id, kept_from)
        self.transfers.forget_rounds(operation_id, let_go)

    def _pools(self) -> object:
        machine = self.machine
        device = machine.qpu.device
        round_period_ticks = device.clock.period_ticks
        generation_ticks = self.folds.generation_ticks(round_period_ticks)
        pool_paths = machine.layout.statistics_path_by_pool()
        run_statistics = machine.observation.statistics
        return measure.pool_measures(
            run_statistics, pool_paths, generation_ticks
        )


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
        if decode.tier != window_records.STRONG_POOL_ID:
            return
        counts = self.counts
        counts.strong_windows += 1
        counts.strong_rounds += decode.round_count
        service = decode.done_ticks - decode.compute_start_ticks
        counts.strong_service_ticks += service

    def note_commit(
        self, frame_record: decoding_records.PauliFrameCommitRecord
    ) -> None:
        counts = self.counts
        counts.last_commit_tick = _higher(
            counts.last_commit_tick, frame_record.committed_ticks
        )

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

    def count_round(self, operation_id: object, round_index: int) -> None:
        latest = self.counts.latest_round_by_operation.get(operation_id, 0)
        self.counts.latest_round_by_operation[operation_id] = max(
            latest, round_index
        )

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

    def rounds_to_let_go(self, operation_id: object, kept_from: int) -> range:
        """The rounds below kept_from not let go yet."""
        counts = self.counts
        let_go_below = counts.let_go_below_by_operation.get(operation_id, 1)
        counts.let_go_below_by_operation[operation_id] = max(
            let_go_below, kept_from
        )
        return range(let_go_below, kept_from)

    def rounds_read_out(self) -> int:
        rounds = self.counts.latest_round_by_operation.values()
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

    def growth_rate(self, tier: window_records.PoolId) -> Optional[float]:
        growth = self.growth_by_tier.get(tier)
        if growth is None:
            return None
        return growth.slope()

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
    latest_round_by_operation: dict = dataclasses.field(default_factory=dict)
    let_go_below_by_operation: dict = dataclasses.field(default_factory=dict)
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
    """One latency's values in us: their count, exact sum and maximum.

    measure_shot averages by statistics.fmean, fsum over the count, and
    fsum is the exact sum correctly rounded. The floats' exact sum, kept
    as a Fraction and rounded once (float(Fraction) rounds correctly),
    is the same number in any order, and a stream of any length keeps
    three numbers per latency.
    """

    def __init__(self) -> None:
        self.count = 0
        self.total = fractions.Fraction(0)
        self.maximum: Optional[float] = None

    def add(self, value: float) -> None:
        self.count += 1
        exact = fractions.Fraction(value)
        self.total += exact
        self.maximum = _higher(self.maximum, value)

    def summary(self) -> dict:
        total = float(self.total)
        mean = total / self.count
        return {"count": self.count, "mean": mean, "maximum": self.maximum}


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

"""Building the two tiers' units and the pool each manager schedules.

The chip's manager has the pool of the tier that decodes the plan's
windows and a switching run gives the host's manager a pool of the
strong tier. The regression lock (tests/regression) runs every other
wiring of this module. No lock shot bounds the strong tier's memory,
and none can tell the machine's clock from the fridge's or the room's,
since both tick at 250 MHz.
"""

import dataclasses

import pytest

import decsim.build.decoders as decoder_build
import decsim.build.escalation as escalation_build
import decsim.confidence.cluster as cluster
import decsim.config as config
import decsim.decoders.schedulers as schedulers
import decsim.decoders.settings as decoder_settings
import decsim.engine as engine_module
import decsim.machine as machine_module
import decsim.records.seeds as seed_records
import decsim.records.windows as window_records
import decsim.topology.graph_layout as graph_layout
import tests.declared_run as declared_run
import tests.topology.test_graph_layout as layout_tests
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)


def test_each_tiers_unit_memory_reaches_the_pool_of_its_own_units():
    switching = declared_run.switching_run(escalates=True)
    settings = switching.settings
    weak_memory = decoder_settings.UnitMemorySettings(bits=12)
    strong_memory = decoder_settings.UnitMemorySettings(bits=30)
    weak = dataclasses.replace(settings.weak_decoder, unit_memory=weak_memory)
    strong = dataclasses.replace(
        settings.strong_decoder, unit_memory=strong_memory
    )
    bounded = dataclasses.replace(
        settings, weak_decoder=weak, strong_decoder=strong
    )

    machine = machine_module.Machine.build(bounded, 0)

    weak_unit = machine.decoders.decoder_manager.pool.units[0]
    strong_manager = machine.decoders.manager_of(window_records.STRONG_POOL_ID)
    strong_unit = strong_manager.pool.units[0]
    assert weak_unit.memory.capacity_bits == 12
    assert strong_unit.memory.capacity_bits == 30


def test_each_managers_statistics_are_kept_at_its_tiers_settings_path():
    """The jobs waiting over both pools are the two queues' sum.

    Each queue's level is set before it reports the change, so the sum
    over the pools holds what the queues hold at every tick: its area is
    the sum of their areas.
    """
    switching = declared_run.switching_run(escalates=True)
    machine = machine_module.Machine.build(switching.settings, 0)
    machine.run()

    groups = machine.observation.statistics.groups
    weak = groups["weak_decoder.queue"].queued
    strong = groups["strong_decoder.queue"].queued
    over_pools = groups[decoder_build.MANAGERS_PATH].queued

    decoder_paths = [path for path in sorted(groups) if "decode" in path]
    assert decoder_paths == [
        "decoder_manager",
        "strong_decoder.decodes",
        "strong_decoder.queue",
        "strong_decoder.units",
        "weak_decoder.decodes",
        "weak_decoder.queue",
        "weak_decoder.units",
    ]
    assert over_pools.area == weak.area + strong.area
    assert groups["strong_decoder.units"].busy_units.area > 0


def test_an_engine_that_names_no_clock_counts_on_the_machines():
    """No preset clock ticks at 300 MHz, so a unit on one fails here."""
    machine_clock = config.Clock.from_megahertz(300.0)
    engine = decoder_settings.EngineSettings()
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings()
    weak = decoder_settings.DecoderPoolSettings(
        algorithm=matching, engine=engine
    )

    unit = decoder_build.build_decoder_unit(
        weak, "weak_decoder", machine_clock, None, None
    )

    assert unit.timing.clock == machine_clock


def test_a_manager_that_names_no_clock_dispatches_on_the_machines():
    """A dispatch cost on no clock of its own is charged on the machine's.

    No preset clock ticks at 300 MHz, so a manager on one fails here.
    """
    machine_clock = config.Clock.from_megahertz(300.0)
    declared = declared_run.weak_only_run()
    manager_settings = decoder_settings.DecoderManagerSettings(
        dispatch_cycles=1
    )
    settings = dataclasses.replace(
        declared.settings,
        clock=machine_clock,
        decoder_manager=manager_settings,
    )

    machine = declared_run.run_machine(settings)

    dispatch_cost = machine.decoders.decoder_manager.service.dispatch_cost
    assert dispatch_cost.clock == machine_clock


class _SeedRecordingScheduler(schedulers.FifoScheduler):
    """A FIFO that keeps every seed the run hands it."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The record that builds this FIFO."""

        def build(self) -> "_SeedRecordingScheduler":
            return _SeedRecordingScheduler()

    def __init__(self):
        self.reserved_seeds = []

    def reserve_run_seed(self, seed):
        self.reserved_seeds.append(seed)
        return seed_records.RunSeedReservation(None)

    def commit_run_seed(self, reservation):
        del reservation

    def cancel_run_seed(self, reservation):
        del reservation


def test_each_managers_scheduler_is_seeded_on_its_own_path():
    """Every shipped scheduler is the unseeded FIFO; this one takes seeds."""
    weak, strong = declared_run.switching_decoders(False)
    engine = declared_run.DECLARED_ENGINE
    weak_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=weak, engine=engine
    )
    strong_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=strong, engine=engine
    )
    scheduler_settings = _SeedRecordingScheduler.Settings()
    manager_settings = decoder_settings.DecoderManagerSettings(
        scheduler=scheduler_settings
    )
    switching = declared_run.switching_run(escalates=True)
    settings = dataclasses.replace(
        switching.settings,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        decoder_manager=manager_settings,
    )

    machine = machine_module.Machine.build(settings, 7)

    chip_scheduler = machine.decoders.decoder_manager.queue.scheduler
    strong_manager = machine.decoders.manager_of(window_records.STRONG_POOL_ID)
    host_scheduler = strong_manager.queue.scheduler
    assert len(chip_scheduler.reserved_seeds) == 1
    assert len(host_scheduler.reserved_seeds) == 1
    assert chip_scheduler.reserved_seeds != host_scheduler.reserved_seeds


def test_a_weak_decoder_short_of_the_signals_evidence_is_named_by_its_record():
    """The refusal names the record the run wrote, not its results word.

    A priced matching row's word is its latency, which names no decoder,
    and matching grows no clusters for the cluster gap to read.
    """
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=0.5
    )
    tier = decoder_settings.DecoderPoolSettings(algorithm=matching)
    signal_settings = cluster.ClusterGap.Settings()
    signal = signal_settings.build(matching, 4.6)

    refusal = "weak_decoder PyMatchingDecoder.Settings cannot serve"
    with pytest.raises(ValueError, match=refusal):
        decoder_build.build_decoder_unit(
            tier, "weak_decoder", None, None, signal
        )


class _FormedBeforeEveryPool:
    """A placement that forms each round before any pool reads it."""

    def forms_at(self, seat: str) -> bool:
        del seat
        return False


def test_bulk_strong_merges_the_queue_of_every_pool_an_edge_leads_to():
    """The entry pool decodes the planner's windows, one job each."""
    chain = layout_tests.chain_machine()
    settings = graph_layout.entry_settings(chain)
    layout = layout_tests.laid_out(chain)
    engine = engine_module.Engine()
    parts = escalation_build.build_switching(layout, engine)
    signals = escalation_build.signal_by_pool(parts)
    placement = _FormedBeforeEveryPool()
    pools = decoder_build.build_pool_units(
        layout, settings.clock, placement, signals
    )
    bulk = decoder_settings.DecoderManagerSettings(bulk_strong=True)
    targets = layout.target_by_source()

    decoders = decoder_build.Decoders.build(
        bulk, settings.clock, engine, pools, targets
    )

    merges = [manager.queue.merges_strong for manager in decoders.managers]
    assert merges == [False, True, True]

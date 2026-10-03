"""Building the two tiers' units and the pool each manager schedules.

The chip's manager has the pool of the tier that decodes the plan's
windows and a switching run gives the host's manager a pool of the
strong tier. The regression lock (tests/regression) runs every other
wiring of this module. No lock shot bounds the strong tier's memory,
and none can tell the machine's clock from the fridge's or the room's,
since both tick at 250 MHz.
"""

import dataclasses

import decsim.build.decoders as decoder_build
import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.machine as machine_module
import tests.declared_run as declared_run
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
    strong_unit = machine.decoders.strong_decoder_manager.pool.units[0]
    assert weak_unit.memory.capacity_bits == 12
    assert strong_unit.memory.capacity_bits == 30


def test_an_engine_that_names_no_clock_counts_on_the_machines():
    """No preset clock ticks at 300 MHz, so a unit on one fails here."""
    machine_clock = config.Clock.from_megahertz(300.0)
    engine = decoder_settings.EngineSettings()
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings()
    weak = decoder_settings.DecoderPoolSettings(
        algorithm=matching, engine=engine
    )

    unit = decoder_build.build_decoder_unit(
        weak, "weak", machine_clock, None, None
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

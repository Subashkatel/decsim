"""Building the two tiers' units and the pool each manager schedules.

The chip's manager has the pool of the tier that decodes the plan's
windows and a switching run gives the host's manager a pool of the
strong tier. The regression lock (tests/regression) runs every other
wiring of this module; no lock shot bounds the strong tier's memory.
"""

import dataclasses

import decsim.decoders.settings as decoder_settings
import decsim.machine as machine_module
import tests.declared_run as declared_run


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

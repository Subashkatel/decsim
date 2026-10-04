"""The qpu part's factory: each settings record builds its own factory.

Each factory is a decision the machine makes before a round is packed,
and its card is checked by its record rather than inside the running
machine, which is where a mistake belongs (STYLE.md rule 4).
"""

import pytest

import decsim.engine as engine_module
import decsim.machine as machine_module
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.settings as machine_settings

_INFINITE = magic_state_factories.InfiniteFactory
_DISTILLATION = magic_state_factories.DistillationFactory
_MULTI_LEVEL = magic_state_factories.MultiLevelDistillationFactory


_ONE_LEVEL = magic_state_factories.DistillLevel(unit_count=1, distance=3)
_NO_DECODE_CARD = _DISTILLATION.Settings(
    unit_count=1,
    attempt_ticks=1000,
    correction_round_count=1,
    correction_decode_count=0,
)
_ELEVEN_DECODE_CARD = _DISTILLATION.Settings(
    unit_count=1, attempt_ticks=1000, correction_round_count=1
)
_ONE_LEVEL_CARD = _MULTI_LEVEL.Settings(levels=(_ONE_LEVEL,))
_MACHINE_BUILT_FACTORIES = (
    _INFINITE.Settings(),
    _NO_DECODE_CARD,
    _ELEVEN_DECODE_CARD,
    _ONE_LEVEL_CARD,
)


@pytest.mark.parametrize("factory_settings", _MACHINE_BUILT_FACTORIES)
def test_every_factory_builds_and_takes_the_decode_queue_it_is_handed(
    factory_settings,
):
    """The qpu part hands every factory the run's decoder manager.

    A factory whose card asks for no correction decode runs beside it.
    What a factory sends through the port is
    tests/qpu/test_magic_state_factories.py's.
    """
    settings = machine_settings.MachineSettings(
        magic_state_factory=factory_settings
    )

    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()

    factory = machine.qpu.factory
    assert factory.decode_queue is machine.decoders.decoder_manager
    assert result.terminal_status == "complete"


def test_the_level_chain_is_built_on_the_runs_round():
    engine = engine_module.Engine()
    card = _MULTI_LEVEL.Settings(levels=(_ONE_LEVEL,))

    factory = card.build(engine, 1234)

    assert factory.card is card
    assert factory.preparation_ticks == 2 * 3 * 1234

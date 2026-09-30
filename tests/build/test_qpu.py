"""The qpu part's factory: one row of the table, built by one call.

Each row is a decision the machine makes before a round is packed, and
each is checked here rather than inside the running machine, which is
where a yaml's mistake belongs (STYLE.md rule 4).
"""

import pytest

import decsim.build.qpu as qpu_part
import decsim.engine as engine_module
import decsim.machine as machine_module
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.qpu.settings as qpu_settings
import decsim.settings as machine_settings

_DISTILLATION = magic_state_factories.DistillationFactory
_MULTI_LEVEL = magic_state_factories.MultiLevelDistillationFactory


def test_two_rows_with_different_cards_are_built_by_the_one_call():
    """One constructor signature, so a row's own keys ride in its Settings."""
    engine = engine_module.Engine()
    no_card = qpu_settings.FactorySettings(kind="infinite")
    card = _DISTILLATION.Settings(
        unit_count=1,
        attempt_ticks=1000,
        correction_round_count=1,
        correction_decode_count=0,
    )
    with_a_card = qpu_settings.FactorySettings(
        kind="distillation", row_settings=card
    )

    always_in_stock = qpu_part.build_factory(no_card, engine, 1000)
    fifteen_to_one = qpu_part.build_factory(with_a_card, engine, 1000)

    assert isinstance(always_in_stock, magic_state_factories.InfiniteFactory)
    assert isinstance(fifteen_to_one, magic_state_factories.DistillationFactory)


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
    qpu_settings.FactorySettings(kind="infinite"),
    qpu_settings.FactorySettings(
        kind="distillation", row_settings=_NO_DECODE_CARD
    ),
    qpu_settings.FactorySettings(
        kind="distillation", row_settings=_ELEVEN_DECODE_CARD
    ),
    qpu_settings.FactorySettings(
        kind="multi_level", row_settings=_ONE_LEVEL_CARD
    ),
)


@pytest.mark.parametrize("factory_settings", _MACHINE_BUILT_FACTORIES)
def test_every_factory_row_builds_and_takes_the_decode_queue_it_is_handed(
    factory_settings,
):
    """A row reads the collaborators it needs and ignores the rest.

    The qpu part hands every row the run's decoder manager as its decode
    queue, so a row whose card asks for no correction decode runs beside
    it.
    """
    settings = machine_settings.MachineSettings(
        magic_state_factory=factory_settings
    )

    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()

    assert result.terminal_status == "complete"


def test_a_factory_kind_that_names_no_row_is_refused():
    engine = engine_module.Engine()
    settings = qpu_settings.FactorySettings(kind="teleported")

    with pytest.raises(ValueError) as refusal:
        qpu_part.build_factory(settings, engine, 1000)

    assert "magic_state_factory.kind" in str(refusal.value)


def test_the_collaborators_record_carries_the_runs_round_and_the_rows_keys():
    engine = engine_module.Engine()
    card = _MULTI_LEVEL.Settings(levels=(_ONE_LEVEL,))
    settings = qpu_settings.FactorySettings(
        kind="multi_level", row_settings=card
    )

    factory = qpu_part.build_factory(settings, engine, 1234)

    assert factory.card is card
    assert factory.preparation_ticks == 2 * 3 * 1234

"""The plan's cadence: where a round period comes from, and its floor.

decsim/frontends/planner.py resolves the round period once, before the
engine starts: the code card's own period if it declares one, else the
run's round_period_microseconds. A card declares a period when its
hardware fixes one (a superconducting surface-code round of about a
microsecond, Google 2207.06431 Fig. 1); a card that declares none
(SurfaceCodeModel by default) leaves the period to the run, which is
what the p-versus-d sweeps vary. The resolved period is then a whole
number of ticks, and a period under one tick is refused, because a
zero-tick cadence never advances the clock.
"""

import pytest

import decsim.machine as machine_module
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.settings as qpu_settings


def test_the_run_period_is_used_when_the_card_declares_none():
    qpu = qpu_settings.QpuSettings(round_period_microseconds=0.75)
    settings = machine_module.MachineSettings(qpu=qpu)
    machine = machine_module.Machine.build(settings)
    assert machine.qpu.cycle_ticks == 750_000


def test_the_card_period_wins_over_the_run_period():
    card = code_geometry.SurfaceCodeModel(round_microseconds=2.0)
    qpu = qpu_settings.QpuSettings(round_period_microseconds=1.25, code=card)
    settings = machine_module.MachineSettings(qpu=qpu)
    machine = machine_module.Machine.build(settings)
    assert machine.qpu.cycle_ticks == 2_000_000


def test_a_period_shorter_than_one_tick_is_refused():
    qpu = qpu_settings.QpuSettings(round_period_microseconds=0.0)
    settings = machine_module.MachineSettings(qpu=qpu)
    with pytest.raises(
        ValueError, match="resolved round cadence must be at least one tick"
    ):
        machine_module.Machine.build(settings)


def test_a_card_period_saves_a_run_period_shorter_than_one_tick():
    card = code_geometry.SurfaceCodeModel(round_microseconds=2.0)
    qpu = qpu_settings.QpuSettings(round_period_microseconds=0.0, code=card)
    settings = machine_module.MachineSettings(qpu=qpu)
    machine = machine_module.Machine.build(settings)
    assert machine.qpu.cycle_ticks == 2_000_000

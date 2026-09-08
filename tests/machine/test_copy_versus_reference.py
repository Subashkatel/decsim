"""The copy-versus-reference settings: each default is today's behaviour.

Design audit note 14 section 8.2 (recommendation R4-12): every hop of
this tree copies today, and each of these keys says whether one hop may
be a reference instead, so the two can be measured against each other.
The sources are classical and quantum both: AFS's on-chip access
(2001.06598 lines 528-531), Collision Clustering's Init unit
(2309.05558 lines 268-271), Toshio's transfer of the assigned data
(2510.25222 lines 1248-1250), Helios's single-writer shared memory
(2301.08419 lines 632-640), Chen's non-blocking frame manager
(2605.30765 lines 1618-1620), Riverlane's polled status register
(2410.05202 lines 1256-1259) and Google's detections formed at the
workstation (2408.13687 lines 474-476).
"""

import dataclasses
import pathlib

import pytest

import decsim.front.experiment as experiment
import decsim.machine as machine_module
import decsim.records.transfers as transfer_records

CONFIGS = pathlib.Path("configs")
WEAK_INPUT_PATH = transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER


def _machine(**weak_changes):
    """The weak baseline at d=3, counting its data movement."""
    config_path = CONFIGS / "weak_decoder_baseline.yaml"
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.001, distance=3, round_period_us=1.0
    )
    weak = dataclasses.replace(settings.weak_decoder, **weak_changes)
    observation = dataclasses.replace(settings.observation, data_movement=True)
    settings = dataclasses.replace(
        settings, weak_decoder=weak, observation=observation
    )
    return machine_module.Machine.build(settings, 0)


def _unit_input_copies(machine) -> int:
    """The copies into a decoder unit's own memory."""
    copies = 0
    for (
        path,
        counts,
    ) in machine.observation.data_movement.copies.by_path.items():
        if "unit" in path:
            copies += counts.events
    return copies


def _weak_input_transfers(machine) -> int:
    snapshot = machine.observation.traffic.snapshot()
    for path_snapshot in snapshot.paths:
        if path_snapshot.path is WEAK_INPUT_PATH:
            return path_snapshot.counters.transfer_count
    return 0


def _observables(result) -> list:
    """Each operation's committed logical observables, in order."""
    observables = []
    for row in result.operation_results:
        observables.append((row.operation_id, row.logical_observables))
    return observables


def test_the_input_default_copies_the_rounds_into_the_unit():
    """Copy is today's behaviour: one copy and one link move per window."""
    machine = _machine()
    machine.run()
    copies = _unit_input_copies(machine)
    transfers = _weak_input_transfers(machine)
    windows = len(machine.observation.windows.windows)
    assert copies == windows
    assert transfers == windows


def test_an_in_place_input_references_the_rounds_and_moves_nothing():
    """In place: no deposit, no link move, the store's hold kept instead."""
    copied = _machine()
    copied_result = copied.run()
    in_place = _machine(input="in_place")
    in_place_result = in_place.run()
    assert _unit_input_copies(in_place) == 0
    assert _weak_input_transfers(in_place) == 0
    copied_references = copied.observation.data_movement.references.events
    in_place_references = in_place.observation.data_movement.references.events
    windows = len(in_place.observation.windows.windows)
    assert in_place_references == copied_references + windows
    copied_observables = _observables(copied_result)
    in_place_observables = _observables(in_place_result)
    assert in_place_observables == copied_observables


def test_an_input_kind_that_is_not_a_row_is_refused_by_name():
    with pytest.raises(ValueError, match="weak_decoder.input 'in-place'"):
        _machine(input="in-place")

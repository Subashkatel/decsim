"""The d = 5 memory the burst detector tests feed, round by round.

Stim's surface_code:rotated_memory_z at p = 1e-3, the circuit decsim's
memory_circuit generates, its rounds in formation order, and the two
rows built on it.
"""

from typing import Optional

import numpy
import stim

import decsim.burst_detectors.event_count.detector as event_count
import decsim.config as config
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.detector_error_model.fault_model_contracts as fault_contracts
import decsim.detector_error_model.window_slicer as window_slicer
import decsim.engine as engine_module
import decsim.frontends.settings as workload_settings
import decsim.ports as ports
import decsim.qpu.stim_device as stim_device
import decsim.records.windows as window_records
from decsim.burst_detectors.masked_regional_cusum import (
    detector as masked_regional_cusum,
)

ROUNDS = 30
DISTANCE = 5
PHYSICAL_ERROR_PROBABILITY = 1e-3
CODE_TASK = "surface_code:rotated_memory_z"
# d = 5 rotated_memory_z: round one forms the 12 Z detectors against the
# prepared state, every later round 24 bulk detectors
FIRST_ROUND_QUIET = (0,) * 12
BULK_ROUND_QUIET = (0,) * 24
BULK_ROUND_LOUD = (1,) * 24
CLOCKS = config.ClockSettings({"fridge": 250.0})
GRAPHLIKE = fault_contracts.FaultRepresentation.GRAPHLIKE
CUSUM = masked_regional_cusum.MaskedRegionalCusumBurstDetector
# a stream long enough for the 64-round mask window and its 100-round
# hold, with a burst loud enough to mask checks
LONG_ROUNDS = 160
LONG_BURST_ONSET = 40


def memory_circuit() -> stim.Circuit:
    return workload_settings.memory_circuit(
        CODE_TASK, ROUNDS, DISTANCE, PHYSICAL_ERROR_PROBABILITY
    )


def event_count_detector(
    settings: event_count.EventCountBurstDetector.Settings,
    engine: Optional[engine_module.Engine] = None,
) -> event_count.EventCountBurstDetector:
    if engine is None:
        engine = engine_module.Engine()
    circuit = memory_circuit()
    circuits = {1: (circuit, ROUNDS)}
    return event_count.EventCountBurstDetector(settings, engine, circuits, 1.0)


def cusum_detector(
    settings: masked_regional_cusum.MaskedRegionalCusumBurstDetector.Settings,
    rounds: int = ROUNDS,
    engine: Optional[engine_module.Engine] = None,
) -> masked_regional_cusum.MaskedRegionalCusumBurstDetector:
    if engine is None:
        engine = engine_module.Engine()
    circuit = workload_settings.memory_circuit(
        CODE_TASK, rounds, DISTANCE, PHYSICAL_ERROR_PROBABILITY
    )
    circuits = {1: (circuit, rounds)}
    return CUSUM(settings, engine, circuits, 1.0)


def formation_table() -> detector_formation.FormationTable:
    circuit = memory_circuit()
    return detector_formation.build_formation_table(circuit, ROUNDS)


def rounds_of_events(
    circuit: stim.Circuit, round_count: int, sampled_events: numpy.ndarray
) -> list:
    """One shot's detection events cut into rounds, in formation order."""
    table = detector_formation.build_formation_table(circuit, round_count)
    rounds = []
    after_last_round = round_count + 1
    for round_index in range(1, after_last_round):
        recipes = table.detectors_of_round(round_index)
        values = []
        for recipe in recipes:
            value = sampled_events[recipe.detector_index]
            values.append(value)
        rounds.append(values)
    return rounds


def sampled_rounds(circuit: stim.Circuit, seed: int) -> list:
    sampler = circuit.compile_detector_sampler(seed=seed)
    shots = sampler.sample(1)
    base_circuit = memory_circuit()
    return rounds_of_events(base_circuit, ROUNDS, shots[0])


def whole_patch_burst(onset_round: int, probability: float) -> stim.Circuit:
    circuit = memory_circuit()
    table = formation_table()
    burst = stim_device.BurstStimDevice.Settings(
        burst_onset_round=onset_round,
        burst_error_probability=probability,
    )
    return stim_device.burst_circuit(circuit, table, burst)


def feed(detector: ports.BurstDetector, rounds: list) -> None:
    for index, events in enumerate(rounds):
        round_index = index + 1
        detector.observe_round(1, round_index, events)


def quiet_rounds(round_count: int) -> list:
    bulk_round_count = round_count - 1
    bulk_rounds = [BULK_ROUND_QUIET] * bulk_round_count
    return [FIRST_ROUND_QUIET, *bulk_rounds]


def window(first_round: int, last_round: int) -> window_records.Window:
    round_span = last_round - first_round
    round_count = round_span + 1
    return window_records.Window(
        operation_id=1,
        window_index=0,
        commit_lo=first_round,
        commit_hi=last_round,
        buffer_hi=last_round,
        round_count=round_count,
    )


def window_model(
    first_round: int, last_round: int
) -> fault_contracts.WindowErrorModel:
    """The window's model as the planner slices it: rows and columns."""
    circuit = memory_circuit()
    slicer = window_slicer.WindowSlicer(
        circuit,
        round_count=ROUNDS,
        fault_model_requirement=fault_contracts.GRAPHLIKE_FAULT_MODEL_REQUIRED,
    )
    return slicer.slice_window(
        first_round, first_round, last_round, last_round, is_last=False
    )


def long_burst_shot(seed: int) -> tuple:
    """A d = 5 shot of LONG_ROUNDS rounds, bursting from LONG_BURST_ONSET."""
    circuit = workload_settings.memory_circuit(
        CODE_TASK, LONG_ROUNDS, DISTANCE, PHYSICAL_ERROR_PROBABILITY
    )
    table = detector_formation.build_formation_table(circuit, LONG_ROUNDS)
    burst = stim_device.BurstStimDevice.Settings(
        burst_onset_round=LONG_BURST_ONSET, burst_error_probability=0.05
    )
    shot_circuit = stim_device.burst_circuit(circuit, table, burst)
    sampler = shot_circuit.compile_detector_sampler(seed=seed)
    shots = sampler.sample(1)
    return circuit, shots[0]


def bulk_rows(
    circuit: stim.Circuit, sampled: numpy.ndarray, positions: tuple
) -> numpy.ndarray:
    """(bulk rounds, checks) of 0/1, read off Stim's detector coordinates.

    Stim's time coordinate 0 is the first round and its last is the
    data readout, so the bulk rows are the times between.
    """
    coordinates = circuit.get_detector_coordinates()
    column_by_position = {}
    for column, position in enumerate(positions):
        column_by_position[position] = column
    times = [coordinates[detector][2] for detector in coordinates]
    readout_time = int(max(times))
    shape = (readout_time - 1, len(positions))
    rows = numpy.zeros(shape, dtype=int)
    for detector, coordinate in coordinates.items():
        time = int(coordinate[2])
        if 1 <= time < readout_time:
            position = (coordinate[0], coordinate[1])
            column = column_by_position[position]
            rows[time - 1, column] = sampled[detector]
    return rows


def bank_inputs(
    detector: masked_regional_cusum.MaskedRegionalCusumBurstDetector,
) -> tuple:
    """The bank's positions, pairs and floored usual rates."""
    calibration = detector.charts_by_operation[1].calibration
    bank = calibration.bank
    positions = calibration.layout.positions
    pair_rows = bank.pair_incidence
    pairs = []
    for row in pair_rows:
        checks = numpy.flatnonzero(row)
        pairs.append(tuple(checks))
    return positions, pairs, bank.usual_rates


def flagged_event_count_detector(
    raise_strong_priors: bool,
) -> event_count.EventCountBurstDetector:
    """Every detector loud from round 12 to 17: the whole patch in burst."""
    settings = event_count.EventCountBurstDetector.Settings(
        raise_strong_priors=raise_strong_priors
    )
    detector = event_count_detector(settings)
    quiet_before = quiet_rounds(11)
    loud = [BULK_ROUND_LOUD] * 6
    rounds = [*quiet_before, *loud]
    feed(detector, rounds)
    return detector


def flagged_cusum_detector(
    raise_strong_priors: bool,
) -> masked_regional_cusum.MaskedRegionalCusumBurstDetector:
    """Every check loud from round 12 to 17: the whole patch in burst."""
    settings = CUSUM.Settings(raise_strong_priors=raise_strong_priors)
    detector = cusum_detector(settings)
    quiet_before = quiet_rounds(11)
    loud = [BULK_ROUND_LOUD] * 6
    rounds = [*quiet_before, *loud]
    feed(detector, rounds)
    return detector

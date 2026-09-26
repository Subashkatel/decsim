"""The d = 5 memory the burst detector tests feed, round by round.

Stim's surface_code:rotated_memory_z at p = 1e-3, the circuit decsim's
memory_circuit generates, its rounds in formation order, and the two
rows built on it.
"""

import decsim.burst_detectors.event_count.detector as event_count
import decsim.config as config
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.detector_error_model.fault_model_contracts as fault_contracts
import decsim.detector_error_model.window_slicer as window_slicer
import decsim.engine as engine_module
import decsim.frontends.settings as workload_settings
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


def memory_circuit():
    return workload_settings.memory_circuit(
        CODE_TASK, ROUNDS, DISTANCE, PHYSICAL_ERROR_PROBABILITY
    )


def event_count_detector(settings, engine=None):
    if engine is None:
        engine = engine_module.Engine()
    circuit = memory_circuit()
    circuits = {1: (circuit, ROUNDS)}
    return event_count.EventCountBurstDetector(settings, engine, circuits, 1.0)


def cusum_detector(settings, rounds=ROUNDS, engine=None):
    if engine is None:
        engine = engine_module.Engine()
    circuit = workload_settings.memory_circuit(
        CODE_TASK, rounds, DISTANCE, PHYSICAL_ERROR_PROBABILITY
    )
    circuits = {1: (circuit, rounds)}
    return CUSUM(settings, engine, circuits, 1.0)


def formation_table():
    circuit = memory_circuit()
    return detector_formation.build_formation_table(circuit, ROUNDS)


def rounds_of_events(circuit, round_count, sampled_events):
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


def sampled_rounds(circuit, seed):
    sampler = circuit.compile_detector_sampler(seed=seed)
    shots = sampler.sample(1)
    base_circuit = memory_circuit()
    return rounds_of_events(base_circuit, ROUNDS, shots[0])


def whole_patch_burst(onset_round, probability):
    circuit = memory_circuit()
    table = formation_table()
    burst = stim_device.BurstStimDevice.Settings(
        burst_onset_round=onset_round,
        burst_error_probability=probability,
    )
    return stim_device.burst_circuit(circuit, table, burst)


def feed(detector, rounds):
    for index, events in enumerate(rounds):
        round_index = index + 1
        detector.observe_round(1, round_index, events)


def quiet_rounds(round_count):
    bulk_round_count = round_count - 1
    bulk_rounds = [BULK_ROUND_QUIET] * bulk_round_count
    return [FIRST_ROUND_QUIET, *bulk_rounds]


def window(first_round, last_round):
    round_count = last_round - first_round + 1
    return window_records.Window(
        operation_id=1,
        window_index=0,
        commit_lo=first_round,
        commit_hi=last_round,
        buffer_hi=last_round,
        round_count=round_count,
    )


def window_model(first_round, last_round):
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

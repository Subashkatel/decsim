"""A finite circuit with no declared schedule has it read off its detectors.

Each measurement takes the round its DETECTOR group announces, its time
coordinate plus one, as a tqec lattice-surgery circuit writes them
(tests/data/tqec_cnot_k1.stim); a detector with no coordinate leaves no
round to read, so such a circuit declares its schedule.
"""

import pathlib

import pytest
import stim

import decsim.records.workload as workload_records

THIS_FILE = pathlib.Path(__file__)
DATA_DIRECTORY = THIS_FILE.parents[1] / "data"


def test_a_tqec_circuit_has_the_rounds_its_detectors_announce():
    circuit_path = DATA_DIRECTORY / "tqec_cnot_k1.stim"
    circuit = stim.Circuit.from_file(circuit_path)

    physical = workload_records.FiniteCircuit(circuit)

    assert physical.round_count() == 12


def test_a_circuit_with_an_untimed_detector_and_no_schedule_is_refused():
    circuit = stim.Circuit("M 0\nDETECTOR(0, 0) rec[-1]\nM 0\nDETECTOR rec[-1]")

    with pytest.raises(ValueError, match="2 detectors, 1 with no coordinate"):
        workload_records.FiniteCircuit(circuit)

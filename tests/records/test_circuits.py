"""Repeated Stim fragments keep their declared measurement delivery groups.

Stim's circuit measurement order remains the physical record order. Selecting
an acquisition partition only names contiguous groups within that record.
"""

import pytest
import stim

import decsim.records.circuits as circuit_records
import decsim.records.rounds as round_records


@pytest.mark.parametrize(
    "round_index, is_final, fragment_name",
    [
        (1, False, "first_round"),
        (2, False, "repeated_round"),
        (2, True, "final_round"),
        (1, True, "single_round"),
    ],
)
def test_each_physical_fragment_selects_its_measurement_partitions(
    round_index: int, is_final: bool, fragment_name: str
) -> None:
    circuit = stim.Circuit("M 0")
    partition = round_records.MeasurementPartition(("patch",), 1)
    partitions = {fragment_name: (partition,)}
    program = circuit_records.RepeatedStimCircuit(
        circuit, circuit, circuit, circuit, readout_partitions=partitions
    )

    selected = program.partitions_for_round(round_index, is_final)
    physical = program.round_circuit(round_index, is_final)

    assert selected == (partition,)
    assert physical == circuit


def test_partition_declarations_are_copied_from_the_callers_mapping() -> None:
    circuit = stim.Circuit("M 0")
    partition = round_records.MeasurementPartition(("patch",), 1)
    groups = [partition]
    partitions = {"first_round": groups}
    program = circuit_records.RepeatedStimCircuit(
        circuit, circuit, circuit, circuit, readout_partitions=partitions
    )
    groups.clear()
    partitions.clear()

    selected = program.partitions_for_round(1, False)
    unspecified = program.partitions_for_round(2, False)

    assert selected == (partition,)
    assert unspecified == ()


def test_an_unknown_partition_fragment_name_is_refused() -> None:
    circuit = stim.Circuit("M 0")
    with pytest.raises(ValueError, match="unknown round fragment"):
        circuit_records.RepeatedStimCircuit(
            circuit, circuit, circuit, circuit, readout_partitions={"typo": ()}
        )

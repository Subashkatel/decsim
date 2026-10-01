"""Repeated Stim fragments: each requested round runs its declared fragment.

A one-round run is the single fragment, a later final round the final
fragment, and every round between the first and the final the repeated one
(decsim/records/circuits.py, RepeatedStimCircuit).
"""

import pytest
import stim

import decsim.records.circuits as circuit_records


@pytest.mark.parametrize(
    "round_index, is_final, fragment_text",
    [
        (1, False, "M 0"),
        (2, False, "M 1"),
        (2, True, "M 2"),
        (1, True, "M 3"),
    ],
)
def test_each_round_runs_its_own_physical_fragment(
    round_index: int, is_final: bool, fragment_text: str
) -> None:
    first_round = stim.Circuit("M 0")
    repeated_round = stim.Circuit("M 1")
    final_round = stim.Circuit("M 2")
    single_round = stim.Circuit("M 3")
    program = circuit_records.RepeatedStimCircuit(
        first_round, repeated_round, final_round, single_round
    )
    expected = stim.Circuit(fragment_text)

    physical = program.round_circuit(round_index, is_final)

    assert physical == expected

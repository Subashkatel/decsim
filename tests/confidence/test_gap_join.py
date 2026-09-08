"""The window's gap as two forced-class jobs, on one unit and on two.

Design audit note 12 sections 4.1 to 4.4: the window side submits both
forced-class jobs at one instant, both queue in the weak pool, each unit
is fed from Buffer 0, and weak_decoder.units alone decides whether the
pair overlaps. The rule the data-movement study rests on is one copy per
unit that reads the window: one unit moves the window's bits once and is
busy for the sum of the two solves, two units move them twice and
overlap, and the committed corrections are the same either way.
"""

import copy
import dataclasses
import pathlib

import decsim.machine as machine_module
import decsim.records.decoding as decoding_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import tests.escalation.declared_fabric as fabric
from tests.escalation.test_strong_window_shapes import GATE_SWITCHING_CARD

WEAK_INPUT_PATH = transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER


def _switching_machine(weak_units: int, weak_microseconds: float = 4.0):
    """The gate's switching card at d=3, priced so its ticks are declared."""
    sections = copy.deepcopy(GATE_SWITCHING_CARD)
    sections["weak_decoder"]["kind"] = weak_microseconds
    sections["weak_decoder"]["units"] = weak_units
    sections["strong_decoder"]["kind"] = 20.0
    base_directory = pathlib.Path(".")
    settings = machine_module.MachineSettings.from_mapping(
        sections, name="switching_validation", base_directory=base_directory
    )
    qpu = dataclasses.replace(
        settings.qpu, distance=3, round_period_microseconds=1.0
    )
    workload = dataclasses.replace(
        settings.workload, physical_error_probability=0.008
    )
    observation = dataclasses.replace(
        settings.observation, record_switching_windows=True
    )
    settings = dataclasses.replace(
        settings, qpu=qpu, workload=workload, observation=observation
    )
    return machine_module.Machine.build(settings, 0)


def _weak_requests(machine) -> list:
    weak = window_records.DecoderTier.WEAK
    records = []
    for record in machine.observation.decode_records.requests:
        if record.request_key.tier is weak:
            records.append(record)
    return records


def _weak_input_traffic(machine) -> tuple:
    """(transfers, bits) that moved into the weak units."""
    snapshot = machine.observation.traffic.snapshot()
    for path_snapshot in snapshot.paths:
        if path_snapshot.path is WEAK_INPUT_PATH:
            counters = path_snapshot.counters
            return counters.transfer_count, counters.known_payload_bits
    raise AssertionError("the weak input path is not on the fabric")


def _committed_observables(machine) -> list:
    committed = []
    snapshot = machine.pauli_frame.snapshot()
    for record in snapshot.records:
        committed.append((record.window_key, record.logical_observables))
    return committed


def test_one_unit_decodes_each_window_twice_and_moves_its_bits_once():
    """Two requests per window, one transfer per window into the unit."""
    machine = _switching_machine(1)
    machine.run()
    windows = len(machine.observation.windows.windows)
    weak_requests = _weak_requests(machine)
    assert len(weak_requests) == 2 * windows
    transfers, _bits = _weak_input_traffic(machine)
    assert transfers == windows


def test_two_units_move_the_bits_twice_and_commit_the_same_corrections():
    """The second unit buys the fast solve and costs a second copy."""
    one_unit = _switching_machine(1)
    one_unit.run()
    two_units = _switching_machine(2)
    two_units.run()
    one_transfers, one_bits = _weak_input_traffic(one_unit)
    two_transfers, two_bits = _weak_input_traffic(two_units)
    assert two_transfers == 2 * one_transfers
    assert two_bits == 2 * one_bits
    assert _committed_observables(two_units) == _committed_observables(one_unit)


def test_one_unit_holds_the_window_for_the_sum_of_its_two_solves():
    """A priced weak card prices one decode; the pair is two of them."""
    machine = _switching_machine(1, weak_microseconds=4.0)
    machine.run()
    starts = fabric.log_lines_containing(machine, "START DECODE mem")
    weak_starts = []
    for line in starts:
        if "strong(" not in line:
            weak_starts.append(line)
    windows = len(machine.observation.windows.windows)
    assert len(weak_starts) == 2 * windows


def test_the_first_solve_is_held_and_the_join_names_the_windows_gap():
    """The held half and the join are named in the run's log."""
    machine = _switching_machine(1)
    machine.run()
    held = fabric.log_lines_containing(machine, "GAP HOLD")
    joined = fabric.log_lines_containing(machine, "GAP JOIN")
    windows = len(machine.observation.windows.windows)
    assert len(held) == windows
    assert len(joined) == windows
    assert "waits for the other class" in held[0]
    assert "gap " in joined[0]


def test_a_windows_two_requests_are_one_attempt_in_the_ledger():
    """One attempt, two forced-class requests, one of them the answer."""
    machine = _switching_machine(1)
    machine.run()
    outcomes = decoding_records.RequestProcessingOutcome
    companion = outcomes.WEAK_FORCED_CLASS_COMPANION
    companions = []
    answered = []
    for record in _weak_requests(machine):
        if record.terminal_processing_outcome is companion:
            companions.append(record)
            continue
        if record.soft_output is not None:
            answered.append(record)
    windows = len(machine.observation.windows.windows)
    assert len(companions) == windows
    assert len(answered) == windows

"""A maker's workload on disk, read back as it was written.

decsim.ops/1 carries every Operation field a maker sets, and the physical
circuit is a finite .stim with its round json or the four live fragments
with physical.json; a field or a readout group the files cannot hold is
refused, since a run folder written without it would rerun otherwise.
"""

import dataclasses

import pytest
import stim

import decsim.frontends.workload_files as workload_files
import decsim.records.circuits as circuit_records
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.workload as workload_records


def _live_workload() -> workload_records.Workload:
    """A prefix that decodes, and a readout that waits for its answer."""
    prefix = program_records.Operation(
        1, "prefix", ("p",), patches=("p",), stream_id=100, stream_offset=0
    )
    readout = program_records.Operation(
        2,
        "readout",
        ("p",),
        patches=("p",),
        blocked_by=1,
        emits_detector_data=False,
    )
    fragment = stim.Circuit("M 0")
    program = circuit_records.RepeatedStimCircuit(
        fragment, fragment, fragment, fragment, round_period_microseconds=1.1
    )
    rounds = {1: 3, 2: 0}
    return workload_records.Workload((prefix, readout), rounds, program)


def test_a_finite_workload_reads_back_what_was_written(tmp_path):
    memory = program_records.Operation(
        1, "memory", (0,), patches=(0,), kind=program_records.OpKind.MEMORY
    )
    readout = program_records.Operation(
        2,
        "readout",
        (0,),
        patches=(0,),
        emits_detector_data=False,
        scheduled_start_round=2,
    )
    circuit = stim.Circuit("M 0\nM 0\nDETECTOR rec[-1] rec[-2]")
    physical = workload_records.FiniteCircuit(circuit, {0: 1, 1: 2})
    workload = workload_records.Workload((memory, readout), {1: 2}, physical)

    keys = workload_files.write_workload(workload, tmp_path)
    operations_path = tmp_path / keys["operations"]
    circuit_path = tmp_path / keys["circuit"]
    rounds_path = tmp_path / keys["measurement_rounds"]
    read = workload_files.read_workload(
        operations_path, circuit_path, rounds_path
    )

    assert read == workload


def test_live_fragments_read_back_what_was_written(tmp_path):
    workload = _live_workload()

    keys = workload_files.write_workload(workload, tmp_path)
    operations_path = tmp_path / keys["operations"]
    fragments_path = tmp_path / keys["fragments"]
    read = workload_files.read_workload(
        operations_path, fragments_path=fragments_path
    )

    assert read == workload


def test_an_operation_field_the_file_form_does_not_carry_is_refused(tmp_path):
    """Writing it would lose it, so a run folder could not rerun the run."""
    staged = program_records.Operation(
        1, "memory", (0,), finalizes_stream_round=True
    )
    workload = workload_records.Workload((staged,))

    with pytest.raises(ValueError, match=r"sets \['finalizes_stream_round'\]"):
        workload_files.write_workload(workload, tmp_path)


def test_live_fragments_with_readout_groups_are_refused(tmp_path):
    """physical.json carries no groups, so a rerun would read out otherwise."""
    workload = _live_workload()
    partition = round_records.MeasurementPartition(("p",), 1)
    program = dataclasses.replace(
        workload.physical, readout_partitions={"repeated_round": (partition,)}
    )
    workload = dataclasses.replace(workload, physical=program)

    with pytest.raises(ValueError, match="name readout_partitions"):
        workload_files.write_workload(workload, tmp_path)

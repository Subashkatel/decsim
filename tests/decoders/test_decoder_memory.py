"""The input memory of one decoder unit: what it holds, and in what order.

A hardware decoder holds the syndrome of the window it is decoding in
its own registers: AFS's syndrome hold registers and STM (Das et al.
2001.06598, lines 619-621, 836-839) and Collision Clustering's input
syndrome registers beside its SRAM tables (Barber et al. 2309.05558,
lines 460-462). decsim prices that as a per-unit memory with a capacity
in rounds (the yaml's unit_memory_rounds), taken when a job's rounds
land and freed when the outcome leaves.

The rounds are ordered here, once, so a decoder reads them in the order
the detector rows were formed in: rounds ascending by operation and
round index, and within one round the fragments in arrival order, which
is the order the QPU stamped their slots in.
"""

import pytest

import decsim.decoders.decoder_memory as decoder_memory
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.rounds as round_records


def fragment(operation_id, round_index, fragment_index, bits=(0, 1)):
    return round_records.RetainedSyndromeFragment(
        operation_id=operation_id,
        patch_id=f"patch-{fragment_index}",
        round_index=round_index,
        bits=bits,
        size_bits=len(bits),
        fragment_index=fragment_index,
    )


def job_of(payloads, label="w0"):
    return decoding_records.DecodeJob(
        op_id=41,
        window_id=7,
        n_rounds=len(payloads),
        payloads=list(payloads),
        label=label,
    )


def timing_only_job(label, round_count):
    payloads = []
    for round_index in range(round_count):
        round_fragment = fragment(1, round_index, 0)
        payloads.append(round_fragment)
    return job_of(payloads, label)


def test_materialization_orders_the_rounds_and_keeps_each_rounds_order():
    first_in_round = fragment(1, 4, 8, bits=(1, 0))
    second_in_round = fragment(1, 4, 2, bits=(0, 1))
    payloads = [
        fragment(2, 3, 0),
        first_in_round,
        fragment(1, 2, 0),
        second_in_round,
        fragment(2, 1, 0),
    ]
    job = job_of(payloads)

    decoder_input = decoder_memory.materialize_decoder_input(job)

    identities = []
    for round_input in decoder_input.rounds:
        identities.append((round_input.operation_id, round_input.round_index))
    assert identities == [(1, 2), (1, 4), (2, 1), (2, 3)]
    assert decoder_input.rounds[1].fragments == (
        first_in_round,
        second_in_round,
    )


def test_a_pool_the_yaml_leaves_out_holds_as_many_rounds_as_it_is_given():
    config = decoder_memory.DecoderMemoryConfig({"default": 6})
    assert config.capacity_for("default") == 6
    assert config.capacity_for("strong") is None


def test_a_deposited_job_occupies_its_rounds_until_it_is_taken():
    memory = decoder_memory.DecoderMemory("default", 0, capacity_rounds=4)
    job = timing_only_job("w0", 3)

    memory.deposit(job)
    assert memory.occupied_rounds == 3
    assert memory.peak_occupied_rounds == 3

    memory.take(job)
    assert memory.occupied_rounds == 0


def test_depositing_one_job_twice_is_refused():
    memory = decoder_memory.DecoderMemory("default", 0, capacity_rounds=4)
    job = timing_only_job("w0", 3)
    memory.deposit(job)
    with pytest.raises(RuntimeError, match="already holds 'w0'"):
        memory.deposit(job)


def test_a_window_wider_than_the_memory_stops_the_run_with_the_numbers():
    memory = decoder_memory.DecoderMemory("default", 1, capacity_rounds=2)
    job = timing_only_job("big", 3)
    with pytest.raises(decoder_memory.DecoderMemoryCapacityError) as caught:
        memory.deposit(job)
    failure = caught.value
    assert failure.pool == "default"
    assert failure.unit == 1
    assert failure.requested_rounds == 3
    assert failure.capacity_rounds == 2
    assert memory.occupied_rounds == 0


def _two_patch_memory_run(rounds_per_unit, unit_count):
    """Two three-round memory operations, one memory of the size given."""
    operations = []
    for operation_id in (1, 2):
        operation = program_records.Operation(
            id=operation_id,
            name=f"op {operation_id}",
            qubits=(operation_id,),
            patches=(operation_id,),
        )
        operations.append(operation)
    rounds_policy = round_policies.FixedRounds(3)
    workload = workload_settings.WorkloadSettings(
        operations=operations, rounds_policy=rounds_policy
    )
    qpu = qpu_settings.QpuSettings(distance=3)
    decoder = decoders.PerRoundDecoder(tau_us=1.0)
    weak_decoder = decoder_settings.DecoderSettings(
        decoder=decoder, units=unit_count
    )
    memory_config = decoder_memory.DecoderMemoryConfig(
        {"default": rounds_per_unit}
    )
    manager = decoder_settings.DecoderManagerSettings(
        decoder_memory=memory_config
    )
    return machine_module.MachineSettings(
        workload=workload,
        qpu=qpu,
        weak_decoder=weak_decoder,
        decoder_manager=manager,
    )


def test_every_unit_has_its_own_memory_and_ends_the_run_empty():
    settings = _two_patch_memory_run(rounds_per_unit=6, unit_count=2)
    machine = machine_module.Machine.build(settings)
    machine.run()
    units = machine.decoder_manager.pool.units()
    names = [unit.name for unit in units]
    occupied = [unit.memory.occupied_rounds for unit in units]
    admissions = [unit.memory.admissions for unit in units]
    assert names == ["default#0", "default#1"]
    assert occupied == [0, 0]
    assert sum(admissions) >= 2


def test_a_unit_too_small_for_its_window_stops_the_run():
    settings = _two_patch_memory_run(rounds_per_unit=1, unit_count=1)
    machine = machine_module.Machine.build(settings)
    with pytest.raises(decoder_memory.DecoderMemoryCapacityError):
        machine.run()

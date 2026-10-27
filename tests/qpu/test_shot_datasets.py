"""A saved dataset holds the error-model source's draws, shot for shot.

The oracle is the live draw itself: Stim's CompiledDemSampler, as
error_model_sampler.ErrorModelSampler.draw calls it, under the seed the
source draws a root seed's shot with.
"""

import numpy
import pytest

import decsim.detector_error_model.error_model_sampler as error_model_sampler
import decsim.producers as producers
import decsim.qpu.shot_datasets as shot_datasets

stim = pytest.importorskip("stim")

CODE_TASK = "surface_code:rotated_memory_z"
DISTANCE = 3
ROUNDS = 9
PROBABILITY = 0.01
# the memory is the workload's one operation, decoded under its own id
DECODE_IDENTITY = 1


def memory_workload():
    """The memory every test draws from: Stim's noise, clean boundaries."""
    return producers.memory_circuit(
        CODE_TASK, ROUNDS, DISTANCE, PROBABILITY, None, True
    )


def memory_circuit():
    workload = memory_workload()
    operation = workload.operations[0]
    return operation.circuit


def saved_dataset(folder, chunks=((0, 4), (4, 4))):
    """A dataset of the memory over chunks of (first root seed, count)."""
    circuit = memory_circuit()
    for first_root_seed, shot_count in chunks:
        shot_datasets.write_chunk(
            folder, circuit, DECODE_IDENTITY, first_root_seed, shot_count
        )
    shot_datasets.seal(folder, circuit, DECODE_IDENTITY, {"test": True})
    return shot_datasets.ShotDataset(folder)


def test_a_saved_shot_is_the_shot_the_error_model_draws_for_its_seed(
    tmp_path,
):
    dataset = saved_dataset(tmp_path)
    circuit = memory_circuit()
    sample_seed = shot_datasets.sample_seed_for(5, DECODE_IDENTITY)
    sampler = error_model_sampler.ErrorModelSampler(circuit)
    drawn_events, drawn_flips, drawn_errors = sampler.draw(sample_seed)
    events, flips, errors = dataset.shot(circuit, sample_seed)
    assert dataset.root_seeds == range(0, 8)
    assert numpy.array_equal(events, drawn_events)
    assert numpy.array_equal(flips, drawn_flips)
    assert numpy.array_equal(errors, drawn_errors)


def test_a_circuit_the_dataset_was_not_drawn_from_is_refused(tmp_path):
    dataset = saved_dataset(tmp_path)
    workload = producers.memory_circuit(
        CODE_TASK, ROUNDS, DISTANCE, 0.02, None, True
    )
    operation = workload.operations[0]
    sample_seed = shot_datasets.sample_seed_for(0, DECODE_IDENTITY)
    with pytest.raises(ValueError, match="drawn from another circuit"):
        dataset.shot(operation.circuit, sample_seed)


def test_a_root_seed_outside_the_dataset_is_refused(tmp_path):
    dataset = saved_dataset(tmp_path)
    circuit = memory_circuit()
    sample_seed = shot_datasets.sample_seed_for(8, DECODE_IDENTITY)
    with pytest.raises(ValueError, match="holds no shot for sample seed"):
        dataset.shot(circuit, sample_seed)


def test_chunks_that_leave_root_seeds_out_are_refused(tmp_path):
    with pytest.raises(ValueError, match="leave out the root seeds from 4"):
        saved_dataset(tmp_path, chunks=((0, 4), (6, 2)))


def test_a_changed_file_fails_verification(tmp_path):
    dataset = saved_dataset(tmp_path)
    flips_path = tmp_path / "chunk_000000000004" / "observable_flips.npy"
    flipped = numpy.ones((4, 1), dtype=numpy.uint8)
    numpy.save(flips_path, flipped)
    with pytest.raises(ValueError, match="not the .* its chunk recorded"):
        dataset.verify()

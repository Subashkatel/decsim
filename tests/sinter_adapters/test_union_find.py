"""The Union-Find sinter adapter against decsim's union_find weak row.

The referent is the row itself on the same Stim samples: the batch
path (decsim/experiments/stim_batch.py) builds the machine's one
whole-circuit window and decodes each shot through the row's Decoder
port. The adapter decodes the model sinter hands a decoder
(sinter/_collection/_collection_worker_state.py:28), so the two agree
shot for shot only if the adapter builds the row's graph.
"""

import math

import numpy
import pytest
import sinter
import stim
import yaml

import decsim.experiments.experiment as experiment
import decsim.experiments.stim_batch as stim_batch
import decsim.sinter_adapters.union_find as union_find_adapter
import tests.experiments.yaml_configs as yaml_configs

BASELINE = (
    yaml_configs.CONFIGS_DIR
    / "experiments"
    / "decoder_baseline"
    / "decoder_baseline.yaml"
)
CODE_TASKS = ["surface_code:rotated_memory_x", "surface_code:rotated_memory_z"]
SHOT_COUNT = 2048
ROUNDS = 5
ERROR_RATE = 0.005
# the baseline's union_find row (decoder_baseline.yaml)
UNION_FIND_ROW = {
    "kind": "union_find",
    "units": 1,
    "unit_memory": {"bits": None},
    "engine": {
        "clock": "fridge",
        "fetch_cycles_per_round": 1,
        "fetch_cycles_per_job": 0,
        "release_cycles_per_job": 10,
        "release_cycles_per_round": 0,
    },
}


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("code_task", CODE_TASKS)
def test_the_adapter_answers_as_the_union_find_row_shot_for_shot(
    tmp_path, distance, code_task
):
    task = union_find_task(tmp_path, distance, code_task)
    circuit = stim_batch.circuit_of(task)
    events, _observables = stim_batch.samples_of_seeds(circuit, 0, SHOT_COUNT)
    window = stim_batch.WholeCircuitWindow(task)
    row = stim_batch.bound_row(task, 0)
    row_answers = row_predictions(window, row, events)
    model = circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True
    )
    decoder = union_find_adapter.UnionFindDecoder()
    compiled = decoder.compile_decoder_for_dem(dem=model)
    packed_events = numpy.packbits(events, axis=1, bitorder="little")

    packed_answers = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed_events
    )

    adapter_answers = numpy.unpackbits(
        packed_answers, axis=1, count=circuit.num_observables, bitorder="little"
    )
    numpy.testing.assert_array_equal(adapter_answers, row_answers)


def test_the_adapter_is_a_sinter_decoder():
    decoder = union_find_adapter.UnionFindDecoder()

    assert isinstance(decoder, sinter.Decoder)


@pytest.mark.parametrize("weight_step", [-1.0, 0.0, math.inf])
def test_a_weight_step_that_is_not_finite_and_positive_is_refused(
    weight_step,
):
    with pytest.raises(ValueError) as refused:
        union_find_adapter.UnionFindDecoder(weight_step=weight_step)

    assert str(refused.value) == (
        "Union-Find weight_step must be finite and positive"
    )


def test_sinter_collects_through_the_adapter():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=ERROR_RATE,
    )
    task = sinter.Task(circuit=circuit, decoder="union-find")
    custom_decoders = {"union-find": union_find_adapter.UnionFindDecoder()}

    (stats,) = sinter.collect(
        num_workers=1,
        tasks=[task],
        max_shots=200,
        custom_decoders=custom_decoders,
    )

    assert stats.shots == 200
    assert stats.errors < 200


def union_find_task(tmp_path, distance: int, code_task: str):
    """The one batch point of a union_find row, short shots."""
    raw = {
        "extends": str(BASELINE),
        "sampling": "stim_batch",
        "workload": {
            "kind": "producer",
            "function": "decsim.producers:memory_circuit",
            "arguments": {
                "code_task": code_task,
                "rounds_per_shot": ROUNDS,
                "distance": "${qpu.distance}",
                "physical_error_probability": ERROR_RATE,
            },
        },
        "sweep": [
            {
                "axes": {
                    "qpu.distance": [distance],
                    "weak_decoder": [UNION_FIND_ROW],
                },
            }
        ],
        "collection": {"max_shots": 1},
    }
    config_path = tmp_path / "union_find.yaml"
    text = yaml.safe_dump(raw, sort_keys=False)
    config_path.write_text(text)
    config = experiment.load_experiment(config_path)
    return config.first_point_task()


def row_predictions(window, row, events) -> numpy.ndarray:
    """Each shot's predicted observables from the row's Decoder port."""
    predictions = []
    for shot_events in events:
        result = window.result_of(row, shot_events)
        predictions.append(result.logical_observables)
    return numpy.asarray(predictions, dtype=numpy.uint8)

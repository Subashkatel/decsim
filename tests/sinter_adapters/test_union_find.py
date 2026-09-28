"""The Union-Find sinter adapter against the machine's union_find weak row.

The referent is the machine itself: each seed's shot runs through the
whole machine (collect.run_shot) under the decoder baseline, whose
naive_online scheme decodes the operation as one window, and the row's
answer is the observables the shot's result carries. The adapter decodes
the events the machine's device drew (observe/sampled_shots.py) with the
model sinter hands a decoder (sinter/_collection/
_collection_worker_state.py:28), so the two agree shot for shot only if
the adapter builds the row's graph.
"""

import math

import numpy
import pytest
import sinter
import stim
import yaml

import decsim.collect as collect
import decsim.experiments.experiment as experiment
import decsim.sinter_adapters.union_find as union_find_adapter
import decsim.windows.built_window_models as built_window_models
import tests.experiments.yaml_configs as yaml_configs

BASELINE = (
    yaml_configs.CONFIGS_DIR
    / "experiments"
    / "decoder_baseline"
    / "decoder_baseline.yaml"
)
CODE_TASKS = ["surface_code:rotated_memory_x", "surface_code:rotated_memory_z"]
# At this rate a shot holds enough defects that the graph's weights
# decide many answers: the adapter on a coarser weight_step (2.0) answers
# 25 to 44 of these 150 shots otherwise in each case.
SHOT_COUNT = 150
ROUNDS = 5
ERROR_RATE = 0.02
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
def test_the_adapter_answers_as_the_machines_union_find_row_shot_for_shot(
    tmp_path, distance, code_task
):
    task = union_find_task(tmp_path, distance, code_task)
    seeds = range(SHOT_COUNT)
    events, machine_answers = machine_shots(task, seeds)
    (operation,) = task.settings.workload.operations
    circuit = operation.circuit
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
    numpy.testing.assert_array_equal(adapter_answers, machine_answers)


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


def test_a_model_with_a_detector_hyperedge_is_refused():
    """An undecomposed model is what sinter hands on when splitting fails."""
    model = stim.DetectorErrorModel("error(0.1) D0 D1 D2 L0")
    decoder = union_find_adapter.UnionFindDecoder()

    with pytest.raises(ValueError) as refused:
        decoder.compile_decoder_for_dem(dem=model)

    assert str(refused.value) == (
        "fault 0 of sinter's detector error model is a detector hyperedge "
        "with detectors (0, 1, 2); this graphlike decoder supports one or "
        "two detectors per fault"
    )


def test_a_logical_error_that_flips_no_detector_is_refused():
    model = stim.DetectorErrorModel("error(0.1) D0 D1\nerror(0.1) L0")
    decoder = union_find_adapter.UnionFindDecoder()

    with pytest.raises(ValueError) as refused:
        decoder.compile_decoder_for_dem(dem=model)

    assert str(refused.value) == (
        "error 1 is a detectorless logical mechanism after instruction-wide "
        "XOR reduction: logical observables (0,)"
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
    """The one machine point of a union_find row, short shots."""
    raw = {
        "extends": str(BASELINE),
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


def machine_shots(task, seeds) -> tuple:
    """Each seed's machine shot: the events drawn, the observables predicted.

    The events are the ones the device drew for the point's one
    operation, and the prediction is the correction the machine
    committed for it.
    """
    built_models = built_window_models.BuiltWindowModels()
    (operation,) = task.settings.workload.operations
    events = []
    answers = []
    for seed in seeds:
        shot = collect.run_shot(task, seed, built_models=built_models)
        sampled_shots = shot.machine.observation.sampled_shots
        sampled = sampled_shots.shots_by_operation[operation.id]
        events.append(sampled.detection_events)
        (operation_result,) = shot.result.operation_results
        answers.append(operation_result.logical_observables)
    event_array = numpy.asarray(events, dtype=numpy.uint8)
    answer_array = numpy.asarray(answers, dtype=numpy.uint8)
    return event_array, answer_array

"""The Relay-BP sinter adapter against decsim's relay_bp weak row.

The referent is the row itself on the same Stim samples: the batch
path (decsim/experiments/stim_batch.py) builds the machine's one
whole-circuit window and decodes each shot through the row's Decoder
port, bound to a run seed at the machine's path. The adapter, given the
gamma seed that binding draws, must answer every shot as the row does.
"""

import numpy
import pytest
import sinter
import stim
import yaml

import decsim.experiments.experiment as experiment
import decsim.experiments.stim_batch as stim_batch
import decsim.records.seeds as seed_records
import decsim.seeding as seeding
import decsim.sinter_adapters.relay_bp as relay_bp_adapter
import tests.experiments.yaml_configs as yaml_configs

BASELINE = (
    yaml_configs.CONFIGS_DIR
    / "experiments"
    / "decoder_baseline"
    / "decoder_baseline.yaml"
)
# the row's keys in the baseline (decoder_baseline.yaml), bases apart
RELAY_SETTINGS = {
    "gamma0": 0.35,
    "gamma_interval": [-0.254, 0.985],
    "pre_iterations": 80,
    "relay_set_count": 300,
    "iterations_per_set": 60,
    "converged_solution_count": 1,
    "bases": "apart",
}
ENGINE = {
    "clock": "fridge",
    "fetch_cycles_per_round": 1,
    "fetch_cycles_per_job": 0,
    "release_cycles_per_job": 10,
    "release_cycles_per_round": 0,
}
ROUNDS = 5
# At this rate the first leg often fails, so the drawn gamma table
# decides some shots (at d 5, 11 of these 300 in X and 4 in Z change
# under another seed), and a wrong table or column order shows.
ERROR_RATE = 0.01
# 300 seeds from 900 cross the sampler's block of 1024 seeds, and
# sinter's batches of 128 shots cross each other
FIRST_SEED = 900
SHOT_COUNT = 300
BATCH_SHOT_COUNT = 128
ROOT_SEED = 7


@pytest.mark.parametrize(
    ("distance", "code_task", "bases"),
    [
        (3, "surface_code:rotated_memory_x", "apart"),
        (3, "surface_code:rotated_memory_z", "apart"),
        (5, "surface_code:rotated_memory_x", "apart"),
        (5, "surface_code:rotated_memory_z", "apart"),
        (3, "surface_code:rotated_memory_z", "together"),
    ],
)
def test_the_adapter_answers_as_the_relay_bp_row_shot_for_shot(
    tmp_path, distance, code_task, bases
):
    settings = {**RELAY_SETTINGS, "bases": bases}
    task = relay_bp_task(tmp_path, distance, code_task, settings)
    circuit = stim_batch.circuit_of(task)
    events, _observables = stim_batch.samples_of_seeds(
        circuit, FIRST_SEED, SHOT_COUNT
    )
    window = stim_batch.WholeCircuitWindow(task)
    row = stim_batch.bound_row(task, ROOT_SEED)
    row_answers = row_predictions(window, row, events)
    seed = gamma_seed(ROOT_SEED)
    decoder = relay_bp_adapter.RelayBeliefPropagationDecoder(
        circuit, settings, seed
    )
    model = circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True
    )
    compiled = decoder.compile_decoder_for_dem(dem=model)

    adapter_answers = batched_predictions(compiled, circuit, events)

    numpy.testing.assert_array_equal(adapter_answers, row_answers)


@pytest.mark.parametrize(
    ("settings", "refusal"),
    [
        (
            {"gamma_interval": [0.9, 0.1]},
            "relay_bp_adapter.gamma_interval must be [low, high], two "
            "finite real numbers with low below high (got [0.9, 0.1])",
        ),
        (
            {"gamma_0": 0.35},
            "relay_bp_adapter does not know ['gamma_0']; its keys are "
            "['alpha', 'alpha_iteration_scaling_factor', 'gamma0', "
            "'pre_iterations', 'relay_set_count', 'iterations_per_set', "
            "'gamma_interval', 'converged_solution_count', 'bases']",
        ),
    ],
)
def test_settings_the_row_does_not_accept_are_refused(settings, refusal):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=3
    )

    with pytest.raises(ValueError) as refused:
        relay_bp_adapter.RelayBeliefPropagationDecoder(circuit, settings, 0)

    assert str(refused.value) == refusal


def test_sinter_collects_through_the_adapter():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=ERROR_RATE,
    )
    task = sinter.Task(circuit=circuit, decoder="relay-bp")
    decoder = relay_bp_adapter.RelayBeliefPropagationDecoder(
        circuit, RELAY_SETTINGS, 0
    )
    custom_decoders = {"relay-bp": decoder}

    (stats,) = sinter.collect(
        num_workers=1,
        tasks=[task],
        max_shots=200,
        custom_decoders=custom_decoders,
    )

    assert stats.shots == 200
    assert stats.errors < 200


def relay_bp_task(tmp_path, distance: int, code_task: str, settings: dict):
    """The one batch point of a relay_bp row, short shots."""
    row = {
        "kind": "relay_bp",
        "units": 1,
        "unit_memory": {"bits": None},
        **settings,
        "engine": ENGINE,
    }
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
                    "weak_decoder": [row],
                },
            }
        ],
        "collection": {"max_shots": 1},
    }
    config_path = tmp_path / "relay_bp.yaml"
    text = yaml.safe_dump(raw, sort_keys=False)
    config_path.write_text(text)
    config = experiment.load_experiment(config_path)
    return config.first_point_task()


def gamma_seed(root_seed: int) -> int:
    """The seed the row's gamma table draws under the machine's path."""
    window_decoder_field = seed_records.RunSeedPathSegment(
        "field", "window_decoder"
    )
    path = (*stim_batch.ROW_SEED_PATH, window_decoder_field)
    return seeding.derive_component_seed(root_seed, path)


def row_predictions(window, row, events) -> numpy.ndarray:
    """Each shot's predicted observables from the row's Decoder port."""
    predictions = []
    for shot_events in events:
        result = window.result_of(row, shot_events)
        predictions.append(result.logical_observables)
    return numpy.asarray(predictions, dtype=numpy.uint8)


def batched_predictions(compiled, circuit, events) -> numpy.ndarray:
    """The adapter's answers, handed the shots in sinter-sized batches."""
    packed_events = numpy.packbits(events, axis=1, bitorder="little")
    answers = []
    for start in range(0, SHOT_COUNT, BATCH_SHOT_COUNT):
        batch = packed_events[start : start + BATCH_SHOT_COUNT]
        packed_answers = compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=batch
        )
        answers.append(packed_answers)
    packed = numpy.concatenate(answers)
    return numpy.unpackbits(
        packed, axis=1, count=circuit.num_observables, bitorder="little"
    )

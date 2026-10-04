"""The Relay-BP sinter adapter against the machine's relay_bp weak row.

The referent is the machine itself: each seed's shot runs through the
whole machine (collect.run_shot) under the decoder baseline, whose
naive_online scheme decodes the operation as one window, and the row's
answer is the observables the shot's result carries. The machine binds
shot s's row to root seed s, so the adapter decoding the events the
machine's device drew (observe/sampled_shots.py), given the gamma seed
that binding draws, must answer every shot as the row does.
"""

import dataclasses

import numpy
import pytest
import sinter
import stim

import decsim.decoders.relay_belief_propagation.decoder as relay_decoder
import decsim.records.seeds as seed_records
import decsim.seeding as seeding
import decsim.sinter_adapters.relay_bp as relay_bp_adapter
import tests.sinter_adapters.test_union_find as union_find_tests

# the baseline's row with Mueller et al.'s surface values, bases apart
RELAY_SETTINGS = relay_decoder.RelayBeliefPropagationDecoder.Settings(
    gamma0=0.35,
    gamma_interval=(-0.254, 0.985),
    pre_iterations=80,
    relay_set_count=300,
    iterations_per_set=60,
    converged_solution_count=1,
    bases="apart",
)
# The machine's seed path to its weak row's gamma table: the decoder's
# seed root (Decoders.seed_roots, primary_decoder), the row's algorithm
# decoder, and the window decoder that draws the table
# (decoders/relay_belief_propagation/window_decoder.py).
GAMMA_TABLE_SEED_PATH = (
    seed_records.RunSeedPathSegment("field", "primary_decoder"),
    seed_records.RunSeedPathSegment("field", "decoder"),
    seed_records.RunSeedPathSegment("field", "window_decoder"),
)
# At this rate the first leg sometimes fails, so the gamma table decides
# some shots and a wrong seed, table or column order shows.
ERROR_RATE = 0.01
FIRST_SEEDS = tuple(range(10))


# Each case's table seeds are the shots whose answer the gamma table
# decides, where the adapter given another seed answers otherwise: every
# one in seeds 0 to 399, and to 1199 at d 3 with bases apart, where they
# are rarer. Another Stim version may draw other samples for a seed,
# which moves the shots the table decides, never the equality.
@pytest.mark.parametrize(
    ("distance", "code_task", "bases", "table_seeds"),
    [
        (3, "surface_code:rotated_memory_x", "apart", (2, 633)),
        (
            3,
            "surface_code:rotated_memory_z",
            "apart",
            (266, 406, 474, 621, 1000),
        ),
        (
            5,
            "surface_code:rotated_memory_x",
            "apart",
            (81, 99, 123, 275, 376, 386),
        ),
        (
            5,
            "surface_code:rotated_memory_z",
            "apart",
            (41, 95, 135, 162, 278, 335),
        ),
        (3, "surface_code:rotated_memory_z", "together", (54, 152, 251, 290)),
    ],
)
def test_the_adapter_answers_as_the_machines_relay_bp_row_shot_for_shot(
    distance, code_task, bases, table_seeds
):
    pytest.importorskip("relay_bp")
    settings = dataclasses.replace(RELAY_SETTINGS, bases=bases)
    task = union_find_tests.baseline_task(
        distance, code_task, ERROR_RATE, settings
    )
    seeds = (*FIRST_SEEDS, *table_seeds)
    events, machine_answers = union_find_tests.machine_shots(task, seeds)
    (operation,) = task.settings.workload.operations

    adapter_answers = seeded_answers(operation.circuit, settings, seeds, events)

    numpy.testing.assert_array_equal(adapter_answers, machine_answers)


def test_sinter_collects_through_the_adapter():
    pytest.importorskip("relay_bp")
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


def seeded_answers(circuit, settings, seeds, events) -> numpy.ndarray:
    """Each shot decoded by an adapter given the gamma seed of its own seed.

    The machine draws each shot's gamma table afresh, so each shot has
    an adapter of its own, handed that one shot as a sinter batch.
    """
    model = circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True
    )
    packed_events = numpy.packbits(events, axis=1, bitorder="little")
    answers = []
    for seed, shot_events in zip(seeds, packed_events, strict=True):
        table_seed = seeding.derive_component_seed(seed, GAMMA_TABLE_SEED_PATH)
        decoder = relay_bp_adapter.RelayBeliefPropagationDecoder(
            circuit, settings, table_seed
        )
        compiled = decoder.compile_decoder_for_dem(dem=model)
        batch = numpy.expand_dims(shot_events, 0)
        (packed_answer,) = compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=batch
        )
        answers.append(packed_answer)
    packed_answers = numpy.asarray(answers)
    return numpy.unpackbits(
        packed_answers, axis=1, count=circuit.num_observables, bitorder="little"
    )

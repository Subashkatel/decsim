"""The code cards count qubits and stabilizers the way the real codes do.

Sources: Stim's generated ``surface_code:rotated_memory_z`` circuit
(src/stim/gen/gen_surface_code.cc) for the rotated surface code: a
distance-d patch has d*d data qubits and d*d - 1 measure qubits, each
measure qubit read out once per round; Bravyi et al. 2308.07915 (not on
disk; from the abstract) for the [[144, 12, 12]] bivariate-bicycle code
on 144 data and 144 check qubits, so n/2 X checks and n/2 Z checks per
round; Skoric 2209.08552 and Tan PRX Quantum 4, 040344 for the (d, d)
window floor.
"""

import pytest

import decsim.qpu.code_geometry as code_geometry

stim = pytest.importorskip("stim")


def stim_counts(distance):
    one_round = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=distance, rounds=1
    )
    two_rounds = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=distance, rounds=2
    )
    measure_qubit_count = (
        two_rounds.num_measurements - one_round.num_measurements
    )
    qubit_coordinates = one_round.get_final_qubit_coordinates()
    data_qubit_count = len(qubit_coordinates) - measure_qubit_count
    return measure_qubit_count, data_qubit_count


def test_a_distance_three_patch_has_eight_stabilizers_and_nine_data_qubits():
    card = code_geometry.SurfaceCodeModel(distance=3)
    assert stim_counts(3) == (8, 9)
    assert card.syndrome_bits_per_round(1) == 8
    assert card.spatial_nodes(1) == 9


def test_a_distance_five_patch_has_24_stabilizers_and_25_data_qubits():
    card = code_geometry.SurfaceCodeModel(distance=5)
    assert stim_counts(5) == (24, 25)
    assert card.syndrome_bits_per_round(1) == 24
    assert card.spatial_nodes(1) == 25


def test_two_patches_add_a_seam_of_d_nodes_and_double_the_syndrome():
    card = code_geometry.SurfaceCodeModel(distance=3)
    assert card.spatial_nodes(2) == 21
    assert card.syndrome_bits_per_round(2) == 16


def test_the_surface_card_logical_cycle_is_d_rounds():
    card = code_geometry.SurfaceCodeModel(distance=5)
    assert card.rounds_per_logical_cycle() == 5


def test_the_surface_card_commits_d_rounds_per_window():
    card = code_geometry.SurfaceCodeModel(distance=5)
    assert card.commit_rounds() == 5


def test_the_surface_card_buffers_d_rounds_per_window():
    card = code_geometry.SurfaceCodeModel(distance=5)
    assert card.buffer_rounds() == 5


def test_the_surface_card_window_floor_is_d_by_d():
    card = code_geometry.SurfaceCodeModel(distance=5)
    assert card.buffering_floor() == (5, 5)


def test_the_surface_card_has_no_cadence_of_its_own_by_default():
    card = code_geometry.SurfaceCodeModel(distance=5)
    assert card.round_period_us() is None


def test_the_surface_card_is_named_by_its_distance():
    card = code_geometry.SurfaceCodeModel(distance=5)
    assert card.name == "rotated surface code (d=5)"


def test_the_bicycle_card_is_the_gross_code_by_default():
    card = code_geometry.BivariateBicycleCodeModel()
    assert (card.qubit_count, card.logical_qubit_count, card.distance) == (
        144,
        12,
        12,
    )


def test_the_gross_code_card_reads_out_all_144_checks_per_round():
    card = code_geometry.BivariateBicycleCodeModel()
    assert card.syndrome_bits_per_round(1) == 144


def test_the_gross_code_card_has_144_decoding_graph_nodes_per_round():
    card = code_geometry.BivariateBicycleCodeModel()
    assert card.spatial_nodes(1) == 144


def test_the_bicycle_card_has_no_window_floor():
    card = code_geometry.BivariateBicycleCodeModel()
    assert card.buffering_floor() == (0, 0)


def test_the_bicycle_card_is_named_by_its_parameters():
    card = code_geometry.BivariateBicycleCodeModel()
    assert card.name == "bivariate-bicycle code [[144,12,12]]"


def test_the_bicycle_card_commits_d_rounds_and_buffers_none_by_default():
    card = code_geometry.BivariateBicycleCodeModel()
    assert card.commit_rounds() == 12
    assert card.buffer_rounds() == 0


def test_the_surface_card_commit_override_replaces_d():
    card = code_geometry.SurfaceCodeModel(distance=5, commit_rounds_override=2)
    assert card.commit_rounds() == 2


def test_the_surface_card_buffer_override_replaces_d():
    card = code_geometry.SurfaceCodeModel(distance=5, buffer_rounds_override=1)
    assert card.buffer_rounds() == 1


def test_the_bicycle_card_commit_override_replaces_d():
    card = code_geometry.BivariateBicycleCodeModel(commit_rounds_override=4)
    assert card.commit_rounds() == 4


def test_the_bicycle_card_buffer_override_replaces_zero():
    card = code_geometry.BivariateBicycleCodeModel(buffer_rounds_override=3)
    assert card.buffer_rounds() == 3


def test_a_card_cadence_is_kept_as_a_float():
    card = code_geometry.SurfaceCodeModel(distance=3, round_microseconds=1)
    period = card.round_period_us()
    assert period == 1.0
    assert type(period) is float


def test_more_logical_than_physical_qubits_is_refused_for_a_bicycle_code():
    with pytest.raises(ValueError, match="logical_qubit_count must not exceed"):
        code_geometry.BivariateBicycleCodeModel(
            qubit_count=24, logical_qubit_count=30, distance=6
        )


def test_a_distance_above_the_qubit_count_is_refused_for_a_bicycle_code():
    with pytest.raises(ValueError, match="distance must not exceed"):
        code_geometry.BivariateBicycleCodeModel(
            qubit_count=24, logical_qubit_count=4, distance=30
        )


def test_a_bicycle_code_without_qubits_is_refused():
    with pytest.raises(ValueError, match="qubit_count must be positive"):
        code_geometry.BivariateBicycleCodeModel(qubit_count=0)


def test_a_negative_buffer_override_is_refused_for_a_bicycle_code():
    with pytest.raises(ValueError, match="must be nonnegative"):
        code_geometry.BivariateBicycleCodeModel(buffer_rounds_override=-1)


def test_an_odd_qubit_count_is_refused_for_a_bicycle_code():
    with pytest.raises(ValueError, match="qubit_count must be even"):
        code_geometry.BivariateBicycleCodeModel(qubit_count=143)


def test_a_blank_window_floor_justification_is_refused():
    with pytest.raises(ValueError, match="non-empty"):
        code_geometry.SurfaceCodeModel(
            distance=3, window_floor_justification=" "
        )

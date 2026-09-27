"""Batch sampling against the decoders' own packages and against the machine.

Referent one is each weak decoder row's own package, run on the same
Stim samples: PyMatching on Stim's decomposed detector error model of
the circuit, the path sinter's pymatching decoder takes (sinter
_decoding_pymatching.py), a different answer allowed only at equal
matching weight; the pure-Python Union-Find in
tests/decoders/union_find_oracle.py on the row's own graph; relay_bp's
RelayDecoderF32 on each basis part with the row's Relay-BP-1 settings
and the gamma table drawn from the seed the row installs; and
tesseract_decoder's own tesseract-short-beam profile
(make_tesseract_sinter_decoders_dict) on the model sinter hands a
decoder. Referent two is the machine: collect.run_shot's own sample,
decoded by the batch path, answers as the machine answered. The
rest pins the collection: shared samples across decoders and piece
cuts, a stop on the exact failure, resume, and ids that do not move
without the key.
"""

import dataclasses
import hashlib
import json
import math

import numpy
import pymatching
import pytest
import scipy.sparse
import yaml

import decsim.collect as collect
import decsim.decoders.backend_outcome as backend_outcome
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.decoders.union_find.window_decoder as union_find_window
import decsim.detector_error_model.basis_split as basis_split
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.experiments.collect_command as collect_command
import decsim.experiments.experiment as experiment
import decsim.experiments.measure as measure
import decsim.experiments.refusal as refusal
import decsim.experiments.run_folder as run_folder
import decsim.experiments.status_command as status_command
import decsim.experiments.stim_batch as stim_batch
import decsim.qpu.stim_device as stim_device
import decsim.records.decoding as decoding_records
import decsim.records.seeds as seed_records
import decsim.seeding as seeding
import tests.decoders.union_find_oracle as union_find_oracle
import tests.experiments.yaml_configs as yaml_configs
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

BASELINE = (
    yaml_configs.CONFIGS_DIR
    / "experiments"
    / "decoder_baseline"
    / "decoder_baseline.yaml"
)
BATCH_BASELINE = (
    yaml_configs.CONFIGS_DIR
    / "experiments"
    / "decoder_baseline_batch"
    / "decoder_baseline_batch.yaml"
)
CODE_TASKS = ("surface_code:rotated_memory_x", "surface_code:rotated_memory_z")
ROUNDS = 5
ERROR_RATE = 0.005
SHOTS = 64
GRAPHLIKE = fault_models.FaultRepresentation.GRAPHLIKE
PHYSICAL = fault_models.FaultRepresentation.PHYSICAL
WINDOW_DECODER_SEGMENT = seed_records.RunSeedPathSegment(
    "field", "window_decoder"
)


class QuietMatching(minimum_weight_perfect_matching.PyMatchingDecoder):
    """PyMatching that commits no correction to a window with any event.

    A backend with no answer is how a shot goes unscored (window_decode_of);
    this row gives that answer on every shot that fires, so the unscored
    seeds are known in advance.
    """

    def decode_window(self, backend, model, faults, syndrome):
        if not syndrome.any():
            matching = minimum_weight_perfect_matching.PyMatchingDecoder
            return matching.decode_window(
                self, backend, model, faults, syndrome
            )
        fault_count = faults.check.shape[1]
        return backend_outcome.no_correction_decode(
            decoding_records.BackendDecodeStatus.INVALID_CORRECTION,
            decoding_records.BackendFailureReason.NO_PERFECT_MATCHING,
            fault_count,
        )


# A part a Python caller builds in place of the yaml's.
PYTHON_BUILT_DECODER = minimum_weight_perfect_matching.PyMatchingDecoder()
PYTHON_BUILT_ROUTER = decoders.CodeRouter(default=PYTHON_BUILT_DECODER)
PYTHON_BUILT_DEVICE = stim_device.StimDevice()
# Each shipped config's configuration id and point ids, one sha256 over
# them in task order, recorded on main at 257d700e before the key existed.
SHIPPED_IDENTITIES = {
    "reference.yaml": (
        "6e8fc0337e84578665781c3b5378981453a70b5efa74936f027f4cbbe120ce9f"
    ),
    "examples/my_first_sweep.yaml": (
        "f6688627de9e18758790baec1c282d97e454bdd4a9fdd1ec583d0f92268c84d4"
    ),
    "examples/priced_cards_example.yaml": (
        "42d5e924af1acaf017d53fcd9c1dd601049fded4b6587ee765762493769f1795"
    ),
    "examples/two_tiers.yaml": (
        "8ec77525f25f0af35872e874fbd2ccee0784a8058128b7a01e6bb21397ec15a1"
    ),
    "experiments/burst_detection/burst_detection.yaml": (
        "0f0c59735dff2658df771e74b65bc4738ed2623a3b8483b3a7fc2bbde099084f"
    ),
    "experiments/data_movement/data_movement.yaml": (
        "147de3a00ff6edde8f781084ee76b1400c53c6489b874e8423dd26f2fac2f8d6"
    ),
    "experiments/decoder_baseline/decoder_baseline.yaml": (
        "3fbbedc03ed2d9e325b6dd0e03705e9f9bf05c8ea5679c81b99201ab091c5811"
    ),
    "experiments/switching/cluster_gap_switching.yaml": (
        "05e86c58f9ca9c57f6cd80fe47d770c6a19eaa61c7ee5d0966f8e9d7ed12451f"
    ),
    "experiments/switching/seam_pinned_switching.yaml": (
        "05e1103f9deb8b5c69526425727a1bb234a9e0c0188acd81b06c98f332ec0975"
    ),
}


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("code_task", CODE_TASKS)
def test_the_pymatching_row_answers_as_sinters_pymatching_up_to_ties(
    tmp_path, distance, code_task
):
    task = batch_task(tmp_path, "pymatching", distance, code_task)
    circuit = stim_batch.circuit_of(task)
    events, _observables = stim_batch.samples_of_seeds(circuit, 0, SHOTS)
    window = stim_batch.WholeCircuitWindow(task)
    row = stim_batch.bound_row(task, 0)
    batch = batch_predictions(window, row, events)
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(
        detector_error_model
    )
    package = matching.decode_batch(events)

    untied = disagreements_off_ties(window, row, matching, events, package)

    assert batch.shape == package.shape
    assert untied == []


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("code_task", CODE_TASKS)
def test_the_union_find_row_answers_as_the_python_union_find(
    tmp_path, distance, code_task
):
    task = batch_task(tmp_path, "union_find", distance, code_task)
    circuit = stim_batch.circuit_of(task)
    events, _observables = stim_batch.samples_of_seeds(circuit, 0, SHOTS)
    window = stim_batch.WholeCircuitWindow(task)
    row = stim_batch.bound_row(task, 0)
    batch = batch_predictions(window, row, events)
    graph = union_find_window.graph_from_model(
        window.model.graphlike_faults,
        location="reference window",
        weight_step=0.1,
    )

    oracle = union_find_oracle_predictions(window, graph, events)

    numpy.testing.assert_array_equal(batch, oracle)


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("code_task", CODE_TASKS)
def test_the_relay_row_answers_as_relay_bp_with_the_gammas_it_installs(
    tmp_path, distance, code_task
):
    relay_bp = pytest.importorskip("relay_bp")
    task = batch_task(tmp_path, "relay_bp", distance, code_task)
    circuit = stim_batch.circuit_of(task)
    events, _observables = stim_batch.samples_of_seeds(circuit, 0, SHOTS)
    window = stim_batch.WholeCircuitWindow(task)
    row = stim_batch.bound_row(task, 11)
    batch = batch_predictions(window, row, events)
    decoder_path = stim_batch.ROW_SEED_PATH + (WINDOW_DECODER_SEGMENT,)
    gamma_seed = seeding.derive_component_seed(11, decoder_path)

    package = relay_predictions(relay_bp, window, gamma_seed, events)

    numpy.testing.assert_array_equal(batch, package)


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("code_task", CODE_TASKS)
def test_the_tesseract_row_answers_as_the_short_beam_profile(
    tmp_path, distance, code_task
):
    tesseract_decoder = pytest.importorskip("tesseract_decoder")
    task = batch_task(tmp_path, "tesseract", distance, code_task)
    circuit = stim_batch.circuit_of(task)
    events, _observables = stim_batch.samples_of_seeds(circuit, 0, SHOTS)
    window = stim_batch.WholeCircuitWindow(task)
    row = stim_batch.bound_row(task, 0)
    batch = batch_predictions(window, row, events)
    detector_error_model = circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True
    )
    profiles = tesseract_decoder.make_tesseract_sinter_decoders_dict()
    profile = profiles["tesseract-short-beam"]
    compiled = profile.compile_decoder_for_dem(dem=detector_error_model)
    packed = numpy.packbits(events, axis=1, bitorder="little")

    predicted = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed
    )

    unpacked = numpy.unpackbits(predicted, axis=1, bitorder="little")
    package = unpacked[:, : circuit.num_observables].astype(bool)
    numpy.testing.assert_array_equal(batch, package)


@pytest.mark.parametrize("kind", ["union_find", "pymatching"])
@pytest.mark.parametrize("distance", [3, 5])
def test_the_machines_own_samples_decode_to_the_machines_answers(
    tmp_path, kind, distance
):
    task = machine_task(tmp_path, kind, distance)

    machine, batch = machine_and_batch_answers(task, range(20))

    assert batch == machine


@pytest.mark.parametrize("distance", [3, 5])
def test_the_machines_own_samples_decode_to_relays_machine_answers(
    tmp_path, distance
):
    pytest.importorskip("relay_bp")
    task = machine_task(tmp_path, "relay_bp", distance)

    machine, batch = machine_and_batch_answers(task, range(20))

    assert batch == machine


@pytest.mark.parametrize("distance", [3, 5])
def test_the_machines_own_samples_decode_to_tesseracts_machine_answers(
    tmp_path, distance
):
    pytest.importorskip("tesseract_decoder")
    task = machine_task(tmp_path, "tesseract", distance)

    machine, batch = machine_and_batch_answers(task, range(20))

    assert batch == machine


def test_a_numbered_kind_decodes_as_the_machines_matching(tmp_path):
    task = machine_task(tmp_path, "numbered", 3)

    machine, batch = machine_and_batch_answers(task, range(5))

    assert batch == machine


def test_every_decoder_of_a_point_decodes_the_same_samples(tmp_path):
    pytest.importorskip("relay_bp")
    pytest.importorskip("tesseract_decoder")
    kinds = ["union_find", "pymatching", "relay_bp", "tesseract"]
    collection = {"max_shots": 50, "piece_rounds": 1000}
    config_path = write_batch_config(tmp_path, kinds, collection)
    experiment_dir = tmp_path / "run"

    collect_command.run_experiment(config_path, experiment_dir)

    digests = piece_values(experiment_dir, "sample_sha256")
    assert len(digests) == 4
    assert len(set(digests)) == 1


def test_a_seeds_sample_is_the_same_whatever_piece_holds_it(tmp_path):
    task = batch_task(tmp_path, "pymatching", 3, CODE_TASKS[1])
    circuit = stim_batch.circuit_of(task)

    events, observables = stim_batch.samples_of_seeds(circuit, 1000, 100)
    whole_events, whole_observables = stim_batch.samples_of_seeds(
        circuit, 0, 2048
    )

    numpy.testing.assert_array_equal(events, whole_events[1000:1100])
    numpy.testing.assert_array_equal(observables, whole_observables[1000:1100])


@pytest.mark.parametrize(
    "kind, package",
    [("relay_bp", "relay_bp"), ("tesseract", "tesseract_decoder")],
)
def test_a_shots_answer_is_the_same_in_every_piece_cut(tmp_path, kind, package):
    """Cuts inside a block and across the block edge at seed 1024."""
    pytest.importorskip(package)
    config_path = write_batch_config(
        tmp_path, [kind], {"max_shots": 1}, error_rate=0.03
    )
    config = experiment.load_experiment(config_path)
    task = config.first_point_task()

    whole_inside = cut_answers(task, [(0, 40)])
    cut_inside = cut_answers(task, [(0, 15), (15, 25)])
    whole_across = cut_answers(task, [(1000, 48)])
    cut_across = cut_answers(task, [(1000, 10), (1010, 30), (1040, 8)])

    assert cut_inside == whole_inside
    assert cut_across == whole_across
    assert whole_inside[0] != []
    assert whole_across[0] != []


def test_a_shot_its_decode_leaves_uncorrected_is_unscored(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(
        decoder_settings.DECODERS, "matching_on_quiet_windows", QuietMatching
    )
    rows = batch_rows()
    quiet_row = dict(rows["pymatching"], kind="matching_on_quiet_windows")
    task = task_of_row(tmp_path, quiet_row)
    circuit = stim_batch.circuit_of(task)
    events, _observables = stim_batch.samples_of_seeds(circuit, 0, 30)
    fired = events.any(axis=1)
    fired_seeds = numpy.flatnonzero(fired)
    unit = collect.Unit(task, 0, 30)

    outcome = stim_batch.run_unit(unit, None)

    (counts,) = outcome.rows
    assert counts["unscored_seeds"] == fired_seeds.tolist()
    assert counts["scored_shots"] == 30 - len(fired_seeds)
    assert counts["failure_seeds"] == []


def test_a_time_cap_stops_a_point_at_its_first_pieces_end(tmp_path):
    collection = {
        "max_core_seconds": 1e-9,
        "max_shots": 1000,
        "piece_rounds": 100,
    }
    config_path = write_batch_config(tmp_path, ["pymatching"], collection)
    experiment_dir = tmp_path / "run"

    _run_dir, rows = collect_command.run_experiment(config_path, experiment_dir)

    (row,) = rows
    assert row["state"] == "time cap"
    assert row["prefix_shots"] == 100 // ROUNDS
    assert row["shots"] == 100 // ROUNDS


def test_a_point_stops_on_the_shot_of_its_last_wanted_failure(tmp_path):
    collection = {"max_failures": 5, "max_shots": 100000, "piece_rounds": 200}
    config_path = write_batch_config(
        tmp_path, ["pymatching"], collection, error_rate=0.02
    )
    config = experiment.load_experiment(config_path)
    task = config.first_point_task()
    failure_seeds = failing_seeds(task, 400)
    experiment_dir = tmp_path / "run"

    _run_dir, rows = collect_command.run_experiment(config_path, experiment_dir)

    (row,) = rows
    assert row["state"] == "target"
    assert row["prefix_failures"] == 5
    assert row["prefix_shots"] == failure_seeds[4] + 1


def test_two_piece_cuts_fold_to_the_same_counts(tmp_path):
    small = {"max_shots": 120, "piece_rounds": 100}
    large = {"max_shots": 120, "piece_rounds": 600}
    small_folder = tmp_path / "small"
    large_folder = tmp_path / "large"
    small_path = write_batch_config(
        small_folder, ["union_find"], small, error_rate=0.02
    )
    large_path = write_batch_config(
        large_folder, ["union_find"], large, error_rate=0.02
    )
    small_experiment = tmp_path / "small_run"
    large_experiment = tmp_path / "large_run"

    _small_dir, small_rows = collect_command.run_experiment(
        small_path, small_experiment
    )
    _large_dir, large_rows = collect_command.run_experiment(
        large_path, large_experiment
    )

    assert counted(small_rows) == counted(large_rows)
    assert small_rows[0]["prefix_shots"] == 120


def test_a_raised_target_resumes_from_the_saved_pieces(tmp_path):
    first = {"max_failures": 2, "max_shots": 100000, "piece_rounds": 25}
    raised = dict(first, max_failures=4)
    first_folder = tmp_path / "first"
    raised_folder = tmp_path / "raised"
    first_path = write_batch_config(
        first_folder, ["pymatching"], first, error_rate=0.02
    )
    raised_path = write_batch_config(
        raised_folder, ["pymatching"], raised, error_rate=0.02
    )
    experiment_dir = tmp_path / "run"
    collect_command.run_experiment(first_path, experiment_dir)
    saved = piece_texts(experiment_dir)

    _run_dir, rows = collect_command.run_experiment(raised_path, experiment_dir)

    (row,) = rows
    after = piece_texts(experiment_dir)
    kept = {name: after[name] for name in saved}
    assert row["prefix_failures"] == 4
    assert row["state"] == "target"
    assert kept == saved
    assert len(after) > len(saved)


def test_status_folds_a_batch_experiment_from_its_counts(tmp_path):
    collection = {"max_failures": 3, "max_shots": 100000, "piece_rounds": 200}
    config_path = write_batch_config(
        tmp_path, ["pymatching"], collection, error_rate=0.02
    )
    experiment_dir = tmp_path / "run"
    _run_dir, rows = collect_command.run_experiment(config_path, experiment_dir)

    status_rows = status_command.fold_the_experiment(experiment_dir)

    (status_row,) = status_rows
    assert status_row["state"] == "target"
    assert status_row["prefix_shots"] == rows[0]["prefix_shots"]
    assert status_row["rounds"] == ROUNDS * status_row["shots"]


@pytest.mark.parametrize("name", sorted(SHIPPED_IDENTITIES))
def test_a_config_without_the_key_keeps_every_id(name):
    config_path = yaml_configs.CONFIGS_DIR / name
    config = experiment.load_experiment(config_path)

    digest = identities_digest(config)

    assert digest == SHIPPED_IDENTITIES[name]


def test_the_key_gives_a_config_and_its_points_new_ids():
    machine = experiment.load_experiment(BASELINE)
    batch = experiment.load_experiment(BATCH_BASELINE)

    machine_ids = identities(machine)
    batch_ids = identities(batch)

    shared = set(batch_ids) & set(machine_ids)
    assert len(batch_ids) == len(machine_ids)
    assert shared == set()


def test_the_batch_baseline_runs_the_machine_baselines_points():
    machine = experiment.load_experiment(BASELINE)
    batch = experiment.load_experiment(BATCH_BASELINE)

    machine_ids = identities(machine)
    unkeyed_ids = point_ids_without_the_key(batch)

    assert unkeyed_ids == machine_ids[1:]


def test_a_window_scheme_other_than_naive_online_is_refused(tmp_path):
    overrides = batch_overrides()
    overrides["windows"] = {
        "kind": "sliding",
        "commit_rounds": None,
        "buffer_rounds": None,
    }
    config_path = yaml_configs.write_config(tmp_path, overrides)

    experiment_dir = tmp_path / "run"

    with pytest.raises(refusal.RefusalError) as refused:
        collect_command.run_experiment(config_path, experiment_dir)

    assert str(refused.value) == (
        "sampling stim_batch decodes one window over the whole circuit on "
        "the weak tier alone, from the circuit's own samples, which is the "
        "machine's decode only under windows.kind naive_online; this point "
        "sets windows.kind 'sliding'"
    )


def test_an_escalation_other_than_the_weak_baseline_is_refused(tmp_path):
    overrides = batch_overrides()
    switching = yaml_configs.fixed_threshold_switching()
    overrides.update(switching)
    config_path = yaml_configs.write_config(tmp_path, overrides)

    experiment_dir = tmp_path / "run"

    with pytest.raises(refusal.RefusalError) as refused:
        collect_command.run_experiment(config_path, experiment_dir)

    assert str(refused.value) == (
        "sampling stim_batch decodes one window over the whole circuit on "
        "the weak tier alone, from the circuit's own samples, which is the "
        "machine's decode only under escalation.kind weak_baseline; this "
        "point sets escalation.kind 'switching'"
    )


def test_a_qpu_other_than_the_stim_device_is_refused(tmp_path):
    overrides = batch_overrides()
    overrides["qpu"] = {"kind": "recorded_stim"}
    config_path = yaml_configs.write_config(tmp_path, overrides)

    experiment_dir = tmp_path / "run"

    with pytest.raises(refusal.RefusalError) as refused:
        collect_command.run_experiment(config_path, experiment_dir)

    assert str(refused.value) == (
        "sampling stim_batch decodes one window over the whole circuit on "
        "the weak tier alone, from the circuit's own samples, which is the "
        "machine's decode only under qpu.kind stim_device; this point sets "
        "qpu.kind 'recorded_stim'"
    )


def test_a_window_check_is_refused(tmp_path):
    overrides = batch_overrides()
    overrides["observation"] = {"check_windows_with": "tesseract"}
    config_path = yaml_configs.write_config(tmp_path, overrides)
    experiment_dir = tmp_path / "run"

    with pytest.raises(refusal.RefusalError) as refused:
        collect_command.run_experiment(config_path, experiment_dir)

    assert str(refused.value) == (
        "sampling stim_batch decodes one window over the whole circuit on "
        "the weak tier alone, from the circuit's own samples, which is the "
        "machine's decode only under observation.check_windows_with none; "
        "this point sets observation.check_windows_with 'tesseract'"
    )


@pytest.mark.parametrize(
    "section, key, built",
    [
        ("weak_decoder", "decoder", PYTHON_BUILT_DECODER),
        ("decoder_manager", "router", PYTHON_BUILT_ROUTER),
        ("qpu", "device", PYTHON_BUILT_DEVICE),
    ],
)
def test_a_python_built_part_is_refused(tmp_path, section, key, built):
    task = batch_task(tmp_path, "pymatching", 3, CODE_TASKS[1])
    part = getattr(task.settings, section)
    built_part = dataclasses.replace(part, **{key: built})
    settings = dataclasses.replace(task.settings, **{section: built_part})
    built_task = dataclasses.replace(task, settings=settings)

    with pytest.raises(refusal.RefusalError) as refused:
        stim_batch.check_task(built_task)

    assert str(refused.value) == (
        "sampling stim_batch builds the decoder from the yaml at the "
        f"machine's seed path, so it takes no Python-built {section}.{key}; "
        "this point sets one"
    )


def test_a_workload_of_two_operations_is_refused(tmp_path):
    overrides = batch_overrides()
    overrides["workload"] = {
        "kind": "producer",
        "function": "decsim.producers:memory_patches",
        "arguments": {
            "code_task": "surface_code:rotated_memory_z",
            "rounds_per_shot": ROUNDS,
            "patch_count": 2,
            "distance": "${qpu.distance}",
        },
    }
    config_path = yaml_configs.write_config(tmp_path, overrides)

    experiment_dir = tmp_path / "run"

    with pytest.raises(refusal.RefusalError) as refused:
        collect_command.run_experiment(config_path, experiment_dir)

    assert str(refused.value) == (
        "sampling stim_batch decodes one operation's circuit a shot, and "
        "this point's workload has 2 operations"
    )


def test_a_sampling_off_the_table_is_refused(tmp_path):
    overrides = batch_overrides()
    overrides["sampling"] = "stim_batches"
    config_path = yaml_configs.write_config(tmp_path, overrides)

    experiment_dir = tmp_path / "run"

    with pytest.raises(refusal.RefusalError) as refused:
        collect_command.run_experiment(config_path, experiment_dir)

    assert str(refused.value) == (
        "sampling 'stim_batches' is not a row of its table; the rows are "
        "['stim_batch']"
    )


def test_a_sampling_that_is_not_text_is_refused(tmp_path):
    overrides = batch_overrides()
    overrides["sampling"] = ["stim_batch"]
    config_path = yaml_configs.write_config(tmp_path, overrides)

    with pytest.raises(refusal.RefusalError) as refused:
        experiment.load_experiment(config_path)

    assert str(refused.value) == (
        f"{config_path}: sampling is ['stim_batch']; it names a sampling as "
        "text, such as stim_batch, or is left out for the machine"
    )


def batch_rows() -> dict:
    """The shipped baseline's four weak decoder rows, by kind.

    "numbered" is its PyMatching row priced by a number in place of a
    kind, the machine's fixed-latency matching.
    """
    text = BASELINE.read_text()
    raw = yaml.safe_load(text)
    axes = raw["sweep"][0]["axes"]
    rows = {}
    for row in axes["weak_decoder"]:
        rows[row["kind"]] = row
    rows["numbered"] = dict(rows["pymatching"], kind=0.028)
    return rows


def write_batch_config(
    folder,
    kinds: list,
    collection: dict,
    *,
    distances=(3,),
    code_tasks=(CODE_TASKS[1],),
    error_rate: float = ERROR_RATE,
    sampling="stim_batch",
):
    """A yaml over the shipped baseline, short shots, its sweep as given."""
    rows = batch_rows()
    decoder_rows = [rows[kind] for kind in kinds]
    workload = short_memory_workload(error_rate)
    raw = {
        "extends": str(BASELINE),
        "workload": workload,
        "sweep": [
            {
                "axes": {
                    "workload.arguments.code_task": list(code_tasks),
                    "qpu.distance": list(distances),
                    "weak_decoder": decoder_rows,
                }
            }
        ],
        "collection": collection,
    }
    if sampling is not None:
        raw["sampling"] = sampling
    folder.mkdir(parents=True, exist_ok=True)
    config_path = folder / "batch.yaml"
    text = yaml.safe_dump(raw, sort_keys=False)
    config_path.write_text(text)
    return config_path


def short_memory_workload(error_rate: float) -> dict:
    """The baseline's memory workload at ROUNDS rounds a shot."""
    return {
        "kind": "producer",
        "function": "decsim.producers:memory_circuit",
        "arguments": {
            "code_task": CODE_TASKS[1],
            "rounds_per_shot": ROUNDS,
            "distance": "${qpu.distance}",
            "physical_error_probability": error_rate,
        },
    }


def batch_task(tmp_path, kind: str, distance: int, code_task: str):
    """The one point of a batch yaml of one decoder, distance and basis."""
    config_path = write_batch_config(
        tmp_path,
        [kind],
        {"max_shots": 1},
        distances=(distance,),
        code_tasks=(code_task,),
    )
    config = experiment.load_experiment(config_path)
    return config.first_point_task()


def machine_task(tmp_path, kind: str, distance: int):
    """The one point of the same yaml without the key: the machine's."""
    config_path = write_batch_config(
        tmp_path,
        [kind],
        {"max_shots": 1},
        distances=(distance,),
        sampling=None,
    )
    config = experiment.load_experiment(config_path)
    return config.first_point_task()


def task_of_row(tmp_path, row: dict):
    """The one batch point of one decoder row given whole."""
    raw = {
        "extends": str(BASELINE),
        "sampling": "stim_batch",
        "workload": short_memory_workload(ERROR_RATE),
        "sweep": [{"axes": {"qpu.distance": [3], "weak_decoder": [row]}}],
        "collection": {"max_shots": 1},
    }
    config_path = tmp_path / "row.yaml"
    text = yaml.safe_dump(raw, sort_keys=False)
    config_path.write_text(text)
    config = experiment.load_experiment(config_path)
    return config.first_point_task()


def cut_answers(task, cuts: list) -> tuple:
    """The failed and the unscored seeds of units run cut by cut."""
    failure_seeds = []
    unscored_seeds = []
    for first_seed, count in cuts:
        unit = collect.Unit(task, first_seed, count)
        outcome = stim_batch.run_unit(unit, None)
        (counts,) = outcome.rows
        failure_seeds.extend(counts["failure_seeds"])
        unscored_seeds.extend(counts["unscored_seeds"])
    return failure_seeds, unscored_seeds


def batch_overrides() -> dict:
    """The minimal config as a batch point: naive_online and a real row."""
    weak_decoder = dict(
        yaml_configs.MINIMAL_CONFIG["weak_decoder"], kind="pymatching"
    )
    windows = {
        "kind": "naive_online",
        "commit_rounds": None,
        "buffer_rounds": None,
    }
    return {
        "sampling": "stim_batch",
        "windows": windows,
        "weak_decoder": weak_decoder,
    }


def batch_predictions(window, row, events) -> numpy.ndarray:
    """Each shot's predicted observables through the batch path."""
    predictions = []
    for shot_events in events:
        result = window.result_of(row, shot_events)
        predictions.append(result.logical_observables)
    return numpy.asarray(predictions, dtype=bool)


def disagreements_off_ties(window, row, matching, events, package) -> list:
    """The shots whose answers differ at different minimum weights."""
    faults = window.model.require_faults(GRAPHLIKE)
    backend = row.compiled_for(faults, window.model)
    untied = []
    for shot, shot_events in enumerate(events):
        result = window.result_of(row, shot_events)
        predicted = numpy.asarray(result.logical_observables, dtype=bool)
        if numpy.array_equal(predicted, package[shot]):
            continue
        syndrome = shot_events[window.detector_rows]
        _, row_weight = backend.plain.decode(syndrome, return_weight=True)
        _, package_weight = matching.decode(shot_events, return_weight=True)
        if not math.isclose(row_weight, package_weight):
            untied.append(shot)
    return untied


def union_find_oracle_predictions(window, graph, events) -> numpy.ndarray:
    """Each shot's observables from the pure-Python Union-Find."""
    predictions = []
    for shot_events in events:
        syndrome = shot_events[window.detector_rows].astype(numpy.uint8)
        evidence = union_find_oracle.decode_graph(graph, syndrome)
        predictions.append(evidence.logical_observables)
    return numpy.asarray(predictions, dtype=bool)


def relay_predictions(
    relay_bp, window, gamma_seed: int, events
) -> numpy.ndarray:
    """Relay-BP-1 on each basis part, the gamma table the seed draws."""
    parts = basis_split.split_by_basis(window.model)
    decoders = {}
    faults = {}
    rows = {}
    for basis in ("X", "Z"):
        faults[basis] = parts[basis].require_faults(PHYSICAL)
        rows[basis] = basis_split.rows_of_basis(window.model, basis)
        decoders[basis] = relay_decoder(relay_bp, faults[basis], gamma_seed)
    predictions = []
    for shot_events in events:
        syndrome = shot_events[window.detector_rows].astype(numpy.uint8)
        flips = 0
        for basis in ("X", "Z"):
            answer = decoders[basis].decode_detailed(syndrome[rows[basis]])
            correction = numpy.asarray(answer.decoding, dtype=numpy.uint8)
            flips = flips + faults[basis].observables @ correction
        flip_array = numpy.asarray(flips)
        prediction = flip_array % 2
        predictions.append(prediction)
    return numpy.asarray(predictions, dtype=bool)


def relay_decoder(relay_bp, faults, gamma_seed: int):
    """RelayDecoderF32 with the baseline's Relay-BP-1 arguments."""
    column_count = faults.check.shape[1]
    bit_generator = numpy.random.PCG64(gamma_seed)
    generator = numpy.random.Generator(bit_generator)
    gamma_table = generator.uniform(-0.254, 0.985, size=(300, column_count))
    check = scipy.sparse.csr_matrix(faults.check)
    priors = numpy.asarray(faults.priors, dtype=numpy.float64)
    return relay_bp.RelayDecoderF32(
        check,
        priors,
        alpha=None,
        alpha_iteration_scaling_factor=1.0,
        gamma0=0.35,
        pre_iter=80,
        num_sets=300,
        set_max_iter=60,
        gamma_dist_interval=(-0.254, 0.985),
        explicit_gammas=gamma_table,
        stop_nconv=1,
        stopping_criterion="nconv",
        logging=False,
        seed=gamma_seed,
    )


def machine_and_batch_answers(task, seeds) -> tuple:
    """Each seed's machine answer, and the batch answer on its sample.

    An answer is whether the shot is scored and the observables it
    predicted. The batch row is bound to the shot's seed, the machine's
    root, so a row that draws from the run seed draws alike.
    """
    window = stim_batch.WholeCircuitWindow(task)
    machine = []
    batch = []
    for seed in seeds:
        shot = collect.run_shot(task, seed)
        observation = shot.machine.observation
        sampled = observation.sampled_shots.shots_by_operation[1]
        events = numpy.asarray(sampled.detection_events, dtype=bool)
        (operation_result,) = shot.result.operation_results
        predicted = tuple(operation_result.logical_observables)
        measured = measure.measure_shot(shot)
        machine.append((measured.is_scored, predicted))
        row = stim_batch.bound_row(task, seed)
        result = window.result_of(row, events)
        batch_scored = result.no_correction_reason is None
        batch_predicted = tuple(result.logical_observables)
        batch.append((batch_scored, batch_predicted))
    return machine, batch


def failing_seeds(task, shot_count: int) -> list:
    """The seeds below shot_count whose batch answer misses the truth."""
    circuit = stim_batch.circuit_of(task)
    events, observables = stim_batch.samples_of_seeds(circuit, 0, shot_count)
    window = stim_batch.WholeCircuitWindow(task)
    row = stim_batch.bound_row(task, 0)
    predictions = batch_predictions(window, row, events)
    missed = (predictions != observables).any(axis=1)
    failing = numpy.flatnonzero(missed)
    return failing.tolist()


def piece_values(experiment_dir, name: str) -> list:
    """One piece.json line of every piece of the experiment."""
    values = []
    piece_paths = experiment_dir.glob("pieces/*/*/piece.json")
    for piece_path in sorted(piece_paths):
        text = piece_path.read_text()
        piece = json.loads(text)
        values.append(piece[name])
    return values


def piece_texts(experiment_dir) -> dict:
    """Each piece.json's text, by its folder."""
    texts = {}
    for piece_path in experiment_dir.glob("pieces/*/*/piece.json"):
        folder = piece_path.parent
        texts[folder.name] = piece_path.read_text()
    return texts


def counted(rows: list) -> list:
    """The counts and prefix columns of a fold's rows."""
    names = (
        "shots",
        "logical_failures",
        "scored_shots",
        "unscored_shots",
        "prefix_shots",
        "prefix_failures",
        "state",
    )
    return [{name: row[name] for name in names} for row in rows]


def identities(config) -> list:
    """The configuration id, then every point id in task order."""
    ids = [run_folder.configuration_id(config)]
    for task in config.tasks():
        point_id = task.strong_id()
        ids.append(point_id)
    return ids


def point_ids_without_the_key(config) -> list:
    """Every point id in task order, each task's sampling taken off."""
    ids = []
    for task in config.tasks():
        unkeyed = dataclasses.replace(task, sampling=None)
        point_id = unkeyed.strong_id()
        ids.append(point_id)
    return ids


def identities_digest(config) -> str:
    """One sha256 over a config's ids, a line each."""
    ids = identities(config)
    text = "\n".join(ids)
    encoded = text.encode()
    digest = hashlib.sha256(encoded)
    return digest.hexdigest()

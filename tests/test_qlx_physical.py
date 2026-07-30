"""QLX whole-program PHYSICAL coupling (Gate 2b fixture).

Everything here runs on real QLX-origin artifacts in tests/data/qlx/
(regenerate with ./tools/qlx python3 decsim/tests/data/qlx/
generate_qlx_fixtures.py + dump_decoder_params.py):

  * mem_surface.stim — QLX-emitted whole-program circuit (d=3 surface
    memory, 8 syndrome rounds, 56 coordinate-less DETECTORs, 1 observable)
  * mem_surface_decoder_params.json — emit_decoder_params() output whose
    dem_detector_locs [packet, bit, baseline] provide the detector->round
    map the coordinate-less circuit lacks
  * mem_surface_walker_dem.txt — the analytic walker DEM
  * mem_surface_twin.json — QLX digital-twin reference (ler, det_rate)

Claim level: WHOLE-PROGRAM artifact, detector-bijection, and global-twin
evidence, plus explicit decoder-domain rejection. The emitted fixture
contains a detectorless logical mechanism, so it supports neither
per-window decode-equivalence nor full-engine decoding-success claims.
Nonvacuous detectable decoding evidence is in test_qlx_walker_dem_decode.py.
No per-operation claim is made (compound feedback-bearing programs remain
a QLX exporter boundary — see docs/validation/QLX_UTILIZATION_AND_GAP_
ANALYSIS.md §5).
"""
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from decsim.run_spec import RunSpec, simulate

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")

DATA = pathlib.Path(__file__).resolve().parent / "data/qlx"

pytestmark = pytest.mark.skipif(
    not (DATA / "mem_surface_decoder_params.json").exists(),
    reason="QLX physical fixtures missing; run generate_qlx_fixtures.py "
           "and dump_decoder_params.py via ./tools/qlx")

SHOTS = 5000
SEED = 12345


@pytest.fixture(scope="module")
def circuit():
    return stim.Circuit((DATA / "mem_surface.stim").read_text())


@pytest.fixture(scope="module")
def params():
    return json.loads((DATA / "mem_surface_decoder_params.json").read_text())


@pytest.fixture(scope="module")
def detector_rounds(params):
    """QLX submit-packet index == 1-based comparison round."""
    return {i: loc[0] for i, loc in enumerate(params["dem_detector_locs"])}


@pytest.fixture(scope="module")
def sampled(circuit):
    dets, obs = circuit.compile_detector_sampler(seed=SEED).sample(
        SHOTS, separate_observables=True)
    return dets, obs[:, 0].astype(np.uint8)


def _canon_stim_dem(dem):
    mech = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets, obs = [], []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs.append(t.val)
        key = (tuple(sorted(dets)), tuple(sorted(obs)))
        q = mech.get(key, 0.0)
        mech[key] = q * (1 - p) + p * (1 - q)
    return mech


def _canon_walker_dem(text):
    mech = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("error("):
            continue
        head, _, rest = line.partition(")")
        p = float(head[len("error("):])
        dets = tuple(sorted(int(t[1:]) for t in rest.split()
                            if t.startswith("D")))
        obs = tuple(sorted(int(t[1:]) for t in rest.split()
                           if t.startswith("L")))
        key = (dets, obs)
        q = mech.get(key, 0.0)
        mech[key] = q * (1 - p) + p * (1 - q)
    return mech


def test_detector_locs_cover_all_rounds(circuit, params, detector_rounds):
    assert params["dem_num_detectors"] == circuit.num_detectors == 56
    assert len(detector_rounds) == 56
    assert sorted(set(detector_rounds.values())) == list(range(1, 8))
    # 8 stabilizers per comparison round (4 X + 4 Z; dem_num_sx = 4)
    assert params["dem_num_sx"] == 4
    per_round = {}
    for det, rnd in detector_rounds.items():
        per_round[rnd] = per_round.get(rnd, 0) + 1
    assert all(count == 8 for count in per_round.values())


def test_stim_dem_semantically_consistent_with_walker(circuit):
    """Every stim-derived mechanism appears in the walker DEM under the
    current numeric detector ids. NOTE (Codex Gate-2 review, finding 1):
    this is a SEMANTIC-consistency check, NOT an order proof — any
    same-per-round bit permutation also satisfies it. The order proof is
    test_detector_bijection_from_measurement_records. (The walker
    superset is idle-depolarization the stim emitter does not include;
    gap G8, confirmed against EmitStim.cpp which emits nothing for bare
    idle.)"""
    stim_mech = _canon_stim_dem(circuit.detector_error_model())
    walker_mech = _canon_walker_dem(
        (DATA / "mem_surface_walker_dem.txt").read_text())
    missing = set(stim_mech) - set(walker_mech)
    assert not missing, f"stim mechanisms absent from walker DEM: {missing}"
    assert len(walker_mech) > len(stim_mech)   # superset is expected (G8)


def _detector_measurement_semantics(circuit):
    """For each stim DETECTOR (in index order): the absolute measurement
    indices it compares, derived by walking the instruction stream."""
    n_meas = 0
    per_detector = []
    for inst in circuit.flattened():
        if inst.name == "DETECTOR":
            recs = sorted(n_meas + t.value for t in inst.targets_copy())
            per_detector.append(recs)
        else:
            n_meas += inst.num_measurements
    assert n_meas == circuit.num_measurements
    return per_detector


def test_detector_bijection_from_measurement_records(circuit, params):
    """TRUE index-correspondence proof (replaces the subset check that the
    Codex review refuted with a 40,319-permutation counterexample): for
    every stim DETECTOR d, its measurement-record semantics — later
    measurement m2, earlier m1, with checks-per-round measurements per
    syndrome round — must reproduce the walker's dem_detector_locs[d]
    exactly: packet == m2 // checks, bit == m2 % checks, baseline ==
    m1 // checks. Exact per-detector equality is sensitive to ANY
    permutation, per-round bit shuffles included."""
    checks = 2 * params["dem_num_sx"]           # X + Z checks per round
    locs = params["dem_detector_locs"]
    semantics = _detector_measurement_semantics(circuit)
    assert len(semantics) == len(locs) == circuit.num_detectors
    for det, (recs, loc) in enumerate(zip(semantics, locs)):
        assert len(recs) == 2, f"D{det}: expected a comparison pair"
        m1, m2 = recs
        expected = [m2 // checks, m2 % checks, m1 // checks]
        assert loc == expected, \
            f"D{det}: walker locs {loc} != measurement semantics {expected}"


def test_detector_bijection_rejects_bit_permutations(circuit, params):
    """Negative control: a same-round bit permutation (which fooled the
    old subset check) must FAIL the bijection check."""
    checks = 2 * params["dem_num_sx"]
    perm = [(b + 1) % checks for b in range(checks)]   # cyclic shift
    permuted = [[loc[0], perm[loc[1]], loc[2]]
                for loc in params["dem_detector_locs"]]
    semantics = _detector_measurement_semantics(circuit)
    violations = 0
    for recs, loc in zip(semantics, permuted):
        m1, m2 = recs
        if loc != [m2 // checks, m2 % checks, m1 // checks]:
            violations += 1
    assert violations == len(permuted), \
        "a same-round bit permutation slipped past the bijection check"


def test_windowed_decode_equals_global_on_qlx_circuit(
        circuit, detector_rounds, sampled):
    """The vacuous QLX model is measured, then rejected by window decoding.

    G9 leaves a detectorless logical mechanism in this fixture. Global
    PyMatching silently omits it, but decsim must not discard that identity.
    """
    from decsim.detector_error_model import (
        GRAPHLIKE_FAULT_MODEL_REQUIRED,
        build_window_error_models,
    )

    dets, truth = sampled
    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    pred_global = matching.decode_batch(dets)[:, 0].astype(np.uint8)

    plan = [(1, 3, 6), (4, 6, 7), (7, 7, 7)]
    with pytest.raises(ValueError, match="detectorless logical"):
        build_window_error_models(
            circuit,
            plan,
            detector_rounds=detector_rounds,
            fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
            fault_exclusion_ranges=(),
        )

    fails = int((pred_global != truth).sum())
    twin = json.loads((DATA / "mem_surface_twin.json").read_text())
    twin_ler = float(twin["ler"].strip("'"))
    twin_std = float(twin["ler_std"].strip("'()," ).split(",")[0])
    ler = fails / SHOTS
    sigma = math.sqrt(ler * (1 - ler) / SHOTS + twin_std ** 2)
    # QLX digital_twin's default backend builds pymatching from the SAME
    # stim-emitted DEM (it ignores fq.decoder_config — Codex Gate-2 review
    # finding 3), so this is an independent-sampler comparison of the same
    # decoding pipeline; the tolerance covers pure sampling noise.
    assert abs(ler - twin_ler) < 4 * sigma, \
        (f"decsim LER {ler:.5f} vs QLX twin {twin_ler:.5f} beyond 4 sigma "
         f"({sigma:.5f}) — same decoder path, so this exceeds sampling noise")


def test_qlx_circuit_runs_through_the_full_engine(circuit, detector_rounds):
    """The full engine rejects QLX's detectorless logical mechanism."""
    from decsim.adapters.stim_device import StimDevice
    from decsim.codes import SurfaceCodeModel
    from decsim.message import Operation
    from decsim.mwpm_decoder import PyMatchingDecoder
    from decsim.planner import FixedRounds
    from decsim.schemes import SlidingWindowScheme

    class _ZeroLatency:
        def latency(self, job):
            return 1

    rounds = max(detector_rounds.values())          # 7 comparison rounds
    device = StimDevice(detector_rounds={1: detector_rounds})
    op = Operation(id=1, name="qlx-mem", qubits=(0,), clifford=True,
                   circuit=circuit)
    with pytest.raises(ValueError, match="detectorless logical"):
        simulate(RunSpec(
            ops=[op],
            num_units=4,
            rounds_policy=FixedRounds(rounds),
            code=SurfaceCodeModel(d=3),
            scheme=SlidingWindowScheme(),
            device=device,
            decoder=PyMatchingDecoder(_ZeroLatency()),
            seed=900,
        ), verbose=False)


def test_stim_device_no_override_path_unchanged(circuit):
    """Formal regression for the detector_rounds extension: on a
    coordinate-bearing circuit, a StimDevice WITHOUT an override and one
    with an override for a DIFFERENT key produce identical round maps."""
    from decsim.adapters.stim_device import StimDevice
    from decsim.message import Operation

    coord_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=6,
        after_clifford_depolarization=0.001)
    plain = StimDevice(seed=5)
    other = StimDevice(seed=5, detector_rounds={999: {0: 1}})
    for device in (plain, other):
        op = Operation(id=1, name="m", qubits=(0,), clifford=True,
                       circuit=coord_circuit)
        device.begin_operation(op, 6)
    assert plain._by_round[1] == other._by_round[1]
    assert (plain._detector_rounds_for_key(1, coord_circuit, 6)
            == other._detector_rounds_for_key(1, coord_circuit, 6)
            == StimDevice._detector_rounds(coord_circuit, 6))


def test_detection_rate_matches_twin(sampled):
    """Sampler-level agreement: detection fraction vs the twin's det_rate."""
    dets, _ = sampled
    twin = json.loads((DATA / "mem_surface_twin.json").read_text())
    twin_rate = float(twin["det_rate"].strip("'"))
    rate = float(np.asarray(dets, dtype=np.float64).mean())
    n_bits = dets.shape[0] * dets.shape[1]
    sigma = math.sqrt(max(rate * (1 - rate) / n_bits, 1e-12))
    assert abs(rate - twin_rate) < max(5 * sigma, 0.002), \
        f"det rate {rate:.5f} vs twin {twin_rate:.5f}"


def _terminal_schedule():
    names = (
        ["alloc", "prep_z", "reset", "reset", "h"]
        + ["cx"] * 8
        + ["h", "measure_syndrome", "h"]
        + ["cx"] * 8
        + ["h", "measure_syndrome", "h"]
        + ["cx"] * 8
        + ["h", "measure_syndrome", "mz", "dealloc"]
    )
    starts = ([0] * 15) + ([1] * 11) + ([2] * 11) + [3, 3]
    assert len(names) == len(starts) == 39
    entries = []
    for index, (name, start) in enumerate(zip(names, starts)):
        entries.append({
            "op_id": f"{name}_{index}",
            "op_name": f"fabric.{name}",
            "dependencies": (() if index == 0 else
                             (entries[index - 1]["op_id"],)),
            "occupied_slots": (("C0", 0),),
            "duration": 1 if name == "measure_syndrome" else 0,
            "consumes": None,
            "produces": None,
            "protocol": None,
            "start_round": start,
        })
    return {"entries": entries}


def _terminal_circuit_and_metadata():
    circuit = stim.Circuit("""
        M 0 1
        M 2 3
        M 4 5
        X_ERROR(0.25) 6
        M 6 7 8
        DETECTOR rec[-9]
        DETECTOR rec[-8]
        DETECTOR rec[-7] rec[-9]
        DETECTOR rec[-6] rec[-8]
        DETECTOR rec[-5] rec[-7]
        DETECTOR rec[-4] rec[-6]
        DETECTOR rec[-4] rec[-3] rec[-1]
        DETECTOR rec[-5] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-3]
    """)
    metadata = {
        "dem_num_detectors": 8,
        "dem_num_observables": 1,
        "dem_num_sx": 1,
        "dem_hx": [[1]],
        "dem_hz": [[0, 2]],
        "dem_detector_locs": [
            [0, 0, -1], [0, 1, -1],
            [1, 0, 0], [1, 1, 0],
            [2, 0, 1], [2, 1, 1],
            [-1, 0, 2], [-2, 0, 2],
        ],
        "dem_D_sparse": [[999]],
        "dem_O_sparse": [[999]],
        "dem_weights": [999.0],
    }
    return circuit, metadata


def _asymmetric_check_circuit_and_metadata():
    circuit = stim.Circuit()
    measurement_count = 13
    circuit.append("M", range(measurement_count))
    locations = []

    def add_detector(records, location):
        circuit.append("DETECTOR", [
            stim.target_rec(record - measurement_count)
            for record in records
        ])
        locations.append(location)

    check_count = 3
    for submission in range(3):
        for check in range(check_count):
            records = [submission * check_count + check]
            baseline = submission - 1
            if baseline >= 0:
                records.append(baseline * check_count + check)
            add_detector(records, [submission, check, baseline])

    add_detector([6, 9, 11], [-2, 0, 2])
    add_detector([7, 10], [-1, 0, 2])
    add_detector([8, 11, 12], [-1, 1, 2])
    metadata = {
        "dem_num_detectors": 12,
        "dem_num_observables": 0,
        "dem_num_sx": 1,
        "dem_hx": [[0, 2]],
        "dem_hz": [[1], [2, 3]],
        "dem_detector_locs": locations,
    }
    return circuit, metadata


def test_detector_routing_supports_asymmetric_check_counts():
    from decsim.frontends.qlx import _prove_detector_routing

    circuit, metadata = _asymmetric_check_circuit_and_metadata()
    rounds, terminal_ids, terminal_data_count = _prove_detector_routing(
        circuit, metadata, submission_count=3
    )
    assert terminal_ids == (9, 10, 11)
    assert terminal_data_count == 4
    assert rounds[8] == rounds[9] == 3


def test_detector_routing_rejects_inconsistent_x_check_split():
    from decsim.frontends.qlx import _prove_detector_routing

    circuit, metadata = _asymmetric_check_circuit_and_metadata()
    metadata["dem_num_sx"] = 2
    with pytest.raises(ValueError, match="dem_num_sx"):
        _prove_detector_routing(circuit, metadata, submission_count=3)


@pytest.mark.parametrize("value", [True, 1.5, -1])
def test_detector_routing_requires_nonnegative_exact_x_check_count(value):
    from decsim.frontends.qlx import _prove_detector_routing

    circuit, metadata = _asymmetric_check_circuit_and_metadata()
    metadata["dem_num_sx"] = value
    error = TypeError if type(value) is not int else ValueError
    with pytest.raises(error, match="dem_num_sx"):
        _prove_detector_routing(circuit, metadata, submission_count=3)


def test_detector_routing_requires_exact_list_check_matrices():
    from decsim.frontends.qlx import _prove_detector_routing

    circuit, metadata = _asymmetric_check_circuit_and_metadata()
    metadata["dem_hx"] = ([0, 2],)
    with pytest.raises(TypeError, match="dem_hx"):
        _prove_detector_routing(circuit, metadata, submission_count=3)


def _two_submission_metadata(locations):
    return {
        "dem_num_detectors": len(locations),
        "dem_num_observables": 0,
        "dem_num_sx": 0,
        "dem_hx": [],
        "dem_hz": [[]],
        "dem_detector_locs": locations,
    }


def test_detector_routing_rejects_future_baseline_submission():
    from decsim.frontends.qlx import _prove_detector_routing

    circuit = stim.Circuit("""
        M 0
        M 1
        DETECTOR rec[-2] rec[-1]
    """)
    metadata = _two_submission_metadata([[0, 0, 1]])

    with pytest.raises(ValueError, match="precede current submission"):
        _prove_detector_routing(circuit, metadata, submission_count=2)


def test_detector_routing_rejects_nonmonotonic_global_detector_order():
    from decsim.frontends.qlx import _prove_detector_routing

    circuit = stim.Circuit("""
        M 0
        M 1
        DETECTOR rec[-2] rec[-1]
        DETECTOR rec[-2]
    """)
    metadata = _two_submission_metadata([[1, 0, 0], [0, 0, -1]])

    with pytest.raises(ValueError, match="global detector order"):
        _prove_detector_routing(circuit, metadata, submission_count=2)


def _terminal_program():
    from decsim.frontends.qlx import qlx_frontend

    circuit, metadata = _terminal_circuit_and_metadata()
    return qlx_frontend(
        _terminal_schedule(),
        physical_circuit=circuit,
        detector_metadata=metadata,
        decode_operation_id=99,
    )


def test_exact_terminal_schedule_and_physical_lowering():
    program = _terminal_program()
    assert len(program.operations) == 39
    assert sum(program.raw_durations.values()) == 3
    assert sum(value == 0 for value in program.raw_durations.values()) == 36
    assert max(
        operation.scheduled_start_round + program.raw_durations[operation.id]
        for operation in program.operations
    ) == 3
    assert program.rounds_for(program.decoder_operations[0], None) == 3
    emitters = [op for op in program.operations if op.emits_detector_data]
    assert len(emitters) == 4
    assert [op.stream_offset for op in emitters] == [0, 1, 2, 2]
    assert [op.syndrome_fragment_index for op in emitters[-2:]] == [0, 1]


def test_decode_owner_id_cannot_collide_with_schedule_operation_id():
    from decsim.frontends.qlx import qlx_frontend

    circuit, metadata = _terminal_circuit_and_metadata()
    with pytest.raises(ValueError, match="collides"):
        qlx_frontend(
            _terminal_schedule(),
            physical_circuit=circuit,
            detector_metadata=metadata,
            decode_operation_id=0,
        )


def test_physical_lowering_rejects_nonunit_submission_duration():
    from decsim.frontends.qlx import qlx_frontend

    schedule = _terminal_schedule()
    measure = next(
        entry for entry in schedule["entries"]
        if entry["op_name"] == "fabric.measure_syndrome"
    )
    measure["duration"] = 2
    measure_index = schedule["entries"].index(measure)
    for entry in schedule["entries"][measure_index + 1:]:
        entry["start_round"] += 1
    circuit, metadata = _terminal_circuit_and_metadata()
    with pytest.raises(ValueError, match="unit-duration"):
        qlx_frontend(
            schedule,
            physical_circuit=circuit,
            detector_metadata=metadata,
            decode_operation_id=99,
        )


@pytest.mark.parametrize("field", ["locations", "terminal_parity"])
def test_physical_lowering_rejects_detector_routing_mutations(field):
    from decsim.frontends.qlx import qlx_frontend

    circuit, metadata = _terminal_circuit_and_metadata()
    metadata = json.loads(json.dumps(metadata))
    if field == "locations":
        metadata["dem_detector_locs"][0], metadata["dem_detector_locs"][1] = (
            metadata["dem_detector_locs"][1],
            metadata["dem_detector_locs"][0],
        )
    else:
        metadata["dem_hz"][0] = [1]
    with pytest.raises(ValueError, match="measurement records"):
        qlx_frontend(
            _terminal_schedule(),
            physical_circuit=circuit,
            detector_metadata=metadata,
            decode_operation_id=99,
        )


def test_terminal_lowering_rejects_mz_on_another_cell():
    from decsim.frontends.qlx import qlx_frontend

    schedule = _terminal_schedule()
    terminal = next(
        entry for entry in schedule["entries"]
        if entry["op_name"] == "fabric.mz"
    )
    terminal["occupied_slots"] = (("F0", 0),)
    circuit, metadata = _terminal_circuit_and_metadata()
    with pytest.raises(ValueError, match="dependent mz"):
        qlx_frontend(
            schedule, physical_circuit=circuit,
            detector_metadata=metadata, decode_operation_id=99,
        )


def test_sparse_dem_arrays_are_outside_the_consumed_contract():
    from decsim.frontends.qlx import qlx_frontend

    circuit, metadata = _terminal_circuit_and_metadata()
    changed = json.loads(json.dumps(metadata))
    changed["dem_D_sparse"] = [[1, 2, 3]]
    changed["dem_O_sparse"] = []
    changed["dem_weights"] = [-7.0]
    first = qlx_frontend(
        _terminal_schedule(), physical_circuit=circuit,
        detector_metadata=metadata, decode_operation_id=99,
    )
    second = qlx_frontend(
        _terminal_schedule(), physical_circuit=circuit,
        detector_metadata=changed, decode_operation_id=99,
    )
    assert first.detector_rounds_by_stream == second.detector_rounds_by_stream
    assert (first.terminal_detector_ids_by_stream
            == second.terminal_detector_ids_by_stream)


@pytest.mark.parametrize("scheme_name", ["sliding", "parallel"])
def test_terminal_stream_has_three_rounds_in_window_schemes(scheme_name):
    from decsim.adapters.stim_device import StimDevice
    from decsim.codes import SurfaceCodeModel
    from decsim.decoders import PresetLatencyDecoder
    from decsim.schemes import ParallelWindowScheme, SlidingWindowScheme

    program = _terminal_program()
    scheme = (SlidingWindowScheme() if scheme_name == "sliding"
              else ParallelWindowScheme())
    device = StimDevice(
        detector_rounds=program.detector_rounds_by_stream,
        terminal_detector_ids=program.terminal_detector_ids_by_stream,
        terminal_data_bits=program.terminal_data_bits_by_stream,
    )
    result = simulate(RunSpec(
        frontend=program, decode_ops=list(program.decoder_operations),
        rounds_policy=program, code=SurfaceCodeModel(d=3), scheme=scheme,
        device=device, decoder=PresetLatencyDecoder(0.0), seed=41,
    ), verbose=False)
    assert result.window_manager.rounds_for(program.decoder_operations[0]) == 3
    assert result.window_manager.window_count[99] == 1


def test_terminal_stream_matches_global_sample_and_decode():
    from decsim.adapters.stim_device import StimDevice
    from decsim.codes import SurfaceCodeModel
    from decsim.mwpm_decoder import PyMatchingDecoder
    from decsim.schemes import NaiveOnlineScheme

    class ZeroLatency:
        def latency(self, job):
            return 1

    program = _terminal_program()
    device = StimDevice(
        detector_rounds=program.detector_rounds_by_stream,
        terminal_detector_ids=program.terminal_detector_ids_by_stream,
        terminal_data_bits=program.terminal_data_bits_by_stream,
    )
    result = simulate(RunSpec(
        frontend=program,
        decode_ops=list(program.decoder_operations),
        rounds_policy=program,
        code=SurfaceCodeModel(d=3),
        scheme=NaiveOnlineScheme(),
        device=device,
        decoder=PyMatchingDecoder(ZeroLatency()),
        seed=37,
    ), verbose=False)

    matching = pymatching.Matching.from_detector_error_model(
        program.decoder_operations[0].circuit.detector_error_model()
    )
    expected = tuple(int(value) for value in matching.decode(device._dets[99]))
    assert result.window_manager.op_results[99] == expected
    assert len(device._dets[99]) == 8
    assert result.window_manager.rounds_arrived[99] == 3
    assert result.chip.stream_next_round[99] == 3
    transfers = result.result.link_traffic["transfers"]
    final_qc = [
        row for row in transfers
        if row["path"] == "qc" and row["attribution"]["round_lo"] == 3
    ]
    final_cwd = [
        row for row in transfers
        if row["path"] == "cwd" and row["attribution"]["round_lo"] == 3
    ]
    assert [row["payload_bits"] for row in final_qc] == [2, 3]
    assert [row["payload_bits"] for row in final_cwd] == [5]

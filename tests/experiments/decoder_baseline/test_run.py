"""The decoder baseline script against decsim's own workload.

The referent for the circuit is decsim.producers.memory_circuit, the
producer the machine's baseline yaml names, so the offline and the
machine baselines sample one circuit. The rest pins the owner's cap:
24 core-hours a point at its decoder's measured speed, and never more
than a billion shots.
"""

import importlib.util
import pathlib

import pytest

import decsim.producers as producers

pytest.importorskip("relay_bp")
pytest.importorskip("tesseract_decoder")

_THIS_FILE = pathlib.Path(__file__)
_TEST_FILE = _THIS_FILE.resolve()
REPOSITORY_ROOT = _TEST_FILE.parents[3]
SCRIPT = REPOSITORY_ROOT / "experiments" / "decoder_baseline" / "run.py"


def test_the_circuit_is_decsims_memory_circuit():
    run = script_module()

    circuit = run.circuit("z", 5, 0.001)

    workload = producers.memory_circuit(
        "surface_code:rotated_memory_z", 100, 5, 0.001
    )
    (operation,) = workload.operations
    assert str(circuit) == str(operation.circuit)


def test_the_baseline_has_a_point_per_basis_distance_rate_and_decoder():
    run = script_module()

    experiment = run.baseline()

    assert len(experiment.tasks) == 2 * 6 * 6 * 4


def test_a_points_shot_cap_is_a_day_of_its_decoders_shots():
    run = script_module()

    shot_cap = run.max_shots("relay-bp-1", 15)

    seconds_per_shot = run.SECONDS_PER_SHOT["relay-bp-1"][15]
    day_of_shots = 86400 / seconds_per_shot
    assert shot_cap == int(day_of_shots)


def test_no_point_is_capped_past_a_billion_shots(monkeypatch):
    run = script_module()
    monkeypatch.setitem(run.SECONDS_PER_SHOT["pymatching"], 5, 1e-6)

    shot_cap = run.max_shots("pymatching", 5)

    assert shot_cap == 1_000_000_000


def script_module():
    """experiments/decoder_baseline/run.py, imported, its main not run."""
    spec = importlib.util.spec_from_file_location("decoder_baseline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

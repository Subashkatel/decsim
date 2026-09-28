"""The decoder baseline script against decsim's own workload.

The referent for the circuit is decsim.producers.memory_circuit, the
producer the machine's baseline yaml names, so the offline and the
machine baselines sample one circuit. The rest pins the stop rule: 100
errors a point, and a shot limit no time limit reaches, so Slurm's time
limit is the budget.
"""

import importlib.util
import pathlib

import pytest

import decsim.producers as producers
import decsim.sinter_adapters.relay_bp as relay_bp_adapter

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


def test_every_point_stops_at_100_errors_or_a_billion_shots():
    run = script_module()

    experiment = run.baseline()

    limits = set()
    for task in experiment.tasks:
        options = task.collection_options
        limits.add((options.max_errors, options.max_shots))
    assert limits == {(100, 1_000_000_000)}


def test_a_relay_bp_point_decodes_with_decsims_row_built_on_its_circuit():
    run = script_module()

    experiment = run.baseline()

    for task, decoder in zip(
        experiment.tasks, experiment.decoders, strict=True
    ):
        if task.decoder != run.RELAY_BP:
            continue
        assert isinstance(
            decoder, relay_bp_adapter.RelayBeliefPropagationDecoder
        )
        assert decoder.circuit is task.circuit


def script_module():
    """experiments/decoder_baseline/run.py, imported, its main not run."""
    spec = importlib.util.spec_from_file_location("decoder_baseline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

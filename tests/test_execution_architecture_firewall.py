import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
import pytest

from decsim.message import ExecutionProgram, RunOperationBody, Operation
from decsim.qpu import QPUDevice
from decsim import RunSpec


def test_removed_coordinator_and_clocked_device_cannot_return():
    package = Path(__file__).parents[1] / "decsim"
    assert not (package / "chip.py").exists()
    names = {node.name for path in package.glob("*.py")
             for node in ast.walk(ast.parse(path.read_text()))
             if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    assert "Chip" not in names
    assert "ClockedDevice" not in names
    assert "ModularController" not in names


def test_program_and_qpu_command_are_immutable():
    program = ExecutionProgram((Operation(0, "m", (0,)),))
    command = RunOperationBody(program.operations[0], 1, 1, 1)
    with pytest.raises(FrozenInstanceError):
        program.operations = ()
    with pytest.raises(FrozenInstanceError):
        command.round_count = 2


def test_qpu_has_no_classical_runtime_backdoor():
    qpu = QPUDevice(object(), object(), object())
    assert set(vars(qpu)) == {"engine", "model", "readout_receiver", "completion_receiver"}
    for forbidden in ("window_manager", "decoder_manager", "factory", "links", "idle_policy", "program"):
        assert not hasattr(qpu, forbidden)


def test_execution_runtime_is_the_single_live_admission_owner():
    from decsim.controller import Controller
    from decsim.execution_runtime import ExecutionRuntime
    controller_fields = set(Controller.__init__.__code__.co_names)
    source = (Path(__file__).parents[1] / "decsim" / "controller.py").read_text()
    runtime_source = (Path(__file__).parents[1] / "decsim" / "execution_runtime.py").read_text()
    for field in ("body_done_time", "decode_release_time", "op_start_time",
                  "busy_claims", "dependencies_remaining"):
        assert f"self.{field}" not in source
        assert f"self.{field}" in runtime_source
    assert not hasattr(Controller, "on_decision")
    assert hasattr(ExecutionRuntime, "on_decision")


def test_controller_and_runtime_can_be_replaced_without_forwarding_aliases():
    completed = RunSpec(ops=[]).build(verbose=False)
    assert completed.execution_runtime.controller is completed.controller
    assert completed.controller.runtime is completed.execution_runtime
    assert "body_done_time" not in vars(completed.controller)
    assert "body_done_time" in vars(completed.execution_runtime)


def test_controller_is_the_only_qpu_readout_to_binary_boundary():
    package = Path(__file__).parents[1] / "decsim"
    controller_source = (package / "controller.py").read_text()
    qpu_source = (package / "qpu.py").read_text()
    assert "QPUReadout" in controller_source
    assert "SyndromePayload" in controller_source
    assert "relay_qpu_readout" in controller_source
    assert "SyndromePayload" not in qpu_source
    assert "relay_syndrome" not in qpu_source
    assert "QPUReadout" in qpu_source
    assert "self.staging" not in controller_source

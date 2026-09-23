"""The workload section: the kind's row reads its own keys.

STYLE.md rule 7: adding a component is one class, one table row and its
section in the yaml reference, and nothing else. WORKLOADS
(decsim/frontends/settings.py) is the table workload.kind names; the
tests below add a row from outside decsim and run it from a yaml, and
pin the sentence each Python-only shipped row refuses a yaml with.
"""

import dataclasses

import pytest

import decsim.experiments.experiment as experiment
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.qpu.round_policies as round_policies
import decsim.records.program as program_records
import tests.experiments.yaml_configs as yaml_configs


class TwoPatchMemory:
    """A workload row written outside decsim: two patches, fixed rounds."""

    has_frontend = False

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """This row's one key of its own, the rounds each patch runs."""

        patch_rounds: int = 1

        @classmethod
        def from_yaml(cls, section):
            return cls(**section)

    @staticmethod
    def operations(settings, code):
        """Two single-patch memory operations of the settings' rounds."""
        del code
        rounds = settings.row_settings.patch_rounds
        first = program_records.Operation(
            id=1, name="a", qubits=(1,), patches=(1,)
        )
        second = program_records.Operation(
            id=2, name="b", qubits=(2,), patches=(2,)
        )
        policy = round_policies.FixedRounds(rounds)
        return (first, second), policy


def test_a_workload_row_written_outside_decsim_runs_from_a_yaml(
    monkeypatch, tmp_path
):
    """One table row and one yaml kind is the whole edit."""
    monkeypatch.setitem(
        workload_settings.WORKLOADS, "two_patch_memory", TwoPatchMemory
    )
    workload = {"kind": "two_patch_memory", "patch_rounds": 6}
    card = {"workload": workload, "qpu": {"kind": "timing_only"}}
    config_path = yaml_configs.write_config(tmp_path, card)
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.001, distance=3, round_period_us=1.0
    )
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()

    assert result.terminal_status == "complete"
    assert len(machine.operations) == 2
    assert settings.workload.row_settings.patch_rounds == 6


def test_a_kind_off_the_table_is_refused_naming_the_rows(tmp_path):
    """The table's own refusal, at the yaml boundary."""
    workload = {"kind": "not_a_row", "rounds_per_shot": 6}
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})
    with pytest.raises(ValueError, match="workload.kind 'not_a_row' is not"):
        experiment.load_experiment(config_path)


@pytest.mark.parametrize("rounds_per_shot", [0, -1, True, "0d"])
def test_a_shot_of_fewer_than_one_round_is_refused_at_load(
    tmp_path, rounds_per_shot
):
    workload = {**yaml_configs.MINIMAL_CONFIG["workload"]}
    workload["rounds_per_shot"] = rounds_per_shot
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})
    with pytest.raises(ValueError, match="a round count of at least 1"):
        experiment.load_experiment(config_path)


def test_each_python_only_row_refuses_a_yaml_by_name(tmp_path):
    """A row a yaml cannot carry says what it needs and where to build it."""
    sentences = {
        "circuit_list": "a list of Operation records",
        "surgery_ir": "the qubit_to_patch mapping",
    }
    for kind, sentence in sentences.items():
        workload = {"kind": kind}
        config_path = yaml_configs.write_config(
            tmp_path, {"workload": workload}
        )
        with pytest.raises(ValueError, match=sentence):
            experiment.load_experiment(config_path)


def test_a_memory_circuit_key_off_its_list_is_refused():
    section = {
        "kind": "memory_circuit",
        "code_task": "surface_code:rotated_memory_z",
        "round_per_shot": 15,
    }
    sentence = (
        r"workload does not know \['round_per_shot'\]; its keys are "
        r"\['kind', 'code_task', 'rounds_per_shot'\]"
    )
    with pytest.raises(ValueError, match=sentence):
        workload_settings.WorkloadSettings.from_yaml(section)


def test_a_memory_circuit_key_left_out_is_refused_rather_than_defaulted():
    section = {
        "kind": "memory_circuit",
        "code_task": "surface_code:rotated_memory_z",
    }
    sentence = r"memory_circuit needs \['rounds_per_shot'\] beside kind"
    with pytest.raises(ValueError, match=sentence):
        workload_settings.WorkloadSettings.from_yaml(section)


def test_a_workload_without_a_kind_is_refused_naming_the_rows():
    """The one default kind is the dataclass's, for Python-built runs."""
    section = {
        "code_task": "surface_code:rotated_memory_z",
        "rounds_per_shot": 15,
    }
    with pytest.raises(ValueError, match="workload.kind None is not a row"):
        workload_settings.WorkloadSettings.from_yaml(section)

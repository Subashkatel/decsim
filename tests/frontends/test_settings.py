"""The workload section: the kind's row reads its own keys.

STYLE.md rule 7: adding a component is one class, one table row and its
section in the yaml reference, and nothing else. WORKLOADS
(decsim/frontends/settings.py) is the table workload.kind names; the
tests below add a row from outside decsim and run it from a yaml, and
pin the sentence each Python-only shipped row refuses a yaml with.
"""

import pytest

import decsim.front.experiment as experiment
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.qpu.round_policies as round_policies
import decsim.records.program as program_records
import tests.front.yaml_configs as yaml_configs


class TwoPatchMemory:
    """A workload row written outside decsim: two patches, fixed rounds."""

    has_frontend = False

    @staticmethod
    def from_yaml(section):
        """This row reads one key of its own, the rounds each patch runs."""
        return {
            "rounds_per_shot": workload_settings.RoundsPerShot(
                fixed=section["rounds_per_shot"]
            )
        }

    @staticmethod
    def operations(settings, code):
        """Two single-patch memory operations of the settings' rounds."""
        del code
        rounds = settings.rounds_per_shot.rounds_for(1)
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
    workload = {"kind": "two_patch_memory", "rounds_per_shot": 6}
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


def test_a_kind_off_the_table_is_refused_naming_the_rows(tmp_path):
    """The table's own refusal, at the yaml boundary."""
    workload = {"kind": "not_a_row", "rounds_per_shot": 6}
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})
    with pytest.raises(ValueError, match="workload.kind 'not_a_row' is not"):
        experiment.load_experiment(config_path)


def test_each_python_only_row_refuses_a_yaml_by_name(tmp_path):
    """A row a yaml cannot carry says what it needs and where to build it."""
    sentences = {
        "circuit_list": "a list of Operation records",
        "surgery_ir": "the qubit_to_patch mapping",
        "qlx": "a lowered QLX program object",
    }
    for kind, sentence in sentences.items():
        workload = {"kind": kind, "rounds_per_shot": 6}
        config_path = yaml_configs.write_config(
            tmp_path, {"workload": workload}
        )
        with pytest.raises(ValueError, match=sentence):
            experiment.load_experiment(config_path)

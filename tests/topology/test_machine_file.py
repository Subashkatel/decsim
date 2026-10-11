"""A machine written in YAML, and back (topology/machine_file.py).

The referent is the graph built in Python: every regression machine,
written out and read back, is the graph it was written from; a part's
field reads as `--set` reads it; a constant reads with its source; a
typo in either file is refused by name.
"""

import pathlib

import pytest

import decsim.parameters.fields as fields
import decsim.topology.legacy as legacy
import decsim.topology.machine_file as machine_file
import tests.experiments.run_files as run_files
import tests.regression.test_shot_results as regression

WORKLOAD = "workload"
SMALL_MACHINE = """
clock: 250MHz
parts:
  qpu: {type: QpuSettings, distance: 3, round_period: 1us}
  controller: {type: ControllerSettings, packing_per_round: 8ns}
  weak_buffer: {type: SyndromeBufferSettings, placed: fridge}
  weak:
    type: DecoderPoolSettings
    placed: fridge
    unit_count: ${union_find.units}
    algorithm: {type: PyMatchingDecoder, preset_latency: 28ns}
  frame: {type: PauliFrameConfig, write: 1 cycle}
wires:
  - qpu.readout -> controller.readout
  - controller.rounds -> weak_buffer.rounds
  - weak_buffer.reads -> weak.rounds
  - wire: weak.corrections -> frame.corrections
    link: {type: ClockedLink, latency: "${fiber.down}"}
  - frame.decisions -> controller.decisions
  - controller.pulses -> qpu.pulses
fabric:
  type: FridgeHostFabric
  cable: {type: RoundTripLeg, round_trip: 2.305us, echo_payload: 128 bits,
          capacity: 100Gbps}
  host_local: {type: TimedLink}
observation: {log: off}
"""
CONSTANTS = """
union_find:
  units: 2  # two units, an estimate
fiber:
  down: 41 cycles  # 2603.16203 lines 903-904
"""


def _constants(tmp_path: pathlib.Path, text: str = CONSTANTS):
    path = tmp_path / "constants.yaml"
    path.write_text(text)
    return machine_file.read_constants(path)


@pytest.mark.parametrize("name", list(regression.SHAPES))
def test_every_regression_machine_written_out_reads_back(name):
    build = regression.SHAPES[name]
    machine = build()
    workload = machine.part(WORKLOAD)

    text = machine_file.dumps(machine)
    read = machine_file.loads(text, python_parts={WORKLOAD: workload})

    assert f"{WORKLOAD}: python" in text
    assert read.parts == machine.parts
    assert read == machine


def test_a_file_holds_only_what_differs_from_the_defaults():
    settings = run_files.minimal_machine()
    machine = legacy.graph_of(settings)

    text = machine_file.dumps(machine)

    assert "unit_count" not in text
    assert "type: DecoderPool\n" in text


def test_a_part_reads_its_fields_and_constants_as_set_reads_them(tmp_path):
    constants = _constants(tmp_path)

    machine = machine_file.loads(SMALL_MACHINE, constants)

    pool = machine.part("weak")
    controller = machine.part("controller")
    corrections = machine.wires[3]
    assert pool.settings.unit_count == 2
    assert pool.placed == "fridge"
    assert controller.settings.packing_cycles_per_round == 2
    assert corrections.link.latency_cycles == 41
    assert machine.observation.log == "off"
    assert constants.sources["fiber.down"] == "2603.16203 lines 903-904"
    assert machine.unbound_ports() == []


def test_a_constant_the_machine_never_reads_is_refused(tmp_path):
    machine_path = tmp_path / "machine.yaml"
    machine_path.write_text(SMALL_MACHINE)
    constants_path = tmp_path / "constants.yaml"
    unread = CONSTANTS + "spare: 3  # read by nothing\n"
    constants_path.write_text(unread)

    with pytest.raises(fields.ParameterError, match=r"\['spare'\]"):
        machine_file.load(machine_path, constants_path)


def test_a_misspelt_constant_is_refused_with_the_near_names(tmp_path):
    constants = _constants(tmp_path)
    misspelt = SMALL_MACHINE.replace("fiber.down", "fiber.dwn")

    with pytest.raises(fields.ParameterError, match="fiber.down"):
        machine_file.loads(misspelt, constants)


def test_a_misspelt_field_is_refused_with_the_near_names(tmp_path):
    constants = _constants(tmp_path)
    misspelt = SMALL_MACHINE.replace("unit_count", "unit_cont")

    with pytest.raises(fields.ParameterError, match="unit_count"):
        machine_file.loads(misspelt, constants)


def test_a_record_with_no_type_is_refused(tmp_path):
    constants = _constants(tmp_path)
    untyped = SMALL_MACHINE.replace("type: PauliFrameConfig, ", "")

    with pytest.raises(fields.ParameterError, match="names its type"):
        machine_file.loads(untyped, constants)


def test_yaml_1_2_reads_off_as_a_word_and_5e_4_as_a_number():
    document = machine_file._parsed("a: off\nb: 5e-4\nc: 0123\n")

    assert document == {"a": "off", "b": 0.0005, "c": 123}

"""The parameter reference stays true or these tests fail.

One home per fact: experiments/configs/reference.yaml is the canonical
yaml-key listing (and stays runnable); guide/parameter-reference.md is the
canonical RunSpec table. Change a knob, and the failing assertion here
names the file you owe an update to.
"""

import dataclasses
from pathlib import Path

import yaml

from decsim.run_spec import RunSpec
from experiments.build_run import build_run
from experiments.experiment_config import load_experiment
from experiments.measure_shot import measure_shot

REPOSITORY = Path(__file__).parent.parent.parent
REFERENCE_YAML = REPOSITORY / "experiments/configs/reference.yaml"
REFERENCE_DOC = REPOSITORY / "guide/parameter-reference.md"


def test_reference_config_loads_and_runs():
    config = load_experiment(REFERENCE_YAML)
    shot = measure_shot(config, physical_error_probability=0.001,
                        round_period_us=1.0, algorithm_latency_us=0.028,
                        seed=0)
    assert shot.windows >= 1 and not shot.direct_mismatch


def test_buffer_size_keys_reach_the_run_spec(tmp_path):
    raw = yaml.safe_load(REFERENCE_YAML.read_text())
    raw["buffers"] = {"buffer_0_size": 30, "buffer_1_size": 40,
                      "packing_workspace_size": 50}
    raw["decoder"]["unit_buffer_size"] = 24
    config_path = tmp_path / "finite_buffers.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    spec, _ = build_run(load_experiment(config_path),
                        physical_error_probability=0.001,
                        round_period_us=1.0, algorithm_latency_us=0.028,
                        seed=0)
    assert spec.syndrome_buffering.upstream_packet_slots == 30
    assert spec.syndrome_buffering.sb1_packet_slots == 40
    assert spec.syndrome_buffering.packing_assembly_slots == 50
    assert spec.decoder_memory.capacity_for("default") == 24


def test_trace_io_narrates_every_component_without_changing_the_run(tmp_path):
    """The I/O trace adds receive/hold/emit lines for each store and unit,
    and nothing else: every measured field stays identical."""
    import dataclasses

    raw = yaml.safe_load(REFERENCE_YAML.read_text())
    measurements = {}
    for io_trace in (False, True):
        raw["trace_io"] = io_trace
        config_path = tmp_path / f"io_{io_trace}.yaml"
        config_path.write_text(yaml.safe_dump(raw))
        measurements[io_trace] = measure_shot(
            load_experiment(config_path), physical_error_probability=0.001,
            round_period_us=1.0, algorithm_latency_us=0.028, seed=0)
    changed = [field.name for field in dataclasses.fields(measurements[False])
               if field.name != "sim_wall_seconds"
               and getattr(measurements[False], field.name)
               != getattr(measurements[True], field.name)]
    assert not changed, f"trace_io changed measured fields: {changed}"

    raw["trace_io"] = True
    config_path = tmp_path / "io_lines.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    spec, _ = build_run(load_experiment(config_path),
                        physical_error_probability=0.001, round_period_us=1.0,
                        algorithm_latency_us=0.028, seed=0)
    lines = spec.build(io_trace=True).engine.log_lines
    assert any("Buffer 0: received round" in line for line in lines)
    assert any("SRAM: receiving" in line for line in lines)
    assert any("SRAM: memory" in line and "input landed" in line for line in lines)
    assert any("SRAM: emitted" in line for line in lines)
    assert any("PauliFrame: committed window" in line for line in lines)
    spec, _ = build_run(load_experiment(config_path),
                        physical_error_probability=0.001, round_period_us=1.0,
                        algorithm_latency_us=0.028, seed=0)
    quiet_lines = spec.build(io_trace=False).engine.log_lines
    assert not any("SRAM" in line for line in quiet_lines)


def test_reference_config_lists_every_key_the_shipped_configs_use():
    reference_keys = set(yaml.safe_load(REFERENCE_YAML.read_text()))
    for name in ("weak_baseline", "strong_only"):
        config_path = REFERENCE_YAML.parent / f"{name}.yaml"
        used_keys = set(yaml.safe_load(config_path.read_text()))
        missing = used_keys - reference_keys
        assert not missing, (
            f"{name}.yaml uses keys absent from reference.yaml: {missing}")


def test_reference_doc_names_every_runspec_field():
    doc = REFERENCE_DOC.read_text()
    public_fields = [field.name for field in dataclasses.fields(RunSpec)
                     if not field.name.startswith("_")]
    undocumented = [name for name in public_fields
                    if f"`{name}`" not in doc]
    assert not undocumented, (
        f"guide/parameter-reference.md is stale; add rows for: "
        f"{undocumented}")


def test_reference_doc_names_every_yaml_key():
    doc = REFERENCE_DOC.read_text()
    reference_keys = set(yaml.safe_load(REFERENCE_YAML.read_text()))
    unmentioned = [key for key in sorted(reference_keys)
                   if f"`{key}" not in doc and key not in doc]
    assert not unmentioned, (
        f"guide/parameter-reference.md never mentions yaml keys: "
        f"{unmentioned}")

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

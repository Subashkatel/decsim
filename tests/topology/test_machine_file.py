"""A machine written out as YAML (topology/machine_file.py).

The referent is the graph built in Python: every regression machine is
written out with its workload as a row, and a file holds only the
fields that differ from their defaults.
"""

import pytest

import decsim.topology.machine_file as machine_file
import tests.experiments.run_files as run_files
import tests.regression.test_shot_results as regression

WORKLOAD = "workload"


@pytest.mark.parametrize("name", list(regression.SHAPES))
def test_every_regression_machine_is_written_out(name):
    """Its workload is a memory row, written out as the rest is."""
    build = regression.SHAPES[name]
    machine = build()

    text = machine_file.dumps(machine)

    assert f"{WORKLOAD}:\n    type: memory" in text


def test_a_file_holds_only_what_differs_from_the_defaults():
    machine = run_files.minimal_machine()

    text = machine_file.dumps(machine)

    assert "unit_count" not in text
    assert "type: DecoderPool\n" in text

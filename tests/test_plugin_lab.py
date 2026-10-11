"""A lab's own decoder and verdict, end to end (tests/plugin_lab/).

The parts live outside decsim and are named by import path. They are
set from the command line and from YAML, listed by info and written by
describe; the verdict's cut is a sweep axis; their statistics reach the
run's tables; and the decoder's draws repeat for a seed and differ
between seeds. Appendix C of the design (notes/arch_redesign/design.md)
claims each of these.
"""

import csv

import pytest

import decsim.confidence.complementary as complementary
import decsim.confidence.no_signal as no_signal
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.collect as collect
import decsim.experiments.command as command
import decsim.parameters.assignments as assignments
import decsim.parameters.in_force as in_force
import decsim.presets.link_cards as link_cards
import decsim.presets.machines as machines
import decsim.topology.machine_file as machine_file
import tests.plugin_lab.parts as parts

DECODER_PATH = "tests.plugin_lab.parts.WeightedMatching.Settings"
VERDICT_PATH = "tests.plugin_lab.parts.WeightCut.Settings"
CUT = "switching.verdict.cut"
# a graph of two pools runs as the MachineSettings it resolves to, so
# its decoder's statistics are at the slot's path (record_statistics.md)
DECODER_STATISTICS = "weak_decoder.algorithm"
THRESHOLD = threshold_sources.FixedThreshold.Settings(20.0)
RUN_FILE = """
import decsim
import decsim.experiments.sweeps as sweeps
import tests.test_plugin_lab as plugin_tests

sweep = decsim.Sweep(
    "lab",
    plugin_tests.lab_machine,
    {"switching.verdict.cut": [0.0, 1000000.0]},
    "cut{switching.verdict.cut}",
    stop=decsim.CollectionSettings(max_shots=2),
)
experiment = sweep.experiment()
"""


@pytest.fixture(autouse=True)
def _no_assignments_left_behind(monkeypatch):
    """Each test starts with DECSIM_SET empty."""
    monkeypatch.setenv(assignments.ENVIRONMENT_VARIABLE, "")


def lab_machine(physical_error_probability: float = 0.02):
    """A d = 3 memory: the lab's decoder, escalating on the lab's verdict."""
    base = machines.weak_baseline(3, physical_error_probability, 1.0)
    algorithm = parts.WeightedMatching.Settings()
    weak = base.with_settings(machines.WEAK_POOL, algorithm=algorithm)
    confidence = no_signal.NoSignal.Settings()
    verdict = parts.WeightCut.Settings()
    switching = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=THRESHOLD, verdict=verdict
    )
    fabric = link_cards.strong_side(base.fabric)
    return machines.with_strong_tier(
        weak, machines.STRONG_BASELINE_POOL, switching, fabric
    )


def test_the_lab_parts_are_set_by_import_path_on_the_command_line():
    plain = _gap_switching()
    texts = [
        f"weak.algorithm.type={DECODER_PATH}",
        "switching.confidence.type=none",
        f"switching.verdict.type={VERDICT_PATH}",
        "switching.verdict.cut=2.5",
    ]
    parsed = []
    for text in texts:
        assignment = assignments.parse(text, "--set")
        parsed.extend(assignment)

    with in_force.in_force(parsed):
        task = collect.Task("lab", plain)

    pool = task.graph.part(machines.WEAK_POOL)
    edge = task.graph.part(machines.SWITCHING)
    assert pool.settings.algorithm == parts.WeightedMatching.Settings()
    assert edge.settings.verdict == parts.WeightCut.Settings(cut=2.5)


def test_the_lab_machine_reads_back_from_yaml_by_import_path():
    machine = lab_machine()

    text = machine_file.dumps(machine)
    read = machine_file.loads(text)

    assert DECODER_PATH in text
    assert VERDICT_PATH in text
    assert read == machine


def test_info_and_describe_name_the_lab_parts(tmp_path, capsys):
    run_file = tmp_path / "lab_run.py"
    run_file.write_text(RUN_FILE)

    command.main(["info", DECODER_PATH])
    command.main(["describe", str(run_file)])

    printed = capsys.readouterr()
    assert f"weighted_matching: {DECODER_PATH}" in printed.out
    assert f"type: {VERDICT_PATH}" in printed.out


def test_the_verdicts_cut_is_a_sweep_axis_and_its_stats_are_tabled(
    tmp_path,
):
    run_file = tmp_path / "lab_run.py"
    run_file.write_text(RUN_FILE)
    out_dir = tmp_path / "out"

    command.main(["run", str(run_file), "--out", str(out_dir)])

    summary_path = out_dir / "stat_summary.csv"
    totals = _totals(summary_path)
    escalated_at_zero = totals.get(("0.0", "switching.verdict", "escalated"))
    escalated_at_most = totals.get(
        ("1000000.0", "switching.verdict", "escalated")
    )
    assert escalated_at_zero > 0
    assert escalated_at_most in (None, 0.0)
    assert totals["0.0", DECODER_STATISTICS, "weighted"] > 0


def test_the_lab_decoders_draws_repeat_for_a_seed_and_differ_between():
    machine = lab_machine()
    task = collect.Task("lab", machine)

    first = _draws(task, 0)
    again = _draws(task, 0)
    other = _draws(task, 1)

    assert first
    assert first == again
    assert first != other


def _totals(path) -> dict:
    """(cut, path, statistic): a count's total, from stat_summary.csv."""
    with path.open() as summary:
        reader = csv.DictReader(summary)
        rows = list(reader)
    totals = {}
    for row in rows:
        if row["measure"] != "total":
            continue
        key = (row[CUT], row["path"], row["statistic"])
        totals[key] = float(row["value"])
    return totals


def _gap_switching():
    """A d = 3 memory escalating on the complementary gap."""
    base = machines.weak_baseline(3, 0.02, 1.0)
    confidence = complementary.ComplementaryGap.Settings()
    switching = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=THRESHOLD
    )
    fabric = link_cards.strong_side(base.fabric)
    return machines.with_strong_tier(
        base, machines.STRONG_BASELINE_POOL, switching, fabric
    )


def _draws(task: collect.Task, seed: int) -> dict:
    shot = collect.run_shot(task, seed)
    groups = shot.machine.observation.statistics.groups
    return groups[DECODER_STATISTICS].draws.value_counts()

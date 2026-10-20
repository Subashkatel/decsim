"""A SWIPER .lss schedule as a decsim workload (frontends/lattice_surgery.py).

The referent is SWIPER's own reading (github.com/jviszlai/swiper, commit
1c00e09) of two of its cached schedules, tests/data/swiper_toffoli.lss
and swiper_msd_15to1.lss: tools/swiper_referent.py wrote each one's
rounds at d = 7, its LatticeSurgerySchedule.to_dag edges, the
instructions DeviceManager sets to 0 rounds at seeds 0, 1 and 2, each
instruction's merge faces and whether WindowBuilder decodes its rounds
into the json beside it. The hand-built T gate is one gate as SWIPER's
compiler writes it (lattice_surgery_schedule.py inject_T, merge, S).
"""

import json
import pathlib

import pytest

import decsim.config as config
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.frontends.lattice_surgery as lattice_surgery
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.records.program as program_records
import tests.declared_run as declared_run

HERE = pathlib.Path(__file__)
DATA = HERE.parents[1] / "data"
DISTANCE = 7
# msd_15to1's last discard waits for its fifteen Y measurements to end,
# an edge SWIPER keeps apart from its DAG (conditioned_on_completion_idx)
MSD_COMPLETIONS = frozenset((index, 129) for index in range(99, 114))
SCHEDULES = (("toffoli", frozenset()), ("msd_15to1", MSD_COMPLETIONS))
T_GATE = """\
INJECT_T 0 [(0,1)] D [] [] [] [] [] [] none
MERGE 1 [(0,0),(0,1)] D [] [3,4,5,6] [] [] [((0,0),(0,1))] [] none
DISCARD 2 [(0,1)] 0 [] [] [] [] [] [] none
MERGE 3 [(0,0),(1,0)] D [1] [] [] [] [((0,0),(1,0))] [3,4,5,6] CONDITIONAL_S
Y_MEAS 4 [(1,0)] HALF_D_PLUS_2 [1] [] [] [] [] [3,4,5,6] CONDITIONAL_S
IDLE 5 [(0,0)] HALF_D_PLUS_2 [1] [] [] [] [] [3,4,5,6] CONDITIONAL_S
DISCARD 6 [(1,0)] 0 [1] [] [] [] [] [3,4,5,6] CONDITIONAL_S
DISCARD 7 [(0,0)] 0 [] [] [] [] [] [] none
"""
CONDITIONAL_S = frozenset({3, 4, 5, 6})


def swiper_reading(name: str) -> dict:
    path = DATA / f"swiper_{name}_d{DISTANCE}.json"
    text = path.read_text()
    return json.loads(text)


def rounds_in_order(workload) -> list:
    """Each operation's rounds, in the schedule's order."""
    counts = dict(workload.round_counts)
    return [counts[operation.id] for operation in workload.operations]


def faces_in_order(workload) -> list:
    """Each operation's couplings as json lists, in the schedule's order."""
    return [
        _json_pairs(operation.couplings) for operation in workload.operations
    ]


def _json_pairs(couplings: tuple) -> list:
    return [[list(patch) for patch in pair] for pair in couplings]


def decoded_with_rounds(reading: dict) -> list:
    """Per instruction, whether SWIPER decodes it and it runs rounds."""
    rounds = reading["rounds"]
    runs = [count > 0 for count in rounds]
    pairs = zip(reading["decoded"], runs, strict=True)
    return [all(pair) for pair in pairs]


def lowered_edges(workload) -> set:
    """Every (predecessor, operation) edge once the workload is lowered."""
    settings = workload_settings.WorkloadSettings.running(workload)
    edges = set()
    for operation in settings.operations:
        pairs = [(before, operation.id) for before in operation.predecessors]
        edges.update(pairs)
    return edges


@pytest.mark.parametrize("name, completions", SCHEDULES)
def test_a_schedule_takes_swipers_rounds_and_dependencies(name, completions):
    path = DATA / f"swiper_{name}.lss"
    reading = swiper_reading(name)

    workload = lattice_surgery.read_schedule(path, DISTANCE)

    assert rounds_in_order(workload) == reading["rounds"]
    swiper_edges = {tuple(edge) for edge in reading["edges"]}
    assert lowered_edges(workload) == swiper_edges | completions


@pytest.mark.parametrize("name", ("toffoli", "msd_15to1"))
def test_a_merge_couples_the_patches_of_swipers_merge_faces(name):
    path = DATA / f"swiper_{name}.lss"
    reading = swiper_reading(name)

    workload = lattice_surgery.read_schedule(path, DISTANCE)

    assert faces_in_order(workload) == reading["merge_faces"]


@pytest.mark.parametrize("name", ("toffoli", "msd_15to1"))
def test_an_operation_emits_the_rounds_swipers_window_builder_decodes(name):
    path = DATA / f"swiper_{name}.lss"
    reading = swiper_reading(name)

    workload = lattice_surgery.read_schedule(path, DISTANCE)

    emitting = [op.emits_detector_data for op in workload.operations]
    assert emitting == decoded_with_rounds(reading)


@pytest.mark.parametrize("seed", (0, 1, 2))
@pytest.mark.parametrize("name", ("toffoli", "msd_15to1"))
def test_the_conditional_s_skipped_at_a_seed_is_the_one_swiper_skips(
    name, seed
):
    path = DATA / f"swiper_{name}.lss"
    reading = swiper_reading(name)
    instructions = lattice_surgery.read_instructions(path)

    skipped = lattice_surgery.swiper_skipped_conditional_s(instructions, seed)
    workload = lattice_surgery.read_schedule(path, DISTANCE, skipped)

    counts = workload.round_counts
    zeroed = [index for index, rounds in counts if rounds == 0]
    assert sorted(zeroed) == reading["zeroed_by_seed"][str(seed)]


@pytest.fixture
def t_gate_path(tmp_path):
    path = tmp_path / "t_gate.lss"
    path.write_text(T_GATE)
    return path


def test_a_t_gates_conditional_s_waits_on_the_t_merge(t_gate_path):
    workload = lattice_surgery.read_schedule(t_gate_path, DISTANCE)

    by_id = {operation.id: operation for operation in workload.operations}
    blocked = {key for key, op in by_id.items() if op.blocked_by == 1}
    assert blocked == CONDITIONAL_S
    assert by_id[0].kind == program_records.OpKind.INJECT
    assert by_id[0].needs_magic_state
    assert by_id[2].kind == program_records.OpKind.DISCARD
    emitting = {key for key, op in by_id.items() if op.emits_detector_data}
    assert emitting == {1, 3, 4, 5}


def test_a_skipped_conditional_s_keeps_its_wait_and_emits_nothing(
    t_gate_path,
):
    workload = lattice_surgery.read_schedule(
        t_gate_path, DISTANCE, skipped=CONDITIONAL_S
    )

    skipped = [op for op in workload.operations if op.id in CONDITIONAL_S]
    blockers = {operation.blocked_by for operation in skipped}
    emitting = [operation.emits_detector_data for operation in skipped]
    assert blockers == {1}
    assert not any(emitting)


@pytest.mark.parametrize(
    "line, words",
    [
        (
            "TELEPORT 0 [(0,0)] D [] [] [] [] [] [] none",
            "unknown schedule instruction 'TELEPORT'",
        ),
        ("IDLE 0 [(0,0)] D [] [] []", "a schedule line has 11 fields, got 7"),
        (
            "IDLE 3 [(0,0)] D [] [] [] [] [] [] none",
            "instruction 3 sits at position 0",
        ),
    ],
)
def test_a_line_swiper_never_writes_is_refused(tmp_path, line, words):
    path = tmp_path / "bad.lss"
    path.write_text(line)

    with pytest.raises(ValueError, match=words):
        lattice_surgery.read_schedule(path, DISTANCE)


def test_only_a_conditional_s_group_is_skipped(t_gate_path):
    with pytest.raises(
        ValueError,
        match=r"only a conditional S group is skipped; \[1\] are not in one",
    ):
        lattice_surgery.read_schedule(t_gate_path, DISTANCE, skipped={1})


def test_the_conditional_s_starts_after_the_t_merge_is_decoded(t_gate_path):
    """Timing only: the S group's merge stalls on the T merge's decode."""
    workload = lattice_surgery.read_schedule(t_gate_path, DISTANCE)
    settings = workload_settings.WorkloadSettings.running(workload)
    algorithm = decoders.PresetLatencyDecoder.Settings(5.0)
    pool = decoder_settings.DecoderPoolSettings(
        algorithm=algorithm, engine=declared_run.DECLARED_ENGINE
    )
    base = machines.base(settings)
    graph = machines.with_pool(base, pool)

    machine = machine_module.Machine.build(graph, 0)
    result = machine.run()

    stalls = {stall.operation_id: stall for stall in result.gate_stalls}
    assert set(stalls) == CONDITIONAL_S
    merge = stalls[3]
    stamps = machine.observation.runtime_stamps
    assert merge.blocker_done_ticks == stamps.body_done[1]
    assert merge.started_ticks > merge.released_ticks
    five_microseconds = config.microseconds_to_ticks(5.0)
    assert merge.reaction_ticks > five_microseconds
    assert merge.stall_ticks > 0

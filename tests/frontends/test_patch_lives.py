"""Each patch's life as one decode stream (frontends/patch_lives.py).

The referent is SWIPER's own device (github.com/jviszlai/swiper,
swiper/device_manager.py) run at seed 0 on two of its cached schedules:
tools/swiper_referent.py wrote each patch life it ran, the instructions
that measured rounds in it and its rounds by kind, into the json beside
each schedule. The hand-built T gate is the one in test_lattice_surgery.py.
"""

import json
import pathlib

import pytest

import decsim.frontends.lattice_surgery as lattice_surgery
import decsim.frontends.patch_lives as patch_lives
import decsim.frontends.settings as workload_settings
import decsim.records.program as program_records
import decsim.records.workload as workload_records

HERE = pathlib.Path(__file__)
DATA = HERE.parents[1] / "data"
DISTANCE = 7
SEED = 0
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
# each patch's life, as a discard ends it
T_GATE_LIVES = (
    program_records.PatchLife(("patch", (0, 1), 0), (0, 1), (0, 1, 2)),
    program_records.PatchLife(("patch", (1, 0), 0), (1, 0), (3, 4, 6)),
    program_records.PatchLife(("patch", (0, 0), 0), (0, 0), (1, 3, 5, 7)),
)


def lowered_settings(workload):
    return workload_settings.WorkloadSettings.running(workload)


def seed_zero_workload(name: str):
    """The schedule with the conditional S SWIPER skips at seed 0."""
    path = DATA / f"swiper_{name}.lss"
    instructions = lattice_surgery.read_instructions(path)
    skipped = lattice_surgery.swiper_skipped_conditional_s(instructions, SEED)
    return lattice_surgery.read_schedule(path, DISTANCE, skipped)


def swiper_lives(name: str) -> dict:
    """SWIPER's lives at seed 0, by patch and the instructions run in it."""
    path = DATA / f"swiper_{name}_d{DISTANCE}.json"
    text = path.read_text()
    reading = json.loads(text)
    lives = {}
    for life in reading["patch_lives"]:
        key = _life_key(life["patch"], life["instructions"])
        lives[key] = life["rounds"]
    return lives


def decsim_lives(workload) -> dict:
    """The lowered lives, by patch and the instructions with rounds in it.

    Each maps to the rounds its decoded instructions add to its stream.
    """
    counts = dict(workload.round_counts)
    settings = lowered_settings(workload)
    operation_by_id = {op.id: op for op in settings.operations}
    lives = {}
    for life in settings.patch_lives:
        ran = [index for index in life.operation_ids if counts[index]]
        key = _life_key(life.patch, ran)
        lives[key] = _decoded_rounds(life, operation_by_id, counts)
    return lives


def instruction_rounds_of(swiper: dict) -> dict:
    """Each SWIPER life's rounds of instructions, T injections left out."""
    pairs = swiper.items()
    return {key: rounds.get("instruction", 0) for key, rounds in pairs}


def _life_key(patch, instructions) -> tuple:
    return (tuple(patch), tuple(sorted(instructions)))


def _decoded_rounds(life, operation_by_id: dict, counts: dict) -> int:
    rounds = []
    for index in life.operation_ids:
        if operation_by_id[index].emits_detector_data:
            rounds.append(counts[index])
    return sum(rounds)


def test_a_t_gates_patches_live_until_their_discards(tmp_path):
    path = tmp_path / "t_gate.lss"
    path.write_text(T_GATE)
    workload = lattice_surgery.read_schedule(path, DISTANCE)

    settings = lowered_settings(workload)

    assert settings.patch_lives == T_GATE_LIVES
    owners = [owner.id for owner in settings.dynamic_streams]
    assert owners == [life.stream_id for life in T_GATE_LIVES]


def test_a_life_whose_operations_emit_nothing_has_no_stream(tmp_path):
    path = tmp_path / "t_gate.lss"
    path.write_text(T_GATE)
    workload = lattice_surgery.read_schedule(
        path, DISTANCE, skipped=CONDITIONAL_S
    )

    settings = lowered_settings(workload)

    patches = [life.patch for life in settings.patch_lives]
    assert patches == [(0, 1), (0, 0)]


@pytest.mark.parametrize("name", ("toffoli", "msd_15to1"))
def test_the_lives_are_swipers_with_its_instruction_rounds(name):
    workload = seed_zero_workload(name)
    swiper = swiper_lives(name)

    lives = decsim_lives(workload)

    assert set(lives) == set(swiper)
    assert lives == instruction_rounds_of(swiper)


def test_a_patch_stream_workload_runs_no_circuit():
    operation = program_records.Operation(1, "merge", (0,), patches=(0,))
    operations = (operation,)
    circuit = object()
    workload = workload_records.Workload(
        operations, patch_streams=True, physical=circuit
    )

    with pytest.raises(ValueError, match="timing only: no circuit"):
        lowered_settings(workload)


def test_an_owner_measures_its_patch_open_ended():
    life = program_records.PatchLife(("patch", (2, 3), 1), (2, 3), (4, 5))

    owner = patch_lives.owner(life)

    assert owner.id == ("patch", (2, 3), 1)
    assert owner.patches == ((2, 3),)
    assert owner.kind is program_records.OpKind.MEMORY

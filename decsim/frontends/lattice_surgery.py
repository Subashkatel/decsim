"""A lattice-surgery schedule in SWIPER's .lss form, as a decsim workload.

SWIPER (Viszlai et al. 2412.05115, github.com/jviszlai/swiper) compiles
a Clifford+T circuit through Gridsynth and the Lattice Surgery Compiler
into a list of instructions on grid patches (swiper/
lattice_surgery_schedule.py), one per line:

    name idx patches duration conditioned_on_idx conditional_dependencies
    conditioned_on_completion_idx conditional_completion_dependencies
    merge_faces group_instr_indices group_name

read_schedule turns each instruction into an operation, with SWIPER's
device rules (swiper/device_manager.py _update_active_instructions):
  - an instruction follows the last one on each of its patches, the
    patch order circuit_frontend.lowered gives every workload;
  - one conditioned on an instruction (the conditional S after a T
    injection) waits until that instruction is decoded: blocked_by;
  - one conditioned on an instruction's completion waits for it to end:
    a predecessor;
  - durations D, HALF_D and HALF_D_PLUS_2 are d, d // 2 and d // 2 + 2
    rounds (Duration.get_true_duration), a DISCARD 0 rounds;
  - a merge couples the patches of its merge faces;
  - a DISCARD ends the patch's life (DeviceManager
    _get_initialized_patches);
  - a T injection's rounds run but are not decoded (swiper/
    window_builder.py build_windows), so it emits no detector data.
A conditional S is needed half the time (SWIPER Sec. 4.1): the skipped
ones keep their wait on the decode and take no rounds, as SWIPER sets
their durations to 0 (device_manager.py, its __init__).
swiper_skipped_conditional_s draws them as SWIPER does from a seed.
The schedule is timing only: no instruction carries a circuit. Each
patch's life is decoded as one stream (frontends/patch_lives.py), as
SWIPER's window builder cuts its regions per patch.
"""

import ast
import dataclasses
import pathlib

import numpy as np

import decsim.records.program as program_records
import decsim.records.workload as workload_records

# SWIPER's instruction names, as decsim's operation kinds
KINDS = {
    "MERGE": program_records.OpKind.MERGE,
    "INJECT_T": program_records.OpKind.INJECT,
    "Y_MEAS": program_records.OpKind.MEASURE,
    "IDLE": program_records.OpKind.IDLE,
    "DISCARD": program_records.OpKind.DISCARD,
}
FIELD_COUNT = 11


@dataclasses.dataclass(frozen=True)
class Instruction:
    """The fields of one .lss line read here, its duration still abstract.

    merge_faces holds the (patch, patch) pairs a merge joins, each pair
    and the pairs in sorted order. group holds the indices of the
    conditional S group the instruction belongs to, empty outside one.
    """

    name: str
    index: int
    patches: tuple
    duration: str
    conditioned_on: tuple
    conditioned_on_completion: tuple
    merge_faces: tuple
    group: frozenset

    @property
    def is_conditional_s(self) -> bool:
        """A Y measurement conditioned on a decode: a conditional S."""
        return self.name == "Y_MEAS" and bool(self.conditioned_on)

    def rounds(self, distance: int) -> int:
        """The duration at code distance d."""
        if self.duration == "D":
            return distance
        if self.duration == "HALF_D":
            return distance // 2
        if self.duration == "HALF_D_PLUS_2":
            return distance // 2 + 2
        return int(self.duration)


def read_instructions(path: pathlib.Path) -> tuple:
    """The schedule's instructions, in its order, each index its position."""
    text = path.read_text()
    lines = [line for line in text.splitlines() if line.strip()]
    instructions = [_instruction(line) for line in lines]
    for position, instruction in enumerate(instructions):
        if instruction.index != position:
            raise ValueError(
                f"{path}: instruction {instruction.index} sits at position "
                f"{position}"
            )
    return tuple(instructions)


def swiper_skipped_conditional_s(instructions: tuple, seed: int) -> frozenset:
    """The conditional S instructions SWIPER skips at that seed.

    One draw per conditional S in schedule order from
    numpy.random.default_rng(seed), skipped below 0.5, with the members
    of its group (device_manager.py, its __init__).
    """
    rng = np.random.default_rng(seed)
    skipped = set()
    for instruction in instructions:
        if not instruction.is_conditional_s:
            continue
        if rng.random() < 0.5:
            skipped.add(instruction.index)
            skipped.update(instruction.group)
    return frozenset(skipped)


def read_schedule(
    path: pathlib.Path, distance: int, skipped: frozenset = frozenset()
) -> workload_records.Workload:
    """The schedule at code distance d, the skipped instructions 0 rounds.

    skipped holds instruction indices, as swiper_skipped_conditional_s
    returns them; only a conditional S group may be skipped.
    """
    instructions = read_instructions(path)
    _check_skipped(instructions, skipped)
    operations = []
    round_counts = {}
    for instruction in instructions:
        rounds = 0
        if instruction.index not in skipped:
            rounds = instruction.rounds(distance)
        operation = _operation(instruction, rounds)
        operations.append(operation)
        round_counts[instruction.index] = rounds
    return workload_records.Workload(
        tuple(operations), round_counts, patch_streams=True
    )


def _instruction(line: str) -> Instruction:
    """One line's fields; the dependents, each wait's other side, unread."""
    fields = line.split()
    if len(fields) != FIELD_COUNT:
        raise ValueError(
            f"a schedule line has {FIELD_COUNT} fields, got {len(fields)}: "
            f"{line!r}"
        )
    name = fields[0]
    if name not in KINDS:
        raise ValueError(f"unknown schedule instruction {name!r}")
    patches = ast.literal_eval(fields[2])
    conditioned_on = ast.literal_eval(fields[4])
    conditioned_on_completion = ast.literal_eval(fields[6])
    faces = ast.literal_eval(fields[8])
    group = ast.literal_eval(fields[9])
    index = int(fields[1])
    ordered_patches = tuple(sorted(patches))
    completions = tuple(conditioned_on_completion)
    ordered_faces = _ordered_faces(faces)
    members = frozenset(group)
    return Instruction(
        name=name,
        index=index,
        patches=ordered_patches,
        duration=fields[3],
        conditioned_on=tuple(conditioned_on),
        conditioned_on_completion=completions,
        merge_faces=ordered_faces,
        group=members,
    )


def _ordered_faces(faces: list) -> tuple:
    """Each face's two patches in order, and the faces in order."""
    ordered = [tuple(sorted(face)) for face in faces]
    ordered.sort()
    return tuple(ordered)


def _operation(instruction: Instruction, rounds: int):
    """The instruction as an operation.

    It emits detector data when it runs rounds that are decoded: not a
    skipped or zero-round one, and not a T injection.
    """
    blocked_by = None
    if instruction.conditioned_on:
        (blocked_by,) = instruction.conditioned_on
    word = instruction.name.lower()
    patches = instruction.patches
    kind = KINDS[instruction.name]
    is_injection = instruction.name == "INJECT_T"
    emits = rounds > 0 and not is_injection
    name = f"{word}{instruction.index}"
    return program_records.Operation(
        id=instruction.index,
        name=name,
        qubits=patches,
        patches=patches,
        couplings=instruction.merge_faces,
        kind=kind,
        predecessors=instruction.conditioned_on_completion,
        blocked_by=blocked_by,
        emits_detector_data=emits,
        clifford=not is_injection,
    )


def _check_skipped(instructions: tuple, skipped: frozenset) -> None:
    """Only a conditional S group's instructions are skipped."""
    allowed = set()
    for instruction in instructions:
        if instruction.is_conditional_s:
            allowed.add(instruction.index)
            allowed.update(instruction.group)
    strays = set(skipped) - allowed
    stray = sorted(strays)
    if stray:
        raise ValueError(
            f"only a conditional S group is skipped; {stray} are not in one"
        )

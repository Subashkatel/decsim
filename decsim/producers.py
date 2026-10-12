"""The workload makers decsim ships: Stim's and Deltakit's memories.

A maker is a plain function that returns a records.workload.Workload,
which WorkloadSettings.running lowers for the machine. A maker written
outside decsim has the same shape; these are the ones that ship.

A workload lowered in the run file was made at the distance the run file
gave its maker. A maker row holds a maker's parameters instead and is
lowered when the task is made (topology/graph.py Machine.at_task),
against the qpu the task has, so an assignment to qpu.distance or to the
row's own fields makes the circuit again. gem5 resolves a parameter that
follows another part's the same way, before any object is built: a proxy
such as `Parent.cache_line_size * Self.num_blocks_per_sector`
(src/mem/cache/tags/Tags.py:139) is resolved by SimObject.unproxyParams
(src/python/m5/SimObject.py:1117). The task's id hashes the lowered
workload, so it carries the circuit's text as sinter's strong id does
(sinter/_data/_task.py:193), and a row and the workload it lowers to
have one id.

    memory       Stim's generated memory on the qpu's card at its
                 distance (memory_circuit, memory_patches), or patches
                 with no circuit (memory_without_circuit)
    live_memory  live fragments read from a folder, decoded after some
                 rounds (live_memory); the fragments were made at one
                 distance, so the row fixes the qpu's

A row of one's own is a record with the members of
ports.WorkloadMakerSettings, written by its import path.
"""

import dataclasses
import pathlib
from typing import Optional

import stim

import decsim.frontends.deltakit as deltakit
import decsim.frontends.settings as workload_settings
import decsim.frontends.workload_files as workload_files
import decsim.qpu.settings as qpu_settings
import decsim.records.circuits as circuit_records
import decsim.records.program as program_records
import decsim.records.workload as workload_records

# The stream a live memory's segments extend: an identity local to the
# workload, which also names the stream's substream seed
# (seeding.substream_seed), so it is the one
# examples/live_memory_example.py uses and a run reproduces that example's
# draws.
LIVE_STREAM_ID = 100


def memory_circuit(
    code_task: str,
    rounds_per_shot: int,
    distance: int,
    physical_error_probability: float,
    noise: Optional[workload_settings.StimNoise] = None,
) -> workload_records.Workload:
    """Stim's generated memory circuit, one operation for the whole shot.

    One probability on all four of Stim's noise channels, as Stim's
    guide does (frontends/settings.py memory_circuit), or noise's four
    channels; physical_error_probability is then the task's label.
    """
    circuit = workload_settings.memory_circuit(
        code_task, rounds_per_shot, distance, physical_error_probability, noise
    )
    operation = program_records.Operation(
        id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit
    )
    return workload_records.Workload(
        operations=(operation,),
        round_counts={1: rounds_per_shot},
        physical_error_probability=physical_error_probability,
    )


def memory_patches(
    code_task: str,
    rounds_per_shot: int,
    patch_count: int,
    distance: int,
    physical_error_probability: float,
    noise: Optional[workload_settings.StimNoise] = None,
) -> workload_records.Workload:
    """Independent memory patches run at once.

    patch_count copies of memory_circuit's circuit, each its own
    operation on its own patch and logical qubit, all from round one, so
    their windows share the decoder tiers' units, Elastic's "smaller pool
    of decoders" shared across logical qubits (2406.17995 lines 121-123).
    The copies sit side by side along x in one Stim coordinate frame,
    patch p shifted by p (2 d + 2) with a SHIFT_COORDS ahead of its
    circuit, so a surface-code patch, 2 d units wide, starts one lattice
    step past its neighbour. Each copy draws its own shot. noise is
    memory_circuit's.
    """
    if patch_count < 1:
        raise ValueError(
            f"workload.patch_count is at least 1, got {patch_count}; with "
            "no patch the run would finish having run nothing"
        )
    circuit = workload_settings.memory_circuit(
        code_task, rounds_per_shot, distance, physical_error_probability, noise
    )
    pitch = 2 * distance + 2
    operations = []
    round_counts = {}
    for patch in range(patch_count):
        offset = patch * pitch
        shift = stim.Circuit(f"SHIFT_COORDS({offset}, 0)")
        placed = shift + circuit
        operation_id = patch + 1
        operation = program_records.Operation(
            id=operation_id,
            name=f"memory{patch}",
            qubits=(patch,),
            patches=(patch,),
            circuit=placed,
        )
        operations.append(operation)
        round_counts[operation_id] = rounds_per_shot
    return workload_records.Workload(
        operations=tuple(operations),
        round_counts=round_counts,
        physical_error_probability=physical_error_probability,
    )


def memory_without_circuit(
    rounds_per_shot: int,
    patch_count: int,
    physical_error_probability: Optional[float],
) -> workload_records.Workload:
    """Memory patches with no circuit: their rounds, priced by the card.

    For a code card no Stim generator makes (the bivariate-bicycle
    card) or a timing-only run: each patch one operation, whose rounds
    take their sizes from the card (qpu/settings.py).
    """
    operations = []
    round_counts = {}
    for patch in range(patch_count):
        operation_id = patch + 1
        operation = program_records.Operation(
            id=operation_id,
            name=f"memory{patch}",
            qubits=(patch,),
            patches=(patch,),
        )
        operations.append(operation)
        round_counts[operation_id] = rounds_per_shot
    return workload_records.Workload(
        operations=tuple(operations),
        round_counts=round_counts,
        physical_error_probability=physical_error_probability,
    )


def deltakit_memory(
    rounds: int,
    distance: int,
    physical_error_probability: float,
    family: str = "rotated_surface",
    basis: str = "Z",
    patch: str = "memory-patch",
) -> workload_records.Workload:
    """Deltakit's finite memory, one operation for the whole circuit.

    The Explorer's css_code_memory_circuit with SD6 noise at the sweep's
    probability, exported with its measurement-to-round map
    (frontends/deltakit.py memory_circuit), as examples/deltakit_example.py
    builds it by hand. The optional deltakit extra is imported only when
    this maker runs.
    """
    circuit, measurement_rounds = deltakit.memory_circuit(
        family, distance, rounds, basis, physical_error_probability
    )
    operation = program_records.Operation(
        1, "memory", (patch,), patches=(patch,)
    )
    physical = workload_records.FiniteCircuit(circuit, measurement_rounds)
    return workload_records.Workload(
        (operation,), {1: rounds}, physical, physical_error_probability
    )


def deltakit_live_memory(
    distance: int,
    physical_error_probability: float,
    round_period_microseconds: float,
    decode_after_rounds: int,
    basis: str = "Z",
    noise_model: str = "sd6",
    relaxation_time_microseconds: Optional[float] = None,
    dephasing_time_microseconds: Optional[float] = None,
    patch: str = "memory-patch",
) -> workload_records.Workload:
    """A live Deltakit memory decoded after some rounds, then read out.

    The four fragments of frontends/deltakit.py memory_rounds, their
    period the sweep's, run as the live program live_memory builds.
    """
    program = deltakit.memory_rounds(
        "rotated_surface",
        distance,
        basis,
        physical_error_probability,
        round_period_microseconds=round_period_microseconds,
        noise_model=noise_model,
        relaxation_time_microseconds=relaxation_time_microseconds,
        dephasing_time_microseconds=dephasing_time_microseconds,
    )
    workload = live_memory(program, decode_after_rounds, patch)
    return dataclasses.replace(
        workload, physical_error_probability=physical_error_probability
    )


def live_memory(
    program: circuit_records.RepeatedStimCircuit,
    decode_after_rounds: int,
    patch: str = "memory-patch",
) -> workload_records.Workload:
    """Live fragments decoded after some rounds, then read out.

    The prefix runs the stream's first rounds and is decoded; the patch
    waits, protected, until the decoded result releases the one round
    that resumes it, and the readout ends the stream.
    """
    patches = (patch,)
    prefix = program_records.Operation(
        1,
        "prefix",
        patches,
        patches=patches,
        stream_id=LIVE_STREAM_ID,
        stream_offset=0,
    )
    protect = program_records.Operation(
        2, "protect", patches, patches=patches, emits_detector_data=False
    )
    resume = program_records.Operation(
        3,
        "resume",
        patches,
        patches=patches,
        blocked_by=1,
        emits_detector_data=False,
    )
    readout = program_records.Operation(
        4, "readout", patches, patches=patches, emits_detector_data=False
    )
    operations = (prefix, protect, resume, readout)
    round_counts = {1: decode_after_rounds, 2: 0, 3: 1, 4: 0}
    return workload_records.Workload(operations, round_counts, program)


@dataclasses.dataclass(frozen=True)
class MemorySettings:
    """A memory experiment at the qpu's distance, on one patch or several.

    physical_error_probability is p on each of Stim's four channels; with
    noise the channels are noise's and p is the task's label, which a
    threshold table and the reports read. One of rounds and
    rounds_per_distance is given: a shot's rounds, or that many for each
    unit of the distance. code_task is a Stim task whose circuit is the
    qpu's card's code (code_geometry.py stim_memory_tasks), or None for
    patches with no circuit. patches run at once, side by side
    (memory_patches). decoded False decodes nothing: the
    readout-and-frame baseline.
    """

    physical_error_probability: float
    rounds: Optional[int] = None
    rounds_per_distance: Optional[int] = None
    code_task: Optional[str] = "surface_code:rotated_memory_z"
    patches: int = 1
    noise: Optional[workload_settings.StimNoise] = None
    decoded: bool = True
    name = "memory"
    fixes_the_distance = False

    def __post_init__(self) -> None:
        if (self.rounds is None) != (self.rounds_per_distance is None):
            return
        raise ValueError(
            "a memory states its rounds once: rounds or rounds_per_distance, "
            f"got rounds={self.rounds!r} and "
            f"rounds_per_distance={self.rounds_per_distance!r}"
        )

    def lowered_for(
        self, qpu: qpu_settings.QpuSettings
    ) -> workload_settings.WorkloadSettings:
        """The memory on the qpu's card at its distance, lowered."""
        card = qpu.code_card.build(qpu.distance, None, None)
        workload = self._made(qpu.code_card, card.distance)
        running = workload_settings.WorkloadSettings.running(workload)
        if self.decoded:
            return running
        return dataclasses.replace(running, decode_operations=())

    def _made(
        self, code_card: qpu_settings.CodeCardSettings, distance: int
    ) -> workload_records.Workload:
        """The maker's workload: Stim's circuit, or patches with none."""
        rounds = self._rounds(distance)
        probability = self.physical_error_probability
        if self.code_task is None:
            return memory_without_circuit(rounds, self.patches, probability)
        _refuse_another_codes_task(self.code_task, code_card)
        if self.patches == 1:
            return memory_circuit(
                self.code_task, rounds, distance, probability, self.noise
            )
        return memory_patches(
            self.code_task,
            rounds,
            self.patches,
            distance,
            probability,
            self.noise,
        )

    def _rounds(self, distance: int) -> int:
        if self.rounds is not None:
            return self.rounds
        return self.rounds_per_distance * distance


@dataclasses.dataclass(frozen=True)
class LiveMemorySettings:
    """Live fragments read from a folder, decoded after some rounds.

    fragments is a folder of the four fragments and physical.json
    (workload_files.read_fragments). The prefix is decoded, and the
    patch waits, protected, until the decoded result releases it
    (live_memory); examples/live_memory_example.py runs it on
    StreamingStimDevice. feedback_boundary_mode is the workload's
    (frontends/settings.py).
    """

    fragments: pathlib.Path
    decode_after_rounds: int
    patch: str = "memory-patch"
    feedback_boundary_mode: str = "trailing_buffer"
    name = "live_memory"
    fixes_the_distance = True

    def lowered_for(
        self, qpu: qpu_settings.QpuSettings
    ) -> workload_settings.WorkloadSettings:
        """The live memory over the fragments, lowered."""
        del qpu
        program = workload_files.read_fragments(self.fragments)
        workload = live_memory(program, self.decode_after_rounds, self.patch)
        running = workload_settings.WorkloadSettings.running(workload)
        return dataclasses.replace(
            running, feedback_boundary_mode=self.feedback_boundary_mode
        )


def _refuse_another_codes_task(
    code_task: str, code_card: qpu_settings.CodeCardSettings
) -> None:
    """A Stim task whose circuit is not the card's code is refused."""
    tasks = code_card.stim_memory_tasks
    if code_task in tasks:
        return
    raise ValueError(
        f"workload.code_task {code_task!r} makes a circuit of another code "
        f"than the qpu's card; the card's Stim tasks are {tasks}, and "
        "workload.code_task=none makes patches with no circuit"
    )

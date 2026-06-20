"""Frontend and code-model seams for external IR adapters."""

from decsim import protocols as P
from decsim.codes import BBCodeModel, SurfaceCodeModel
from decsim.config import us
from decsim.controllers import ModularController
from decsim.decoders import PresetLatencyDecoder
from decsim.devices import SyndromeBitDevice, TimingOnlyDevice
from decsim.layouts import UniformLayout
from decsim.message import DecodeResult, Operation
from decsim.planner import FixedRounds
from decsim.wiring import build_and_run


class PhysicalPatchIRFrontend:
    """Test adapter for an already-lowered patch-level physical IR."""

    def __init__(self, records: list[dict]):
        self.records = records
        self.operations: list[Operation] = []

    def build(self) -> list[Operation]:
        """Lower records and wire patch program order."""
        last_operation_on_patch = {}
        operations = []
        for record in self.records:
            operation_id = record["id"]
            patches = tuple(record["patches"])
            predecessors = set(record.get("after", ()))
            for patch in patches:
                if patch in last_operation_on_patch:
                    predecessors.add(last_operation_on_patch[patch])
                last_operation_on_patch[patch] = operation_id

            operation = Operation(
                operation_id,
                record["name"],
                tuple(record.get("qubits", patches)),
                clifford=record.get("clifford", True),
                consumes_magic_state=record.get("consumes_magic_state"),
                patches=patches,
                predecessors=tuple(sorted(predecessors)),
                blocked_by=record.get("blocked_by"),
            )
            operations.append(operation)

        predecessor_ids = {
            predecessor_id
            for operation in operations
            for predecessor_id in operation.predecessors
        }
        for operation in operations:
            operation.has_successor = operation.id in predecessor_ids

        self.operations = operations
        return operations


class LatticeSurgeryToPhysicalIRFrontend:
    """Test adapter that lowers lattice-surgery records into physical patch IR."""

    def __init__(self, instructions: list[dict]):
        self.instructions = instructions
        self.physical_ir: list[dict] = []
        self.physical_frontend: PhysicalPatchIRFrontend | None = None
        self.operations: list[Operation] = []

    def build(self) -> list[Operation]:
        """Lower lattice-surgery records, then let physical IR build operations."""
        self.physical_ir = []
        for instruction in self.instructions:
            physical_record = {
                "id": instruction["id"],
                "name": instruction["name"],
                "patches": tuple(instruction["patches"]),
                "qubits": tuple(instruction.get("logical_qubits", instruction["patches"])),
                "clifford": instruction.get("clifford", True),
                "consumes_magic_state": instruction.get("consumes_magic_state"),
                "blocked_by": instruction.get("blocked_by"),
                "after": tuple(instruction.get("after", ())),
            }
            self.physical_ir.append(physical_record)

        self.physical_frontend = PhysicalPatchIRFrontend(self.physical_ir)
        self.operations = self.physical_frontend.build()
        return self.operations


class BBCodeISAFrontend:
    """Test adapter for a BB-code logical instruction stream."""

    def __init__(self, instructions: list[dict]):
        self.instructions = instructions
        self.operations: list[Operation] = []

    def build(self) -> list[Operation]:
        """Map BB logical instructions onto physical code blocks."""
        last_operation_on_block = {}
        operations = []
        for instruction in self.instructions:
            operation_id = instruction["id"]
            block = instruction["block"]
            logical_qubit = (block, instruction["logical"])
            predecessors = set(instruction.get("after", ()))
            if block in last_operation_on_block:
                predecessors.add(last_operation_on_block[block])
            last_operation_on_block[block] = operation_id

            operation = Operation(
                operation_id,
                f"{instruction['gate']}[{block}:q{instruction['logical']}]",
                (logical_qubit,),
                clifford=instruction.get("clifford", True),
                patches=(block,),
                predecessors=tuple(sorted(predecessors)),
            )
            operations.append(operation)

        predecessor_ids = {
            predecessor_id
            for operation in operations
            for predecessor_id in operation.predecessors
        }
        for operation in operations:
            operation.has_successor = operation.id in predecessor_ids

        self.operations = operations
        return operations


class BBCodeISAToPhysicalIRFrontend:
    """Test adapter that lowers a BB-code ISA into physical patch IR."""

    def __init__(self, instructions: list[dict]):
        self.instructions = instructions
        self.physical_ir: list[dict] = []
        self.physical_frontend: PhysicalPatchIRFrontend | None = None
        self.operations: list[Operation] = []

    def build(self) -> list[Operation]:
        """Lower BB-code ISA records, then let physical IR build operations."""
        self.physical_ir = []
        for instruction in self.instructions:
            block = instruction["block"]
            logical_qubit = (block, instruction["logical"])
            physical_record = {
                "id": instruction["id"],
                "name": f"{instruction['gate']}[{block}:q{instruction['logical']}]",
                "patches": (block,),
                "qubits": (logical_qubit,),
                "clifford": instruction.get("clifford", True),
                "after": tuple(instruction.get("after", ())),
            }
            self.physical_ir.append(physical_record)

        self.physical_frontend = PhysicalPatchIRFrontend(self.physical_ir)
        self.operations = self.physical_frontend.build()
        return self.operations


class RecordingDecoder:
    """Fixed-latency decoder that records job sizing."""

    def __init__(self):
        self.jobs = []

    def latency(self, job):
        return us(0.1)

    def decode(self, job):
        self.jobs.append(job)
        return DecodeResult(job.op_id, job.window_id, logical_value=0)


def _zero_link_controller(engine):
    return ModularController(
        engine, t_qc=0, t_cd=0, t_dd=0, t_do=0, t_oc=0, t_cq=0,
        log_syndromes=False)


def test_patch_level_physical_ir_frontend_runs_without_builtin_surgery_parser():
    """A patch-level IR adapter can feed the standard runtime through InputFrontend."""
    frontend = PhysicalPatchIRFrontend([
        {"id": 10, "name": "prep patch_a", "patches": ("patch_a",), "qubits": (0,)},
        {"id": 11, "name": "measure link", "patches": ("patch_a", "patch_b"),
         "qubits": (0, 1), "clifford": False, "consumes_magic_state": False},
        {"id": 12, "name": "conditional correction", "patches": ("patch_b",),
         "qubits": (1,), "clifford": False, "blocked_by": 11,
         "consumes_magic_state": False},
    ])
    code = SurfaceCodeModel(d=3)

    result = build_and_run(
        frontend=frontend, code=code, num_units=2, rounds_policy=FixedRounds(3),
        device=TimingOnlyDevice(), decoder=PresetLatencyDecoder(0.1),
        make_controller=_zero_link_controller, verbose=False)

    assert isinstance(frontend, P.InputFrontend)
    assert frontend.operations[1].predecessors == (10,)
    assert frontend.operations[2].predecessors == (11,)
    assert result["cluster"].window_count == {10: 1, 11: 1, 12: 1}
    assert result["chip"].decode_release_time[12] >= result["cluster"].windows[(11, 0)].t_done
    assert result["fully_done"] > 0


def test_lattice_surgery_ir_can_lower_to_physical_ir_then_run():
    """A lattice-surgery adapter can lower through physical IR before runtime."""
    frontend = LatticeSurgeryToPhysicalIRFrontend([
        {"id": 0, "name": "prepare logical patch", "patches": ("patch_a",),
         "logical_qubits": (0,)},
        {"id": 1, "name": "merge patches", "patches": ("patch_a", "patch_b"),
         "logical_qubits": (0, 1), "clifford": False, "consumes_magic_state": False},
        {"id": 2, "name": "conditional split correction", "patches": ("patch_b",),
         "logical_qubits": (1,), "clifford": False, "blocked_by": 1,
         "consumes_magic_state": False},
    ])

    result = build_and_run(
        frontend=frontend, code=SurfaceCodeModel(d=3), num_units=2,
        rounds_policy=FixedRounds(3), device=TimingOnlyDevice(),
        decoder=PresetLatencyDecoder(0.1), make_controller=_zero_link_controller,
        verbose=False)

    assert isinstance(frontend, P.InputFrontend)
    assert frontend.physical_ir[1]["patches"] == ("patch_a", "patch_b")
    assert isinstance(frontend.physical_frontend, PhysicalPatchIRFrontend)
    assert frontend.operations[1].predecessors == (0,)
    assert frontend.operations[2].predecessors == (1,)
    assert frontend.operations[2].blocked_by == 1
    assert result["chip"].decode_release_time[2] >= result["cluster"].windows[(1, 0)].t_done


def test_bb_code_isa_frontend_runs_with_bb_code_model_and_same_components():
    """A BB-code ISA adapter can swap only frontend and code model."""
    frontend = BBCodeISAFrontend([
        {"id": 0, "gate": "bb_x_check", "block": "gross_0", "logical": 0},
        {"id": 1, "gate": "bb_z_check", "block": "gross_0", "logical": 5},
        {"id": 2, "gate": "bb_x_check", "block": "gross_1", "logical": 0},
    ])
    code = BBCodeModel(d=4)
    layout = UniformLayout(code)
    decoder = RecordingDecoder()

    result = build_and_run(
        frontend=frontend, code=code, layout=layout, num_units=2,
        rounds_policy=FixedRounds(4), device=SyndromeBitDevice(code, per_patch=True),
        decoder=decoder, make_controller=_zero_link_controller, verbose=False)

    assert isinstance(frontend, P.InputFrontend)
    assert isinstance(code, P.CodeModel)
    assert result["cluster"].layout is layout
    assert frontend.operations[1].predecessors == (0,)
    assert frontend.operations[2].predecessors == ()
    assert {job.code for job in decoder.jobs} == {code.name}
    assert {job.spatial_nodes for job in decoder.jobs} == {code.spatial_nodes(1)}
    assert all(job.payloads for job in decoder.jobs)
    assert len(result["cluster"].committed_windows) == result["cluster"].total_windows


def test_bb_code_isa_can_lower_to_physical_ir_then_run():
    """A BB-code ISA adapter can lower to physical IR before entering decsim."""
    frontend = BBCodeISAToPhysicalIRFrontend([
        {"id": 0, "gate": "bb_x_check", "block": "gross_0", "logical": 0},
        {"id": 1, "gate": "bb_z_check", "block": "gross_0", "logical": 5},
        {"id": 2, "gate": "bb_x_check", "block": "gross_1", "logical": 0},
    ])
    code = BBCodeModel(d=4)
    layout = UniformLayout(code)
    decoder = RecordingDecoder()

    result = build_and_run(
        frontend=frontend, code=code, layout=layout, num_units=2,
        rounds_policy=FixedRounds(4), device=SyndromeBitDevice(code, per_patch=True),
        decoder=decoder, make_controller=_zero_link_controller, verbose=False)

    assert isinstance(frontend, P.InputFrontend)
    assert frontend.physical_ir[0]["patches"] == ("gross_0",)
    assert isinstance(frontend.physical_frontend, PhysicalPatchIRFrontend)
    assert frontend.operations[1].predecessors == (0,)
    assert frontend.operations[2].predecessors == ()
    assert result["cluster"].code is code
    assert {job.code for job in decoder.jobs} == {code.name}
    assert {job.spatial_nodes for job in decoder.jobs} == {code.spatial_nodes(1)}
    assert len(result["cluster"].committed_windows) == result["cluster"].total_windows

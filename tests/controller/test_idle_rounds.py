"""The idle accounting: every idle round is routed by the policy, claimed once.

Idle rounds are decoder workload in every reference system (SWIPER
2412.05115: one UNWANTED_IDLE round per unused patch per cycle; XQsim;
Terhal's backlog bound via Battistel 2303.00054). The policies' own laws
are test_policies.py; here the accounting's: the rounds emitted on a
patch are claimed once by the operation that takes the patch, and a patch
on a live protected stream emits through the stream, not here.
"""

import types

import pytest

import decsim.controller.idle_rounds as idle_rounds_module
import decsim.controller.policies as policies
import decsim.observe.controller_counters as controller_counters
import decsim.records.program as program_records


class RecordingQpu:
    def __init__(self):
        self.memory_rounds = []

    def emit_feedback_memory_round(self, operation_id, patch, round_index):
        self.memory_rounds.append((operation_id, patch, round_index))


class RecordingStreams:
    def __init__(self, live_patches=()):
        self.live_patches = set(live_patches)

    def is_live_protected_patch(self, patch):
        return patch in self.live_patches

    def extend_live_stream(self, operation, patch):
        del operation, patch
        return False


class RecordingDecodeQueue:
    def __init__(self):
        self.demands = []

    def enqueue_without_input(
        self, round_count, on_done, label, code, spatial_nodes
    ):
        del on_done
        self.demands.append(
            {
                "rounds": round_count,
                "code": code,
                "spatial_nodes": spatial_nodes,
                "label": label,
            }
        )


def patch_record():
    geometry = types.SimpleNamespace(
        commit_round_count=2, buffer_round_count=1, code_name="surface-code"
    )
    return types.SimpleNamespace(code_geometry=geometry, spatial_node_count=17)


def accounting_with(policy, streams=None):
    if streams is None:
        streams = RecordingStreams()
    qpu = RecordingQpu()
    demand = RecordingDecodeQueue()
    geometry_by_patch = {"patch-a": patch_record(), "patch-b": patch_record()}
    accounting = idle_rounds_module.IdleRoundAccounting(
        policy, demand, geometry_by_patch, streams, qpu
    )
    memory = program_records.Operation(
        7, "memory", ("patch-a",), patches=("patch-a",)
    )
    other = program_records.Operation(
        8, "other", ("patch-b",), patches=("patch-b",)
    )
    program = program_records.ExecutionProgram((memory, other))
    accounting.load(program)
    return accounting, qpu, demand


def counters_on(accounting):
    """A counters listener on the accounting's one source."""
    counters = controller_counters.ControllerCounters()
    accounting.idle_round_emitted.connect(counters.idle_round_emitted)
    return counters


def test_the_rounds_emitted_on_a_patch_are_claimed_once_by_the_operation():
    ignore = policies.Ignore()
    accounting, qpu, _demand = accounting_with(ignore)
    counters = counters_on(accounting)
    operation = accounting.operation_by_id[7]

    accounting.emit_idle_round(7, "patch-a", 1)
    accounting.emit_idle_round(7, "patch-a", 2)
    accounting.emit_idle_round(8, "patch-b", 1)
    first_claim = accounting.claim(operation)
    second_claim = accounting.claim(operation)
    accounting.emit_idle_round(7, "patch-a", 3)
    third_claim = accounting.claim(operation)

    assert (first_claim, second_claim, third_claim) == (2, 0, 1)
    assert counters.idle_rounds == 4
    assert qpu.memory_rounds == [
        (7, "patch-a", 1),
        (7, "patch-a", 2),
        (8, "patch-b", 1),
        (7, "patch-a", 3),
    ]


def test_a_claim_that_fails_on_a_later_patch_keeps_the_earlier_claims():
    """A claim is not a transaction: a patch already claimed stays claimed.

    An operation whose second patch identity is unhashable fails at that
    patch; the four rounds of its first patch are already claimed and are
    not restored, and the other patch's round is untouched.
    """
    ignore = policies.Ignore()
    accounting, _qpu, _demand = accounting_with(ignore)
    for round_index in (1, 2, 3, 4):
        accounting.emit_idle_round(7, "patch-a", round_index)
    accounting.emit_idle_round(8, "patch-b", 1)
    malformed = program_records.Operation(
        9, "malformed", ("q",), patches=("patch-a", [])
    )

    with pytest.raises(TypeError):
        accounting.claim(malformed)

    unclaimed_by_patch = {
        patch: idle.unclaimed
        for patch, idle in accounting.idle_by_patch.items()
    }
    assert unclaimed_by_patch == {"patch-a": 0, "patch-b": 1}


def test_an_operation_without_patches_claims_by_its_qubits():
    ignore = policies.Ignore()
    accounting, _qpu, _demand = accounting_with(ignore)
    operation = program_records.Operation(9, "bare", ("patch-a", "patch-b"))

    accounting.emit_idle_round(7, "patch-a", 1)
    accounting.emit_idle_round(8, "patch-b", 1)

    assert accounting.claim(operation) == 2


def test_a_patch_on_a_live_protected_stream_emits_through_the_stream():
    streams = RecordingStreams(live_patches=("patch-a",))
    ignore = policies.Ignore()
    accounting, qpu, _demand = accounting_with(ignore, streams)
    counters = counters_on(accounting)

    accounting.emit_idle_round(7, "patch-a", 1)

    assert qpu.memory_rounds == []
    assert counters.idle_rounds == 0


def test_the_charged_policy_costs_one_job_per_region_and_the_remainder():
    charged = policies.SeparateDecodeJobs()
    accounting, _qpu, demand = accounting_with(charged)
    operation = accounting.operation_by_id[7]

    for round_index in (1, 2, 3, 4, 5):
        accounting.emit_idle_round(7, "patch-a", round_index)
    accounting.end_idle_period(operation, "patch-a")

    rounds_and_labels = [
        (job["rounds"], job["label"]) for job in demand.demands
    ]
    assert rounds_and_labels == [
        (3, "mem(memory,r2)"),
        (3, "mem(memory,r4)"),
        (2, "mem(memory,r5)"),
    ]

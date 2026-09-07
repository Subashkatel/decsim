from types import SimpleNamespace

import pytest

from decsim.controller import policies
from decsim.controller.idle_rounds import IdleRoundAccounting
from decsim.controller.policies import (
    Eager,
    ExtendStream,
    Held,
    Ignore,
    SeparateDecodeJobs,
)
from decsim.controller.settings import IdlePolicySettings
from decsim.decoders.settings import DecoderSettings
from decsim.escalation.policies import Baseline, Switching
from decsim.escalation.threshold_sources import FixedThreshold
from decsim.frontends.settings import WorkloadSettings
from decsim.machine import Machine, MachineSettings
from decsim.message import RunSeedReservation, RunShape, SoftOutputSource
from decsim.observe.controller_counters import ControllerCounters
from decsim.ports import IdlePolicy
from decsim.qpu.settings import QpuSettings
from decsim.windows.settings import WindowSettings
from decsim.windows.window_manager import BoundaryPolicy
from decsim.windows.windowing_schemes import (
    SlidingTerminalPolicy,
    SlidingWindowScheme,
)


class ExternalBoundaryPolicy:
    def on_commit(self, window, *, final):
        return final


class ExternalIdlePolicy:
    def __init__(self):
        self.relayed = []

    def relay(self, controller, operation, patch, round_index):
        self.relayed.append((operation, patch, round_index))
        controller.emit_memory_round(operation, patch, round_index)

    def end_idle_period(self, controller, operation, patch):
        pass


class EngineProbe:
    def __init__(self):
        self.now = 0
        self.scheduled = []
        self.logs = []

    def schedule(self, delay, action, **metadata):
        self.scheduled.append((delay, action, metadata))

    def log(self, owner, text):
        self.logs.append((owner, text))


class QPUProbe:
    def __init__(self):
        self.feedback_rounds = []
        self.stream_rounds = []

    def emit_feedback_memory_round(self, operation_id, patch, round_index):
        self.feedback_rounds.append((operation_id, patch, round_index))

    def emit_idle_stream_round(self, operation, stream_id, round_index, patch):
        self.stream_rounds.append((operation, stream_id, round_index, patch))


class WindowManagerProbe:
    def __init__(self, live_streams=()):
        self.live_streams = set(live_streams)
        self.idle_demands = []

    def has_dynamic_stream(self, stream_id):
        return stream_id in self.live_streams

    def enqueue_without_input(
        self, round_count, on_done, label, code, spatial_nodes
    ):
        del on_done
        self.idle_demands.append(
            {
                "rounds": round_count,
                "code": code,
                "spatial_nodes": spatial_nodes,
                "label": label,
            }
        )


class StreamsProbe:
    """Stream bookkeeping stand-in: a binding per operation, a set of live
    protected patches, and the streams the window manager knows."""

    def __init__(self, qpu, window_manager):
        self.qpu = qpu
        self.window_manager = window_manager
        self.bindings = {}
        self.live_protected_patches = set()
        self.stream_next_round = {}

    def binding_for(self, operation_id):
        return self.bindings.get(operation_id)

    def is_live_protected_patch(self, patch):
        return patch in self.live_protected_patches

    def extend_live_stream(self, operation, patch):
        binding = self.bindings.get(operation.id)
        stream_id = None if binding is None else binding.stream_id
        if stream_id is None or not self.window_manager.has_dynamic_stream(
            stream_id
        ):
            return False
        global_round = self.stream_next_round.get(stream_id, 0) + 1
        self.stream_next_round[stream_id] = global_round
        self.qpu.emit_idle_stream_round(
            operation, stream_id, global_round, patch
        )
        return True


def make_controller(idle_policy, *, live_streams=()):
    geometry = SimpleNamespace(
        distance=3,
        commit_round_count=2,
        buffer_round_count=1,
        code_name="surface-code",
    )
    patch = SimpleNamespace(
        patch_identity="patch-a",
        round_ticks=11,
        code_geometry=geometry,
        spatial_node_count=17,
    )
    engine = EngineProbe()
    qpu = QPUProbe()
    window_manager = WindowManagerProbe(live_streams)
    controller = IdleRoundAccounting(
        idle_policy,
        window_manager,
        {"patch-a": patch},
        StreamsProbe(qpu, window_manager),
        qpu,
    )
    controller.operation_by_id[7] = SimpleNamespace(id=7, name="logical-cnot")
    return controller, engine, qpu, window_manager


def switching_source():
    return SoftOutputSource(
        method="matching-gap",
        cluster_origin="decoder",
        growth_schedule="uniform",
        gap_units="natural-log",
        correction="minimum-weight",
        weight_step_natural_log=1.0,
        references=(),
    )


def test_boundary_policies_decide_without_argument_validation():
    """Boundary policies accept unchecked inputs and make their documented commit decisions."""
    falsey_final = []
    truthy_final = object()
    unchecked_window = object()

    assert Eager().on_commit(unchecked_window, final=False) is True
    assert Eager().on_commit(unchecked_window, final=True) is True
    assert Eager().on_commit(unchecked_window, final=falsey_final) is True
    assert Eager().on_commit(unchecked_window, final=truthy_final) is True
    assert Held().on_commit(unchecked_window, final=False) is False
    assert Held().on_commit(unchecked_window, final=True) is True
    assert (
        Held().on_commit(unchecked_window, final=falsey_final) is falsey_final
    )
    assert (
        Held().on_commit(unchecked_window, final=truthy_final) is truthy_final
    )


def test_policies_are_stateless():
    for policy in (Ignore(), ExtendStream(), SeparateDecodeJobs()):
        assert vars(policy) == {}
    assert vars(Eager()) == {}
    assert vars(Held()) == {}


def test_builtin_and_external_policies_satisfy_runtime_protocols():
    """Built-in and structurally compatible external objects satisfy the runtime policy protocols."""
    for boundary_policy in (Eager(), Held(), ExternalBoundaryPolicy()):
        assert isinstance(boundary_policy, BoundaryPolicy)
    for idle_policy in (
        Ignore(),
        ExtendStream(),
        SeparateDecodeJobs(),
        ExternalIdlePolicy(),
    ):
        assert isinstance(idle_policy, IdlePolicy)


def test_runspec_builds_fresh_policy_defaults():
    """Each run with omitted policies receives fresh eager and charged-idle
    defaults: idle rounds are decoder workload in every reference system
    (SWIPER, XQsim, Terhal backlog), so the default costs them."""
    first = Machine.build(MachineSettings())
    first.run()
    second = Machine.build(MachineSettings())
    second.run()

    assert isinstance(first.window_manager.courier.boundary_policy, Eager)
    assert isinstance(first.idle_rounds.policy, SeparateDecodeJobs)
    assert isinstance(second.window_manager.courier.boundary_policy, Eager)
    assert isinstance(second.idle_rounds.policy, SeparateDecodeJobs)
    assert (
        first.window_manager.courier.boundary_policy
        is not second.window_manager.courier.boundary_policy
    )
    assert first.idle_rounds.policy is not second.idle_rounds.policy


def test_runspec_preserves_truthy_custom_policies_on_independent_axes():
    """RunSpec preserves truthy custom policies and wires the two axes independently."""
    boundary_policy = ExternalBoundaryPolicy()
    boundary_run = Machine.build(
        MachineSettings(windows=WindowSettings(boundary_policy=boundary_policy))
    )
    boundary_run.run()
    idle_policy = ExternalIdlePolicy()
    idle_run = Machine.build(
        MachineSettings(idle_policy=IdlePolicySettings(policy=idle_policy))
    )
    idle_run.run()

    assert (
        boundary_run.window_manager.courier.boundary_policy is boundary_policy
    )
    assert isinstance(boundary_run.idle_rounds.policy, SeparateDecodeJobs)
    assert idle_run.idle_rounds.policy is idle_policy
    assert isinstance(idle_run.window_manager.courier.boundary_policy, Eager)


def test_policy_module_has_no_registry_or_string_selector():
    """The policy module exposes neither a built-in registry nor a string selector."""
    assert not hasattr(policies, "MODES")
    assert not hasattr(policies, "from_mode")


def _run_shape(
    boundary_policy, *, is_double_window=False, has_dynamic_streams=False
):
    scheme = SlidingWindowScheme(
        terminal_policy=SlidingTerminalPolicy.REGULAR_STRIDE_LOOKAHEAD
    )
    return RunShape(
        scheme=scheme,
        boundary_policy=boundary_policy,
        operations=(),
        commit_round_count=3,
        buffer_round_count=3,
        is_double_window=is_double_window,
        is_bulk_strong=False,
        has_dynamic_streams=has_dynamic_streams,
        has_static_decode_plan=False,
        has_frontend=False,
    )


def test_switching_validates_builtin_boundary_contexts():
    """Switching rejects boundary policies that conflict with serial or double windows."""
    switching = Switching(FixedThreshold(1.0), switching_source())
    eager_serial = _run_shape(Eager())
    held_streams = _run_shape(Held(), has_dynamic_streams=True)
    held_double_window = _run_shape(Held(), is_double_window=True)
    eager_streams = _run_shape(Eager(), has_dynamic_streams=True)

    with pytest.raises(ValueError, match="serial switching requires Held"):
        switching.check_plan(eager_serial)
    switching.check_plan(held_streams)

    with pytest.raises(ValueError, match="Held boundary policy"):
        switching.check_plan(held_double_window)

    baseline = Baseline()
    baseline.check_plan(eager_streams)
    baseline.check_plan(held_streams)


def test_extend_stream_relays_idle_rounds_into_a_live_stream():
    """ExtendStream routes idle data into an existing live stream."""
    controller, _, qpu, _ = make_controller(
        ExtendStream(), live_streams=("stream-a",)
    )
    operation = controller.operation_by_id[7]
    controller.streams.bindings[7] = SimpleNamespace(stream_id="stream-a")

    controller.emit_idle_round(7, "patch-a", 9)

    assert qpu.feedback_rounds == []
    assert qpu.stream_rounds == [(operation, "stream-a", 1, "patch-a")]
    assert controller.streams.stream_next_round == {"stream-a": 1}


def test_extend_stream_falls_back_without_a_live_stream():
    """ExtendStream falls back to memory rounds without creating or reopening a stream."""
    unbound, _, unbound_qpu, _ = make_controller(
        ExtendStream(), live_streams=("stream-a",)
    )
    unbound.emit_idle_round(7, "patch-a", 3)

    closed, _, closed_qpu, window_manager = make_controller(ExtendStream())
    closed.streams.bindings[7] = SimpleNamespace(stream_id="stream-a")
    closed.emit_idle_round(7, "patch-a", 4)

    assert unbound_qpu.feedback_rounds == [(7, "patch-a", 3)]
    assert closed_qpu.feedback_rounds == [(7, "patch-a", 4)]
    assert unbound.streams.stream_next_round == {}
    assert closed.streams.stream_next_round == {}
    assert window_manager.live_streams == set()


def test_separate_decode_jobs_submits_only_complete_idle_regions():
    """SeparateDecodeJobs emits memory rounds and submits load-only jobs only at complete increments."""
    controller, _, qpu, window_manager = make_controller(SeparateDecodeJobs())

    for round_index in (1, 2, 3, 4, 5):
        controller.emit_idle_round(7, "patch-a", round_index)

    assert qpu.feedback_rounds == [(7, "patch-a", r) for r in (1, 2, 3, 4, 5)]
    assert window_manager.idle_demands == [
        {
            "rounds": 3,
            "code": "surface-code",
            "spatial_nodes": 17,
            "label": "mem(logical-cnot,r2)",
        },
        {
            "rounds": 3,
            "code": "surface-code",
            "spatial_nodes": 17,
            "label": "mem(logical-cnot,r4)",
        },
    ]


def test_separate_decode_jobs_charges_the_trailing_idle_region_when_the_patch_is_claimed():
    """Idle rounds left over after the last complete commit region are
    still decoded: when an operation claims the patch, the remainder costs
    one load-only job sized to those rounds plus the buffer. A final window
    may be smaller than a regular one (Tan et al. 2209.09219, Skoric et al.
    2209.08552), and no validated system leaves the end of a stream
    undecoded (Google's streaming decoder, LILLIPUT's per-cycle decode)."""
    controller, _, _, window_manager = make_controller(SeparateDecodeJobs())
    operation = controller.operation_by_id[7]

    for round_index in (1, 2, 3, 4, 5):
        controller.emit_idle_round(7, "patch-a", round_index)
    controller.end_idle_period(operation, "patch-a")
    controller.end_idle_period(operation, "patch-a")

    assert [demand["rounds"] for demand in window_manager.idle_demands] == [
        3,
        3,
        2,
    ]
    assert window_manager.idle_demands[-1]["label"] == "mem(logical-cnot,r5)"


def test_an_external_idle_policy_relays_through_the_controller():
    """An external policy owns its relay; the controller offers memory rounds,
    live-stream extension and idle decode demand."""
    policy = ExternalIdlePolicy()
    controller, _, qpu, window_manager = make_controller(
        policy, live_streams=("stream-a",)
    )
    controller.streams.bindings[7] = SimpleNamespace(stream_id="stream-a")

    controller.emit_idle_round(7, "patch-a", 2)

    assert qpu.feedback_rounds == [(7, "patch-a", 2)]
    assert qpu.stream_rounds == []
    assert window_manager.idle_demands == []
    assert policy.relayed == [(controller.operation_by_id[7], "patch-a", 2)]


def test_controller_accounts_every_idle_round_except_on_a_live_protected_stream():
    """Every idle cycle the QPU reports is emitted once; a patch on a live
    protected stream emits through that stream instead."""
    idle_policy = ExternalIdlePolicy()
    controller, engine, qpu, _ = make_controller(idle_policy)
    counters = ControllerCounters()
    controller.idle_round_emitted.connect(counters.idle_round_emitted)

    controller.emit_idle_round(7, "patch-a", 1)
    controller.emit_idle_round(7, "patch-a", 2)
    controller.streams.live_protected_patches.add("patch-a")
    controller.emit_idle_round(7, "patch-a", 3)

    assert qpu.feedback_rounds == [(7, "patch-a", 1), (7, "patch-a", 2)]
    assert counters.idle_rounds == 2
    assert engine.scheduled == []


def test_run_seed_binding_uses_distinct_policy_paths():
    """Boundary and idle consumers reserve distinct derived seeds before either commits."""
    events = []

    class SeededBoundary(ExternalBoundaryPolicy):
        def reserve_run_seed(self, seed):
            events.append(("reserve", "boundary", seed))
            return RunSeedReservation("derived", seed, None)

        def commit_run_seed(self, reservation):
            events.append(("commit", "boundary", reservation.proposed_seed))

        def cancel_run_seed(self, reservation):
            events.append(("cancel", "boundary", reservation.proposed_seed))

    class SeededIdle(ExternalIdlePolicy):
        def reserve_run_seed(self, seed):
            events.append(("reserve", "idle", seed))
            return RunSeedReservation("derived", seed, None)

        def commit_run_seed(self, reservation):
            events.append(("commit", "idle", reservation.proposed_seed))

        def cancel_run_seed(self, reservation):
            events.append(("cancel", "idle", reservation.proposed_seed))

    boundary_policy = SeededBoundary()
    idle_policy = SeededIdle()
    completed = Machine.build(
        MachineSettings(
            windows=WindowSettings(boundary_policy=boundary_policy),
            idle_policy=IdlePolicySettings(policy=idle_policy),
        ),
        23,
    )
    completed.run()

    assert completed.window_manager.courier.boundary_policy is boundary_policy
    assert completed.idle_rounds.policy is idle_policy
    assert [event[0] for event in events] == [
        "reserve",
        "reserve",
        "commit",
        "commit",
    ]
    reserved = {
        owner: seed for action, owner, seed in events if action == "reserve"
    }
    assert set(reserved) == {"boundary", "idle"}
    assert reserved["boundary"] != reserved["idle"]


def test_sliding_tail_follows_qldpc_rule():
    """The last window starts when fewer than W + F rounds remain and is
    never shorter than W (qLDPC SlidingWindowDecoder, sinter.py
    `while start < end - (W + s - 1)`)."""
    from decsim.windows.windowing_schemes import (
        _finite_forward_window_geometries,
    )

    def qldpc_windows(round_count, width, stride):
        start, windows = 0, []
        while start < round_count - (width + stride - 1):
            windows.append((start + 1, start + stride, start + width))
            start += stride
        windows.append((start + 1, round_count, round_count))
        return windows

    for round_count in (5, 13, 20, 30, 31, 32, 33):
        for commit, buffer in ((3, 6), (3, 3), (2, 4), (5, 5)):
            ours = [
                (g.commit_lo, g.commit_hi, g.buffer_hi)
                for g in _finite_forward_window_geometries(
                    round_count, commit, buffer
                )
            ]
            assert ours == qldpc_windows(round_count, commit + buffer, commit)
    last = _finite_forward_window_geometries(31, 3, 6)[-1]
    assert (last.commit_lo, last.commit_hi) == (22, 31)


# ---- idle rounds as decoder workload (2026-08-26 study) ----------------------
#
# References validated against, line for line:
# - SWIPER-SIM (ISCA 2025, arXiv 2412.05115): device_manager emits one
#   UNWANTED_IDLE syndrome round per unused patch per cycle and the window
#   builder has no idle special-case; idle volume is decode volume.
# - XQsim (ISCA 2022): the error decode unit consumes every patch under each
#   RUN_ESM; nothing is exempt.
# - Terhal's backlog bound (via Battistel, arXiv 2303.00054): the decoder
#   must cover the generation rate, so deleting idle volume undercounts.
# - Bombin et al. (arXiv 2303.04846) and the RT system stack (arXiv
#   2605.30765): a stall before a feed-forward decision generates more
#   syndrome; buffer-region content is decoded in every reference.


def _feedback_chain(idle_policy=None):
    """The T-gate feed-forward chain of the study: T1 blocked on T0's outcome
    idles the patch while T0's last window waits for its trailing buffer,
    exactly SWIPER Fig. 1 / Bombin's stall. Mirrors the refactor-lock
    feedback_chain spec."""
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.frontends.circuit_frontend import CircuitFrontend
    from decsim.message import Operation
    from decsim.qpu.code_geometry import SurfaceCodeModel
    from decsim.qpu.round_policies import FixedRounds

    ops = CircuitFrontend(
        [
            Operation(
                0, "T0", (0,), clifford=False, consumes_magic_state=False
            ),
            Operation(
                1,
                "T1",
                (0,),
                clifford=False,
                consumes_magic_state=False,
                blocked_by=0,
            ),
        ]
    ).build()
    settings = MachineSettings(
        workload=WorkloadSettings(
            operations=ops,
            rounds_policy=FixedRounds(3),
            feedback_boundary_mode="trailing_buffer",
        ),
        qpu=QpuSettings(
            code=SurfaceCodeModel(distance=3), round_period_microseconds=1.0
        ),
        windows=WindowSettings(
            scheme=SlidingWindowScheme(
                terminal_policy=SlidingTerminalPolicy.REGULAR_STRIDE_LOOKAHEAD
            )
        ),
        weak_decoder=DecoderSettings(
            decoder=PresetLatencyDecoder(2.0), units=1
        ),
        idle_policy=IdlePolicySettings(policy=idle_policy),
    )
    machine = Machine.build(settings, 13)
    machine.run()
    return machine


def _idle_decode_labels(completed) -> set:
    """The distinct synthetic idle-decode jobs the run charged."""
    labels = set()
    for line in completed.observation.log.lines:
        start = line.find("mem(")
        if start != -1:
            labels.add(line[start : line.index(")", start) + 1])
    return labels


def test_idle_rounds_cost_decode_jobs_by_default():
    """The default charges idle volume like the references; Ignore does not.

    SWIPER windows idle syndrome exactly like operation syndrome and XQsim
    decodes every patch each cycle, so the charged default must produce
    synthetic idle decode jobs wherever the patch idles, and the optimistic
    card must produce none while the rounds themselves still travel."""
    charged = _feedback_chain()
    optimistic = _feedback_chain(idle_policy=Ignore())

    assert _idle_decode_labels(charged), "the default charged no idle work"
    assert not _idle_decode_labels(optimistic)
    assert optimistic.observation.controller_counters.idle_rounds > 0
    assert (
        charged.observation.controller_counters.idle_rounds
        >= optimistic.observation.controller_counters.idle_rounds
    )


def test_memory_filled_trailing_buffer_is_flagged():
    """A trailing buffer satisfied by memory rounds alone is a time-only
    release with no syndrome content behind it, where every reference
    decodes the buffer region's content; the window carries the
    approximation flag and the manager counts it."""
    completed = _feedback_chain()

    filled_lines = [
        line
        for line in completed.observation.log.lines
        if "buffer filled by memory rounds" in line
    ]
    assert len(filled_lines) >= 1
    # the log marks the approximation; the release itself stands
    for window in completed.observation.windows.windows.values():
        assert window.t_data_complete is not None
        assert window.t_done is not None


def test_single_operation_run_charges_no_idle_work():
    """A workload whose one op keeps its patch busy every round emits no idle
    rounds, so the charged default is inert there: the single-op experiment
    sweeps are unchanged by the policy flip."""
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.frontends.circuit_frontend import CircuitFrontend
    from decsim.message import Operation
    from decsim.qpu.code_geometry import SurfaceCodeModel
    from decsim.qpu.round_policies import FixedRounds

    ops = CircuitFrontend(
        [
            Operation(0, "M", (0,), clifford=True, consumes_magic_state=False),
        ]
    ).build()
    settings = MachineSettings(
        workload=WorkloadSettings(operations=ops, rounds_policy=FixedRounds(6)),
        qpu=QpuSettings(
            code=SurfaceCodeModel(distance=3), round_period_microseconds=1.0
        ),
        windows=WindowSettings(scheme=SlidingWindowScheme()),
        weak_decoder=DecoderSettings(
            decoder=PresetLatencyDecoder(2.0), units=1
        ),
    )
    completed = Machine.build(settings, 13)
    completed.run()

    assert completed.observation.controller_counters.idle_rounds == 0
    assert not any(
        "buffer filled by memory rounds" in line
        for line in completed.observation.log.lines
    )
    assert not _idle_decode_labels(completed)

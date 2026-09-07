"""The root: components wired by hand as gem5 assigns ports, and by table.

Referent: gem5's learning_gem5/part1/simple.py names each component once
and assigns its ports (system.cpu.icache_port = system.membus...); here
a QPU device and a receiver are wired the same way and the readouts
arrive in cycle order. The table test follows sinter's BUILT_IN_DECODERS
(sinter/_decoding/_decoding_all_built_in_decoders.py): a new decoder is
one class and one row.
"""

import dataclasses
import pathlib

import pytest

import decsim.confidence.cluster as cluster
import decsim.config as config
import decsim.controller.policies as boundary_policies
import decsim.controller.settings as controller_settings
import decsim.decoders.decoder as decoder_module
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.decoders.staged_decoder as staged_decoder
import decsim.decoders.union_find.decoder as union_find_decoder
import decsim.engine as engine_module
import decsim.front.experiment as experiment
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.qpu.cycle_clock as cycle_clock
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.seeds as seed_records
import decsim.seeding as seeding
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings
import decsim.windows.settings as window_settings

THIS_FILE = pathlib.Path(__file__)
TESTS_DIRECTORY = THIS_FILE.parents[1]
CONFIGS = TESTS_DIRECTORY.parent / "configs"
CYCLE_TICKS = config.microseconds_to_ticks(1.0)


class RecordingReceiver:
    """A readout receiver that keeps every readout with its arrival tick."""

    def __init__(self, engine: engine_module.Engine):
        self.engine = engine
        self.arrivals = []

    def accept_qpu_readout(self, readout, route) -> None:
        del route
        self.arrivals.append((self.engine.now, readout.round_index))


class FakeWeakDecoder(decoder_module.DecoderBase):
    """A table row for the plug-in test: fixed latency, no correction."""

    def __init__(self, latency_model=None):
        del latency_model

    def latency(self, job: decoding_records.DecodeJob) -> int:
        del job
        return CYCLE_TICKS

    def decode(
        self, job: decoding_records.DecodeJob
    ) -> decoding_records.DecodeResult:
        return decoding_records.DecodeResult(job.op_id, job.window_id)


def test_readouts_reach_the_receiver_in_cycle_order_cycle_ticks_apart():
    engine = engine_module.Engine()
    receiver = RecordingReceiver(engine)
    device = syndrome_devices.TimingOnlyDevice()

    def finish_the_program(operation: program_records.Operation) -> None:
        del operation
        qpu.finish()

    qpu = cycle_clock.QPUDevice(
        engine,
        device,
        CYCLE_TICKS,
        readout_receiver=receiver,
        completion_receiver=finish_the_program,
    )
    operation = program_records.Operation(
        id=1, name="memory", qubits=(0,), patches=(0,)
    )
    body = program_records.RunOperationBody(
        operation,
        round_ticks=CYCLE_TICKS,
        round_count=3,
        source_round_count=3,
    )
    qpu.issue(body)
    engine.run()
    assert receiver.arrivals == [
        (CYCLE_TICKS, 1),
        (2 * CYCLE_TICKS, 2),
        (3 * CYCLE_TICKS, 3),
    ]


def test_a_new_decoder_is_one_class_and_one_table_row():
    """Gate point 1's settings run to completion on a decoder added as a row."""
    config_path = CONFIGS / "weak_decoder_baseline.yaml"
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.003, distance=3, round_period_us=1.0
    )
    weak_decoder = dataclasses.replace(settings.weak_decoder, kind="fake")
    settings = dataclasses.replace(settings, weak_decoder=weak_decoder)
    machine_module.DECODERS["fake"] = FakeWeakDecoder
    try:
        machine = machine_module.Machine.build(settings, 0)
        result = machine.run()
    finally:
        del machine_module.DECODERS["fake"]
    assert result.terminal_status == "complete"
    assert type(machine.active_decoder.decoder) is FakeWeakDecoder
    decode_lines = [
        line for line in machine.observation.log.lines if "decode" in line
    ]
    assert decode_lines


def test_a_second_table_row_runs_gate_point_one():
    """Gate point 1's settings run to completion on the union_find row."""
    config_path = CONFIGS / "weak_decoder_baseline.yaml"
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.003, distance=3, round_period_us=1.0
    )
    weak_decoder = dataclasses.replace(settings.weak_decoder, kind="union_find")
    settings = dataclasses.replace(settings, weak_decoder=weak_decoder)
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    assert result.terminal_status == "complete"
    inner = machine.active_decoder.decoder
    assert type(inner) is union_find_decoder.UnionFindDecoder
    assert result.operation_results[0].logical_observables is not None


def test_a_decoder_kind_off_the_table_is_refused_naming_the_rows():
    weak_decoder = decoder_settings.DecoderSettings(kind="lookup_table")
    settings = machine_module.MachineSettings(weak_decoder=weak_decoder)
    with pytest.raises(
        ValueError,
        match="weak_decoder.kind 'lookup_table' is not a row of its table; "
        r"the rows are \['belief_matching', 'bposd', 'pymatching', "
        r"'relay_bp', 'tesseract', 'union_find', 'unweighted_pymatching'\]",
    ):
        machine_module.Machine.build(settings)


def test_a_strong_store_kind_off_the_table_is_refused_even_when_unused():
    # The weak baseline never reads the strong store, but the yaml still
    # names its kind, and a kind off the table is a mistake in the yaml.
    strong_round_store = round_store_settings.RoundStoreSettings(
        kind="off_table"
    )
    settings = machine_module.MachineSettings(
        strong_round_store=strong_round_store
    )
    with pytest.raises(
        ValueError,
        match="strong_round_store.kind 'off_table' is not a row of its "
        r"table; the rows are \['round_store'\]",
    ):
        machine_module.Machine.build(settings)


def test_a_yaml_section_nobody_owns_is_refused_naming_the_sections():
    with pytest.raises(
        ValueError, match=r"the yaml has no section \['buffers'\]; the sections"
    ):
        machine_module.MachineSettings.from_mapping(
            {"buffers": {}}, name="x", base_directory=None
        )


def test_the_constructor_wiring_reaches_its_components():
    """Every cross-reference is made by constructor, none left None."""
    settings = machine_module.MachineSettings()
    machine = machine_module.Machine.build(settings)
    assert machine.qpu.readout_receiver is machine.controller
    assert machine.execution_runtime.issuer is machine.issuer
    manager = machine.decoder_manager
    assert machine.window_manager.requester.decode_queue is manager


MEMORY_ROUNDS = 6
MEMORY_CIRCUIT = workload_settings.memory_circuit(
    "surface_code:rotated_memory_z", MEMORY_ROUNDS, 3, 0.003
)


def _memory_on_stim_device(
    weak_decoder, operations
) -> machine_module.MachineSettings:
    """The operations as a six-round d=3 memory on the Stim device."""
    rounds = round_policies.FixedRounds(MEMORY_ROUNDS)
    workload = workload_settings.WorkloadSettings(
        operations=operations, rounds_policy=rounds
    )
    device = stim_device.StimDevice()
    qpu = qpu_settings.QpuSettings(distance=3, device=device)
    return machine_module.MachineSettings(
        workload=workload, qpu=qpu, weak_decoder=weak_decoder
    )


def _two_patch_memory(weak_decoder) -> machine_module.MachineSettings:
    """Two memory operations on two patches; the second starts at round 4.

    Patch 1 idles for four rounds first, and the default idle policy
    (separate_decode_jobs) charges that idle region as one load-only
    decode job, a job without a window model.
    """
    first = program_records.Operation(
        id=1, name="mem0", qubits=(0,), patches=(0,), circuit=MEMORY_CIRCUIT
    )
    late = program_records.Operation(
        id=2,
        name="mem1",
        qubits=(1,),
        patches=(1,),
        circuit=MEMORY_CIRCUIT,
        scheduled_start_round=4,
    )
    return _memory_on_stim_device(weak_decoder, (first, late))


def test_a_load_only_job_on_a_measured_unit_holds_it_for_zero_algorithm_ticks():
    """A job without a model spends no host time on the measured row.

    The unit's algorithm stage is zero ticks for it, while every window
    with a model holds the unit for its measured time.
    """
    weak_decoder = decoder_settings.DecoderSettings(
        kind="pymatching", engine_megahertz=250.0
    )
    settings = _two_patch_memory(weak_decoder)
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    assert result.terminal_status == "complete"
    algorithm = [
        record
        for record in machine.observation.stages.records
        if record.stage == staged_decoder.ALGORITHM_STAGE
    ]
    operation_ids = {1, 2}
    load_only = [r for r in algorithm if r.op_id not in operation_ids]
    windows = [r for r in algorithm if r.op_id in operation_ids]
    assert len(load_only) == 1
    assert len(windows) == 2
    idle_ticks = [r.end_ticks - r.start_ticks for r in load_only]
    assert idle_ticks == [0]
    assert all(r.end_ticks > r.start_ticks for r in windows)
    idle_lines = [
        line for line in machine.observation.log.lines if "mem(" in line
    ]
    assert any("algorithm mem(" in line for line in idle_lines)


def test_the_cluster_gap_decoder_runs_as_a_python_built_tier():
    """The cluster-gap wrapper is a row of the port.

    The manager reads pipeline_depth at dispatch and occupancy at its
    free-time prediction; a wrapper without them stopped the run at the
    first window with AttributeError.
    """
    latency_model = decoders.PresetLatencyDecoder(1.0)
    base = union_find_decoder.UnionFindDecoder(latency_model)
    decoder = cluster.UnionFindClusterGapDecoder(base)
    weak_decoder = decoder_settings.DecoderSettings(decoder=decoder)
    operation = program_records.Operation(
        id=1, name="memory", qubits=(0,), patches=(0,), circuit=MEMORY_CIRCUIT
    )
    settings = _memory_on_stim_device(weak_decoder, (operation,))
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    assert result.terminal_status == "complete"
    observables = result.operation_results[0].logical_observables
    assert observables == (0,)
    assert len(machine.observation.log.lines) == 19


TIER_ROWS = (
    r"\['belief_matching', 'bposd', 'pymatching', 'relay_bp', 'tesseract', "
    r"'union_find', 'unweighted_pymatching'\]"
)


@pytest.mark.parametrize(
    "escalation_kind, tier",
    [
        ("weak_baseline", "weak"),
        ("strong_only", "strong"),
        ("switching", "weak"),
        ("switching", "strong"),
    ],
)
def test_the_cluster_gap_is_not_a_tier_kind_under_any_escalation(
    escalation_kind, tier
):
    """The cluster gap is the Python-built Decoder of the test above.

    It reports its own soft output in decibels at its weight step, where
    switching decides on the complementary gap in nats, and no tier
    that does not switch reads a soft output at all; so it is not a row
    of the tier table under any escalation kind.
    """
    tiers = {
        "weak": decoder_settings.DecoderSettings(
            kind="pymatching", engine_megahertz=100.0
        ),
        "strong": decoder_settings.DecoderSettings(
            kind="belief_matching", engine_megahertz=100.0
        ),
    }
    tiers[tier] = decoder_settings.DecoderSettings(
        kind="union_find_cluster_gap", engine_megahertz=100.0
    )
    escalation = decoder_settings.EscalationSettings(kind=escalation_kind)
    if escalation_kind == "switching":
        escalation = decoder_settings.EscalationSettings(
            kind="switching", gap_threshold_nats=1.0
        )
    settings = machine_module.MachineSettings(
        weak_decoder=tiers["weak"],
        strong_decoder=tiers["strong"],
        escalation=escalation,
    )
    sentence = (
        f"{tier}_decoder.kind 'union_find_cluster_gap' is not a row of its "
        "table; the rows are " + TIER_ROWS
    )
    with pytest.raises(ValueError, match=sentence):
        machine_module.Machine.build(settings, 0)


class CountingRoundStore(round_store_module.RoundStore):
    """A table row for the plug-in test: the store, counting its writes."""

    def __init__(self, settings, *, on_slot_freed=None):
        round_store_module.RoundStore.__init__(
            self, settings, on_slot_freed=on_slot_freed
        )
        self.stored_count = 0

    def accept_packed_round(self, packet, *, publication_tick):
        self.stored_count += 1
        round_store_module.RoundStore.accept_packed_round(
            self, packet, publication_tick=publication_tick
        )


def test_a_new_round_store_is_one_class_and_one_table_row():
    """Gate point 1's settings run to completion on a store added as a row."""
    config_path = CONFIGS / "weak_decoder_baseline.yaml"
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.003, distance=3, round_period_us=1.0
    )
    round_store = dataclasses.replace(settings.round_store, kind="counting")
    settings = dataclasses.replace(settings, round_store=round_store)
    machine_module.ROUND_STORES["counting"] = CountingRoundStore
    try:
        machine = machine_module.Machine.build(settings, 0)
        result = machine.run()
    finally:
        del machine_module.ROUND_STORES["counting"]
    assert result.terminal_status == "complete"
    assert type(machine.round_store) is CountingRoundStore
    fired = [
        line for line in machine.observation.log.lines if "fires round" in line
    ]
    assert machine.round_store.stored_count == len(fired)
    assert fired


class RecordingBoundaryPolicy:
    """A boundary policy written outside decsim: one method, plain names."""

    def on_commit(self, window, final: bool) -> bool:
        del window
        return final


class RecordingIdlePolicy:
    """An idle policy written outside decsim: the two the port asks for."""

    def relay(self, idle_rounds, operation, patch, round_index: int) -> None:
        idle_rounds.emit_memory_round(operation, patch, round_index)

    def end_idle_period(self, idle_rounds, operation, patch) -> None:
        del idle_rounds, operation, patch


def test_the_default_policies_are_eager_boundaries_and_charged_idle_rounds():
    """Two builds of the same settings each get their own policy objects."""
    settings = machine_module.MachineSettings()
    first = machine_module.Machine.build(settings)
    second = machine_module.Machine.build(settings)
    first_boundary = first.window_manager.courier.boundary_policy
    second_boundary = second.window_manager.courier.boundary_policy
    assert type(first_boundary) is boundary_policies.Eager
    assert type(second_boundary) is boundary_policies.Eager
    assert first_boundary is not second_boundary
    assert type(first.idle_rounds.policy) is (
        boundary_policies.SeparateDecodeJobs
    )
    assert first.idle_rounds.policy is not second.idle_rounds.policy


def test_a_policy_written_outside_decsim_is_used_on_its_own_axis():
    """The two policy axes are independent: one given, the other default."""
    boundary_policy = RecordingBoundaryPolicy()
    windows = window_settings.WindowSettings(boundary_policy=boundary_policy)
    boundary_settings = machine_module.MachineSettings(windows=windows)
    with_boundary = machine_module.Machine.build(boundary_settings)
    idle_policy = RecordingIdlePolicy()
    idle_row = controller_settings.IdlePolicySettings(policy=idle_policy)
    idle_settings = machine_module.MachineSettings(idle_policy=idle_row)
    with_idle = machine_module.Machine.build(idle_settings)
    assert with_boundary.window_manager.courier.boundary_policy is (
        boundary_policy
    )
    assert type(with_boundary.idle_rounds.policy) is (
        boundary_policies.SeparateDecodeJobs
    )
    assert with_idle.idle_rounds.policy is idle_policy
    assert type(with_idle.window_manager.courier.boundary_policy) is (
        boundary_policies.Eager
    )


class SeedRecordingPolicy:
    """A boundary policy that records the seed the root derived for it."""

    def __init__(self):
        self.reserved_seeds = []

    def on_commit(self, window, final: bool) -> bool:
        del window
        return final

    def reserve_run_seed(self, seed):
        self.reserved_seeds.append(seed)
        source = "derived"
        if seed is None:
            source = "entropy"
        return seed_records.RunSeedReservation(source, seed, None)

    def commit_run_seed(self, reservation):
        del reservation

    def cancel_run_seed(self, reservation):
        del reservation


def _machine_with_seed(policy, seed):
    windows = window_settings.WindowSettings(boundary_policy=policy)
    settings = machine_module.MachineSettings(windows=windows)
    return machine_module.Machine.build(settings, seed)


def test_a_component_gets_the_seed_derived_from_the_runs_seed_and_its_path():
    policy = SeedRecordingPolicy()
    _machine_with_seed(policy, 31)
    path = (seed_records.RunSeedPathSegment("field", "boundary_policy"),)
    expected = seeding.derive_component_seed(31, path)
    assert policy.reserved_seeds == [expected]


def test_a_run_without_a_seed_leaves_every_component_on_entropy():
    policy = SeedRecordingPolicy()
    _machine_with_seed(policy, None)
    assert policy.reserved_seeds == [None]


def test_an_integral_seed_of_another_type_is_taken_as_its_value():
    """A numpy integer or an int subclass from a sweep is one int here."""

    class IntegerSubclass(int):
        pass

    policy = SeedRecordingPolicy()
    swept_seed = IntegerSubclass(31)
    _machine_with_seed(policy, swept_seed)
    path = (seed_records.RunSeedPathSegment("field", "boundary_policy"),)
    expected = seeding.derive_component_seed(31, path)
    assert policy.reserved_seeds == [expected]


def test_a_seed_outside_the_unsigned_64_bit_range_is_refused():
    settings = machine_module.MachineSettings()
    one_past_the_widest_seed = 1 << 64
    with pytest.raises(ValueError, match=r"seed must be in \[0, 2\*\*64\)"):
        machine_module.Machine.build(settings, one_past_the_widest_seed)


def test_a_negative_seed_is_refused():
    settings = machine_module.MachineSettings()
    with pytest.raises(ValueError, match=r"seed must be in \[0, 2\*\*64\)"):
        machine_module.Machine.build(settings, -1)


def test_a_seed_that_is_not_a_number_is_refused():
    settings = machine_module.MachineSettings()
    with pytest.raises(ValueError):
        machine_module.Machine.build(settings, "x")

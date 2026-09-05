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
import decsim.decoders.decoder as decoder_module
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.decoders.staged_decoder as staged_decoder
import decsim.decoders.union_find.decoder as union_find_decoder
import decsim.engine as engine_module
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.message as message
import decsim.qpu.cycle_clock as cycle_clock
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings
import experiments.experiment_config as experiment_config

THIS_FILE = pathlib.Path(__file__)
TESTS_DIRECTORY = THIS_FILE.parents[1]
CONFIGS = TESTS_DIRECTORY.parent / "experiments" / "configs"
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

    def latency(self, job: message.DecodeJob) -> int:
        del job
        return CYCLE_TICKS

    def decode(self, job: message.DecodeJob) -> message.DecodeResult:
        return message.DecodeResult(job.op_id, job.window_id)


def test_readouts_reach_the_receiver_in_cycle_order_cycle_ticks_apart():
    engine = engine_module.Engine(verbose=False)
    receiver = RecordingReceiver(engine)
    device = syndrome_devices.TimingOnlyDevice()

    def finish_the_program(operation: message.Operation) -> None:
        del operation
        qpu.finish()

    qpu = cycle_clock.QPUDevice(
        engine,
        device,
        CYCLE_TICKS,
        readout_receiver=receiver,
        completion_receiver=finish_the_program,
    )
    operation = message.Operation(
        id=1, name="memory", qubits=(0,), patches=(0,)
    )
    body = message.RunOperationBody(
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
    experiment = experiment_config.load_experiment(config_path)
    settings = experiment.point_settings(
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
        line for line in machine.engine.log_lines if "decode" in line
    ]
    assert decode_lines


def test_a_second_table_row_runs_gate_point_one():
    """Gate point 1's settings run to completion on the union_find row."""
    config_path = CONFIGS / "weak_decoder_baseline.yaml"
    experiment = experiment_config.load_experiment(config_path)
    settings = experiment.point_settings(
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


def test_the_post_construction_binds_reach_their_components():
    """The debt slices 6 and 7 retire: the binds are made, not left None."""
    settings = machine_module.MachineSettings()
    machine = machine_module.Machine.build(settings)
    assert machine.qpu.readout_receiver is machine.controller
    assert machine.execution_runtime.issuer is machine.issuer
    manager = machine.decoder_manager
    assert machine.window_manager.requester.decode_queue is manager
    assert manager.services is machine.window_manager.escalation


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
    first = message.Operation(
        id=1, name="mem0", qubits=(0,), patches=(0,), circuit=MEMORY_CIRCUIT
    )
    late = message.Operation(
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
    unit = machine.active_decoder
    algorithm = [
        record
        for record in unit.stage_records
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
    idle_lines = [line for line in machine.engine.log_lines if "mem(" in line]
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
    operation = message.Operation(
        id=1, name="memory", qubits=(0,), patches=(0,), circuit=MEMORY_CIRCUIT
    )
    settings = _memory_on_stim_device(weak_decoder, (operation,))
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    assert result.terminal_status == "complete"
    observables = result.operation_results[0].logical_observables
    assert observables == (0,)
    assert len(machine.engine.log_lines) == 19


class CountingRoundStore(round_store_module.RoundStore):
    """A table row for the plug-in test: the store, counting its writes."""

    def __init__(self, settings, *, on_slot_freed=None, listener=None):
        round_store_module.RoundStore.__init__(
            self, settings, on_slot_freed=on_slot_freed, listener=listener
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
    experiment = experiment_config.load_experiment(config_path)
    settings = experiment.point_settings(
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
    fired = [line for line in machine.engine.log_lines if "fires round" in line]
    assert machine.round_store.stored_count == len(fired)
    assert fired

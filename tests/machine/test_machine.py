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

import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.engine as engine_module
import decsim.machine as machine_module
import decsim.message as message
import decsim.qpu.cycle_clock as cycle_clock
import decsim.qpu.syndrome_devices as syndrome_devices
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


class FakeWeakDecoder:
    """A table row for the plug-in test: fixed latency, no correction."""

    fault_model_requirement = fault_models.NO_FAULT_MODEL_REQUIRED

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


def test_a_decoder_kind_off_the_table_is_refused_naming_the_rows():
    weak_decoder = decoder_settings.DecoderSettings(kind="union_find")
    settings = machine_module.MachineSettings(weak_decoder=weak_decoder)
    with pytest.raises(
        ValueError,
        match="weak_decoder.kind 'union_find' is not a row of its table; the "
        r"rows are \['belief_matching', 'pymatching'\]",
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


def test_the_three_post_construction_binds_reach_their_components():
    """The debt slices 5 and 6 retire: the binds are made, not left None."""
    settings = machine_module.MachineSettings()
    machine = machine_module.Machine.build(settings)
    assert machine.qpu.readout_receiver is machine.controller
    assert machine.controller.runtime is machine.execution_runtime
    manager = machine.decoder_manager
    assert machine.window_manager.release_service == manager.release_parked
    assert machine.window_manager.withdraw_decode == manager.withdraw_window

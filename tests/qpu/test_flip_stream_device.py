"""The flip stream: Stim's FlipSimulator a round at a time, kept flat.

The referents are Stim's own. With stabilizer randomization off, a
rollover changes no bit, so every round's raw bits, the detection events
formed from them and the observables' flips must equal those of one
FlipSimulator.do over the whole assembled circuit under the same seed
(probes p2 and p4, Adroit config_proto/probes5). The tables must be the
live source's, which reads them off the circuit run so far. A checkpoint
after round t must equal Stim's own generated circuit of t rounds whose
readout is noiseless, run under the same seed.
"""

import dataclasses

import numpy
import pytest
import stim

import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.qpu.flip_stream_device as flip_stream_device
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.streaming_stim_device as streaming_stim_device
import decsim.records.program as program_records
import decsim.records.windows as window_records
import tests.declared_run as declared_run
import tests.qpu.memory_programs as memory_programs
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

SEED = 5
# high enough that a 9-round d=3 memory fires detectors and flips
NOISY = 0.02


def _program():
    return memory_programs.memory_program(3, "Z", NOISY)


def _streamed(program, round_count: int, rollover_rounds: int):
    """(history, raw packets by round) of one stream, randomization off."""
    plan = flip_stream_device.RoundPlan(program, rollover_rounds)
    history = flip_stream_device.FlipHistory(plan, SEED, False)
    packets = {}
    after_last_round = round_count + 1
    for round_index in range(1, after_last_round):
        is_final = round_index == round_count
        packets[round_index] = history.run_round(is_final)
    return history, packets


def _formed(table, packets: dict) -> list:
    """Every detector's event, formed round by round off the packets."""
    former = detector_formation.StreamingDetectorFormer(table)
    events = {}
    for round_index, bits in packets.items():
        for detector_index, bit in former.feed_packet(round_index, bits):
            events[detector_index] = bit
    return [events[index] for index in sorted(events)]


def _whole_circuit_run(program, round_count: int) -> tuple:
    """(raw bits, detector flips, observable flips) of one whole run."""
    circuit, _measurement_rounds = program.assemble(round_count)
    simulator = stim.FlipSimulator(
        batch_size=1, seed=SEED, disable_stabilizer_randomization=True
    )
    simulator.do(circuit)
    flips = simulator.get_measurement_flips()
    reference = circuit.reference_sample()
    raw = numpy.logical_xor(flips[:, 0], reference)
    detectors = simulator.get_detector_flips()
    observables = simulator.get_observable_flips()
    return (
        [int(bit) for bit in raw],
        [int(bit) for bit in detectors[:, 0]],
        tuple(int(bit) for bit in observables[:, 0]),
    )


@pytest.mark.parametrize("round_count", [1, 2, 9, 23])
@pytest.mark.parametrize("rollover_rounds", [1, 3, 1000])
def test_the_stream_gives_the_whole_circuits_bits_events_and_truth(
    round_count: int, rollover_rounds: int
):
    program = _program()
    history, packets = _streamed(program, round_count, rollover_rounds)
    raw, detectors, observables = _whole_circuit_run(program, round_count)

    streamed_raw = [bit for bits in packets.values() for bit in bits]
    table = history.table()
    assert streamed_raw == raw
    assert _formed(table, packets) == detectors
    assert history.truth() == observables
    assert sum(detectors) > 0 or round_count < 9


@pytest.mark.parametrize("rollover_rounds", [1, 3, 1000])
def test_a_checkpoint_is_stims_circuit_of_its_rounds_read_out_noiseless(
    rollover_rounds: int,
):
    program = _program()
    plan = flip_stream_device.RoundPlan(program, rollover_rounds)
    history = flip_stream_device.FlipHistory(plan, SEED, False)

    snapshots = [_checkpoint_after_a_round(history) for _ in range(24)]

    # Stim writes a REPEAT block from three rounds on
    closed = [
        (snapshot.readout_events, snapshot.observable_flips)
        for snapshot in snapshots[2:]
    ]
    referents = [_noiseless_readout_run(rounds) for rounds in range(3, 25)]
    truths = [truth for _events, truth in referents]
    assert closed == referents
    assert (0,) in truths
    assert (1,) in truths


def test_the_last_checkpoint_is_the_real_readout():
    program = _program()
    history, _packets = _streamed(program, 9, 3)

    snapshot = history.snapshot(100)

    _raw, detectors, observables = _whole_circuit_run(program, 9)
    readout_count = len(snapshot.readout_events)
    assert snapshot.is_final
    assert list(snapshot.readout_events) == detectors[-readout_count:]
    assert snapshot.observable_flips == observables


def test_a_checkpoint_fires_after_every_round_while_someone_listens():
    program = _program()
    owner = program_records.Operation(100, "memory", (0,), patches=(0,))
    device = flip_stream_device.FlipStreamDevice({100: program}, seed=SEED)
    device.declare_stream(owner, 0)
    heard = []
    device.checkpoint_reached.connect(heard.append)

    _run_device_rounds(device, owner, 5)

    rounds = [(snapshot.round_index, snapshot.is_final) for snapshot in heard]
    assert rounds == [(1, False), (2, False), (3, False), (4, False), (5, True)]


def test_its_tables_are_the_live_sources_round_by_round():
    """Each round's table, live then final, as the live source reads it."""
    program = _program()
    owner = program_records.Operation(100, "memory", (0,), patches=(0,))
    flip = flip_stream_device.FlipStreamDevice({100: program}, seed=SEED)
    live = streaming_stim_device.StreamingStimDevice({100: program}, SEED)
    flip.declare_stream(owner, 0)
    live.declare_stream(owner, 0)
    round_count = 12
    after_last_round = round_count + 1

    for round_index in range(1, after_last_round):
        is_final = round_index == round_count
        flip.idle_round_payloads(
            owner, 100, round_index, is_final=is_final, round_period_ticks=1
        )
        live.idle_round_payloads(
            owner, 100, round_index, is_final=is_final, round_period_ticks=1
        )
        flip_table = flip.formation_table(100)
        live_table = live.formation_table(100)
        assert flip_table.detectors == live_table.detectors
        assert flip_table.observables == live_table.observables
        assert flip_table.live_reach == live_table.live_reach


def test_a_protected_region_reads_the_flip_stream_out_in_a_machine():
    """The live source's workload runs on the flip stream to its readout."""
    final_round = 14
    program = _program()
    device = flip_stream_device.FlipStreamDevice({100: program})
    workload = _workload(final_round)
    machine = _machine(workload, device)
    packets = []
    machine.qpu.device.trace.round_emitted.connect(packets.append)

    result = machine.run()

    assert result.terminal_status == "complete"
    assert result.decode_work_settled
    assert len(packets) == final_round
    owner_result = result.operation_results[-1]
    assert owner_result.operation_id == 100
    assert owner_result.observable_truth == device.logical_observable_truth(100)
    assert owner_result.logical_failure is not None


def test_the_stream_runs_its_rounds_once_and_in_order():
    program = _program()
    owner = program_records.Operation(100, "memory", (0,), patches=(0,))
    device = flip_stream_device.FlipStreamDevice({100: program})
    device.declare_stream(owner, 0)
    with pytest.raises(RuntimeError, match="runs its rounds in order"):
        device.idle_round_payloads(
            owner, 100, 2, is_final=False, round_period_ticks=1
        )
    device.idle_round_payloads(
        owner, 100, 1, is_final=True, round_period_ticks=1
    )
    with pytest.raises(RuntimeError, match="nothing after its readout"):
        device.idle_round_payloads(
            owner, 100, 2, is_final=False, round_period_ticks=1
        )


def test_a_static_window_plan_is_not_implemented_yet():
    program = _program()
    device = flip_stream_device.FlipStreamDevice({100: program})
    owner = program_records.Operation(100, "memory", (0,), patches=(0,))
    window = window_records.Window(
        stream_id=100,
        window_index=0,
        commit_lo=1,
        commit_hi=3,
        buffer_hi=3,
        round_count=3,
    )
    with pytest.raises(ValueError, match="not implemented yet"):
        device.window_models_for_operation(
            owner,
            [window],
            3,
            fault_model_requirement=None,
            fault_exclusion_ranges=(),
            window_protocol=None,
        )


@pytest.mark.parametrize(
    "values, sentence",
    [
        ({"rollover_rounds": 0}, "flip_stream.rollover_rounds must be"),
        (
            {"stabilizer_randomization": "yes"},
            "flip_stream.stabilizer_randomization must be true or false",
        ),
        ({"drift_round": 0}, "flip_stream.drift_round must be"),
        ({"drift_factor": -1.0}, "flip_stream.drift_factor must be"),
    ],
)
def test_the_settings_refuse_a_value_by_name(values, sentence):
    with pytest.raises(ValueError, match=sentence):
        flip_stream_device.FlipStreamDevice.Settings(**values)


def test_a_drift_by_two_is_the_program_at_twice_the_probability():
    """Stim's own memory at 2p is the one at p with its noise doubled."""
    program = memory_programs.memory_program(3, "Z", 0.01)
    doubled = memory_programs.memory_program(3, "Z", 0.02)

    for name in ("first_round", "repeated_round", "final_round"):
        fragment = getattr(program, name)
        scaled = flip_stream_device.scaled_noise(fragment, 2.0)
        assert scaled == getattr(doubled, name)


def test_a_drifted_stream_is_the_plain_one_up_to_its_drift_round():
    program = _program()
    plain = flip_stream_device.RoundPlan(program, 3)
    drifted = flip_stream_device.RoundPlan(program, 3, (5, 3.0))

    plain_bits = _run_plan(plain, 9)
    drifted_bits = _run_plan(drifted, 9)

    assert drifted_bits[:5] == plain_bits[:5]
    assert drifted_bits[5:] != plain_bits[5:]


def test_a_drift_to_no_noise_repeats_each_round_from_two_after_it():
    """With no new error, a stabilizer reads the same every round.

    The round after the drift round still reads the errors of the round
    before it, which its detectors compare against.
    """
    program = _program()
    plan = flip_stream_device.RoundPlan(program, 3, (4, 0.0))

    bits = _run_plan(plan, 12)

    repeated = bits[5:11]
    assert all(round_bits == repeated[0] for round_bits in repeated)
    assert bits[0:4] != [bits[0]] * 4


def _checkpoint_after_a_round(history):
    """The next round run, then the stream as if read out after it."""
    history.run_round(False)
    return history.snapshot(100)


def _noiseless_readout_run(round_count: int) -> tuple:
    """(readout events, observable flips) of Stim's circuit, read noiseless.

    Stim's generated memory of round_count rounds, its rounds as written
    and the instructions after its REPEAT block, the readout, without
    noise, run by one FlipSimulator under the stream's seed.
    """
    circuit = memory_programs.memory_circuit(round_count, 3, "Z", NOISY)
    instructions = list(circuit)
    repeat_index = [instruction.name for instruction in instructions].index(
        "REPEAT"
    )
    rounds = circuit[: repeat_index + 1]
    readout = circuit[repeat_index + 1 :]
    quiet_readout = readout.without_noise()
    simulator = stim.FlipSimulator(
        batch_size=1, seed=SEED, disable_stabilizer_randomization=True
    )
    simulator.do(rounds)
    simulator.do(quiet_readout)
    detectors = simulator.get_detector_flips()
    observables = simulator.get_observable_flips()
    readout_count = quiet_readout.num_detectors
    events = detectors[detectors.shape[0] - readout_count :, 0]
    return (
        tuple(int(bit) for bit in events),
        tuple(int(bit) for bit in observables[:, 0]),
    )


def _run_device_rounds(device, owner, round_count: int) -> None:
    """Run the stream's rounds on the device, the last its readout."""
    after_last_round = round_count + 1
    for round_index in range(1, after_last_round):
        is_final = round_index == round_count
        device.idle_round_payloads(
            owner, 100, round_index, is_final=is_final, round_period_ticks=1
        )


def _run_plan(plan, round_count: int) -> list:
    """Each round's raw bits, the last round the final one."""
    history = flip_stream_device.FlipHistory(plan, SEED, False)
    bits = []
    after_last_round = round_count + 1
    for round_index in range(1, after_last_round):
        is_final = round_index == round_count
        round_bits = history.run_round(is_final)
        bits.append(round_bits)
    return bits


def _workload(final_round: int) -> workload_settings.WorkloadSettings:
    """A live stream, protected from round 1, read out at final_round."""
    owner = program_records.Operation(100, "memory", (0,), patches=(0,))
    begin = program_records.Operation(
        1, "protect", (0,), patches=(0,), emits_detector_data=False
    )
    finish = dataclasses.replace(
        begin,
        id=2,
        name="readout",
        predecessors=(1,),
        scheduled_start_round=final_round,
    )
    region = program_records.ProtectedRegion(100, 1, 2)
    policy = round_policies.PerOperationRounds(((100, 0), (1, 0), (2, 0)))
    return workload_settings.WorkloadSettings(
        operations=(begin, finish),
        dynamic_streams=(owner,),
        protected_regions=(region,),
        rounds_policy=policy,
    )


def _machine(workload, device) -> machine_module.Machine:
    """A d=3 run on the device, one-microsecond rounds, PyMatching."""
    source = declared_run.GivenSource(device)
    qpu = qpu_settings.QpuSettings(
        distance=3, source=source, round_period_microseconds=1.0
    )
    clock = config.Clock(1000)
    engine = decoder_settings.EngineSettings(clock=clock)
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=0.1
    )
    decoder = decoder_settings.DecoderPoolSettings(
        algorithm=matching, engine=engine
    )
    base = machines.base(workload, qpu)
    graph = machines.with_pool(base, decoder)
    return machine_module.Machine.build(graph, 17)

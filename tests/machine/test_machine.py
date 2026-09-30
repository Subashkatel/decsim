"""The root: components wired by hand as gem5 assigns ports, and by table.

Referent: gem5's learning_gem5/part1/simple.py names each component once
and assigns its ports (system.cpu.icache_port = system.membus...); here
a QPU device and a receiver are wired the same way and the readouts
arrive in cycle order. The table test follows sinter's BUILT_IN_DECODERS
(sinter/_decoding/_decoding_all_built_in_decoders.py): a new decoder is
one class and one row.

Strong-primary stream checks compare readout with Stim and terminal
prediction with PyMatching on the same circuit. They pin the direct strong
route, bounded storage lifetime and configured service/link timings.

The laws at the end of the file are the ones only the whole machine
holds: they run a declared card (tests/declared_run.py, whose every
latency is a number the test states) and read what the composition of
the controller, the stores, the windows and the frame produced.
"""

import dataclasses
import functools
import importlib.util
import pathlib
import re

import ldpc
import ldpc.ckt_noise.dem_matrices as dem_matrices
import numpy
import pymatching
import pytest
import stim

import decsim.build.control as control_part
import decsim.build.decoders as decoders_part
import decsim.build.escalation as escalation_build
import decsim.build.plan as plan_build
import decsim.build.qpu as qpu_part
import decsim.build.readout as readout_part
import decsim.build.windows as windows_part
import decsim.collect as collect
import decsim.config as config
import decsim.controller.policies as idle_policies
import decsim.controller.settings as controller_settings
import decsim.decoders.belief_propagation_osd.decoder as belief_propagation_osd
import decsim.decoders.decoder as decoder_module
import decsim.decoders.decoder_memory as decoder_memory
import decsim.decoders.decoders as decoders
import decsim.decoders.minimum_weight_perfect_matching.decoder as mwpm
import decsim.decoders.settings as decoder_settings
import decsim.decoders.staged_decoder as staged_decoder
import decsim.decoders.union_find.decoder as union_find_decoder
import decsim.detector_error_model.detector_chronology as detector_chronology
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.engine as engine_module
import decsim.escalation.policies as escalation_policies
import decsim.escalation.settings as escalation_settings
import decsim.experiments.experiment as experiment
import decsim.frontends.deltakit as deltakit
import decsim.frontends.settings as workload_settings
import decsim.links.settings as link_settings
import decsim.machine as machine_module
import decsim.observe.settings as observe_settings
import decsim.ports as ports
import decsim.producers as producers
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.cycle_clock as cycle_clock
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.records.circuits as circuit_records
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.results as result_records
import decsim.records.rounds as round_records
import decsim.records.seeds as seed_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.seeding as seeding
import decsim.settings as machine_settings
import decsim.syndrome_buffer.ported_syndrome_buffer as ported_syndrome_buffer
import decsim.syndrome_buffer.settings as syndrome_buffer_settings
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import decsim.trace_source as trace_source
import decsim.windows.boundary_policies as boundary_policies
import decsim.windows.built_window_models as built_window_models
import decsim.windows.schemes.naive_online as naive_online
import decsim.windows.settings as window_settings
import tests.declared_run as declared_run
import tests.experiments.yaml_configs as yaml_configs
import tests.machine.decoder_arrangements as decoder_arrangements
import tests.qpu.memory_programs as memory_programs
import tools.deltakit_example as finite_example
import tools.live_memory_example as live_example

THIS_FILE = pathlib.Path(__file__)
TESTS_DIRECTORY = THIS_FILE.parents[1]
CONFIGS = TESTS_DIRECTORY.parent / "configs"
CYCLE_TICKS = config.microseconds_to_ticks(1.0)
# the syndrome bits of one round of a distance-three patch
BITS_PER_ROUND = 8
# The live memory program at distance three, as a six-round window
# ending on its final round occupies a memory. Raw (formed at the
# decoder): five rounds of eight check bits and the final round's eight
# plus the nine data qubits. Formed at the controller: five rounds of
# eight events and the final round's eight plus the four Z checks the
# data readout closes.
SIX_ROUND_WINDOW_BITS = {
    "decoder": 5 * BITS_PER_ROUND + BITS_PER_ROUND + 9,
    "controller": 5 * BITS_PER_ROUND + BITS_PER_ROUND + 4,
}
# A decoder that forms the events reads the raw round before a window's
# first, and the window's store holds it with the window's own rounds.
ROUND_BEFORE_BITS = {"decoder": BITS_PER_ROUND, "controller": 0}
# the decoder engines of these runs: 250 MHz and 100 MHz
FAST_ENGINE_CLOCK = config.Clock(4000)
FAST_ENGINE_CARD = decoder_settings.EngineSettings(clock=FAST_ENGINE_CLOCK)
ENGINE_CLOCK = config.Clock(10_000)
ENGINE_CARD = decoder_settings.EngineSettings(clock=ENGINE_CLOCK)
# The run helpers' default root seed: one call is one repeatable shot.
RUN_SEED = 17


def tier_rows() -> str:
    """The decoder table's rows as a refusal prints them, as a pattern.

    Read from the table, so a row added to it breaks no test here:
    that is what the plug-in page promises a new decoder costs.
    """
    rows = sorted(decoder_settings.DECODERS)
    return re.escape(repr(rows))


class RecordingReceiver:
    """A readout receiver that keeps every readout with its arrival tick."""

    def __init__(self, engine: engine_module.Engine):
        self.engine = engine
        self.arrivals = []

    def accept_qpu_readout(self, readout, route) -> None:
        del route
        self.arrivals.append((self.engine.now, readout.round_index))


class StreamlessIdleRounds:
    """The idle accounting of a run with no streams and no idle patch."""

    def emit_idle_round(self, operation_id, patch, round_index) -> None:
        del operation_id, patch, round_index

    def start_command(self, command):
        return command


class FinishingRuntime:
    """A runtime whose one move is to stop the clock when a body ends."""

    def __init__(self, qpu: cycle_clock.QPUDevice):
        self.qpu = qpu

    def body_done(self, operation: program_records.Operation) -> None:
        del operation
        self.qpu.finish()


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
        return decoding_records.DecodeResult(job.operation_id, job.window_id)


def test_readouts_reach_the_receiver_in_cycle_order_cycle_ticks_apart():
    engine = engine_module.Engine()
    receiver = RecordingReceiver(engine)
    code = code_geometry.SurfaceCodeModel(distance=3)
    device = syndrome_devices.TimingOnlyDevice(code)
    cycle_clock_domain = config.Clock(CYCLE_TICKS)
    qpu = cycle_clock.QPUDevice(engine, cycle_clock_domain, code)
    runtime = FinishingRuntime(qpu)
    qpu.syndrome_source = device
    qpu.readout_receiver = receiver
    qpu.runtime = runtime
    qpu.idle_rounds = StreamlessIdleRounds()
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


def test_a_new_decoder_is_one_class_and_one_table_row(monkeypatch):
    """Gate point 1's settings run to completion on a decoder added as a row."""
    config_path = CONFIGS / "bases/weak_decoder_baseline.yaml"
    config = experiment.load_experiment(config_path)
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.003,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    weak_decoder = dataclasses.replace(settings.weak_decoder, kind="fake")
    settings = dataclasses.replace(settings, weak_decoder=weak_decoder)
    monkeypatch.setitem(decoder_settings.DECODERS, "fake", FakeWeakDecoder)
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    assert result.terminal_status == "complete"
    assert type(machine.decoders.primary_decoder.decoder) is FakeWeakDecoder
    decode_lines = [
        line for line in machine.observation.log.lines if "decode" in line
    ]
    assert decode_lines


def test_a_second_table_row_runs_gate_point_one():
    """Gate point 1's settings run to completion on the union_find row."""
    config_path = CONFIGS / "bases/weak_decoder_baseline.yaml"
    config = experiment.load_experiment(config_path)
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.003,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    weak_decoder = dataclasses.replace(settings.weak_decoder, kind="union_find")
    settings = dataclasses.replace(settings, weak_decoder=weak_decoder)
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    assert result.terminal_status == "complete"
    inner = machine.decoders.primary_decoder.decoder
    assert type(inner) is union_find_decoder.UnionFindDecoder
    assert result.operation_results[0].logical_observables is not None


def test_no_component_queues_an_event_until_the_machine_is_started():
    """Build wires the graph; start queues the first events.

    gem5 splits the constructor, which takes a component's
    collaborators, from startup, "the appropriate place to schedule
    initial event(s)"
    (gem5 src/sim/sim_object.hh lines 194 and 280). With
    that split the order the root builds its components in cannot move a
    tick, because no component has queued anything while the rest of the
    machine is still being built.
    """
    config_path = CONFIGS / "bases/weak_decoder_baseline.yaml"
    config = experiment.load_experiment(config_path)
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.003,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    machine = machine_module.Machine.build(settings, 0)
    assert machine.engine.idle is True

    machine.start()

    assert machine.engine.idle is False


def test_a_decoder_kind_off_the_table_is_refused_naming_the_rows():
    weak_decoder = decoder_settings.DecoderSettings(kind="lookup_table")
    settings = machine_settings.MachineSettings(weak_decoder=weak_decoder)
    sentence = (
        "weak_decoder.kind 'lookup_table' is not a row of its table; "
        "the rows are " + tier_rows()
    )
    with pytest.raises(ValueError, match=sentence):
        machine_module.Machine.build(settings)


def test_a_strong_store_kind_off_the_table_is_refused_even_when_unused():
    # The weak baseline never reads the strong store, but the yaml still
    # names its kind, and a kind off the table is a mistake in the yaml.
    strong_syndrome_buffer = syndrome_buffer_settings.SyndromeBufferSettings(
        kind="off_table"
    )
    settings = machine_settings.MachineSettings(
        strong_syndrome_buffer=strong_syndrome_buffer
    )
    with pytest.raises(
        ValueError,
        match="strong_syndrome_buffer.kind 'off_table' is not a row of its "
        r"table; the rows are \['ported_syndrome_buffer', 'syndrome_buffer'\]",
    ):
        machine_module.Machine.build(settings)


def test_a_yaml_section_nobody_owns_is_refused_naming_the_sections():
    with pytest.raises(
        ValueError, match=r"the yaml has no section \['buffers'\]; the sections"
    ):
        machine_settings.MachineSettings.from_mapping(
            {"buffers": {}}, name="x", section_folders={}
        )


@pytest.mark.parametrize(
    "qpu_entry,sentence",
    [
        ({}, r"the yaml needs the sections \['qpu'\]"),
        ({"qpu": 3}, "the yaml section qpu holds 3; a section is a mapping"),
    ],
)
def test_a_section_missing_or_not_a_mapping_is_refused_with_a_sentence(
    qpu_entry, sentence
):
    sections = _required_sections_but_the_qpu()
    sections.update(qpu_entry)
    with pytest.raises(ValueError, match=sentence):
        machine_settings.MachineSettings.from_mapping(
            sections, name="x", section_folders={}
        )


def test_the_wiring_reaches_its_components():
    """Every cross-reference is bound, by port or by constructor."""
    settings = machine_settings.MachineSettings()
    machine = machine_module.Machine.build(settings)
    control = machine.control
    assert machine.qpu.device.readout_receiver is machine.readout.controller
    assert control.execution_runtime.issuer is control.issuer
    manager = machine.decoders.decoder_manager
    assert machine.windows.window_manager.requester.decode_queue is manager


@pytest.mark.parametrize(
    "run", [declared_run.weak_only_run, declared_run.switching_run]
)
def test_every_required_port_is_bound_once_the_parts_connect(run):
    """No component of any part meets an unbound neighbour at run time.

    OMNeT++ refuses a gate left unconnected before a network runs, and
    lets one marked @loose stay so (src/sim/cmodule.cc
    checkInternalConnections, lines 1253-1282); an optional port is the
    loose gate here, and a required one raises its own name when read.
    """
    machine = run()

    _read_every_required_port(machine)


def test_the_trace_names_the_escalation_distance_and_seed_it_is_of():
    qpu = declared_run.declared_qpu()
    observation = observe_settings.ObservationSettings(trace="chrome")
    settings = machine_settings.MachineSettings(
        qpu=qpu, observation=observation
    )

    machine = machine_module.Machine.build(settings, 7)

    trace_writer = machine.observation.trace_writer
    assert trace_writer.process_name == "decsim weak_baseline d3 seed7"


def test_a_machine_built_part_by_part_runs_as_the_one_call_does():
    """The steps of docs/tutorials/build_a_machine.md, and what they print.

    The parts are whole inside once built, a port to another part is
    bound only by assemble, and the machine assembled by hand gives the
    result Machine.build gives.
    """
    config = experiment.load_experiment("configs/examples/two_tiers.yaml")
    point = config.first_point_task()
    settings = point.shot_settings()
    engine = engine_module.Engine()
    escalation_policy = escalation_build.build_escalation_policy(
        settings.escalation, settings.weak_decoder
    )
    plan = plan_build.build_plan(settings, escalation_policy)
    burst_detector = escalation_build.build_burst_detector(
        settings, engine, plan, escalation_policy
    )
    detection_events = readout_part.build_detection_events(
        settings, plan.device, escalation_policy, burst_detector
    )
    pool = decoders_part.build_decoder_pool(
        settings, plan, escalation_policy, detection_events
    )
    links = machine_module.build_links(settings, engine)
    qpu = qpu_part.Qpu.build(settings.magic_state_factory, engine, plan)
    control = control_part.Control.build(
        settings.controller, settings.pauli_frame, engine, plan, links
    )
    readout = readout_part.Readout.build(
        settings, engine, escalation_policy, detection_events, links
    )
    windows = windows_part.Windows.build(
        settings, engine, plan, escalation_policy, burst_detector, links
    )
    decoders = decoders_part.Decoders.build(
        settings.decoder_manager, engine, pool, escalation_policy
    )
    sender = readout.syndrome_round_sender
    unbound = "SyndromeRoundSender.windows was read before it was bound"

    assert isinstance(escalation_policy, escalation_policies.Switching)
    assert plan.round_ticks == 1_000_000
    assert readout.controller.assembler is readout.assembler
    with pytest.raises(RuntimeError, match=unbound):
        assert sender.windows is None

    machine = machine_module.Machine.assemble(
        settings, engine, plan, links, qpu, control, readout, windows, decoders
    )
    result = machine.run()
    whole_machine = machine_module.Machine.build(settings, 0)
    whole = whole_machine.run()

    requester = windows.window_manager.requester
    assert sender.windows is windows.window_manager
    assert requester.decode_queue is decoders.decoder_manager
    assert result.terminal_status == "complete"
    assert result.fully_done_ticks == 67_608_000
    assert result.operation_results[0].logical_failure is False
    assert result == whole


def _read_every_required_port(machine) -> None:
    """Read each required port of the links and of every part's component.

    A part's None field is a neighbour the run does not have, and None
    declares no port.
    """
    parts = (
        machine.qpu,
        machine.control,
        machine.readout,
        machine.windows,
        machine.decoders,
    )
    components = [machine.links]
    for part in parts:
        fields = vars(part)
        components += fields.values()
    for component in components:
        names = _required_port_names(component)
        for name in names:
            getattr(component, name)


def _required_port_names(component) -> list:
    """The required ports the component's class and its bases declare."""
    component_class = type(component)
    declared = []
    for owner in component_class.__mro__:
        namespace = vars(owner)
        declared += namespace.values()
    names = []
    for value in declared:
        if isinstance(value, ports.Port) and not value.optional:
            names.append(value.name)
    return names


MEMORY_ROUNDS = 6
MEMORY_CIRCUIT = workload_settings.memory_circuit(
    "surface_code:rotated_memory_z", MEMORY_ROUNDS, 3, 0.003
)


@pytest.mark.parametrize("producer", ["stim", "deltakit"])
@pytest.mark.parametrize("placement", ["controller", "decoder"])
@pytest.mark.parametrize("history", ["finite", "live"])
def test_streams_decode_only_on_the_strong_path_with_actual_truth(
    producer: str, placement: str, history: str
) -> None:
    program = _program(producer)
    settings = _settings(program, history, placement)
    run = _run(settings)
    _assert_direct_strong_path(run)
    _assert_actual_truth(run)
    _assert_drained(run)
    readout_transfers = _transfers(run.result, "qpu_to_controller")
    store_transfers = _transfers(run.result, "controller_to_strong_buffer")
    assert len(readout_transfers) == len(run.packets)
    assert len(store_transfers) == len(run.packets)
    expected_bits = _stored_bit_count(run, placement)
    _assert_stored_bit_count(store_transfers, expected_bits)


def test_a_string_stream_owner_beside_integer_operations_is_captured():
    """Result rows follow the canonical identity order, ints before strs.

    records/identity.py orders every identity by its type-tagged bytes,
    since Python refuses to compare an int with a str.
    """
    program = memory_programs.memory_program()
    settings = live_example.live_settings(
        program,
        distance=3,
        round_period_microseconds=1.1,
        prefix_round_count=3,
        patch="memory-patch",
        feedback_microseconds=4.0,
        decoder_microseconds=0.1,
    )
    workload = producers.live_memory(program, 3)
    first, *rest = workload.operations
    prefix = dataclasses.replace(first, stream_id="live")
    operations = (prefix, *rest)
    named = dataclasses.replace(workload, operations=operations)
    section = workload_settings.WorkloadSettings()
    lowered = section.running(named)
    settings = dataclasses.replace(settings, workload=lowered)

    machine = machine_module.Machine.build(settings, 17)
    result = machine.run()

    rows = result.operation_results
    owner = rows[-1]
    assert [row.operation_id for row in rows] == [1, 2, 3, 4, "live"]
    assert owner.observable_truth is not None


@pytest.mark.parametrize("final_round", [4, 6, 7, 9])
@pytest.mark.parametrize("placement", ["controller", "decoder"])
def test_zero_delay_terminal_round_uses_its_actual_strong_model(
    final_round: int, placement: str
) -> None:
    program = memory_programs.memory_program()
    settings = _settings(program, "live", placement)
    physical = settings.workload.physical_circuits
    workload = _scheduled_workload(final_round, physical)
    links = _zero_delay_links(settings.links)
    settings = dataclasses.replace(settings, workload=workload, links=links)
    run = _run(settings)
    assert len(run.packets) == final_round
    assert run.packets[-1].round_index == final_round
    _assert_direct_strong_path(run)
    _assert_actual_truth(run)
    _assert_drained(run)


@pytest.mark.parametrize("period_microseconds", [0.7, 1.25])
def test_strong_settings_price_the_route_and_clock_the_feedback(
    period_microseconds: float,
) -> None:
    program = memory_programs.memory_program()
    settings = _settings(program, "live", "controller", period_microseconds)
    settings = _declared_timing(settings)
    run = _run(settings)
    first = run.decodes.finished[0]
    period_ticks = config.microseconds_to_ticks(period_microseconds)
    sixth_round_ticks = 6 * period_ticks
    expected_dispatch_ticks = sixth_round_ticks + 250_000
    assert first.dispatch_ticks == expected_dispatch_ticks
    input_arrival_ticks = expected_dispatch_ticks + 350_000
    expected_terminal_ticks = input_arrival_ticks + 200_000
    assert first.finish_ticks == expected_terminal_ticks
    arrival = _command_tick(run, "ARRIVED", 3)
    started = _command_tick(run, "STARTED", 3)
    clock = config.Clock(period_ticks)
    assert started == clock.edge(0, arrival)
    _assert_direct_strong_path(run)
    _assert_actual_truth(run)


def test_longer_strong_feedback_preserves_the_executed_prefix() -> None:
    program = memory_programs.memory_program()
    faster = _settings(program, "live", "controller")
    slower = _settings(program, "live", "controller")
    feedback = slower.links.frame_to_controller
    channel = dataclasses.replace(
        feedback.channel, propagation_latency_ticks=8_000_000
    )
    path = dataclasses.replace(feedback, channel=channel)
    links = dataclasses.replace(slower.links, frame_to_controller=path)
    slower = dataclasses.replace(slower, links=links)
    first = _run(faster)
    second = _run(slower)
    assert len(second.packets) > len(first.packets)
    first_prefix = _raw_bits(first.packets[:-1])
    second_record = _raw_bits(second.packets)
    assert first_prefix == second_record[: len(first_prefix)]
    assert (
        second.result.execution_done_ticks > first.result.execution_done_ticks
    )
    _assert_actual_truth(first)
    _assert_actual_truth(second)


@pytest.mark.parametrize("placement", ["controller", "decoder"])
def test_bounded_strong_storage_and_unit_memory_drain_without_weak_data(
    placement: str,
) -> None:
    program = memory_programs.memory_program()
    settings = _settings(program, "live", placement)
    window_bits = SIX_ROUND_WINDOW_BITS[placement]
    round_before_bits = ROUND_BEFORE_BITS[placement]
    store_bits = window_bits + round_before_bits
    buffer = syndrome_buffer_settings.SyndromeBufferSettings(bits=store_bits)
    memory = decoder_settings.UnitMemorySettings(bits=window_bits)
    decoder = dataclasses.replace(settings.strong_decoder, unit_memory=memory)
    # a three microsecond strong input keeps the live stream running
    # fifteen rounds, so its last window holds six rounds ending on the
    # final one
    links = _price_path(
        settings.links, "strong_buffer_to_strong_decoder", 3_000_000
    )
    observation = dataclasses.replace(
        settings.observation, decoder_memory_occupancy=True
    )
    settings = dataclasses.replace(
        settings,
        strong_syndrome_buffer=buffer,
        strong_decoder=decoder,
        links=links,
        observation=observation,
    )
    run = _run(settings)
    assert max(run.strong_occupancies) <= store_bits
    units = run.machine.decoders.decoder_manager.pool.units
    occupancy = run.machine.observation.decoder_memory_occupancy.result()
    memory = occupancy["per_unit"][units[0].name]
    assert memory["capacity_bits"] == window_bits
    assert memory["peak_occupied_bits"] == window_bits
    assert memory["admissions"] > 1
    _assert_direct_strong_path(run)
    _assert_actual_truth(run)
    _assert_drained(run)


@pytest.mark.parametrize("commit_rounds", [1, 2])
def test_an_unbuffered_live_reader_forms_each_window_from_its_own_ring(
    commit_rounds: int,
) -> None:
    """The round before a window is one its decoder formed and keeps.

    With no buffer, the previous window's read is released before the
    next window starts; its decoder formed that round, so the next read
    is given nothing from the store.
    """
    program = memory_programs.memory_program()
    settings = _settings(program, "live", "decoder")
    windows = dataclasses.replace(
        settings.windows, commit_rounds=commit_rounds, buffer_rounds=0
    )
    settings = dataclasses.replace(settings, windows=windows)

    run = _run(settings)

    _assert_drained(run)


def test_a_strong_unit_cannot_admit_a_window_wider_than_its_memory() -> None:
    program = memory_programs.memory_program()
    settings = _settings(program, "live", "controller")
    five_rounds_bits = 5 * BITS_PER_ROUND
    memory = decoder_settings.UnitMemorySettings(bits=five_rounds_bits)
    decoder = dataclasses.replace(settings.strong_decoder, unit_memory=memory)
    settings = dataclasses.replace(settings, strong_decoder=decoder)
    # the first window's events: four from the first round, eight a round
    first_window_bits = 4 + 5 * BITS_PER_ROUND
    message = (
        f"holds {five_rounds_bits} bits; the window needs {first_window_bits}"
    )
    with pytest.raises(
        decoder_memory.DecoderMemoryCapacityError, match=message
    ):
        _run(settings)


@pytest.mark.parametrize(
    ("idle_policy", "load_job_count"),
    [("separate_decode_jobs", 2), ("ignore", 0)],
)
def test_static_idle_rounds_use_strong_slots_until_the_decoder_arrival(
    idle_policy: str, load_job_count: int
) -> None:
    settings = _static_idle_settings(idle_policy)
    run, release_ticks_by_round = _run_static(settings)
    _assert_direct_strong_path(run)
    _assert_drained(run)
    arrivals = _memory_arrivals(run)
    assert len(arrivals) == 4
    _assert_memory_slot_lifetimes(run, arrivals, release_ticks_by_round)
    _assert_idle_algorithm_charge(run, load_job_count)


@pytest.mark.parametrize("placement", ["controller", "decoder"])
def test_single_terminal_window_matches_direct_pymatching(
    placement: str,
) -> None:
    program = memory_programs.memory_program(physical_error_probability=0.08)
    settings = _settings(program, "live", placement)
    physical = settings.workload.physical_circuits
    workload = _scheduled_workload(2, physical)
    settings = dataclasses.replace(settings, workload=workload)
    run = _run(settings)
    circuit = run.shots[0][0].circuit
    detector_model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(detector_model)
    events = run.machine.qpu.syndrome_source.sampled_detection_events(100)
    assert any(events)
    prediction = matching.decode(events)
    assert tuple(prediction) == (1,)
    result = run.result.operation_results[-1]
    assert result.logical_observables == tuple(prediction)
    assert result.logical_failure is False
    assert len(run.decodes.finished) == 1
    _assert_actual_truth(run)


def test_full_strong_storage_retries_held_live_rounds_without_loss() -> None:
    program = memory_programs.memory_program()
    settings = _settings(program, "live", "controller")
    window_bits = SIX_ROUND_WINDOW_BITS["controller"]
    buffer = syndrome_buffer_settings.SyndromeBufferSettings(bits=window_bits)
    memory = decoder_settings.UnitMemorySettings(bits=window_bits)
    decoder = dataclasses.replace(
        settings.strong_decoder, kind=5.0, unit_memory=memory
    )
    links = _price_path(
        settings.links, "strong_buffer_to_strong_decoder", 1_000_000
    )
    settings = dataclasses.replace(
        settings,
        strong_syndrome_buffer=buffer,
        strong_decoder=decoder,
        links=links,
    )
    run = _run(settings)
    _assert_held_rounds_retried(run)
    assert max(run.strong_occupancies) <= window_bits
    _assert_direct_strong_path(run)
    _assert_actual_truth(run)
    _assert_drained(run)


@pytest.mark.parametrize("placement", ["controller", "decoder"])
@pytest.mark.parametrize("is_entangled", [False, True])
def test_joint_live_rounds_share_one_history_and_one_strong_route(
    placement: str, is_entangled: bool
) -> None:
    program = memory_programs.joint_repetition_program(is_entangled)
    settings = _joint_settings(program, placement)

    run = _run(settings)

    _assert_direct_strong_path(run)
    _assert_actual_truth(run)
    _assert_drained(run)
    _assert_joint_acquisitions(run)


def test_longer_joint_feedback_protects_both_blocks_without_resampling() -> (
    None
):
    program = memory_programs.joint_repetition_program(True)
    faster = _joint_settings(program, "controller")
    slower = _joint_settings(program, "controller")
    feedback = slower.links.frame_to_controller
    channel = dataclasses.replace(
        feedback.channel, propagation_latency_ticks=8_000_000
    )
    path = dataclasses.replace(feedback, channel=channel)
    links = dataclasses.replace(slower.links, frame_to_controller=path)
    slower = dataclasses.replace(slower, links=links)

    first = _run(faster)
    second = _run(slower)

    assert len(second.packets) > len(first.packets)
    first_prefix = _raw_bits(first.packets[:-1])
    second_record = _raw_bits(second.packets)
    assert first_prefix == second_record[: len(first_prefix)]
    assert (
        second.result.execution_done_ticks > first.result.execution_done_ticks
    )
    _assert_actual_truth(first)
    _assert_actual_truth(second)
    _assert_joint_acquisitions(second)


@pytest.mark.parametrize("placement", ["controller", "decoder"])
def test_channel_reordering_preserves_live_detector_formation(
    placement: str,
) -> None:
    """Keep channel delays, arrival order and the quantum oracle together.

    This complete scenario stays in one function so each observed timing
    can be read beside the channel card that causes it.
    """
    program = _partitioned_joint_program()
    settings = _joint_settings(program, placement)
    base_path = settings.links.qpu_to_controller
    joint = _readout_route(base_path, ("left", "right"), "joint", 4_500_000)
    left = _readout_route(base_path, ("left",), "left", 100_000)
    right = _readout_route(base_path, ("right",), "right", 200_000)
    links = dataclasses.replace(
        settings.links, readout_routes=(joint, left, right)
    )
    settings = dataclasses.replace(settings, links=links)

    run = _run(settings)

    _assert_actual_truth(run)
    _assert_direct_strong_path(run)
    _assert_drained(run)
    events = run.machine.observation.round_events.events
    binary_events = [
        event for event in events if event.kind == "BINARY_AVAILABLE"
    ]
    first_round_arrival_ticks = [
        event.tick for event in binary_events if event.round_index == 1
    ]
    second_round_arrival_ticks = [
        event.tick for event in binary_events if event.round_index == 2
    ]
    last_second_arrival_ticks = max(second_round_arrival_ticks)
    first_arrival_ticks = min(first_round_arrival_ticks)
    assert last_second_arrival_ticks < first_arrival_ticks
    packed = [event.round_index for event in events if event.kind == "PACKED"]
    after_last_round = len(packed) + 1
    expected_rounds = list(range(1, after_last_round))
    assert packed == expected_rounds
    transfers = run.result.link_traffic["transfers"]
    readouts = [row for row in transfers if row["path"] == "qpu_to_controller"]
    delay_ticks = {row["total_delay_ticks"] for row in readouts}
    assert delay_ticks == {
        100_000,
        200_000,
        4_500_000,
    }
    assert len(readouts) == len(run.packets)
    raw_bits = _raw_bits(run.packets)
    payload_bits = sum(row["payload_bits"] for row in readouts)
    assert payload_bits == len(raw_bits)


@pytest.mark.parametrize(
    "basis, has_logical_flips", [("X", False), ("Z", True)]
)
@pytest.mark.parametrize("placement", ["controller", "decoder"])
def test_bb_block_decodes_all_eight_outputs_against_direct_bp_osd(
    basis: str, has_logical_flips: bool, placement: str
) -> None:
    pytest.importorskip("deltakit_explorer")
    program = _bb_memory(basis)
    settings = _bb_settings(program, placement, commit_round_count=10)
    physical = settings.workload.physical_circuits
    workload = _scheduled_workload(3, physical)
    workload = _bb_logical_qubits(workload)
    settings = dataclasses.replace(settings, workload=workload)

    run = _run(settings)

    _assert_actual_truth(run)
    _assert_direct_strong_path(run)
    _assert_drained(run)
    expected = _direct_bp_osd_prediction(run)
    owner = run.result.operation_results[-1]
    assert len(expected) == 8
    assert owner.logical_observables == expected
    assert owner.observable_truth == expected
    assert any(expected) == has_logical_flips
    assert len(run.decodes.finished) == 1


def test_bb_live_feedback_preserves_the_full_logical_vector() -> None:
    pytest.importorskip("deltakit_explorer")
    program = _bb_memory("Z")
    settings = _bb_settings(program, "controller", commit_round_count=2)
    rounds = round_policies.PerOperationRounds({100: 0, 1: 2, 2: 0, 3: 1, 4: 0})
    workload = dataclasses.replace(settings.workload, rounds_policy=rounds)
    settings = dataclasses.replace(settings, workload=workload)

    run = _run(settings)

    _assert_actual_truth(run)
    _assert_direct_strong_path(run)
    _assert_drained(run)
    owner = run.result.operation_results[-1]
    assert len(owner.logical_observables) == 8
    assert len(owner.observable_truth) == 8
    assert len(run.packets) > 3


@pytest.mark.parametrize("placement", ["controller", "decoder"])
def test_bb_higher_logical_fault_is_reported_as_a_failure(
    placement: str,
) -> None:
    pytest.importorskip("deltakit_explorer")
    program = _bb_memory("Z")
    circuit, mapping = program.assemble(3)
    measurements = _bb_fault_measurements(program)
    source = stim_device.RecordedStimDevice(
        measurements, 0, measurement_rounds={100: mapping}
    )
    settings = _bb_settings(program, placement, commit_round_count=10)
    qpu = dataclasses.replace(settings.qpu, device=source)
    owner = program_records.Operation(
        100, "memory", tuple(range(8)), patches=("block",), circuit=circuit
    )
    rounds = round_policies.FixedRounds(3)
    workload = workload_settings.WorkloadSettings(
        operations=(owner,), rounds_policy=rounds
    )
    settings = dataclasses.replace(settings, qpu=qpu, workload=workload)

    run = _run(settings)

    _assert_actual_truth(run)
    _assert_direct_strong_path(run)
    _assert_drained(run)
    result = run.result.operation_results[0]
    events = source.sampled_detection_events(100)
    assert not any(events)
    assert result.logical_observables == (0,) * 8
    assert result.observable_truth[1] == 1
    assert result.logical_failure is True


@pytest.mark.parametrize("placement", ["controller", "decoder"])
def test_interleaved_joint_recording_matches_complete_stim_conversion(
    placement: str,
) -> None:
    """Keep the physical record, chronology and decoder oracle together.

    The explicit setup shows that three acquisition fragments carry one
    quantum history, with the same fault and measurement map at both ends.
    It stays in one function so the complete recorded input and its decoder
    oracle remain visible beside the declared partitions.
    """
    program = memory_programs.joint_repetition_program(True, 0.12)
    circuit, mapping = program.assemble(3)
    sampler = circuit.compile_sampler(seed=73)
    measurements = sampler.sample(shots=1)
    table = detector_formation.build_formation_table(
        circuit, 3, measurement_rounds=mapping
    )
    detector_rounds = table.detector_rounds()
    first = round_records.MeasurementPartition(("left",), 1)
    middle = round_records.MeasurementPartition(("left", "right"), 2)
    right = round_records.MeasurementPartition(("right",), 1)
    last = round_records.MeasurementPartition(("left", "right"), 7)
    partitions = {1: (first, middle, right), 2: (first, middle, right)}
    partitions[3] = (first, middle, last)
    source = stim_device.RecordedStimDevice(
        measurements,
        0,
        measurement_rounds={100: mapping},
        detector_rounds={100: detector_rounds},
        readout_partitions={100: partitions},
    )
    settings = _joint_settings(program, placement)
    code = finite_example.RepetitionMemory(3, 3)
    qpu = dataclasses.replace(
        settings.qpu, device=source, code=code, distance=None
    )
    owner = program_records.Operation(
        100, "joint", (0, 1), patches=("left", "right"), circuit=circuit
    )
    rounds = round_policies.FixedRounds(3)
    workload = workload_settings.WorkloadSettings(
        operations=(owner,), rounds_policy=rounds
    )
    settings = dataclasses.replace(settings, workload=workload, qpu=qpu)

    run = _run(settings)

    _assert_actual_truth(run)
    _assert_direct_strong_path(run)
    _assert_drained(run)
    raw = _raw_bits(run.packets)
    assert raw == tuple(measurements[0])
    expected = _direct_matching_prediction(circuit, source)
    result = run.result.operation_results[0]
    assert expected == (1,)
    assert result.logical_observables == expected
    assert result.observable_truth == expected


@pytest.mark.parametrize("placement", ["controller", "decoder"])
def test_separate_terminal_emitters_preserve_the_complete_record(
    placement: str,
) -> None:
    settings, measurements, circuit = _terminal_partitioned_settings(placement)

    run = _run(settings)

    _assert_actual_truth(run)
    _assert_direct_strong_path(run)
    _assert_drained(run)
    raw_bits = _raw_bits(run.packets)
    expected_raw_bits = tuple(measurements[0])
    assert raw_bits == expected_raw_bits
    final_packets = [
        packet for packet in run.packets if packet.round_index == 3
    ]
    final_indices = [packet.fragment_index for packet in final_packets]
    final_fragment_counts = [packet.fragment_count for packet in final_packets]
    final_width_bits = [packet.size_bits for packet in final_packets]
    assert final_indices == [0, 1, 2, 3]
    assert final_fragment_counts == [4, 4, 4, 4]
    assert final_width_bits == [4, 4, 3, 6]
    source = settings.qpu.device
    expected_prediction = _direct_matching_prediction(circuit, source)
    result = run.result.operation_results[-1]
    assert result.operation_id == 100
    assert result.logical_observables == expected_prediction


def test_a_timing_only_terminal_fragment_reads_the_round_out_whole():
    """The split last round crosses as the unsplit one: 17 raw, 12 events.

    The checks fragment reads 8 bits and the terminal fragment the 9
    data qubits, and the controller forms the round's 8 and 4 events
    into one round of 12, as a three-round operation's last round is.
    """
    distance = 3
    round_count = 3
    code = code_geometry.SurfaceCodeModel(distance=distance)
    source = syndrome_devices.TimingOnlyDevice(code)
    split = _circuit_less_terminal_run(
        source, (0,), is_split=True, round_count=round_count
    )
    source = syndrome_devices.TimingOnlyDevice(code)
    whole = _circuit_less_terminal_run(
        source, (0,), is_split=False, round_count=round_count
    )

    assert split.terminal_status == "complete"
    assert _payload_bits_on(split, "qpu_to_controller") == [8, 8, 8, 9]
    assert _payload_bits_on(whole, "qpu_to_controller") == [8, 8, 17]
    split_events = _payload_bits_on(split, "controller_to_weak_buffer")
    whole_events = _payload_bits_on(whole, "controller_to_weak_buffer")
    assert split_events == whole_events == [4, 8, 12]


@pytest.mark.parametrize(
    ("is_split", "last_round_bits"),
    [(False, [17, 17]), (True, [8, 8, 9, 9])],
)
def test_a_payload_per_patch_reads_each_patch_out_whole(
    is_split: bool, last_round_bits: list
):
    """Two patches, one payload each: 17 raw bits each on the last round.

    The operation's declared slots are the ones its payloads fill, so
    its last round reads the data out; split, the checks leave two
    slots and the terminal fragment fills them with 9 data bits a patch.
    """
    distance = 3
    round_count = 3
    code = code_geometry.SurfaceCodeModel(distance=distance)
    source = syndrome_devices.SyndromeBitDevice(
        code, one_payload_per_patch=True
    )
    result = _circuit_less_terminal_run(
        source, (0, 1), is_split=is_split, round_count=round_count
    )

    assert result.terminal_status == "complete"
    readouts = _payload_bits_on(result, "qpu_to_controller")
    expected_readouts = [8, 8, 8, 8] + last_round_bits
    assert readouts == expected_readouts


def _circuit_less_terminal_run(
    source, patches: tuple, is_split: bool, round_count: int
) -> result_records.RunResult:
    """round_count rounds of stream 100 on the patches, circuit-less."""
    owner = program_records.Operation(100, "memory", patches, patches=patches)
    operations, counts = _stream_rounds(owner, is_split, round_count)
    policy = round_policies.PerOperationRounds(counts)
    workload = workload_settings.WorkloadSettings(
        operations=operations, decode_operations=(owner,), rounds_policy=policy
    )
    distance = source.code.distance
    qpu = qpu_settings.QpuSettings(distance=distance, device=source)
    clock = config.Clock(1000)
    engine = decoder_settings.EngineSettings(clock=clock)
    decoder = decoder_settings.DecoderSettings(kind=0.1, engine=engine)
    settings = machine_settings.MachineSettings(
        workload=workload, qpu=qpu, weak_decoder=decoder
    )
    machine = machine_module.Machine.build(settings, 0)
    return machine.run()


def _stream_rounds(
    owner: program_records.Operation, is_split: bool, round_count: int
) -> tuple:
    """(operations, round counts) of round_count rounds of the stream.

    Each operation declares the fragment slots its payloads fill, one
    payload a patch. Split: every round but the last, then the last
    round's checks in the first half of the slots and its data readout
    as the terminal fragment in the second.
    """
    payload_count = len(owner.patches)
    memory = dataclasses.replace(
        owner,
        id=1,
        name="memory",
        stream_id=100,
        stream_offset=0,
        syndrome_fragment_index=0,
        syndrome_fragment_count=payload_count,
    )
    if not is_split:
        return (memory,), {100: round_count, 1: round_count}
    split_count = 2 * payload_count
    last_round_offset = round_count - 1
    checks = dataclasses.replace(
        memory,
        id=2,
        name="last checks",
        stream_offset=last_round_offset,
        predecessors=(1,),
        syndrome_fragment_count=split_count,
    )
    readout = dataclasses.replace(
        checks,
        id=3,
        name="data readout",
        predecessors=(2,),
        finalizes_stream_round=True,
        syndrome_fragment_index=payload_count,
    )
    counts = {100: round_count, 1: last_round_offset, 2: 1, 3: 0}
    return (memory, checks, readout), counts


def _payload_bits_on(result: result_records.RunResult, path: str) -> list:
    bits = []
    for transfer in result.link_traffic["transfers"]:
        if transfer["path"] == path:
            bits.append(transfer["payload_bits"])
    return bits


@pytest.mark.parametrize("placement", ["controller", "decoder"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_deltakit_bell_memory_uses_shared_live_execution(
    placement: str,
    basis: str,
) -> None:
    pytest.importorskip("deltakit_explorer")
    program = deltakit.bell_memory_rounds(
        3, basis, 0.001, round_period_microseconds=1.1
    )
    settings = _joint_settings(program, placement)

    run = _run(settings)

    _assert_actual_truth(run)
    _assert_direct_strong_path(run)
    _assert_drained(run)
    _assert_joint_acquisitions(run)
    owner = run.result.operation_results[-1]
    assert len(owner.logical_observables) == 1
    assert len(owner.observable_truth) == 1


def test_joint_segments_can_name_the_same_footprint_in_another_order() -> None:
    program = memory_programs.joint_repetition_program(True)
    settings = _joint_settings(program, "controller")
    operations = _operations_on_group(
        settings.workload.operations, ("right", "left")
    )
    workload = dataclasses.replace(settings.workload, operations=operations)
    settings = dataclasses.replace(settings, workload=workload)

    run = _run(settings)

    _assert_actual_truth(run)
    _assert_joint_acquisitions(run)
    _assert_drained(run)


@pytest.mark.parametrize("placement", ["controller", "decoder"])
def test_joint_live_memory_can_use_the_weak_primary_tier(
    placement: str,
) -> None:
    program = memory_programs.joint_repetition_program(True)
    settings = _joint_settings(program, placement)
    escalation = escalation_settings.EscalationSettings(kind="weak_baseline")
    settings = dataclasses.replace(
        settings, escalation=escalation, weak_decoder=settings.strong_decoder
    )

    run = _run(settings)

    _assert_actual_truth(run)
    _assert_joint_acquisitions(run)
    assert run.result.terminal_status == "complete"
    assert run.result.event_queue_empty
    assert run.machine.readout.weak_syndrome_buffer.occupancy == 0
    assert run.weak_writes
    paths = _transfer_paths(run)
    assert "controller_to_weak_buffer" in paths
    assert "controller_to_strong_buffer" not in paths


def _memory_on_stim_device(
    weak_decoder, operations
) -> machine_settings.MachineSettings:
    """The operations as a six-round d=3 memory on the Stim device."""
    rounds = round_policies.FixedRounds(MEMORY_ROUNDS)
    workload = workload_settings.WorkloadSettings(
        operations=operations, rounds_policy=rounds
    )
    device = stim_device.StimDevice()
    qpu = qpu_settings.QpuSettings(distance=3, device=device)
    return machine_settings.MachineSettings(
        workload=workload, qpu=qpu, weak_decoder=weak_decoder
    )


def _two_patch_memory(weak_decoder) -> machine_settings.MachineSettings:
    """Two memory operations on two patches; the second starts at round 4.

    Patch 0 idles for the four rounds after the first operation ends,
    while the second runs, and the default idle policy
    (separate_decode_jobs) charges them as load-only decode jobs, jobs
    without a window model: one commit region of three rounds, and the
    one round left when the workload completes.
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
        kind="pymatching",
        engine=FAST_ENGINE_CARD,
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
    load_only = [r for r in algorithm if r.operation_id not in operation_ids]
    windows = [r for r in algorithm if r.operation_id in operation_ids]
    assert len(load_only) == 2
    assert len(windows) == 2
    idle_ticks = [r.end_ticks - r.start_ticks for r in load_only]
    assert idle_ticks == [0, 0]
    assert all(r.end_ticks > r.start_ticks for r in windows)
    idle_lines = [
        line for line in machine.observation.log.lines if "mem(" in line
    ]
    assert any("algorithm mem(" in line for line in idle_lines)


def _switching_memory(weak_kind: str, confidence: str):
    """A d=3 memory whose weak tier reports the named confidence signal."""
    decibels = 20.0
    nats = escalation_settings.decibels_to_nats(decibels)
    escalation = escalation_settings.EscalationSettings(
        kind="switching",
        confidence=confidence,
        gap_threshold_db=decibels,
        gap_threshold_nats=nats,
    )
    weak_decoder = decoder_settings.DecoderSettings(
        kind=weak_kind,
        engine=ENGINE_CARD,
    )
    strong_decoder = decoder_settings.DecoderSettings(
        kind="pymatching",
        engine=ENGINE_CARD,
    )
    operation = program_records.Operation(
        id=1, name="memory", qubits=(0,), patches=(0,), circuit=MEMORY_CIRCUIT
    )
    settings = _memory_on_stim_device(weak_decoder, (operation,))
    observation = observe_settings.ObservationSettings(
        record_switching_windows=True
    )
    return dataclasses.replace(
        settings,
        strong_decoder=strong_decoder,
        escalation=escalation,
        observation=observation,
    )


def _run_hearing_decodes(settings, seed: int):
    """One run, and every decode that gave its unit back in it."""
    machine = machine_module.Machine.build(settings, seed)
    decodes = declared_run.FinishedDecodes()
    decodes.attach(machine)
    result = machine.run()
    return machine, result, decodes


def _decode_span_ticks(decodes: declared_run.FinishedDecodes) -> list:
    """Every decode's dispatch-to-finish ticks, in the order they finished."""
    ticks = []
    for finished in decodes.finished:
        span_ticks = finished.finish_ticks - finished.dispatch_ticks
        ticks.append(span_ticks)
    return ticks


def test_a_union_find_weak_tier_reports_the_gap_of_its_own_growth():
    """The cluster gap is a signal row over the decoder that grew it.

    One decode per window, its confidence read off the growth that
    decode returned (Meister et al. 2405.07433 Algorithm 2 lines
    518-536), and the unit charged that decode. The matching run beside
    it is the comparison the charge is read against: Union-Find's own
    decode is this implementation's, not sparse blossom's. Seed 1 is a
    shot in which no window escalates, so every decode is a weak one.
    """
    settings = _switching_memory("union_find", "cluster_gap")
    machine, result, decodes = _run_hearing_decodes(settings, 1)
    assert result.terminal_status == "complete"
    assert result.operation_results[0].logical_observables == (0,)
    requests = machine.observation.decode_records.requests
    sources = set()
    for record in requests:
        if record.soft_output is not None:
            sources.add(record.soft_output.source.method)
    assert sources == {"cluster_gap"}
    window_count = len(decodes.finished)
    assert len(requests) == window_count
    matching_settings = _switching_memory("pymatching", "complementary_gap")
    _matching, _result, matching_decodes = _run_hearing_decodes(
        matching_settings, 1
    )
    union_find_services = _decode_span_ticks(decodes)
    matching_services = _decode_span_ticks(matching_decodes)
    union_find_ticks = min(union_find_services)
    matching_ticks = max(matching_services)
    assert union_find_ticks > 3 * matching_ticks


def _confidence_charges(machine) -> list:
    """The ticks each CONFIDENCE line of the run charged, in order."""
    charged = []
    for line in machine.observation.log.lines:
        if "CONFIDENCE" not in line:
            continue
        parts = line.split(": ")
        charge_text = parts[-1]
        ticks_text = charge_text.replace(" ticks", "")
        charged.append(int(ticks_text))
    return charged


def test_the_cluster_gaps_walk_is_charged_on_the_unit_that_grew_it():
    """Decision D8: the walk over a decode's growth is the unit's time.

    Toshio et al. 2510.25222 lines 152-160 compute the soft output on
    the weak decoder itself, and Meister's Algorithm 2 walks the
    union-find decode's own edge intervals (2405.07433 lines 518-536),
    so the evidence and its reader are the same hardware. The run
    charges one walk per window on the weak unit, and each weak decode
    gives its unit back after the decode and the walk it fed. Seed 1 is
    a shot in which no window escalates, so every decode is a weak one.
    """
    settings = _switching_memory("union_find", "cluster_gap")
    machine, _result, decodes = _run_hearing_decodes(settings, 1)
    charged = _confidence_charges(machine)
    assert len(charged) == len(decodes.finished)
    for ticks in charged:
        assert ticks > 0
    named = []
    for line in machine.observation.log.lines:
        if "CONFIDENCE" in line:
            named.append(line)
    # the weak tier of this card is the default pool's one unit
    assert "on unit default#0" in named[0]
    service_ticks = _decode_span_ticks(decodes)
    assert min(service_ticks) > max(charged)
    assert sum(service_ticks) > sum(charged)
    for record in machine.observation.decode_records.requests:
        assert record.soft_output is not None


@pytest.mark.parametrize(
    "weak_kind, confidence, sentence",
    [
        ("union_find", "complementary_gap", "arXiv:2510.05795"),
        ("bposd", "complementary_gap", "arXiv:2510.05795"),
        ("relay_bp", "complementary_gap", "arXiv:2510.05795"),
        ("belief_matching", "complementary_gap", "arXiv:2312.04522"),
        ("pymatching", "cluster_gap", "pymatching 2.4.0"),
        ("tesseract", "cluster_gap", "pymatching 2.4.0"),
    ],
)
def test_a_weak_tier_that_cannot_serve_the_confidence_is_refused_by_name(
    weak_kind, confidence, sentence
):
    """The pairing is refused at the yaml boundary, with its citation.

    Union-Find, BP-OSD and relay-BP do not minimise weight inside a
    fixed logical class (Lee et al. arXiv:2510.05795 Sec. 2.1.1);
    belief matching could, but only on the posterior graph it matched
    on (Gidney et al. arXiv:2312.04522 lines 843-846), which is not
    built; and no row but Union-Find reports the radii the cluster gap
    walks (pymatching 2.4.0 Matching).
    """
    settings = _switching_memory(weak_kind, confidence)
    match = re.escape(f"weak_decoder.kind {weak_kind!r}")
    with pytest.raises(ValueError, match=match):
        machine_module.Machine.build(settings, 0)
    citation = re.escape(sentence)
    with pytest.raises(ValueError, match=citation):
        machine_module.Machine.build(settings, 0)


def test_a_weak_tier_that_serves_the_confidence_is_accepted():
    """Each signal's own row builds; the refusal is about the pairing."""
    settings = _switching_memory("union_find", "cluster_gap")
    assert machine_module.Machine.build(settings, 0) is not None
    settings = _switching_memory("pymatching", "complementary_gap")
    assert machine_module.Machine.build(settings, 0) is not None


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
    """The cluster gap is a confidence row, not a decoder row.

    escalation.confidence names it (confidence/signals.py) and
    it reads the growth of whatever weak decoder the tier table names,
    so it is not a kind of that table under any escalation kind.
    """
    tiers = {
        "weak": decoder_settings.DecoderSettings(
            kind="pymatching",
            engine=ENGINE_CARD,
        ),
        "strong": decoder_settings.DecoderSettings(
            kind="belief_matching",
            engine=ENGINE_CARD,
        ),
    }
    tiers[tier] = decoder_settings.DecoderSettings(
        kind="union_find_cluster_gap",
        engine=ENGINE_CARD,
    )
    escalation = escalation_settings.EscalationSettings(kind=escalation_kind)
    if escalation_kind == "switching":
        escalation = escalation_settings.EscalationSettings(
            kind="switching", gap_threshold_nats=1.0
        )
    settings = machine_settings.MachineSettings(
        weak_decoder=tiers["weak"],
        strong_decoder=tiers["strong"],
        escalation=escalation,
    )
    sentence = (
        f"{tier}_decoder.kind 'union_find_cluster_gap' is not a row of its "
        "table; the rows are " + tier_rows()
    )
    with pytest.raises(ValueError, match=sentence):
        machine_module.Machine.build(settings, 0)


class CountingSyndromeBuffer(syndrome_buffer_module.SyndromeBuffer):
    """A table row for the plug-in test: the store, counting its writes."""

    def __init__(self, settings, engine):
        syndrome_buffer_module.SyndromeBuffer.__init__(self, settings, engine)
        self.stored_count = 0

    def accept_packed_round(self, packet, *, publication_tick):
        self.stored_count += 1
        syndrome_buffer_module.SyndromeBuffer.accept_packed_round(
            self, packet, publication_tick=publication_tick
        )


def test_a_new_syndrome_buffer_is_one_class_and_one_table_row(monkeypatch):
    """Gate point 1's settings run to completion on a store added as a row."""
    config_path = CONFIGS / "bases/weak_decoder_baseline.yaml"
    config = experiment.load_experiment(config_path)
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.003,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    counting = dataclasses.replace(
        settings.weak_syndrome_buffer, kind="counting"
    )
    settings = dataclasses.replace(settings, weak_syndrome_buffer=counting)
    monkeypatch.setitem(
        ported_syndrome_buffer.SYNDROME_BUFFERS,
        "counting",
        CountingSyndromeBuffer,
    )
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    assert result.terminal_status == "complete"
    assert type(machine.readout.weak_syndrome_buffer) is CountingSyndromeBuffer
    fired = [
        line for line in machine.observation.log.lines if "fires round" in line
    ]
    assert machine.readout.weak_syndrome_buffer.stored_count == len(fired)
    assert fired


class AlwaysStrongEscalation(escalation_policies.EscalationPolicyBase):
    """An escalation row written outside decsim: the strong tier decodes."""

    requires_strong_context = False
    primary_tier = window_records.DecoderTier.STRONG

    def verdict_for_weak_result(self, job, result):
        """The strong result is final."""
        del job
        del result
        return decoding_records.Verdict.KEEP


def test_a_new_escalation_kind_is_one_class_and_one_table_row(monkeypatch):
    """The row declares its tier, so no second table names the kind."""
    config_path = CONFIGS / "bases/strong_decoder_baseline.yaml"
    config = experiment.load_experiment(config_path)
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.003,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    escalation = dataclasses.replace(settings.escalation, kind="always_strong")
    settings = dataclasses.replace(settings, escalation=escalation)
    monkeypatch.setitem(
        escalation_settings.ESCALATIONS, "always_strong", AlwaysStrongEscalation
    )
    assert escalation_build.primary_tier(escalation) == "strong"
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    assert result.terminal_status == "complete"
    snapshot = machine.control.pauli_frame.snapshot()
    tiers = [record.tier for record in snapshot.records]
    assert tiers
    assert set(tiers) == {"strong"}


class _SilentFactoryTrace:
    """The one source the factory port declares, for a row that never waits."""

    state_delivered = trace_source.SilentSource()


class AlwaysReadyFactory:
    """A factory row written outside decsim: the collaborators alone.

    Its constructor is InfiniteFactory's own shape, one parameter: the
    collaborators record every row is built from. It declares the decode
    queue port the root binds on every factory row, and the trace source
    the port declares, silent because no request waits.
    """

    decode_queue = ports.Port(ports.DecodeQueue)

    def __init__(self, collaborators):
        self.engine = collaborators.engine
        self.requests = []
        self.trace = _SilentFactoryTrace()

    def start(self):
        """Nothing is made ahead of a request, so nothing is queued."""

    def request(self, operation_id, callback):
        """Deliver at once and remember who asked."""
        self.requests.append(operation_id)
        callback()

    def shutdown(self):
        """Nothing runs, so nothing stops."""


def test_a_factory_row_written_outside_decsim_builds_by_its_own_name(
    monkeypatch,
):
    """One constructor call, so a row that reads only the engine builds."""
    config_path = CONFIGS / "bases/weak_decoder_baseline.yaml"
    config = experiment.load_experiment(config_path)
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.003,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    factory_settings = qpu_settings.FactorySettings(kind="always_ready")
    settings = dataclasses.replace(
        settings, magic_state_factory=factory_settings
    )
    monkeypatch.setitem(
        qpu_settings.MAGIC_STATE_FACTORIES, "always_ready", AlwaysReadyFactory
    )
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    assert isinstance(machine.qpu.factory, AlwaysReadyFactory)
    assert result.terminal_status == "complete"


def test_the_run_result_carries_the_factorys_supply_stall():
    """The wait for a distilled state reaches the result the run returns.

    One unit, a 5 us attempt and a 1 us return trip, no correction
    decode: the one operation that needs a state asks at tick 0 and
    waits for both.
    """
    operation = declared_run.memory_operation(1, consumes_magic_state=True)
    workload = declared_run.declared_workload([operation], 6)
    attempt_ticks = config.microseconds_to_ticks(5.0)
    return_ticks = config.microseconds_to_ticks(1.0)
    card = magic_state_factories.DistillationFactory.Settings(
        unit_count=1,
        attempt_ticks=attempt_ticks,
        correction_round_count=0,
        correction_decode_count=0,
        return_ticks=return_ticks,
    )
    factory = qpu_settings.FactorySettings(
        kind="distillation", row_settings=card
    )
    weak_microseconds = declared_run.DECLARED_MICROSECONDS["weak"]
    decoder = decoders.PresetLatencyDecoder(weak_microseconds)
    weak_decoder = decoder_settings.DecoderSettings(decoder=decoder)
    qpu = declared_run.declared_qpu()
    links = declared_run.declared_profile()
    controller = declared_run.declared_controller()
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        weak_decoder=weak_decoder,
        links=links,
        controller=controller,
        magic_state_factory=factory,
    )

    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()

    assert result.magic_state_stall_ticks == attempt_ticks + return_ticks
    built_factory = machine.qpu.factory
    assert result.magic_state_stall_ticks == built_factory.total_stall_ticks


class RecordingBoundaryPolicy:
    """A boundary policy written outside decsim: one method, plain names."""

    ships_provisional_boundaries = False

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
    settings = machine_settings.MachineSettings()
    first = machine_module.Machine.build(settings)
    second = machine_module.Machine.build(settings)
    first_boundary = first.windows.window_manager.courier.boundary_policy
    second_boundary = second.windows.window_manager.courier.boundary_policy
    assert type(first_boundary) is boundary_policies.Eager
    assert type(second_boundary) is boundary_policies.Eager
    assert first_boundary is not second_boundary
    first_idle_policy = first.control.idle_rounds.policy
    second_idle_policy = second.control.idle_rounds.policy
    assert type(first_idle_policy) is idle_policies.SeparateDecodeJobs
    assert first_idle_policy is not second_idle_policy


def test_a_policy_written_outside_decsim_is_used_on_its_own_axis():
    """The two policy axes are independent: one given, the other default."""
    boundary_policy = RecordingBoundaryPolicy()
    windows = window_settings.WindowSettings(boundary_policy=boundary_policy)
    boundary_settings = machine_settings.MachineSettings(windows=windows)
    with_boundary = machine_module.Machine.build(boundary_settings)
    idle_policy = RecordingIdlePolicy()
    idle_row = controller_settings.IdlePolicySettings(policy=idle_policy)
    idle_settings = machine_settings.MachineSettings(idle_policy=idle_row)
    with_idle = machine_module.Machine.build(idle_settings)
    assert with_boundary.windows.window_manager.courier.boundary_policy is (
        boundary_policy
    )
    assert type(with_boundary.control.idle_rounds.policy) is (
        idle_policies.SeparateDecodeJobs
    )
    assert with_idle.control.idle_rounds.policy is idle_policy
    assert type(with_idle.windows.window_manager.courier.boundary_policy) is (
        boundary_policies.Eager
    )


class SeedRecordingPolicy:
    """A boundary policy that records the seed the root derived for it."""

    ships_provisional_boundaries = False

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
    settings = machine_settings.MachineSettings(windows=windows)
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
    settings = machine_settings.MachineSettings()
    one_past_the_widest_seed = 1 << 64
    with pytest.raises(ValueError, match=r"seed must be in \[0, 2\*\*64\)"):
        machine_module.Machine.build(settings, one_past_the_widest_seed)


def test_a_negative_seed_is_refused():
    settings = machine_settings.MachineSettings()
    with pytest.raises(ValueError, match=r"seed must be in \[0, 2\*\*64\)"):
        machine_module.Machine.build(settings, -1)


def test_a_seed_that_is_not_a_number_is_refused():
    settings = machine_settings.MachineSettings()
    with pytest.raises(ValueError):
        machine_module.Machine.build(settings, "x")


class OutsideCodeCard:
    """A code card written outside decsim: the port's methods, nothing else."""

    name = "outside code card"
    distance = 2

    def rounds_per_logical_cycle(self):
        return 2

    def round_period_us(self):
        return None

    def commit_rounds(self):
        return 2

    def buffer_rounds(self):
        return 0

    def spatial_nodes(self, num_patches):
        return 4 * num_patches

    def syndrome_bits_per_round(self, num_patches):
        return 3 * num_patches

    def data_bits_per_readout(self, num_patches):
        return 4 * num_patches


def _resolved_geometry(machine):
    """The geometry the run resolved, read off the issuer's table."""
    resolved = machine.control.issuer.resolved_operation_by_id.values()
    first = next(iter(resolved))
    return first.code_geometry


def _one_memory_operation_on(card) -> machine_settings.MachineSettings:
    """One timing-only memory operation on the code card given."""
    operation = program_records.Operation(id=1, name="memory", qubits=(0,))
    workload = workload_settings.WorkloadSettings(operations=[operation])
    qpu = qpu_settings.QpuSettings(code=card)
    decoder = decoders.PresetLatencyDecoder(0.0)
    weak_decoder = decoder_settings.DecoderSettings(decoder=decoder)
    return machine_settings.MachineSettings(
        workload=workload, qpu=qpu, weak_decoder=weak_decoder
    )


def test_a_code_card_written_outside_decsim_runs_with_no_registration():
    """A card is a port, not a table row: nothing names it but the settings."""
    card = OutsideCodeCard()
    settings = _one_memory_operation_on(card)
    machine = machine_module.Machine.build(settings)
    result = machine.run()
    geometry = _resolved_geometry(machine)
    assert result.terminal_status == "complete"
    assert geometry.code_name == "outside code card"


def test_a_run_without_a_code_card_resolves_the_distance_three_surface():
    settings = _one_memory_operation_on(None)
    machine = machine_module.Machine.build(settings)
    machine.run()
    geometry = _resolved_geometry(machine)
    assert geometry.code_name == "rotated surface code (d=3)"
    assert geometry.distance == 3


# The replay path: recorded raw measurements through the whole loop, with
# Stim, PyMatching and qLDPC as the referents for what comes out.

RECORDED_ROUNDS = 9
RECORDED_DISTANCE = 3
RECORDED_SHOT_COUNT = 12


@pytest.fixture(scope="module")
def recorded_memory():
    """Twelve recorded shots of a nine-round distance-three memory.

    The device replays raw measurements, so Stim's own m2d converter is
    the referent for the detector events and the observable truth it
    should report.
    """
    stim = pytest.importorskip("stim")
    probability = 0.01
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=RECORDED_ROUNDS,
        distance=RECORDED_DISTANCE,
        after_clifford_depolarization=probability,
        before_measure_flip_probability=probability,
        after_reset_flip_probability=probability,
        before_round_data_depolarization=probability,
    )
    sampler = circuit.compile_sampler(seed=5)
    measurements = sampler.sample(RECORDED_SHOT_COUNT)
    converter = circuit.compile_m2d_converter()
    detectors, observables = converter.convert(
        measurements=measurements, separate_observables=True
    )
    return circuit, measurements, detectors, observables


def as_bits(flags):
    """One row of numpy flags as a tuple of ints."""
    bits = []
    for flag in flags:
        bits.append(int(flag))
    return tuple(bits)


def replayed_run(circuit, measurements, shot):
    """One completed run of the recorded shot through the weak tier."""
    memory = program_records.Operation(
        id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit
    )
    rounds_policy = round_policies.FixedRounds(RECORDED_ROUNDS)
    workload = workload_settings.WorkloadSettings(
        operations=[memory], rounds_policy=rounds_policy
    )
    device = stim_device.RecordedStimDevice(measurements, shot)
    qpu = qpu_settings.QpuSettings(distance=RECORDED_DISTANCE, device=device)
    inner = decoders.PresetLatencyDecoder(0.028)
    decoder = mwpm.PyMatchingDecoder(inner)
    weak_decoder = decoder_settings.DecoderSettings(decoder=decoder)
    settings = machine_settings.MachineSettings(
        workload=workload, qpu=qpu, weak_decoder=weak_decoder
    )
    machine = machine_module.Machine.build(settings, shot)
    run = machine.run()
    return machine, run.operation_results[0]


def test_a_replayed_shots_truth_is_the_flips_stims_converter_reports(
    recorded_memory,
):
    """The run's truth is the shot's own observable flips, shot for shot.

    Stim's m2d converter is what a decoder is scored against, so the
    loop must report exactly its observable flips for the shot it
    replayed, never the flips of another shot.
    """
    circuit, measurements, detectors, observables = recorded_memory
    reported = []
    expected = []
    for shot in range(len(detectors)):
        _machine, result = replayed_run(circuit, measurements, shot)
        truth = tuple(result.observable_truth)
        flips = as_bits(observables[shot])
        reported.append(truth)
        expected.append(flips)
    assert reported == expected


def test_the_windowed_decode_agrees_with_the_whole_shot_decode(
    recorded_memory,
):
    """Windowing changes when a correction is known, not what it is.

    A window decodes a slice of the shot, so an overlapping-recovery run
    may differ from a whole-shot solve on a shot whose fault chain
    crosses a seam (Skoric 2209.08552), and on no more than that: over
    twelve shots at most one may differ.
    """
    pymatching = pytest.importorskip("pymatching")
    circuit, measurements, detectors, _observables = recorded_memory
    model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)
    agreement_count = 0
    for shot in range(len(detectors)):
        _machine, result = replayed_run(circuit, measurements, shot)
        whole_shot = matching.decode(detectors[shot])
        prediction = as_bits(whole_shot)
        agreement_count += prediction == result.logical_observables
    assert agreement_count >= RECORDED_SHOT_COUNT - 1


def test_the_sliding_windows_predict_what_qldpcs_decoder_predicts(
    recorded_memory,
):
    """Two implementations of one rule agree on every shot.

    decsim's sliding windows and qLDPC's SlidingWindowDecoder both
    implement overlapping recovery (Dennis quant-ph/0110143, Skoric
    2209.08552): fed the same shots, the same geometry (window of commit
    plus buffer, stride of commit), the same round grouping and
    PyMatching with parallel faults merged as independent errors, they
    predict the same bits.
    """
    circuit, measurements, detectors, _observables = recorded_memory
    reference_predictions = qldpc_sliding_predictions(circuit, detectors)
    predictions = []
    expected = []
    for shot in range(len(detectors)):
        _machine, result = replayed_run(circuit, measurements, shot)
        reference_bits = as_bits(reference_predictions[shot])
        predictions.append(result.logical_observables)
        expected.append(reference_bits)
    assert predictions == expected


def qldpc_sliding_predictions(circuit, detectors):
    """What qLDPC's own sliding-window decoder predicts for every shot."""
    qldpc_decoders = pytest.importorskip("qldpc.decoders")
    round_of_detector = detector_chronology.resolve_detector_rounds(
        circuit, None, RECORDED_ROUNDS
    )

    def time_of_detector(detector):
        return int(round_of_detector[detector])

    window_size = 2 * RECORDED_DISTANCE
    reference = qldpc_decoders.SlidingWindowDecoder(
        window_size=window_size,
        stride=RECORDED_DISTANCE,
        detector_to_time=time_of_detector,
        decompose_errors=True,
        with_MWPM=True,
        merge_strategy="independent",
    )
    model = circuit.detector_error_model(decompose_errors=True)
    compiled = reference.compile_decoder_for_dem(model)
    events = detectors.astype(numpy.uint8)
    return compiled.decode_shots(events)


def bits_by_round_on(transfers, path):
    """The payload bits each round put on one link path."""
    bits = {}
    for record in transfers:
        if record.path is not path:
            continue
        first_round = record.attribution.first_round
        carried = bits.get(first_round, 0)
        bits[first_round] = carried + record.transfer.payload_bits
    return bits


def test_every_measured_bit_crosses_the_link_exactly_once(recorded_memory):
    """The wire carries one bit per measure qubit per round, and no more.

    Google 2207.06431 and 2408.13687: every measure qubit is read out
    each cycle, which is d*d - 1 bits, and the final round adds the
    d*d data-qubit readout. The bits the run puts on the link add up to
    Stim's own measurement count for the circuit, so no round is lost
    and none is sent twice.
    """
    circuit, measurements, _detectors, _observables = recorded_memory
    machine, _result = replayed_run(circuit, measurements, 0)
    traffic = machine.observation.traffic.snapshot()
    upward = transfer_records.LinkPath.QPU_TO_CONTROLLER

    bits_by_round = bits_by_round_on(traffic.transfers, upward)

    round_bits = bits_by_round.values()
    total_bits = sum(round_bits)
    assert bits_by_round == {
        1: 8,
        2: 8,
        3: 8,
        4: 8,
        5: 8,
        6: 8,
        7: 8,
        8: 8,
        9: 17,
    }
    assert total_bits == circuit.num_measurements


# The whole machine on the declared card: what only the composition of
# the controller, the stores, the windows and the frame decides.

# each round's publication in microseconds, rounds 1 to 12, under each
# packing bound: publication(r) = max(5 + r, publication(r - b)) + 4
PUBLISHED_MICROSECONDS_BY_BOUND = {
    1: [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54],
    2: [10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31],
    3: [10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24],
    4: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
}
TWELVE_ROUND_RUN_END_TICK = 54_000_000


def published_rounds(machine):
    """(round index, tick) of every round published, in publication order."""
    published = []
    for event in machine.observation.round_events.events:
        if event.kind != "PUBLISHED":
            continue
        published.append((event.round_index, event.tick))
    return published


def publication_ticks(published):
    """The tick of every publication, in publication order."""
    ticks = []
    for _round_index, tick in published:
        ticks.append(tick)
    return ticks


def test_a_full_syndrome_buffer_stalls_the_controller_instead_of_dropping():
    """A round with no slot waits upstream and is published in order.

    A real-time decoder backpressures its source rather than discarding
    syndromes: the Rigetti sequencer polls the decoder's status register
    and stalls (Caune et al. 2410.05202 lines 1255-1257), and a QubiC
    core halts on its idle instruction until the measurement is in
    (Fruitwala et al. 2404.15260 lines 297-302, 566-567). While the
    store has room round r is
    published at r plus qpu_to_controller 2 plus readout_to_bits 3 plus
    controller_to_weak_buffer 4, so rounds 1 to 7 fill a store of
    seven at 16 us and round 8 waits past 17 us for the first window's
    input to land instead of being dropped.
    """
    seven_rounds_bits = 7 * BITS_PER_ROUND
    seven_rounds = syndrome_buffer_settings.SyndromeBufferSettings(
        bits=seven_rounds_bits
    )
    machine = declared_run.weak_only_run(
        rounds=12, weak_syndrome_buffer=seven_rounds
    )
    published = published_rounds(machine)
    ticks = publication_ticks(published)
    publication_tick_by_round = dict(published)
    round_indices = list(publication_tick_by_round)
    seventh_publication = config.microseconds_to_ticks(16.0)
    eighth_publication_with_room = config.microseconds_to_ticks(17.0)
    tiers = declared_run.frame_tiers(machine)

    assert round_indices == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert ticks == sorted(ticks)
    assert publication_tick_by_round[7] == seventh_publication
    assert publication_tick_by_round[8] > eighth_publication_with_room
    assert tiers == [((1, 0), "weak"), ((1, 1), "weak"), ((1, 2), "weak")]


def test_both_stores_settle_empty_at_the_end_of_an_escalating_run():
    """The last commit releases every hold either store placed.

    A store keeps a round while any holder still needs it and frees it
    on the last release (decsim/syndrome_buffer/syndrome_buffer.py), so a
    round left behind is a leak that grows over a sweep of shots
    without ever failing a run. Escalating every window places the
    longest-lived holds the machine has: the room-side context of a
    window is held across its whole strong decode.
    """
    machine = declared_run.switching_run(escalates=True, rounds=9)

    assert machine.readout.weak_syndrome_buffer.occupancy == 0
    assert machine.readout.strong_syndrome_buffer.occupancy == 0
    receiver = machine.readout.strong_syndrome_round_receiver
    assert receiver.reserved_bits_by_round == {}


def test_the_execution_and_the_decoding_views_agree_on_the_workload():
    """One resolved workload reaches the sequencer and the windows.

    The root resolves the workload once and hands the same plan to the
    sequencer, which issues the operations, and to the window tracker,
    which accounts for their rounds (decsim/machine.py, _plan). An id
    in one view and not the other, or a planned round count the
    arrivals never meet, leaves rounds with no readiness account.
    """
    machine = declared_run.weak_only_run(rounds=6)
    sequencer = machine.control.execution_runtime
    tracker = machine.windows.window_manager.tracker
    planner = machine.windows.window_manager.planner
    planned_round_count = planner.round_count_of(1)
    arrived_round_count = tracker.rounds_arrived(1)

    assert set(sequencer.schedule.operations) == {1}
    assert set(tracker.operation_by_id) == {1}
    assert planned_round_count == 6
    assert arrived_round_count == 6


def test_every_program_operation_is_registered_even_with_no_detector_data():
    """An operation that emits nothing still gets its accounts.

    The decode plan only covers operations with detector data, so the
    execution-side registration is the only one that reaches a quiet
    operation; without it the operation would run with no readiness
    account and nothing would report its body done.
    """
    quiet = program_records.Operation(
        id=1,
        name="quiet",
        qubits=(1,),
        patches=(1,),
        emits_detector_data=False,
    )
    measured = declared_run.memory_operation(2)
    operations = [quiet, measured]
    machine = declared_run.weak_only_run(rounds=6, operations=operations)
    tracker = machine.windows.window_manager.tracker
    stamps = machine.observation.runtime_stamps

    assert set(tracker.operation_by_id) == {1, 2}
    assert 1 in tracker.arrivals_by_operation
    assert set(stamps.body_done) == {1, 2}


def twelve_rounds_with_packing_bound(bound):
    """A twelve-round weak-only run on the declared card, that bound set."""
    controller = declared_run.declared_controller(
        packing_rounds_in_flight=bound
    )
    operation = declared_run.memory_operation(1)
    workload = declared_run.declared_workload([operation], 12)
    weak_microseconds = declared_run.DECLARED_MICROSECONDS["weak"]
    decoder = decoders.PresetLatencyDecoder(weak_microseconds)
    weak_decoder = decoder_settings.DecoderSettings(decoder=decoder)
    links = declared_run.declared_profile()
    qpu = declared_run.declared_qpu()
    frame = declared_run.declared_frame()
    return machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        weak_decoder=weak_decoder,
        links=links,
        controller=controller,
        pauli_frame=frame,
    )


@pytest.mark.parametrize("bound", [1, 2, 3, 4])
def test_a_full_packing_stage_holds_each_round_until_one_leaves(bound):
    """At most b rounds are in flight; the next enters as one is published.

    controller.packing_rounds_in_flight bounds the whole packing stage: a
    round counts from its emission until it is published. On the declared
    card round r is emitted at r, its bits arrive at r plus
    qpu_to_controller 2 plus readout_to_bits 3, and it is published four
    microseconds after both its bits and its place are there. Under a
    bound of b it takes its place at max(r, publication of round r - b),
    so it is published at max(5 + r, publication of round r - b) + 4: the
    window law of credit flow control, b credits and a four microsecond
    return (garnet's OutVcState credit count,
    src/mem/ruby/network/garnet/OutVcState.hh:51-54). No round is lost.
    """
    settings = twelve_rounds_with_packing_bound(bound)
    machine = machine_module.Machine.build(settings, 0)

    result = machine.run()

    published = published_rounds(machine)
    ticks = publication_ticks(published)
    expected_microseconds = PUBLISHED_MICROSECONDS_BY_BOUND[bound]
    expected_ticks = [
        config.microseconds_to_ticks(microseconds)
        for microseconds in expected_microseconds
    ]
    assert result.terminal_status == "complete"
    assert ticks == expected_ticks


def test_a_packing_bound_of_six_clears_a_twelve_round_run():
    """Five rounds overlap on this card, so six slots never fill.

    Round r occupies the stage from 5 + r us to 9 + r us, so at most
    five rounds are in flight at once and no round is ever held; the
    run then ends at the last window's commit, 54 us.
    """
    settings = twelve_rounds_with_packing_bound(6)
    machine = machine_module.Machine.build(settings, 0)

    result = machine.run()

    assert result.terminal_status == "complete"
    assert machine.engine.now == TWELVE_ROUND_RUN_END_TICK


def link_totals(machine, path):
    """(transfers, payload bits) one link path carried in a whole run."""
    traffic = machine.observation.traffic.snapshot()
    transfers = 0
    payload_bits = 0
    for record in traffic.transfers:
        if record.path is not path:
            continue
        transfers += 1
        payload_bits += record.transfer.payload_bits
    return transfers, payload_bits


def reference_run(escalation_kind):
    """One shot of reference.yaml at d=3, p=0.001, on one escalation kind."""
    config_path = CONFIGS / "reference.yaml"
    config = experiment.load_experiment(config_path)
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    escalation = dataclasses.replace(settings.escalation, kind=escalation_kind)
    settings = dataclasses.replace(settings, escalation=escalation)
    machine = machine_module.Machine.build(settings, 0)
    machine.run()
    return machine


def test_the_controller_writes_every_round_into_buffer_0_over_a_priced_hop():
    """Fifteen rounds are fifteen transfers carrying the round's own bits.

    Toshio et al. 2510.25222 price the syndrome data of each round
    between the system controller and the weak decoder as T_comm^weak
    (Table I), so the write into the weak syndrome buffer is a transfer like
    every other hop rather than a free store: reference.yaml at d = 3
    puts its fifteen rounds on controller_to_weak_buffer. The two hops
    carry different widths, because the card's row forms the detection
    events at the controller: the readout hop carries the 129 measured
    outcomes, and the store hop the 120 events they formed, eight per
    round.
    """
    machine = reference_run("weak_baseline")
    store_hop = transfer_records.LinkPath.CONTROLLER_TO_WEAK_BUFFER
    readout_hop = transfer_records.LinkPath.QPU_TO_CONTROLLER
    room_hop = transfer_records.LinkPath.CONTROLLER_TO_STRONG_BUFFER

    assert link_totals(machine, store_hop) == (15, 120)
    assert link_totals(machine, readout_hop) == (15, 129)
    assert link_totals(machine, room_hop) == (0, 0)


def test_a_strong_only_run_writes_every_round_into_buffer_1_over_a_priced_hop():
    """The same law on the strong syndrome buffer, the only one it fills.

    T_comm^strong is Toshio's symbol for the same transport to the
    strong decoder, and a strong-primary plan reads its windows from
    strong syndrome buffer, so every round takes controller_to_strong_buffer
    once, at the width it leaves the controller, and the weak syndrome buffer
    sees none of them.
    """
    machine = reference_run("strong_only")
    store_hop = transfer_records.LinkPath.CONTROLLER_TO_WEAK_BUFFER
    readout_hop = transfer_records.LinkPath.QPU_TO_CONTROLLER
    room_hop = transfer_records.LinkPath.CONTROLLER_TO_STRONG_BUFFER

    assert link_totals(machine, room_hop) == (15, 120)
    assert link_totals(machine, readout_hop) == (15, 129)
    assert link_totals(machine, store_hop) == (0, 0)


def strong_primary_settings(escalation):
    """The declared strong-primary run, escalated by that section."""
    operation = declared_run.memory_operation(1)
    workload = declared_run.declared_workload([operation], 6)
    latency = declared_run.DECLARED_MICROSECONDS["strong"]
    decoder = decoders.PresetLatencyDecoder(latency)
    strong_decoder = decoder_settings.DecoderSettings(decoder=decoder)
    qpu = declared_run.declared_qpu()
    links = declared_run.declared_profile()
    controller = declared_run.declared_controller()
    frame = declared_run.declared_frame()
    return machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        strong_decoder=strong_decoder,
        escalation=escalation,
        links=links,
        controller=controller,
        pauli_frame=frame,
    )


def test_a_policy_object_decodes_where_its_name_decodes():
    """The built policy is the authority, so both forms are one run.

    sinter resolves the caller's own decoders before its built-in table
    (sinter/_collection/_mux_sampler.py:33-40), and a run named
    strong_only and a run given StrongOnly() are the same machine: the
    same decoder, the same bits, the same ticks.
    """
    named = escalation_settings.EscalationSettings(kind="strong_only")
    policy = escalation_policies.StrongOnly(escalation_policies.NO_CONFIDENCE)
    by_object = escalation_settings.EscalationSettings(policy=policy)
    named_settings = strong_primary_settings(named)
    named_machine = machine_module.Machine.build(named_settings, 0)
    named_result = named_machine.run()
    object_settings = strong_primary_settings(by_object)
    object_machine = machine_module.Machine.build(object_settings, 0)
    object_result = object_machine.run()
    assert object_machine.decoders.primary_decoder is not None
    assert type(object_machine.decoders.primary_decoder) is type(
        named_machine.decoders.primary_decoder
    )
    assert object_result.fully_done_ticks == named_result.fully_done_ticks
    named_observables = named_result.operation_results[0].logical_observables
    object_observables = object_result.operation_results[0].logical_observables
    assert object_observables == named_observables
    assert declared_run.frame_tiers(object_machine) == declared_run.frame_tiers(
        named_machine
    )


def test_a_policy_object_whose_tier_names_no_decoder_is_refused():
    """The refusal reads the policy's tier, not the section's name."""
    policy = escalation_policies.StrongOnly(escalation_policies.NO_CONFIDENCE)
    escalation = escalation_settings.EscalationSettings(policy=policy)
    settings = strong_primary_settings(escalation)
    no_decoder = decoder_settings.DecoderSettings()
    settings = dataclasses.replace(settings, strong_decoder=no_decoder)
    with pytest.raises(ValueError) as refusal:
        machine_module.Machine.build(settings, 0)
    assert "decodes windows on the strong tier" in str(refusal.value)


# ---- a room-side landing that arrives after its operation closed

# The reproducer's links: every hop of the weak path one fridge cycle,
# so the weak tier commits the last window before round 15's copy
# finishes crossing controller_to_strong_buffer, whose 0.26 us is the
# reference card's (Caune 2410.05202 Fig. 1a F).
FRIDGE_CYCLE = {"latency_cycles": 1, "clock": "fridge", "bits_per_cycle": None}
LATE_LANDING_LINKS = {
    "qpu_to_controller": FRIDGE_CYCLE,
    "controller_to_weak_buffer": FRIDGE_CYCLE,
    "weak_buffer_to_weak_decoder": FRIDGE_CYCLE,
    "decoder_to_decoder": FRIDGE_CYCLE,
    "weak_decoder_to_frame": FRIDGE_CYCLE,
}
FREE_CROSSING = {"latency_cycles": 0, "clock": "fridge", "bits_per_cycle": None}
LATE_LANDING_ESCALATION = {
    "kind": "switching",
    "confidence": "complementary_gap",
    "gap_threshold_db": 20.0,
    "threshold_source": "fixed",
    "strong_window": "redo_window",
}


def late_landing_shot(directory, links):
    """One seeded shot of the reproducer, its room-side crossing as given."""
    directory.mkdir(parents=True, exist_ok=True)
    strong_decoder = yaml_configs.strong_unit(30.0)
    card = {
        "escalation": LATE_LANDING_ESCALATION,
        "links": links,
        **strong_decoder,
    }
    config_path = yaml_configs.write_config(directory, card)
    experiment_config = experiment.load_experiment(config_path)
    task = experiment_config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    return collect.run_shot(task, 0)


def frame_commit_ticks(machine):
    """The tick every frame record committed on, in commit order."""
    snapshot = machine.control.pauli_frame.snapshot()
    ticks = []
    for record in snapshot.records:
        ticks.append(record.committed_ticks)
    return tuple(ticks)


def test_a_landing_after_its_operations_close_costs_the_result_nothing(
    tmp_path,
):
    """The weak tier decided the result; the late copy changes no tick.

    On a 0.028 us weak card the last window commits at 15.224 us, so
    every reader of round 15 has resolved and the operation closes
    before that round's copy lands on the room side 0.26 us later. The
    landing is dropped at the door, the run still settles (Machine.run
    checks every component), and the same run with a free crossing,
    where every landing precedes the close, commits on exactly the same
    ticks.
    """
    priced_directory = tmp_path / "priced"
    priced = late_landing_shot(priced_directory, LATE_LANDING_LINKS)
    free_links = {
        **LATE_LANDING_LINKS,
        "controller_to_strong_buffer": FREE_CROSSING,
    }
    free_directory = tmp_path / "free"
    free = late_landing_shot(free_directory, free_links)

    assert priced.result.terminal_status == "complete"
    receiver = priced.machine.readout.strong_syndrome_round_receiver
    assert receiver.store.occupancy == 0
    assert frame_commit_ticks(priced.machine) == frame_commit_ticks(
        free.machine
    )
    priced_result = priced.result.operation_results[0]
    free_result = free.result.operation_results[0]
    assert priced_result.logical_observables == free_result.logical_observables


# IBM's verification rule on the sixteen decoder arrangements: the priced,
# windowed, timed run's correction beside an untimed software model's,
# window by window.

ARRANGEMENT_DISTANCE = 3
ARRANGEMENT_ROUND_COUNT = 30  # the machine's rounds_per_shot, 10d at d 3
ARRANGEMENT_PHYSICAL_ERROR = 0.001
ARRANGEMENT_ROUND_PERIOD_US = 1.0
ARRANGEMENT_SHOT_COUNT = 8
# seeds 9 to 16: the eight shots include windows the switching
# arrangements escalate, which the kept-window rule needs beside it
ARRANGEMENT_FIRST_SEED = 9
ARRANGEMENT_SEED_END = ARRANGEMENT_FIRST_SEED + ARRANGEMENT_SHOT_COUNT
ARRANGEMENT_SEEDS = range(ARRANGEMENT_FIRST_SEED, ARRANGEMENT_SEED_END)
ARRANGEMENT_WINDOW_COUNT = 10  # 30 rounds committed 3 at a time
RELAY_BP_SPECIFICATION = importlib.util.find_spec("relay_bp")
LACKS_RELAY_BP = RELAY_BP_SPECIFICATION is None
# relay_bp is the bb-decoders extra, which a test install leaves out
NEEDS_RELAY_BP = pytest.mark.skipif(
    LACKS_RELAY_BP, reason="could not import 'relay_bp'"
)
SINGLE_TIER_ARRANGEMENTS = (
    "pymatching_weak",
    "pymatching_strong",
    "union_find_weak",
    "union_find_strong",
    "bposd_weak",
    "bposd_strong",
    "belief_matching_strong",
    pytest.param("relay_bp_weak", marks=NEEDS_RELAY_BP),
    pytest.param("relay_bp_strong", marks=NEEDS_RELAY_BP),
)
SWITCHING_ARRANGEMENTS = (
    "pymatching_bposd_switching",
    "pymatching_belief_matching_switching",
    pytest.param("pymatching_relay_bp_switching", marks=NEEDS_RELAY_BP),
    "union_find_bposd_switching",
    "union_find_belief_matching_switching",
    "union_find_pymatching_switching",
    pytest.param("union_find_relay_bp_switching", marks=NEEDS_RELAY_BP),
)
MATCHING_ARRANGEMENTS = ("pymatching_weak", "pymatching_strong")
KEPT_WEAK_REQUEST = (
    decoding_records.RequestProcessingOutcome.PRIMARY_FORWARDED_FOR_DELIVERY
)
ESCALATED_WEAK_REQUEST = (
    decoding_records.RequestProcessingOutcome.WEAK_AWAITED_STRONG
)


def arrangement_point_task(folder, arrangement):
    """One decoder arrangement's point, read from its yaml."""
    config_path = decoder_arrangements.write_arrangement(folder, arrangement)
    config = experiment.load_experiment(config_path)
    return config.point_task(
        {
            yaml_configs.ERROR_RATE_PATH: ARRANGEMENT_PHYSICAL_ERROR,
            "qpu.distance": ARRANGEMENT_DISTANCE,
            "qpu.round_period_microseconds": ARRANGEMENT_ROUND_PERIOD_US,
        },
    )


def record_one_decode(decodes, reference, job, result, outcome, ended_ticks):
    """The run's answer on one request beside the untimed model's answer."""
    del ended_ticks
    reference_result = reference.decode(job)
    run_correction = tuple(result.correction)
    model_correction = tuple(reference_result.correction)
    decodes.append((outcome, job.window_id, run_correction, model_correction))


def timed_and_untimed_decodes(folder, arrangement):
    """Every request of the arrangement's shots, beside the untimed model's.

    The software model is a second Machine built from the same settings
    and the same seed and never run: its decoder is the same table row
    carrying the same run-derived state, so a row that draws from the
    run seed (relay_bp's gamma table, window_decoder.py `_gamma_seed`)
    reads the table the priced run read.
    """
    task = arrangement_point_task(folder, arrangement)
    models = built_window_models.BuiltWindowModels()
    decodes = []
    for seed in ARRANGEMENT_SEEDS:
        settings = task.shot_settings(models)
        machine = machine_module.Machine.build(settings, seed)
        untimed = machine_module.Machine.build(settings, seed)
        reference = untimed.decoders.primary_decoder.decoder
        listener = functools.partial(record_one_decode, decodes, reference)
        outcomes = machine.decoders.decoder_manager.outcomes
        outcomes.trace.request_ended.connect(listener)
        machine.run()
    return decodes


def corrections_that_differ(decodes):
    """Every recorded decode the untimed model did not answer bit for bit."""
    differing = []
    for outcome, window_id, run_correction, model_correction in decodes:
        if run_correction == model_correction:
            continue
        differing.append((outcome, window_id))
    return differing


def decodes_with_outcome(decodes, outcome):
    """The recorded decodes that ended in one processing outcome."""
    selected = []
    for decode in decodes:
        if decode[0] is not outcome:
            continue
        selected.append(decode)
    return selected


@pytest.mark.parametrize("arrangement", SINGLE_TIER_ARRANGEMENTS)
def test_a_single_tier_arrangement_commits_what_the_model_decodes(
    tmp_path, arrangement
):
    """IBM's rule for their gross-code FPGA decoder, on every window.

    "The approach to validate the Relay-BP and the surrounding sliding
    window decoder is to make sure that hardware results match exactly
    the software emulation model", checked on the estimated error vector
    of each window (IBM arXiv 2510.21600 lines 488-495). decsim's
    hardware is the priced run: the links, the syndrome buffers, the
    decoder pool and the staged unit all sit between the sampled shot
    and the correction. Its software model is the same decoder row
    called directly on the same window error model and the same landed
    syndrome, which is PyMatching's own untimed entry point
    (decode_detection_events, src/pymatching/sparse_blossom/driver/
    mwpm_decoding.h lines 55-79 at PyMatching commit 6f63b2b9). Every
    window of eight shots at p = 0.001 and d = 3, ten windows a shot,
    agrees bit for bit, so no change to timing, buffers, links or pools
    can move a correction without this failing.
    """
    decodes = timed_and_untimed_decodes(tmp_path, arrangement)

    differing = corrections_that_differ(decodes)

    expected_count = ARRANGEMENT_SHOT_COUNT * ARRANGEMENT_WINDOW_COUNT
    assert len(decodes) == expected_count
    assert differing == []


@pytest.mark.parametrize("arrangement", SWITCHING_ARRANGEMENTS)
def test_a_switching_arrangements_kept_windows_match_the_weak_model(
    tmp_path, arrangement
):
    """The same rule where the weak tier's answer stands.

    A switching arrangement escalates a low-confidence window to the strong
    tier (Toshio et al. 2510.25222 Sec. III A), so only the windows the
    weak verdict kept are the weak model's to answer; the escalated ones
    are counted and left to the strong tier. Eight shots at p = 0.001
    and d = 3 escalate one of their eighty windows, and every
    window the verdict kept carries the correction the untimed weak row
    decodes from the same window error model and the same syndrome
    (IBM arXiv 2510.21600 lines 488-495).
    """
    decodes = timed_and_untimed_decodes(tmp_path, arrangement)
    kept = decodes_with_outcome(decodes, KEPT_WEAK_REQUEST)
    escalated = decodes_with_outcome(decodes, ESCALATED_WEAK_REQUEST)

    differing = corrections_that_differ(kept)

    assert differing == []
    assert len(escalated) > 0


def qldpc_arrangement_predictions(circuit, shot_events):
    """What qLDPC's sliding-window decoder predicts for the same shots."""
    qldpc_decoders = pytest.importorskip("qldpc.decoders")
    round_of_detector = detector_chronology.resolve_detector_rounds(
        circuit, None, ARRANGEMENT_ROUND_COUNT
    )

    def time_of_detector(detector):
        return int(round_of_detector[detector])

    window_size = 2 * ARRANGEMENT_DISTANCE
    reference = qldpc_decoders.SlidingWindowDecoder(
        window_size=window_size,
        stride=ARRANGEMENT_DISTANCE,
        detector_to_time=time_of_detector,
        decompose_errors=True,
        with_MWPM=True,
        merge_strategy="independent",
    )
    model = circuit.detector_error_model(decompose_errors=True)
    compiled = reference.compile_decoder_for_dem(model)
    detection_events = numpy.array(shot_events, dtype=numpy.uint8)
    decoded = compiled.decode_shots(detection_events)
    predictions = []
    for row in decoded:
        bits = as_bits(row)
        predictions.append(bits)
    return predictions


def arrangement_predictions_and_events(folder, arrangement):
    """The arrangement's prediction per shot, the circuit, and the events."""
    task = arrangement_point_task(folder, arrangement)
    models = built_window_models.BuiltWindowModels()
    predictions = []
    shot_events = []
    sampled = None
    for seed in ARRANGEMENT_SEEDS:
        settings = task.shot_settings(models)
        machine = machine_module.Machine.build(settings, seed)
        result = machine.run()
        sampled = machine.observation.sampled_shots.shots_by_operation[1]
        shot_events.append(sampled.detection_events)
        operation_result = result.operation_results[0]
        predictions.append(operation_result.logical_observables)
    return predictions, sampled.circuit, shot_events


@pytest.mark.parametrize("arrangement", MATCHING_ARRANGEMENTS)
def test_a_matching_arrangement_predicts_what_qldpcs_sliding_windows_predict(
    tmp_path, arrangement
):
    """The priced arrangement run beside an outside sliding-window decoder.

    qLDPC's SlidingWindowDecoder (qldpc/decoders/sinter.py, class
    SlidingWindowDecoder) decodes the same sampled shots untimed, with
    window of commit plus buffer, stride of commit, and parallel faults
    merged as independent errors, as
    test_the_sliding_windows_predict_what_qldpcs_decoder_predicts above
    reads it. The two window plans are not the same at the tail: the
    arrangements' `terminal_policy: lookahead` lays ten windows over thirty
    rounds where qLDPC's stride lays nine, so the comparison is at the
    run's final observable rather than per window. The eight shots at
    p = 0.001 agree exactly; a difference would have to be a
    minimum-weight matching tie, the only kind of disagreement ever
    seen against this reference, and this test refuses one rather than
    allowing it.
    """
    predictions, circuit, shot_events = arrangement_predictions_and_events(
        tmp_path, arrangement
    )

    reference_predictions = qldpc_arrangement_predictions(circuit, shot_events)

    assert len(predictions) == ARRANGEMENT_SHOT_COUNT
    assert predictions == reference_predictions


def test_a_sampled_stream_segment_has_no_independent_accuracy_result() -> None:
    circuit = workload_settings.memory_circuit(
        "surface_code:rotated_memory_z", 24, 3, 0.001
    )
    source = stim_device.StimDevice()
    settings = _protected_memory_settings(circuit, source)
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    prefix = result.operation_results[0]
    owner = result.operation_results[-1]
    assert prefix.logical_observables is not None
    assert prefix.stream_offset == 0
    assert prefix.observable_truth is None
    assert prefix.logical_failure is None
    assert owner.observable_truth is not None
    assert owner.logical_failure is not None


def test_a_recorded_stream_scores_the_full_prediction_against_full_truth() -> (
    None
):
    """Sinter's StimThenDecodeSampler scores equal full-shot output scopes."""
    circuit = workload_settings.memory_circuit(
        "surface_code:rotated_memory_z", 24, 3, 0.001
    )
    noiseless = circuit.without_noise()
    sampler = noiseless.compile_sampler(seed=10)
    measurements = sampler.sample(1)
    # Stim's first final data bit belongs to logical Z and a final check.
    measurements[0, -9] ^= True
    source = stim_device.RecordedStimDevice(measurements, 0)
    settings = _protected_memory_settings(circuit, source)
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    prefix = result.operation_results[0]
    owner = result.operation_results[-1]
    assert prefix.logical_observables == (0,)
    assert prefix.observable_truth is None
    assert prefix.logical_failure is None
    assert owner.logical_observables == (1,)
    assert owner.observable_truth == (1,)
    assert owner.logical_failure is False


def test_a_released_pulse_on_a_protected_patch_waits_for_one_boundary():
    """The pulse leaves at the decision and starts on the next boundary.

    The controller's branch arms the pulse as the decision lands and the
    QPU plays it at its next timing point (QubiC 2404.15260 lines
    286-290, eQASM 1808.02449 lines 535-545). The protected patch keeps
    measuring meanwhile (Quantum Machines 2412.00289 lines 524-531).
    """
    circuit = workload_settings.memory_circuit(
        "surface_code:rotated_memory_z", 24, 3, 0.001
    )
    source = stim_device.StimDevice()
    settings = _protected_memory_settings(circuit, source)
    machine = machine_module.Machine.build(settings, 0)
    outputs = {}
    commands = {}

    def heard_output(event):
        outputs[(event.kind, event.operation_id)] = event.tick

    def heard_command(event):
        commands[(event.kind, event.command.operation.id)] = event.tick

    machine.control.instruction_output.trace.output_event.connect(heard_output)
    machine.qpu.device.trace.command_event.connect(heard_command)
    machine.run()

    decided = outputs[("DECISION_AVAILABLE", 3)]
    issued = outputs[("CONTROL_PULSE_COMMAND_ISSUED", 3)]
    arrived = commands[("ARRIVED", 3)]
    started = commands[("STARTED", 3)]
    assert issued == decided
    boundary = machine.qpu.device.boundary_at_or_after(arrived)
    assert started == boundary == 7_700_000


def test_a_released_feedback_source_binds_to_the_round_it_starts_at():
    """The source reads the protected stream from the boundary it starts on.

    A released command is sent when its decision lands and starts on the
    first boundary after it arrives, so its place in the stream is known
    only then, as a segment's is (bind_at_start). With a one microsecond
    controller_to_qpu the resume pulse sent at 6.872 us arrives after the
    7.7 us boundary and starts at 8.8 us, after the prefix's three rounds
    and the protected rounds of 4.4, 5.5, 6.6 and 7.7 us.
    """
    circuit = workload_settings.memory_circuit(
        "surface_code:rotated_memory_z", 24, 3, 0.001
    )
    source = stim_device.StimDevice()
    settings = _protected_memory_settings(circuit, source)
    prefix, protect, resume, finish = settings.workload.operations
    check = program_records.Operation(
        5,
        "check",
        (0,),
        patches=(0,),
        predecessors=(3,),
        blocked_by=3,
        emits_detector_data=False,
    )
    finish = dataclasses.replace(finish, predecessors=(5,))
    rounds = {100: 24, 1: 3, 2: 0, 3: 1, 5: 1, 4: 0}
    policy = round_policies.PerOperationRounds(rounds)
    operations = (prefix, protect, resume, check, finish)
    workload = dataclasses.replace(
        settings.workload, operations=operations, rounds_policy=policy
    )
    links = _price_path(settings.links, "controller_to_qpu", 1_000_000)
    settings = dataclasses.replace(settings, workload=workload, links=links)
    machine = machine_module.Machine.build(settings, 0)
    starts = {}

    def heard_command(event):
        starts[(event.kind, event.command.operation.id)] = event.tick

    machine.qpu.device.trace.command_event.connect(heard_command)
    machine.run()

    binding = machine.control.issuer.stream_binding_for(3)
    assert starts[("STARTED", 3)] == 8_800_000
    assert (binding.stream_id, binding.stream_offset) == (100, 7)


def test_an_operation_claims_every_idle_cycle_before_it_starts():
    """The rounds its patches measure until it starts are its history.

    Two memories on patches 1 and 2 end and their patches idle. The
    operation on both is released by the first memory's decision, sent
    at 32 us and started on the 34 us boundary. A waiting patch keeps
    measuring and those rounds must be decoded too (Quantum Machines
    2412.00289 lines 524-531), so the operation's one window batches
    every idle cycle through its start boundary, 31 of them.
    """
    first = declared_run.memory_operation(1)
    second = declared_run.memory_operation(2)
    both = program_records.Operation(
        3,
        "both",
        (1, 2),
        patches=(1, 2),
        predecessors=(1, 2),
        blocked_by=1,
    )
    workload = declared_run.declared_workload([first, second, both], 3)
    scheme = naive_online.NaiveOnlineScheme()
    windows = window_settings.WindowSettings(scheme=scheme)
    decoder = decoders.PresetLatencyDecoder(10.0)
    weak_decoder = decoder_settings.DecoderSettings(decoder=decoder)
    ignore = idle_policies.Ignore()
    idle = controller_settings.IdlePolicySettings(policy=ignore)
    qpu = declared_run.declared_qpu()
    links = declared_run.declared_profile()
    controller = declared_run.declared_controller()
    frame = declared_run.declared_frame()
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        weak_decoder=weak_decoder,
        links=links,
        controller=controller,
        pauli_frame=frame,
        windows=windows,
        idle_policy=idle,
    )
    machine = machine_module.Machine.build(settings, 0)
    idle_patches = []

    def heard_idle_round(operation_id, patch, round_index):
        del operation_id, round_index
        idle_patches.append(patch)

    idle_rounds = machine.control.idle_rounds
    idle_rounds.trace.idle_round_emitted.connect(heard_idle_round)
    machine.run()

    window = machine.windows.window_manager.planner.plan.windows[(3, 0)]
    assert idle_patches.count(1) == idle_patches.count(2) == 31
    assert window.batched_preceding_idle_round_count == 31


def _protected_memory_settings(circuit, source):
    """One physical stream with a prefix that releases a waiting operation."""
    owner = program_records.Operation(
        100, "memory", (0,), patches=(0,), circuit=circuit
    )
    prefix = program_records.Operation(
        1,
        "prefix",
        (0,),
        patches=(0,),
        circuit=circuit,
        stream_id=100,
        stream_offset=0,
    )
    begin = program_records.Operation(
        2,
        "protect",
        (0,),
        patches=(0,),
        predecessors=(1,),
        emits_detector_data=False,
    )
    resume = program_records.Operation(
        3,
        "resume",
        (0,),
        patches=(0,),
        predecessors=(2,),
        blocked_by=1,
        emits_detector_data=False,
    )
    finish = program_records.Operation(
        4,
        "readout",
        (0,),
        patches=(0,),
        predecessors=(3,),
        scheduled_start_round=24,
        emits_detector_data=False,
    )
    region = program_records.ProtectedRegion(100, 2, 4)
    policy = round_policies.PerOperationRounds(
        {100: 24, 1: 3, 2: 0, 3: 1, 4: 0}
    )
    workload = workload_settings.WorkloadSettings(
        operations=(prefix, begin, resume, finish),
        dynamic_streams=(owner,),
        protected_regions=(region,),
        rounds_policy=policy,
    )
    qpu = qpu_settings.QpuSettings(distance=3, device=source)
    decoder = decoder_settings.DecoderSettings(kind=0.1, engine=ENGINE_CARD)
    return machine_settings.MachineSettings(
        workload=workload, qpu=qpu, weak_decoder=decoder
    )


@dataclasses.dataclass
class _Run:
    """One completed run and the public readout/storage observations."""

    machine: machine_module.Machine
    result: result_records.RunResult
    packets: list[round_records.QPUReadout]
    shots: list
    weak_writes: list
    strong_occupancies: list[int]
    decodes: declared_run.FinishedDecodes


def _program(producer: str) -> circuit_records.RepeatedStimCircuit:
    if producer == "stim":
        return memory_programs.memory_program()
    pytest.importorskip("deltakit_explorer")
    return deltakit.memory_rounds(
        "rotated_surface", 3, "Z", 0.003, round_period_microseconds=1.1
    )


# the seats a placement word names: the controller, or both decoders,
# which serves a run of either primary tier
SEATS_BY_PLACEMENT = {
    "controller": ("controller",),
    "decoder": ("weak_decoder", "strong_decoder"),
}


def _settings(
    program: circuit_records.RepeatedStimCircuit,
    history: str,
    placement: str,
    period_microseconds: float = 1.1,
) -> machine_settings.MachineSettings:
    base = _physical_settings(program, history, period_microseconds)
    strong = dataclasses.replace(base.weak_decoder, kind=0.2)
    weak = decoder_settings.DecoderSettings()
    escalation = escalation_settings.EscalationSettings(kind="strong_only")
    seats = SEATS_BY_PLACEMENT[placement]
    detection_events = dataclasses.replace(
        base.detection_events, formed_at=seats
    )
    observation = dataclasses.replace(
        base.observation, record_switching_windows=True
    )
    return dataclasses.replace(
        base,
        weak_decoder=weak,
        strong_decoder=strong,
        escalation=escalation,
        detection_events=detection_events,
        observation=observation,
    )


def _physical_settings(
    program: circuit_records.RepeatedStimCircuit,
    history: str,
    period_microseconds: float,
) -> machine_settings.MachineSettings:
    if history == "live":
        return live_example.live_settings(
            program,
            distance=3,
            round_period_microseconds=period_microseconds,
            prefix_round_count=3,
            patch="stream-patch",
            feedback_microseconds=4.0,
            decoder_microseconds=0.1,
        )
    circuit, mapping = program.assemble(24)
    workload = finite_example.protection_workload(
        circuit, mapping, 24, 3, "stream-patch"
    )
    return finite_example.supplied_settings(
        workload,
        distance=3,
        round_count=24,
        period_microseconds=period_microseconds,
        feedback_microseconds=4.0,
    )


def _run(
    settings: machine_settings.MachineSettings, seed: int = RUN_SEED
) -> _Run:
    """One run of the machine: one shot of the source at the seed."""
    machine = machine_module.Machine.build(settings, seed)
    shots = []
    source = machine.qpu.syndrome_source
    source.shot_sampled.connect(lambda *sample: shots.append(sample))
    packets = []
    weak_writes = []
    strong_occupancies = []
    machine.qpu.device.trace.round_emitted.connect(packets.append)
    machine.readout.weak_syndrome_buffer.trace.round_stored.connect(
        lambda *event: weak_writes.append(event)
    )
    observe = functools.partial(
        _record_strong_occupancy, machine, strong_occupancies
    )
    if machine.readout.strong_syndrome_buffer is not None:
        strong_store = machine.readout.strong_syndrome_buffer
        strong_store.trace.round_stored.connect(observe)
    decodes = declared_run.FinishedDecodes()
    decodes.attach(machine)
    result = machine.run()
    return _Run(
        machine,
        result,
        packets,
        shots,
        weak_writes,
        strong_occupancies,
        decodes,
    )


def _record_strong_occupancy(
    machine: machine_module.Machine,
    taken_bits: list[int],
    _key: tuple,
    _packet: round_records.SyndromeRoundPacket,
) -> None:
    """The strong store's bits held and reserved, as its room test sums them."""
    stored_bits = machine.readout.strong_syndrome_buffer.occupied_bits
    receiver = machine.readout.strong_syndrome_round_receiver
    reserved_widths = receiver.reserved_bits_by_round.values()
    reserved_bits = sum(reserved_widths)
    taken = stored_bits + reserved_bits
    taken_bits.append(taken)


def _assert_direct_strong_path(run: _Run) -> None:
    assert run.result.terminal_status == "complete"
    assert run.result.decode_work_settled
    assert run.result.event_queue_empty
    requests = run.machine.observation.decode_records.requests
    tiers = {request.request_key.tier for request in requests}
    assert tiers == {window_records.DecoderTier.STRONG}
    paths = _transfer_paths(run)
    forbidden = {
        "controller_to_weak_buffer",
        "weak_buffer_to_weak_decoder",
        "weak_decoder_to_frame",
        "weak_decoder_to_strong_decoder",
    }
    assert not paths.intersection(forbidden)
    assert "controller_to_strong_buffer" in paths
    assert "strong_buffer_to_strong_decoder" in paths
    assert "strong_decoder_to_frame" in paths
    assert run.weak_writes == []


def _assert_actual_truth(run: _Run) -> None:
    assert len(run.shots) == 1
    operation = run.shots[0][0]
    circuit = operation.circuit
    measurements = _raw_bits(run.packets)
    assert len(measurements) == circuit.num_measurements
    raw = numpy.array([measurements], dtype=numpy.bool_)
    converter = circuit.compile_m2d_converter()
    events, truth = converter.convert(
        measurements=raw, separate_observables=True
    )
    source = run.machine.qpu.syndrome_source
    actual_events = source.sampled_detection_events(100)
    actual_truth = source.logical_observable_truth(100)
    numpy.testing.assert_array_equal(actual_events, events[0])
    numpy.testing.assert_array_equal(actual_truth, truth[0])
    owner_result = run.result.operation_results[-1]
    assert owner_result.operation_id == 100
    assert owner_result.observable_truth == actual_truth
    assert owner_result.logical_failure is not None


def _assert_drained(run: _Run) -> None:
    assert run.machine.readout.weak_syndrome_buffer.occupancy == 0
    assert run.machine.readout.strong_syndrome_buffer.occupancy == 0
    receiver = run.machine.readout.strong_syndrome_round_receiver
    assert receiver.reserved_bits_by_round == {}
    units = run.machine.decoders.decoder_manager.pool.units
    occupied = [unit.memory.occupied_bits for unit in units]
    assert occupied == [0] * len(units)


def _stored_bit_count(run: _Run, placement: str) -> int:
    circuit = run.shots[0][0].circuit
    if placement == "controller":
        return circuit.num_detectors
    return circuit.num_measurements


def _raw_bits(packets: list[round_records.QPUReadout]) -> tuple[int, ...]:
    return tuple(bit for packet in packets for bit in packet.bits)


def _transfers(result: result_records.RunResult, path: str) -> list[dict]:
    return [
        row for row in result.link_traffic["transfers"] if row["path"] == path
    ]


def _zero_delay_links(
    links: link_settings.FabricSettings,
) -> link_settings.FabricSettings:
    paths_by_name = {}
    for path in transfer_records.LinkPath:
        settings = links.path_settings(path)
        channel = dataclasses.replace(
            settings.channel, propagation_latency_ticks=0, capacity=None
        )
        paths_by_name[path.value] = dataclasses.replace(
            settings, channel=channel
        )
    return dataclasses.replace(links, **paths_by_name)


def _declared_timing(
    settings: machine_settings.MachineSettings,
) -> machine_settings.MachineSettings:
    links = _zero_delay_links(settings.links)
    links = _price_path(links, "controller_to_strong_buffer", 250_000)
    links = _price_path(links, "strong_buffer_to_strong_decoder", 350_000)
    links = _price_path(links, "strong_decoder_to_frame", 400_000)
    engine = dataclasses.replace(
        settings.strong_decoder.engine,
        fetch_cycles_per_round=0,
        fetch_cycles_per_job=0,
        release_cycles_per_job=0,
        release_cycles_per_round=0,
    )
    decoder = dataclasses.replace(settings.strong_decoder, engine=engine)
    return dataclasses.replace(settings, links=links, strong_decoder=decoder)


def _price_path(
    links: link_settings.FabricSettings, name: str, latency_ticks: int
) -> link_settings.FabricSettings:
    path = getattr(links, name)
    # a pure delay, so the path's time is the latency written here
    channel = dataclasses.replace(
        path.channel, propagation_latency_ticks=latency_ticks, capacity=None
    )
    priced = dataclasses.replace(path, channel=channel)
    return dataclasses.replace(links, **{name: priced})


def _command_tick(run: _Run, kind: str, operation_id: int) -> int:
    events = run.machine.observation.command_events.events
    ticks = [
        event.tick
        for event in events
        if event.command.operation.id == operation_id and event.kind == kind
    ]
    assert len(ticks) == 1
    return ticks[0]


def _scheduled_workload(
    final_round: int, physical_circuits
) -> workload_settings.WorkloadSettings:
    """Protected from the first round, with no segment the lowering reads.

    The live source is built from the physical circuits the replaced
    section carried (decsim/build/plan.py _syndrome_source).
    """
    owner = program_records.Operation(
        100, "memory", ("stream-patch",), patches=("stream-patch",)
    )
    begin = program_records.Operation(
        1,
        "protect",
        ("stream-patch",),
        patches=("stream-patch",),
        emits_detector_data=False,
    )
    finish = program_records.Operation(
        2,
        "readout",
        ("stream-patch",),
        patches=("stream-patch",),
        predecessors=(1,),
        scheduled_start_round=final_round,
        emits_detector_data=False,
    )
    region = program_records.ProtectedRegion(100, 1, 2)
    policy = round_policies.PerOperationRounds({100: 0, 1: 0, 2: 0})
    return workload_settings.WorkloadSettings(
        operations=(begin, finish),
        dynamic_streams=(owner,),
        protected_regions=(region,),
        rounds_policy=policy,
        physical_circuits=physical_circuits,
    )


def _static_idle_settings(idle_policy: str) -> machine_settings.MachineSettings:
    """Declare the two-patch workload beside the timings it exercises.

    The operation definitions and card remain together so the four idle
    rounds and the absence of any configured weak decoder are visible.
    """
    circuit = memory_programs.memory_circuit(6)
    first = program_records.Operation(
        1, "memory", (0,), patches=(0,), circuit=circuit
    )
    later = program_records.Operation(
        2,
        "memory",
        (1,),
        patches=(1,),
        circuit=circuit,
        scheduled_start_round=4,
    )
    rounds = round_policies.FixedRounds(6)
    workload = workload_settings.WorkloadSettings(
        operations=(first, later), rounds_policy=rounds
    )
    source = stim_device.StimDevice()
    qpu = qpu_settings.QpuSettings(distance=3, device=source)
    clock = config.Clock(4000)
    engine = decoder_settings.EngineSettings(clock=clock)
    strong = decoder_settings.DecoderSettings(kind=0.2, engine=engine)
    escalation = escalation_settings.EscalationSettings(kind="strong_only")
    idle = controller_settings.IdlePolicySettings(kind=idle_policy)
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        strong_decoder=strong,
        escalation=escalation,
        idle_policy=idle,
    )
    observation = dataclasses.replace(
        settings.observation, record_switching_windows=True
    )
    links = _price_path(
        settings.links, "strong_buffer_to_strong_decoder", 350_000
    )
    return dataclasses.replace(settings, observation=observation, links=links)


def _run_static(
    settings: machine_settings.MachineSettings, seed: int = RUN_SEED
) -> tuple[_Run, dict]:
    machine = machine_module.Machine.build(settings, seed)
    weak_writes = []
    machine.readout.weak_syndrome_buffer.trace.round_stored.connect(
        lambda *event: weak_writes.append(event)
    )
    release_ticks_by_round = {}
    released = functools.partial(
        _record_release, machine, release_ticks_by_round
    )
    strong_store = machine.readout.strong_syndrome_buffer
    strong_store.trace.round_released.connect(released)
    decodes = declared_run.FinishedDecodes()
    decodes.attach(machine)
    result = machine.run()
    run = _Run(machine, result, [], [], weak_writes, [], decodes)
    return run, release_ticks_by_round


def _record_release(
    machine: machine_module.Machine, release_ticks_by_round: dict, key: tuple
) -> None:
    release_ticks_by_round[key] = machine.engine.now


def _assert_memory_slot_lifetimes(
    run: _Run,
    arrivals: list[round_records.RoundEvent],
    release_ticks_by_round: dict,
) -> None:
    stored = run.machine.observation.round_events.stored_rounds
    stored_ticks_by_round = {
        (operation, index): tick for tick, operation, index in stored
    }
    for event in arrivals:
        key = (event.operation_id, event.round_index)
        assert event.tick == release_ticks_by_round[key]
        expected_delivery = stored_ticks_by_round[key] + 350_000
        assert event.tick == expected_delivery


def _assert_idle_algorithm_charge(run: _Run, load_job_count: int) -> None:
    stages = run.machine.observation.stages.records
    algorithm = [
        row for row in stages if row.stage == staged_decoder.ALGORITHM_STAGE
    ]
    memory = [row for row in algorithm if row.operation_id not in (1, 2)]
    assert len(memory) == load_job_count
    durations = [row.end_ticks - row.start_ticks for row in memory]
    assert durations == [200_000] * load_job_count


def _assert_stored_bit_count(transfers: list[dict], expected_bits: int) -> None:
    stored_bits = sum(row["payload_bits"] for row in transfers)
    assert stored_bits == expected_bits


def _memory_arrivals(run: _Run) -> list[round_records.RoundEvent]:
    events = run.machine.observation.round_events.events
    return [
        event for event in events if event.kind == "FEEDBACK_MEMORY_DELIVERED"
    ]


def _assert_held_rounds_retried(run: _Run) -> None:
    events = run.machine.observation.round_events.events
    kinds = {event.kind for event in events}
    assert "STALLED" in kinds
    assert "RELEASED" in kinds


def _partitioned_joint_program() -> circuit_records.RepeatedStimCircuit:
    """Keep the first acquisition joint, then separate the two blocks."""
    program = memory_programs.joint_repetition_program(True, 0.04)
    left_checks = round_records.MeasurementPartition(("left",), 2)
    right_checks = round_records.MeasurementPartition(("right",), 2)
    left_data = round_records.MeasurementPartition(("left",), 3)
    right_data = round_records.MeasurementPartition(("right",), 3)
    bulk = (left_checks, right_checks)
    final = (left_checks, right_checks, left_data, right_data)
    partitions = {"repeated_round": bulk, "final_round": final}
    return dataclasses.replace(program, readout_partitions=partitions)


def _readout_route(
    base: link_settings.PathSettings,
    patches: tuple,
    channel_name: str,
    propagation_ticks: int,
) -> link_settings.ReadoutRoute:
    channel = dataclasses.replace(
        base.channel,
        name=channel_name,
        propagation_latency_ticks=propagation_ticks,
    )
    path = dataclasses.replace(base, channel=channel)
    return link_settings.ReadoutRoute(patches, path)


def _joint_settings(
    program: circuit_records.RepeatedStimCircuit, placement: str
) -> machine_settings.MachineSettings:
    base = _settings(program, "live", placement)
    patches = ("left", "right")
    operations = _operations_on_group(base.workload.operations, patches)
    owners = _operations_on_group(base.workload.dynamic_streams, patches)
    workload = dataclasses.replace(
        base.workload, operations=operations, dynamic_streams=owners
    )
    return dataclasses.replace(base, workload=workload)


def _terminal_partitioned_settings(placement: str) -> tuple:
    program = memory_programs.memory_program(physical_error_probability=0.12)
    circuit, _mapping = program.assemble(3)
    sampler = circuit.compile_sampler(seed=73)
    measurements = sampler.sample(shots=1)
    table = detector_formation.build_formation_table(circuit, 3)
    detector_rounds = table.detector_rounds()
    terminal_ids = tuple(
        recipe.detector_index
        for recipe in table.detectors
        if recipe.kind is detector_formation.LayerKind.READOUT
    )
    syndrome_group = round_records.MeasurementPartition(("stream-patch",), 4)
    first_data_group = round_records.MeasurementPartition(("stream-patch",), 3)
    last_data_group = round_records.MeasurementPartition(("stream-patch",), 6)
    source = stim_device.RecordedStimDevice(
        measurements,
        0,
        detector_rounds={100: detector_rounds},
        terminal_detector_ids={100: terminal_ids},
        readout_partitions={
            2: {3: (syndrome_group, syndrome_group)},
            3: {3: (first_data_group, last_data_group)},
        },
    )
    settings = _settings(program, "live", placement)
    qpu = dataclasses.replace(settings.qpu, device=source)
    workload = _terminal_partitioned_workload(circuit)
    settings = dataclasses.replace(settings, qpu=qpu, workload=workload)
    return settings, measurements, circuit


def _terminal_partitioned_workload(
    circuit: stim.Circuit,
) -> workload_settings.WorkloadSettings:
    owner = program_records.Operation(
        100,
        "memory",
        ("stream-patch",),
        patches=("stream-patch",),
        circuit=circuit,
    )
    prefix = dataclasses.replace(
        owner, id=1, name="prefix", stream_id=100, stream_offset=0
    )
    syndrome = dataclasses.replace(
        prefix,
        id=2,
        name="last syndrome",
        stream_offset=2,
        predecessors=(1,),
        syndrome_fragment_index=0,
        syndrome_fragment_count=4,
    )
    finalizer = dataclasses.replace(
        syndrome,
        id=3,
        name="data readout",
        predecessors=(2,),
        finalizes_stream_round=True,
        syndrome_fragment_index=2,
    )
    policy = round_policies.PerOperationRounds({100: 3, 1: 2, 2: 1, 3: 0})
    return workload_settings.WorkloadSettings(
        operations=(prefix, syndrome, finalizer),
        decode_operations=(owner,),
        rounds_policy=policy,
    )


def _operations_on_group(operations: tuple, patches: tuple) -> tuple:
    grouped = []
    for operation in operations:
        replacement = dataclasses.replace(
            operation, qubits=patches, patches=patches
        )
        grouped.append(replacement)
    return tuple(grouped)


def _assert_joint_acquisitions(run: _Run) -> None:
    source = run.machine.qpu.syndrome_source
    identities = [
        (packet.operation_id, packet.round_index) for packet in run.packets
    ]
    assert len(identities) == len(set(identities))
    footprints = {packet.patch_ids for packet in run.packets}
    assert footprints == {("left", "right")}
    measurements = source.sampled_measurements(100)
    emitted = _raw_bits(run.packets)
    assert emitted == measurements
    assert len(run.shots) == 1
    paths = run.result.link_traffic["transfers"]
    acquisitions = [row for row in paths if row["path"] == "qpu_to_controller"]
    assert len(acquisitions) == len(run.packets)


def _bb_memory(basis: str) -> circuit_records.RepeatedStimCircuit:
    import deltakit_explorer.codes as codes

    code = codes.BivariateBicycleCode(3, 5, [1, 1, 4], [0, 1, 2])
    return deltakit.css_memory_rounds(
        code, basis, 0.001, round_period_microseconds=1.1
    )


def _bb_settings(
    program: circuit_records.RepeatedStimCircuit,
    placement: str,
    commit_round_count: int,
) -> machine_settings.MachineSettings:
    settings = _settings(program, "live", placement)
    # Both check matrices have nonzero columns. The public first X logical
    # has weight two and anticommutes with a Z logical, establishing d=2.
    card_settings = code_geometry.BivariateBicycleCodeModel.Settings(
        qubit_count=30, logical_qubit_count=8
    )
    code = code_geometry.BivariateBicycleCodeModel(
        settings=card_settings,
        distance=2,
        commit_rounds_override=commit_round_count,
        buffer_rounds_override=2,
    )
    qpu = dataclasses.replace(settings.qpu, distance=None, code=code)
    price = decoders.PresetLatencyDecoder(0.2)
    backend = belief_propagation_osd.BeliefPropagationOsdDecoder(price)
    decoder = dataclasses.replace(
        settings.strong_decoder, kind=None, decoder=backend
    )
    workload = _bb_logical_qubits(settings.workload)
    return dataclasses.replace(
        settings, qpu=qpu, strong_decoder=decoder, workload=workload
    )


def _bb_logical_qubits(
    workload: workload_settings.WorkloadSettings,
) -> workload_settings.WorkloadSettings:
    logical_qubits = tuple(range(8))
    operations = tuple(
        dataclasses.replace(operation, qubits=logical_qubits)
        for operation in workload.operations
    )
    owners = tuple(
        dataclasses.replace(owner, qubits=logical_qubits)
        for owner in workload.dynamic_streams
    )
    return dataclasses.replace(
        workload, operations=operations, dynamic_streams=owners
    )


def _direct_bp_osd_prediction(run: _Run) -> tuple[int, ...]:
    """Ldpc's published Stim converter retains the undecomposed hyperedges."""
    circuit = run.shots[0][0].circuit
    model = circuit.detector_error_model(approximate_disjoint_errors=False)
    matrices = dem_matrices.detector_error_model_to_check_matrices(
        model, allow_undecomposed_hyperedges=True
    )
    backend = ldpc.BpOsdDecoder(
        matrices.check_matrix,
        error_channel=list(matrices.priors),
        max_iter=2,
        bp_method="product_sum",
        schedule="serial",
        osd_method="osd_cs",
        osd_order=0,
    )
    source = run.machine.qpu.syndrome_source
    events = source.sampled_detection_events(100)
    syndrome = numpy.array(events, dtype=numpy.uint8)
    correction = backend.decode(syndrome)
    prediction = matrices.observables_matrix @ correction
    prediction %= 2
    return tuple(int(bit) for bit in prediction)


def _bb_fault_measurements(
    program: circuit_records.RepeatedStimCircuit,
) -> numpy.ndarray:
    """Apply a public logical X after preparation, with no other faults."""
    import deltakit_explorer.codes as codes

    code = codes.BivariateBicycleCode(3, 5, [1, 1, 4], [0, 1, 2])
    qubits = sorted(code.qubits, key=_bb_qubit_identifier)
    index_by_qubit = {qubit: index for index, qubit in enumerate(qubits)}
    logical = code.x_logical_operators[1]
    targets = [index_by_qubit[pauli.qubit] for pauli in logical]
    first = program.first_round.without_noise()
    repeated = program.repeated_round.without_noise()
    final = program.final_round.without_noise()
    simulator = stim.TableauSimulator(seed=619)
    simulator.do(first)
    simulator.x(*targets)
    simulator.do(repeated)
    simulator.do(final)
    measurements = simulator.current_measurement_record()
    return numpy.array([measurements], dtype=numpy.bool_)


def _bb_qubit_identifier(qubit) -> str:
    return repr(qubit.unique_identifier)


def _direct_matching_prediction(
    circuit: stim.Circuit, source: stim_device.StimDevice
) -> tuple:
    model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)
    events = source.sampled_detection_events(100)
    prediction = matching.decode(events)
    return tuple(prediction)


def _transfer_paths(run: _Run) -> set[str]:
    return {row["path"] for row in run.result.link_traffic["transfers"]}


def _required_sections_but_the_qpu() -> dict:
    """Every required section as an empty mapping, the qpu left out."""
    sections = {}
    for name in machine_settings.REQUIRED_SECTIONS:
        sections[name] = {}
    del sections["qpu"]
    return sections

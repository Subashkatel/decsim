"""The root: components wired by hand as gem5 assigns ports, and by table.

Referent: gem5's learning_gem5/part1/simple.py names each component once
and assigns its ports (system.cpu.icache_port = system.membus...); here
a QPU device and a receiver are wired the same way and the readouts
arrive in cycle order. The table test follows sinter's BUILT_IN_DECODERS
(sinter/_decoding/_decoding_all_built_in_decoders.py): a new decoder is
one class and one row.

The laws at the end of the file are the ones only the whole machine
holds: they run a declared card (tests/declared_run.py, whose every
latency is a number the test states) and read what the composition of
the controller, the stores, the windows and the frame produced.
"""

import dataclasses
import pathlib
import re

import numpy
import pytest

import decsim.build.escalation as escalation_build
import decsim.config as config
import decsim.controller.policies as boundary_policies
import decsim.controller.settings as controller_settings
import decsim.decoders.decoder as decoder_module
import decsim.decoders.decoders as decoders
import decsim.decoders.minimum_weight_perfect_matching.decoder as mwpm
import decsim.decoders.settings as decoder_settings
import decsim.decoders.staged_decoder as staged_decoder
import decsim.decoders.union_find.decoder as union_find_decoder
import decsim.detector_error_model.detector_chronology as detector_chronology
import decsim.engine as engine_module
import decsim.escalation.policies as escalation_policies
import decsim.escalation.settings as escalation_settings
import decsim.front.experiment as experiment
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.observe.settings as observe_settings
import decsim.qpu.cycle_clock as cycle_clock
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.seeds as seed_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.seeding as seeding
import decsim.settings as machine_settings
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings
import decsim.windows.settings as window_settings
import tests.declared_run as declared_run

THIS_FILE = pathlib.Path(__file__)
TESTS_DIRECTORY = THIS_FILE.parents[1]
CONFIGS = TESTS_DIRECTORY.parent / "configs"
CYCLE_TICKS = config.microseconds_to_ticks(1.0)


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
    decoder_settings.DECODERS["fake"] = FakeWeakDecoder
    try:
        machine = machine_module.Machine.build(settings, 0)
        result = machine.run()
    finally:
        del decoder_settings.DECODERS["fake"]
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
    strong_round_store = round_store_settings.RoundStoreSettings(
        kind="off_table"
    )
    settings = machine_settings.MachineSettings(
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
        machine_settings.MachineSettings.from_mapping(
            {"buffers": {}}, name="x", base_directory=None
        )


def test_the_constructor_wiring_reaches_its_components():
    """Every cross-reference is made by constructor, none left None."""
    settings = machine_settings.MachineSettings()
    machine = machine_module.Machine.build(settings)
    assert machine.qpu.receivers.readout is machine.controller
    assert machine.execution_runtime.issuer is machine.issuer
    manager = machine.decoder_manager
    assert machine.window_manager.requester.decode_queue is manager


MEMORY_ROUNDS = 6
MEMORY_CIRCUIT = workload_settings.memory_circuit(
    "surface_code:rotated_memory_z", MEMORY_ROUNDS, 3, 0.003
)


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
    load_only = [r for r in algorithm if r.operation_id not in operation_ids]
    windows = [r for r in algorithm if r.operation_id in operation_ids]
    assert len(load_only) == 1
    assert len(windows) == 2
    idle_ticks = [r.end_ticks - r.start_ticks for r in load_only]
    assert idle_ticks == [0]
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
        gap_threshold_decibels=decibels,
        gap_threshold_nats=nats,
    )
    weak_decoder = decoder_settings.DecoderSettings(
        kind=weak_kind, engine_megahertz=100.0
    )
    strong_decoder = decoder_settings.DecoderSettings(
        kind="pymatching", engine_megahertz=100.0
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


def _weak_service_ticks(machine) -> list:
    """Every decode service's charged ticks, in the order they ended."""
    ticks = []
    for service in machine.observation.decode_records.services:
        ticks.append(service.service_ticks)
    return ticks


def test_a_union_find_weak_tier_reports_the_gap_of_its_own_growth():
    """The cluster gap is a signal row over the decoder that grew it.

    One decode per window, its confidence read off the growth that
    decode returned (Meister et al. 2405.07433 Algorithm 2 lines
    518-536), and the unit charged that decode. The matching run beside
    it is the comparison the charge is read against: Union-Find's own
    decode is this implementation's, not sparse blossom's.
    """
    settings = _switching_memory("union_find", "cluster_gap")
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    assert result.terminal_status == "complete"
    assert result.operation_results[0].logical_observables == (0,)
    requests = machine.observation.decode_records.requests
    sources = set()
    for record in requests:
        if record.soft_output is not None:
            sources.add(record.soft_output.source.method)
    assert sources == {"cluster_gap"}
    window_count = len(machine.observation.decode_records.services)
    assert len(requests) == window_count
    matching_settings = _switching_memory("pymatching", "complementary_gap")
    matching = machine_module.Machine.build(matching_settings, 0)
    matching.run()
    union_find_services = _weak_service_ticks(machine)
    matching_services = _weak_service_ticks(matching)
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
    charges one walk per window on the weak unit, and each weak service
    ends after the decode and the walk it fed.
    """
    settings = _switching_memory("union_find", "cluster_gap")
    machine = machine_module.Machine.build(settings, 0)
    machine.run()
    charged = _confidence_charges(machine)
    services = machine.observation.decode_records.services
    assert len(charged) == len(services)
    for ticks in charged:
        assert ticks > 0
    named = []
    for line in machine.observation.log.lines:
        if "CONFIDENCE" in line:
            named.append(line)
    # the weak tier of this card is the default pool's one unit
    assert "on unit default#0" in named[0]
    service_ticks = _weak_service_ticks(machine)
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
            kind="pymatching", engine_megahertz=100.0
        ),
        "strong": decoder_settings.DecoderSettings(
            kind="belief_matching", engine_megahertz=100.0
        ),
    }
    tiers[tier] = decoder_settings.DecoderSettings(
        kind="union_find_cluster_gap", engine_megahertz=100.0
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
    round_store_module.ROUND_STORES["counting"] = CountingRoundStore
    try:
        machine = machine_module.Machine.build(settings, 0)
        result = machine.run()
    finally:
        del round_store_module.ROUND_STORES["counting"]
    assert result.terminal_status == "complete"
    assert type(machine.round_store) is CountingRoundStore
    fired = [
        line for line in machine.observation.log.lines if "fires round" in line
    ]
    assert machine.round_store.stored_count == len(fired)
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


def test_a_new_escalation_kind_is_one_class_and_one_table_row():
    """The row declares its tier, so no second table names the kind."""
    config_path = CONFIGS / "strong_decoder_baseline.yaml"
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.003, distance=3, round_period_us=1.0
    )
    escalation = dataclasses.replace(settings.escalation, kind="always_strong")
    settings = dataclasses.replace(settings, escalation=escalation)
    escalation_settings.ESCALATIONS["always_strong"] = AlwaysStrongEscalation
    try:
        assert escalation_build.primary_tier(escalation) == "strong"
        machine = machine_module.Machine.build(settings, 0)
        result = machine.run()
    finally:
        del escalation_settings.ESCALATIONS["always_strong"]
    assert result.terminal_status == "complete"
    snapshot = machine.pauli_frame.snapshot()
    tiers = []
    for record in snapshot.records:
        tiers.append(record.tier)
    assert tiers
    assert set(tiers) == {"strong"}


class AlwaysReadyFactory:
    """A factory row written outside decsim: the collaborators alone.

    Its constructor is InfiniteFactory's own shape, one parameter, which
    is the shape the root refused before every row was built from one
    collaborators record.
    """

    def __init__(self, collaborators):
        self.engine = collaborators.engine
        self.requests = []

    def request(self, operation_id, callback):
        """Deliver at once and remember who asked."""
        self.requests.append(operation_id)
        callback()
        return magic_state_factories.Ticket(operation_id, (), self)

    def cancel(self, ticket):
        """Nothing is ever pending."""
        del ticket
        return False

    def shutdown(self):
        """Nothing runs, so nothing stops."""


def test_a_factory_row_written_outside_decsim_builds_by_its_own_name():
    """One constructor call, so a row that reads only the engine builds."""
    config_path = CONFIGS / "weak_decoder_baseline.yaml"
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.003, distance=3, round_period_us=1.0
    )
    factory_settings = qpu_settings.FactorySettings(kind="always_ready")
    settings = dataclasses.replace(
        settings, magic_state_factory=factory_settings
    )
    qpu_settings.MAGIC_STATE_FACTORIES["always_ready"] = AlwaysReadyFactory
    try:
        machine = machine_module.Machine.build(settings, 0)
        result = machine.run()
    finally:
        del qpu_settings.MAGIC_STATE_FACTORIES["always_ready"]
    assert isinstance(machine.factory, AlwaysReadyFactory)
    assert result.terminal_status == "complete"


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
    boundary_settings = machine_settings.MachineSettings(windows=windows)
    with_boundary = machine_module.Machine.build(boundary_settings)
    idle_policy = RecordingIdlePolicy()
    idle_row = controller_settings.IdlePolicySettings(policy=idle_policy)
    idle_settings = machine_settings.MachineSettings(idle_policy=idle_row)
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

    def buffering_floor(self):
        return (0, 0)

    window_floor_justification = None

    def spatial_nodes(self, num_patches):
        return 4 * num_patches

    def syndrome_bits_per_round(self, num_patches):
        return 3 * num_patches


def _resolved_geometry(machine):
    """The geometry the run resolved, read off the issuer's table."""
    resolved = machine.issuer.resolved_operation_by_id.values()
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

PACKING_BOUND_STOP_TICKS = {
    1: 7_000_000,
    2: 8_000_000,
    3: 9_000_000,
    4: 10_000_000,
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


def test_a_full_round_store_stalls_the_controller_instead_of_dropping():
    """A round with no slot waits upstream and is published in order.

    A real-time decoder backpressures its source rather than discarding
    syndromes: the Rigetti sequencer polls the decoder's status register
    and stalls (Caune et al. 2410.05202), and QubiC's cores block in
    WAIT_MEAS until the readout is consumed (Fruitwala et al.
    2404.15260); that is the STALL policy of
    decsim/controller/settings.py. While the store has room round r is
    published at r plus qpu_to_controller 2 plus readout_to_bits 3 plus
    controller_to_weak_buffer 4, so rounds 1 to 7 fill a store of
    seven at 16 us and round 8 waits past 17 us for the first window's
    input to land instead of being dropped.
    """
    seven_rounds = round_store_settings.RoundStoreSettings(rounds=7)
    machine = declared_run.weak_only_run(rounds=12, round_store=seven_rounds)
    published = published_rounds(machine)
    ticks = publication_ticks(published)
    publication_tick_by_round = dict(published)
    round_indices = list(publication_tick_by_round)
    seventh_publication = config.microseconds_to_ticks(16.0)
    eighth_publication_with_room = config.microseconds_to_ticks(17.0)
    tiers = declared_run.frame_tiers(machine)

    assert machine.observation.round_events.packing_drops == 0
    assert round_indices == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert ticks == sorted(ticks)
    assert publication_tick_by_round[7] == seventh_publication
    assert publication_tick_by_round[8] > eighth_publication_with_room
    assert tiers == [((1, 0), "weak"), ((1, 1), "weak"), ((1, 2), "weak")]


def test_both_stores_settle_empty_at_the_end_of_an_escalating_run():
    """The last commit releases every hold either store placed.

    A store keeps a round while any holder still needs it and frees it
    on the last release (decsim/syndrome_buffer/round_store.py), so a
    round left behind is a leak that grows over a sweep of shots
    without ever failing a run. Escalating every window places the
    longest-lived holds the machine has: the room-side context of a
    window is held across its whole strong decode.
    """
    machine = declared_run.switching_run(escalation_probability=1.0, rounds=9)

    assert machine.round_store.occupancy == 0
    assert machine.strong_round_store.occupancy == 0
    assert machine.strong_round_writer.writes_in_flight == 0


def test_the_execution_and_the_decoding_views_agree_on_the_workload():
    """One resolved workload reaches the sequencer and the windows.

    The root resolves the workload once and hands the same plan to the
    sequencer, which issues the operations, and to the window tracker,
    which accounts for their rounds (decsim/machine.py, _plan). An id
    in one view and not the other, or a planned round count the
    arrivals never meet, leaves rounds with no readiness account.
    """
    machine = declared_run.weak_only_run(rounds=6)
    sequencer = machine.execution_runtime
    tracker = machine.window_manager.tracker
    planner = machine.window_manager.planner
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
    tracker = machine.window_manager.tracker
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
def test_the_packing_bound_stops_the_run_at_the_tick_it_fills(bound):
    """A round counts against the bound until the windows hear of it.

    controller.packing_rounds_in_flight bounds the whole packing stage,
    not the assembly step alone: a round counts from its first fragment
    until it is published (decsim/controller/settings.py). On the
    declared card round r reaches assembly at r plus qpu_to_controller 2
    plus readout_to_bits 3 and is published four microseconds later, so
    under a bound of b round 1 + b arrives at 6 + b us while round 1 is
    still on its route, and the run stops there naming the setting.
    """
    settings = twelve_rounds_with_packing_bound(bound)
    machine = machine_module.Machine.build(settings, 0)
    stop_tick = PACKING_BOUND_STOP_TICKS[bound]
    sentence = f"the packing workspace is full at tick {stop_tick}: "

    with pytest.raises(RuntimeError, match=sentence) as refusal:
        machine.run()

    named_setting = f"controller.packing_rounds_in_flight is {bound}"
    assert named_setting in str(refusal.value)
    assert machine.engine.now == stop_tick


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
    assert machine.observation.round_events.packing_drops == 0
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
    settings = config.point_settings(
        physical_error_probability=0.001, distance=3, round_period_us=1.0
    )
    escalation = dataclasses.replace(settings.escalation, kind=escalation_kind)
    settings = dataclasses.replace(settings, escalation=escalation)
    machine = machine_module.Machine.build(settings, 0)
    machine.run()
    return machine


def test_the_controller_writes_every_round_into_buffer_0_over_a_priced_hop():
    """Fifteen rounds are fifteen transfers carrying the round's own bits.

    Toshio et al. 2510.25222 price the syndrome data of each round
    between the system controller and the weak decoder as T_comm^weak
    (Table I), so the write into syndrome buffer 0 is a transfer like
    every other hop rather than a free store: reference.yaml at d = 3
    puts its fifteen rounds and their 129 measured bits on
    controller_to_weak_buffer, the same bits they put on
    qpu_to_controller.
    """
    machine = reference_run("weak_baseline")
    store_hop = transfer_records.LinkPath.CONTROLLER_TO_WEAK_BUFFER
    readout_hop = transfer_records.LinkPath.QPU_TO_CONTROLLER
    room_hop = transfer_records.LinkPath.CONTROLLER_TO_STRONG_BUFFER

    assert link_totals(machine, store_hop) == (15, 129)
    assert link_totals(machine, readout_hop) == (15, 129)
    assert link_totals(machine, room_hop) == (0, 0)


def test_a_strong_only_run_writes_every_round_into_buffer_1_over_a_priced_hop():
    """The same law on the room-side store, the only one it fills.

    T_comm^strong is Toshio's symbol for the same transport to the
    strong decoder, and a strong-primary plan reads its windows from
    syndrome buffer 1, so every round takes controller_to_strong_buffer
    once and syndrome buffer 0 sees none of them.
    """
    machine = reference_run("strong_only")
    store_hop = transfer_records.LinkPath.CONTROLLER_TO_WEAK_BUFFER
    readout_hop = transfer_records.LinkPath.QPU_TO_CONTROLLER
    room_hop = transfer_records.LinkPath.CONTROLLER_TO_STRONG_BUFFER

    assert link_totals(machine, room_hop) == (15, 129)
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
    policy = escalation_policies.StrongOnly()
    by_object = escalation_settings.EscalationSettings(policy=policy)
    named_settings = strong_primary_settings(named)
    named_machine = machine_module.Machine.build(named_settings, 0)
    named_result = named_machine.run()
    object_settings = strong_primary_settings(by_object)
    object_machine = machine_module.Machine.build(object_settings, 0)
    object_result = object_machine.run()
    assert object_machine.active_decoder is not None
    assert type(object_machine.active_decoder) is type(
        named_machine.active_decoder
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
    policy = escalation_policies.StrongOnly()
    escalation = escalation_settings.EscalationSettings(policy=policy)
    settings = strong_primary_settings(escalation)
    no_decoder = decoder_settings.DecoderSettings()
    settings = dataclasses.replace(settings, strong_decoder=no_decoder)
    with pytest.raises(ValueError) as refusal:
        machine_module.Machine.build(settings, 0)
    assert "decodes windows on the strong tier" in str(refusal.value)

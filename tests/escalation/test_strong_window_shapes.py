"""The strong-window shapes plan the region the paper gives.

The redo window is the commit region and one buffer ahead, its
past face pinned on the earlier neighbour's committed correction
(Bombin 2303.04846 lines 775-788, 1456-1458). Toshio et al. 2510.25222:
the double window starts at the
escalated commit, absorbs the windows it covers, and is decoded once
both of its boundaries are weak-determined: the restart window's commit, or the
terminal data (Sec. III C, Fig. 12). A d=3 sliding window commits 3
rounds and buffers 3, so r_strong = r_com + 2 r_buf is 9 rounds
(Sec. III C, lines 1250-1251).

The restart window's weak decode re-reads
escalation.restart_reread_buffer_regions buffer regions of the strong
region from the weak syndrome buffer (Sec. III C: the weak decoder resumes past
the strong region once its rounds are stored), so at width 1 the last
absorbed window's commit rounds must still be stored when the plan
lands, in a backlog regime where the absorbed windows' inputs are in
flight or have already landed in a unit. At width 0 the restart begins
on the round after the strong region. The gate's switching card, priced
on both tiers so the run is deterministic, puts a run in that regime.
"""

import dataclasses

import pytest

import decsim.confidence.complementary as complementary
import decsim.config as config
import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.links.link_profiles as link_profiles
import decsim.machine as machine_module
import decsim.observe.settings as observe_settings
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.ports as ports
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.records.decoding as decoding_records
import decsim.records.identity as identity_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.trace_source as trace_source
import decsim.windows.decode_requests as decode_requests
import decsim.windows.schemes.sliding as sliding_scheme
import decsim.windows.settings as window_settings
import decsim.windows.window_boundaries as window_boundaries
import tests.declared_run as declared_run
import tests.escalation.declared_fabric as fabric
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

# The gate's switching card: both clocks at 250 MHz, each hop one cycle
# of its side's clock on an unbounded wire.
FRIDGE_CLOCK = config.Clock.from_megahertz(250.0)
ROOM_CLOCK = config.Clock.from_megahertz(250.0)
GATE_PATH_CLOCKS = {
    "qpu_to_controller": FRIDGE_CLOCK,
    "controller_to_weak_buffer": FRIDGE_CLOCK,
    "controller_to_strong_buffer": ROOM_CLOCK,
    "weak_buffer_to_weak_decoder": FRIDGE_CLOCK,
    "weak_decoder_to_strong_decoder": ROOM_CLOCK,
    "strong_buffer_to_strong_decoder": ROOM_CLOCK,
    "decoder_to_decoder": FRIDGE_CLOCK,
    "weak_decoder_to_frame": FRIDGE_CLOCK,
    "strong_decoder_to_frame": ROOM_CLOCK,
    "strong_decoder_to_weak_decoder": ROOM_CLOCK,
}
GATE_PHYSICAL_ERROR_PROBABILITY = 0.008


def one_cycle_path(links, path_name: str, clock, latency_cycles: int = 1):
    """One path of the card on an unbounded wire of one lane."""
    return link_profiles.path_card(
        links,
        path_name,
        clock=clock,
        latency_cycles=latency_cycles,
        bits_per_cycle=None,
        source="the gate's switching card",
        lane_count=1,
        setup_cycles_per_transfer=0,
        header_bits_per_transfer=0,
        protocol=None,
    )


def gate_switching(
    distance: int = 3,
    rounds_per_shot=None,
    strong_window=declared_run.REDO_WINDOW,
    physical_error_probability: float = GATE_PHYSICAL_ERROR_PROBABILITY,
) -> machine_settings.MachineSettings:
    """The gate's switching card, at p 0.008 unless given, 1 us rounds.

    PyMatching on the chip keeps a window whose complementary gap is at
    least 20 dB and escalates any other to belief matching on the host,
    both charged their measured wall clock. A shot is 10 d rounds unless
    rounds_per_shot says otherwise.
    """
    if rounds_per_shot is None:
        rounds_per_shot = 10 * distance
    stim = stim_device.StimDevice.Settings()
    qpu = qpu_settings.QpuSettings(
        source=stim, round_period_microseconds=1.0, distance=distance
    )
    links = gate_links()
    plain_windows = window_settings.WindowSettings()
    windows = window_settings.switching_windows(plain_windows, strong_window)
    weak_engine = decoder_settings.EngineSettings(
        clock=FRIDGE_CLOCK, release_cycles_per_job=10
    )
    pymatching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings()
    weak_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=pymatching, engine=weak_engine
    )
    pauli_frame = pauli_frame_module.PauliFrameConfig(
        write_cycles=1, clock=FRIDGE_CLOCK
    )
    workload = machine_settings.memory_workload(
        distance, physical_error_probability, rounds_per_shot
    )
    observation = observe_settings.ObservationSettings(log_component_io=True)
    strong_decoder = host_belief_matching()
    switching = switching_slot(strong_window)
    return machine_settings.MachineSettings(
        clock=FRIDGE_CLOCK,
        qpu=qpu,
        links=links,
        windows=windows,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        switching=switching,
        pauli_frame=pauli_frame,
        workload=workload,
        observation=observation,
    )


def weak_base_switching(
    distance: int,
    physical_error_probability: float,
    round_period_microseconds: float,
) -> machine_settings.MachineSettings:
    """The weak base switching on the gate's slot, over one-cycle hops.

    The switching experiment's redo window task: real PyMatching on the
    weak base's chip unit escalates to belief matching on the host, the
    strong side's five hops one room cycle each.
    """
    base = machine_settings.weak_decoder_baseline(
        distance, physical_error_probability, round_period_microseconds
    )
    pymatching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings()
    weak_decoder = dataclasses.replace(base.weak_decoder, algorithm=pymatching)
    links = machine_settings.one_cycle_strong_side(base.links)
    strong_window = declared_run.REDO_WINDOW
    windows = window_settings.switching_windows(base.windows, strong_window)
    strong_decoder = host_belief_matching()
    switching = switching_slot(strong_window)
    return dataclasses.replace(
        base,
        links=links,
        windows=windows,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        switching=switching,
    )


def host_belief_matching() -> decoder_settings.DecoderPoolSettings:
    """Belief matching on one host unit, charged its measured wall clock."""
    engine = decoder_settings.EngineSettings(
        clock=ROOM_CLOCK, release_cycles_per_job=10
    )
    belief = belief_matching.BeliefMatchingDecoder.Settings(
        max_iterations=30, belief_propagation_method="product_sum"
    )
    return decoder_settings.DecoderPoolSettings(algorithm=belief, engine=engine)


def switching_slot(strong_window) -> escalation_settings.SwitchingSettings:
    """Keep a complementary gap of 20 dB or more, escalate any other."""
    confidence = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=20.0
    )
    return escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold, strong_window=strong_window
    )


def gate_links():
    """The reference card with each of the gate's hops one cycle."""
    reference = link_profiles.logical_reference_profile()
    cards = {}
    for path_name, clock in GATE_PATH_CLOCKS.items():
        cards[path_name] = one_cycle_path(reference, path_name, clock)
    return dataclasses.replace(reference, **cards)


def priced_pool(pool, microseconds: float, unit_count: int = 1):
    """The pool on PyMatching at a fixed latency, on unit_count units."""
    card = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=microseconds
    )
    return dataclasses.replace(pool, algorithm=card, unit_count=unit_count)


def _heard_requests(machine) -> declared_run.EndedRequests:
    """A probe on the machine's decode requests, attached before it runs."""
    requests = declared_run.EndedRequests()
    requests.attach(machine)
    return requests


def _strong_job(requests: declared_run.EndedRequests, window_id: int):
    """The strong decode job one window's escalation sent, as it ended."""
    strong = requests.of_tier(window_records.DecoderTier.STRONG)
    for ended in strong:
        if ended.job.request_key.window_id == window_id:
            return ended.job
    raise AssertionError(f"no strong request for window {window_id}")


def test_the_double_window_leaves_once_its_far_boundary_is_determined():
    machine = fabric.switching_machine(
        rounds=15,
        escalated_windows={1},
        strong_window=declared_run.DOUBLE_WINDOW,
        round_microseconds=4.0,
    )
    machine.run()
    carried_lines = fabric.log_lines_containing(
        machine, "rounds 4-12 of op 1 carried up with the escalation"
    )
    submitted_lines = fabric.log_lines_containing(
        machine, "far-side weak boundary determined -> strong window submitted"
    )
    restart_lines = fabric.log_lines_containing(machine, "DECODE DONE mem1 W4")
    # the restart window W4 decodes from 74 to 84 us and is determined
    # then; its correction reaches the frame 2 us of weak_decoder_to_frame
    # and 1 us of frame write later, which the strong region does not wait
    # for, and the region lands 3 us of weak_decoder_to_strong_decoder
    # after it left
    assert len(carried_lines) == 1
    assert carried_lines[0].startswith("[ 84.000 us]")
    assert len(submitted_lines) == 1
    assert submitted_lines[0].startswith("[ 87.000 us]")
    assert restart_lines[0].startswith("[ 87.000 us]")
    assert not machine.windows.window_manager.strong_redecode.has_pending()


def test_a_region_at_a_back_to_back_seam_waits_for_its_own_weak_commit():
    """Its near boundary condition is the escalated window's own commit.

    A held region names every commit it waits on, and at a back-to-back
    seam the near face's commit is the escalated window's own (Toshio et
    al. 2510.25222 lines 1248-1250), which has not happened when the
    region is planned.
    """
    machine = fabric.switching_machine(
        rounds=15,
        escalated_windows={1, 4},
        strong_window=declared_run.DOUBLE_WINDOW,
        round_microseconds=4.0,
    )
    waits = _recorded_waits(machine)
    machine.run()
    own_commit = "commit of window (1, 4)"
    assert own_commit in waits[(1, 4)]


def _recorded_waits(machine) -> dict:
    """What each held strong window said it waits for."""
    waits = {}

    def held(_request_key, window_key, waits_for, _rounds) -> None:
        waits[window_key] = waits_for

    redecode = machine.windows.window_manager.strong_redecode
    redecode.trace.strong_window_held.connect(held)
    return waits


@pytest.mark.parametrize("commit_rounds, buffer_rounds", [(3, 4), (4, 3)])
def test_a_double_window_crossing_a_later_commit_region_stops_the_run(
    commit_rounds, buffer_rounds
):
    """The region is commit plus two buffers from the escalated commit.

    When twice the buffer is not a multiple of the commit, the region
    ends inside a later window's commit region, and that window commits
    across the region's end with no owner.
    """
    scheme = sliding_scheme.SlidingWindowScheme.Settings(
        commit_rounds=commit_rounds, buffer_rounds=buffer_rounds
    )
    machine = fabric.switching_machine(
        rounds=24,
        escalated_windows={1},
        strong_window=declared_run.DOUBLE_WINDOW,
        scheme=scheme,
    )

    with pytest.raises(RuntimeError, match="across the strong-region edge"):
        machine.run()


# ---- the restart window's the weak syndrome buffer claim under a backlog


def _gate_double_window_machine(
    commit_rounds: int,
    buffer_rounds: int,
    weak_microseconds: float,
    strong_microseconds: float,
    weak_units: int,
    reread_buffer_regions: int,
) -> machine_module.Machine:
    """The gate's switching card with a double window, both tiers priced.

    The gate's card at p 0.008, d 3, 1 us rounds, 30 rounds, seed 1,
    a shot whose first escalation is W3. The weak tier at 40 us per
    window against 3 us of rounds is the backlog regime: every later
    window is requested, and staged as units free, long before the
    escalated window's verdict.
    """
    strong_window = strong_window_shapes.DoubleWindow.Settings(
        restart_reread_buffer_regions=reread_buffer_regions
    )
    settings = gate_switching(strong_window=strong_window)
    scheme = sliding_scheme.SlidingWindowScheme.Settings(
        commit_rounds=commit_rounds, buffer_rounds=buffer_rounds
    )
    windows = dataclasses.replace(settings.windows, scheme=scheme)
    weak_decoder = priced_pool(
        settings.weak_decoder, weak_microseconds, weak_units
    )
    strong_decoder = priced_pool(settings.strong_decoder, strong_microseconds)
    settings = dataclasses.replace(
        settings,
        windows=windows,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
    )
    return machine_module.Machine.build(settings, 1)


def _run_statuses(result) -> list:
    statuses = []
    for row in result.operation_results:
        statuses.append((row.operation_id, row.result_status))
    return statuses


def _log_index(machine, needle: str) -> int:
    lines = fabric.log_lines_containing(machine, needle)
    assert lines, needle
    return machine.observation.log.lines.index(lines[0])


def _claim(machine, window_index: int):
    store = machine.windows.window_manager.retention.weak_store
    claim = decoding_records.PotentialRestart((1, window_index))
    if not store.has_hold(claim):
        return None
    return store.hold_round_identities(claim)


def test_a_double_window_plan_claims_the_rounds_a_restart_would_read():
    """The plan's claims at the default re-read width, which is 1.

    A bounded window claims exactly the rounds its restart would read:
    its own reads and the buffer region behind its commit, the last
    block of the strong region before it (Toshio 2510.25222 Fig. 12
    step 5). The first window, and every window of an ordinary run,
    claims nothing.
    """
    forward = fabric.switching_machine(
        rounds=15,
        escalated_windows=set(),
        strong_window=declared_run.DOUBLE_WINDOW,
    )
    # W1 commits 4-6 and reads to 9; the re-read adds 1-3
    assert _claim(forward, 1) == tuple((1, index) for index in range(1, 10))
    # W4 commits 13-15, reads to 18 clipped at the operation's end, and
    # re-reads 10-12
    assert _claim(forward, 4) == tuple((1, index) for index in range(10, 16))
    assert _claim(forward, 0) is None
    ordinary = fabric.switching_machine(rounds=15, escalated_windows=set())
    ordinary_claims = [
        _claim(ordinary, window_index) for window_index in range(5)
    ]
    assert ordinary_claims == [None, None, None, None, None]


def test_the_restart_window_keeps_its_re_read_rounds_across_the_withdrawals():
    """Commit 3, buffer 3, weak 40 us.

    W3 escalates at 166 us with W4's input landed and W5's in flight;
    the strong window 10-18 absorbs W4 and W5, and the restart W6
    re-reads 16-18, W5's commit rounds, whose last the weak syndrome buffer
    holder was W5's request. W6's own claim carries the rounds across
    W5's withdrawal, and W6's stale request is withdrawn and rebuilt.
    """
    machine = _gate_double_window_machine(3, 3, 40.0, 5.0, 1, 1)
    result = machine.run()
    assert _run_statuses(result) == [(1, "logical_observables")]
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 1), "weak"),
        ((1, 2), "weak"),
        ((1, 3), "strong"),
        ((1, 6), "strong"),
        ((1, 9), "strong"),
    ]
    resliced = fabric.log_lines_containing(machine, "re-sliced")
    assert (
        "restart window (1, 6) re-sliced across strong window edge 18 "
        "(reads rounds 16-24; crossing faults owned by restart_window)"
    ) in resliced[0]
    withdrawn_restart = _log_index(machine, "WITHDRAW memory W6")
    assert withdrawn_restart < machine.observation.log.lines.index(resliced[0])
    assert not machine.windows.window_manager.strong_redecode.has_pending()


def test_the_re_read_rounds_survive_with_commit_four_and_buffer_four():
    """Commit 4, buffer 4: W2 escalates, W5 re-reads 17-20."""
    machine = _gate_double_window_machine(4, 4, 40.0, 5.0, 1, 1)
    result = machine.run()
    assert _run_statuses(result) == [(1, "logical_observables")]
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 1), "weak"),
        ((1, 2), "strong"),
        ((1, 5), "strong"),
    ]
    resliced = fabric.log_lines_containing(machine, "re-sliced")
    assert (
        "restart window (1, 5) re-sliced across strong window edge 20 "
        "(reads rounds 17-28; crossing faults owned by restart_window)"
    ) in resliced[0]


def test_the_re_read_rounds_survive_the_absorbed_inputs_landing_first():
    """Four weak units: the absorbed W5 and the restart W6 land first.

    Both inputs are in unit memory before W3's verdict. A landed
    input's the weak syndrome buffer request hold ends at the landing, so with
    one holder the re-read rounds 16-18 and W6's own 19-21 would be gone
    before the plan runs; W6's claim keeps them past both landings.
    Four units hold two windows at once, since a window's confidence
    is two forced-class solves and each takes a unit.
    """
    machine = _gate_double_window_machine(3, 3, 40.0, 5.0, 4, 1)
    result = machine.run()
    landed_absorbed = _log_index(
        machine, "memory W5 [commit 16-18] input landed"
    )
    landed_restart = _log_index(
        machine, "memory W6 [commit 19-21] input landed"
    )
    escalated = _log_index(machine, "WITHDRAW memory W4")
    assert landed_absorbed < escalated
    assert landed_restart < escalated
    assert _run_statuses(result) == [(1, "logical_observables")]
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 1), "weak"),
        ((1, 2), "weak"),
        ((1, 3), "strong"),
        ((1, 6), "strong"),
        ((1, 9), "strong"),
    ]


def test_width_zero_restarts_on_the_round_after_the_strong_region():
    """Re-read width 0 with two weak units, both restart inputs landed.

    The absorbed W5 and the restart W6 land in unit memory before W3's
    verdict. With no re-read W6 begins at round 19, the round after the
    strong region, and owns the faults of the rounds it reads (Toshio
    2510.25222 Sec. III C); the run completes, and commits the
    same tiers, as it does with one buffer region of re-read.
    """
    machine = _gate_double_window_machine(3, 3, 40.0, 5.0, 2, 0)
    result = machine.run()
    assert _run_statuses(result) == [(1, "logical_observables")]
    resliced = fabric.log_lines_containing(machine, "re-sliced")
    assert (
        "restart window (1, 6) re-sliced across strong window edge 18 "
        "(reads rounds 19-24; crossing faults owned by restart_window)"
    ) in resliced[0]
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 1), "weak"),
        ((1, 2), "weak"),
        ((1, 3), "strong"),
        ((1, 6), "strong"),
        ((1, 9), "strong"),
    ]


def test_a_re_read_width_with_no_referent_is_refused():
    """Toshio 2510.25222 lines 1229-1235 allow 0 and 1; 2 would run on."""
    with pytest.raises(ValueError, match="restart_reread_buffer_regions"):
        strong_window_shapes.DoubleWindow.Settings(
            restart_reread_buffer_regions=2
        )


def test_the_double_window_lands_in_the_declared_backlog_regime():
    """1 us rounds against a 10 us weak decode: W1 escalates with W2 landed.

    The regime is a declared one, so the run completes; it is not
    refused.
    """
    machine = fabric.switching_machine(
        rounds=15,
        escalated_windows={1},
        strong_window=declared_run.DOUBLE_WINDOW,
    )
    machine.run()
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 4), "weak"),
        ((1, 1), "strong"),
    ]
    resliced = fabric.log_lines_containing(machine, "re-sliced")
    assert (
        "restart window (1, 4) re-sliced across strong window edge 12"
        in (resliced[0])
    )
    assert not machine.windows.window_manager.strong_redecode.has_pending()


# ---- a shape row added from outside decsim


class RecordingRedoWindow(strong_window_shapes.RedoWindow):
    """A shape row a study adds: the redo window, its plans noted.

    It fills the StrongWindowShape port on the redo window's own
    layout and ports, which is what a new row does: one class and its
    record, and nothing else changes.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings(strong_window_shapes.RedoWindow.Settings):
        """The redo window's record, building this row."""

        def build(self, engine) -> "RecordingRedoWindow":
            return RecordingRedoWindow(engine)

    def __init__(self, engine) -> None:
        self.engine = engine
        self.planned_windows = []
        self.assignments = []

    def plan(self, weak_job):
        """Note the window, then plan it as the redo window does."""
        self.planned_windows.append(weak_job.window_id)
        redo_window = strong_window_shapes.RedoWindow
        assignment = redo_window.plan(self, weak_job)
        self.assignments.append(assignment)
        return assignment


def test_a_shape_row_added_from_outside_runs_by_its_record():
    """A new strong window shape is one class and its Settings record.

    The machine builds the record's row with the components the port
    needs, as it builds a shipped row.
    """
    recording_row = RecordingRedoWindow.Settings()
    machine = fabric.switching_machine(
        rounds=9, escalated_windows={2}, strong_window=recording_row
    )
    machine.run()
    shape = machine.windows.window_manager.strong_redecode.shape
    assert type(shape) is RecordingRedoWindow
    assert shape.planned_windows == [2]
    # the row pins its past face on W1, the neighbour that committed
    # the round before W2's commit region
    assert shape.assignments[0].folded_boundaries == ((1, 1),)
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 1), "weak"),
        ((1, 2), "strong"),
    ]


class RecordingDoubleWindow(strong_window_shapes.DoubleWindow):
    """A shape row a study adds that absorbs the windows it covers.

    It takes the same one constructor and the same ports the shipped
    rows take, although its layout reads the planner, the requester and
    the ledger that the redo window never touches.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings(strong_window_shapes.DoubleWindow.Settings):
        """The double window's record, building this row."""

        def build(self, engine) -> "RecordingDoubleWindow":
            return RecordingDoubleWindow(engine)

    def __init__(self, engine) -> None:
        self.engine = engine
        self.window_absorbed = trace_source.TraceSource()
        self.planned_windows = []

    def plan(self, weak_job):
        """Note the window, then plan it as the double window does."""
        self.planned_windows.append(weak_job.window_id)
        forward = strong_window_shapes.DoubleWindow
        return forward.plan(self, weak_job)


def test_an_absorbing_row_added_from_outside_builds_through_the_same_call():
    """One constructor signature, whatever the row's geometry.

    The shipped rows read different components: the redo window
    reads the regions, the retention, the builder and the courier; the
    double window
    also re-slices on the planner, withdraws on the requester and
    rewrites the ledger. Both take the engine alone and the same ports,
    so the root builds a row without branching on its geometry.
    """
    recording_row = RecordingDoubleWindow.Settings()
    machine = fabric.switching_machine(
        rounds=9, escalated_windows={0}, strong_window=recording_row
    )
    machine.run()
    shape = machine.windows.window_manager.strong_redecode.shape
    assert type(shape) is RecordingDoubleWindow
    assert shape.planned_windows == [0]
    assert fabric.frame_tiers(machine) == [((1, 0), "strong")]


def test_the_shipped_components_fill_the_shapes_six_window_side_ports():
    """A row written outside decsim programs against the ports, not classes.

    StrongWindowPorts types its six window-side ports as Protocols in
    decsim/ports.py, so the promise only means something if the classes
    the root binds there answer the whole port.
    """
    machine = fabric.switching_machine(rounds=9, escalated_windows=set())
    shape = machine.windows.window_manager.strong_redecode.shape
    assert isinstance(shape.planner, ports.WindowPlan)
    assert isinstance(shape.retention, ports.WindowRetention)
    assert isinstance(shape.builder, ports.WindowJobBuilder)
    assert isinstance(shape.requester, ports.WindowRequests)
    assert isinstance(shape.ledger, ports.LogicalLedger)
    assert isinstance(shape.courier, ports.BoundaryCourier)


def test_a_courier_that_only_pins_a_face_fills_the_courier_port():
    """The port lists the one method the escalation side calls.

    strong_job_payloads calls pin_strong_face and nothing else
    (strong_window_shapes.py, the walk over folded_boundaries), so a
    courier supplied from outside decsim answers the whole port with
    that one method. STYLE.md rule 7: a port carries the methods one
    component needs from another.
    """
    courier = _RecordingCourier()
    assert isinstance(courier, ports.BoundaryCourier)
    assert not hasattr(courier, "committed")


def test_a_pinned_strong_decode_starts_no_earlier_than_its_pin_lands():
    """Toshio 2510.25222 lines 1248-1250, priced on the wire.

    The strong decoder starts after the boundary conditions have been
    determined, and Bombin 2303.04846 lines 782-788 make the neighbour's
    correction part of the input task j reads, so a decode whose face is
    pinned may not begin before the message that carries the pin has
    crossed decoder_to_decoder. Run on a card whose decoder_to_decoder
    latency is 400 fridge cycles, so the delivery is far from the
    landing of the rounds.
    """
    machine = _slow_boundary_machine(declared_run.REDO_WINDOW)
    engine = machine.engine
    delivered_ticks = {}
    started_ticks = {}
    _watch_pin_and_start(engine, delivered_ticks, started_ticks)
    machine.run()
    pinned_keys = set(delivered_ticks) & set(started_ticks)
    waits = [started_ticks[key] - delivered_ticks[key] for key in pinned_keys]
    assert pinned_keys
    assert min(waits) >= 0


def _watch_pin_and_start(engine, delivered_ticks, started_ticks):
    """The tick each window's last pin lands, and its strong decode starts."""
    original_delivery = window_boundaries.BoundaryCourier._pin_delivered
    original_mask = decode_requests.WindowInputGate.mask_input

    def delivered(self, destination, transfer):
        delivered_ticks[destination.key] = engine.now
        original_delivery(self, destination, transfer)

    def start(self, job):
        original_mask(self, job)
        if job.kind is decoding_records.DecodeJobKind.STRONG_REDECODE:
            started_ticks[job.window.key] = engine.now

    window_boundaries.BoundaryCourier._pin_delivered = delivered
    decode_requests.WindowInputGate.mask_input = start


def _slow_boundary_machine(strong_window) -> machine_module.Machine:
    """The gate's switching card with a long decoder_to_decoder hop."""
    settings = gate_switching(strong_window=strong_window)
    slow_path = one_cycle_path(
        settings.links, "decoder_to_decoder", FRIDGE_CLOCK, latency_cycles=400
    )
    links = dataclasses.replace(settings.links, decoder_to_decoder=slow_path)
    settings = dataclasses.replace(settings, links=links)
    return machine_module.Machine.build(settings, 0)


def _gate_machine(strong_window) -> machine_module.Machine:
    """The gate's own switching point, on the strong window row given.

    p 0.008, d 3, 1 us rounds, seed 0: the point where the weak tier
    commits a correction that flips a seam detector of a window that
    later escalates, so the two-solve case a raw-read row must not fold
    is on the run, and so is the seam a pinned row must carry.
    """
    settings = gate_switching(strong_window=strong_window)
    return machine_module.Machine.build(settings, 0)


def _masked_jobs(monkeypatch) -> list:
    """Records (kind, window key, whether the gate changed the input)."""
    original = decode_requests.WindowInputGate.mask_input
    masked = []

    def watch(self, job):
        before = _input_bits(job.decoder_input)
        original(self, job)
        after = _input_bits(job.decoder_input)
        key = None
        if job.window is not None:
            key = job.window.key
        changed = before != after
        masked.append((job.kind, key, changed))

    monkeypatch.setattr(decode_requests.WindowInputGate, "mask_input", watch)
    return masked


def _input_bits(decoder_input) -> tuple:
    """Every fragment's bits of a job's input, in read order."""
    if decoder_input is None:
        return ()
    bits = []
    for round_input in decoder_input.rounds:
        for fragment in round_input.fragments:
            bits.append(fragment.bits)
    return tuple(bits)


def test_a_redo_window_strong_job_is_masked_where_its_weak_job_is(
    monkeypatch,
):
    """The row pins its past face on the neighbour its weak job pins on.

    Bombin et al. 2303.04846 lines 775-788: the input of a later task is
    the syndrome plus the corrections the tasks before it committed. The
    row reads no round behind its commit region, so the seam
    the neighbour's commit flips reaches its input only as that mask,
    and every escalated window whose weak job was masked has its strong
    job masked too.
    """
    masked = _masked_jobs(monkeypatch)
    machine = _gate_machine(declared_run.REDO_WINDOW)
    machine.run()
    weak_masked = _keys_of(masked, strong=False, changed_only=True)
    strong_keys = _keys_of(masked, strong=True, changed_only=False)
    strong_masked = _keys_of(masked, strong=True, changed_only=True)
    masked_and_escalated = weak_masked & strong_keys
    assert masked_and_escalated
    assert masked_and_escalated <= strong_masked


def _keys_of(masked: list, *, strong: bool, changed_only: bool) -> set:
    """The window keys of the recorded jobs of one tier."""
    strong_redecode = decoding_records.DecodeJobKind.STRONG_REDECODE
    keys = set()
    for kind, key, changed in masked:
        is_strong = kind is strong_redecode
        if is_strong is not strong:
            continue
        if changed_only and not changed:
            continue
        keys.add(key)
    return keys


def test_the_redo_window_row_reads_its_commit_region_and_one_buffer():
    """Bombin 2303.04846 lines 1456-1458: a pinned face needs no buffer.

    W1 of a d=3 run commits rounds 4-6. Its past face is pinned on W0's
    committed correction, so no buffer behind it is read: the row reads
    4-9 and commits the same rounds.
    """
    machine = fabric.switching_machine(
        rounds=12,
        escalated_windows={1},
        strong_window=declared_run.REDO_WINDOW,
    )
    requests = _heard_requests(machine)
    machine.run()
    strong = _strong_job(requests, 1)
    assert (strong.window.start_round, strong.window.buffer_hi) == (4, 9)
    assert strong.round_count == 6
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 1), "strong"),
        ((1, 2), "weak"),
        ((1, 3), "weak"),
    ]


def test_the_redo_window_row_pins_nothing_at_the_operations_first_window():
    """W0 has no earlier neighbour, so its past face is the readout's.

    The first window's oldest layer is the operation's own first round
    layer, closed by the initialisation, so the row declares no face
    and the job reads its commit region and one buffer.
    """
    machine = fabric.switching_machine(
        rounds=12,
        escalated_windows={0},
        strong_window=declared_run.REDO_WINDOW,
    )
    requests = _heard_requests(machine)
    machine.run()
    strong = _strong_job(requests, 0)
    assert (strong.window.start_round, strong.window.buffer_hi) == (1, 6)
    assert strong.round_count == 6


def _pinned_boundary_transfers(result) -> list:
    """(window index, payload bits) of every pinned face on the wire.

    A weak delivery is attributed to the window that produced it; a
    pinned face is attributed to the strong window it lands in, so a
    transfer attributed to its own delivery's destination is a pin, its
    own weak commit at a back-to-back seam included.
    """
    pinned = []
    for transfer in result.link_traffic["transfers"]:
        if transfer["path"] != "decoder_to_decoder":
            continue
        attribution = transfer["attribution"]
        if not _is_pinned_face(attribution):
            continue
        pinned.append((attribution["window_id"], transfer["payload_bits"]))
    return pinned


def _is_pinned_face(attribution: dict) -> bool:
    """Whether the transfer is a face pinned on a committed correction."""
    relation = attribution["relation"]
    recorded_key = relation["destination_window_key"]
    destination_key = identity_records.stable_identity_from_json(recorded_key)
    destination_window_id = destination_key[1]
    return attribution["window_id"] == destination_window_id


def test_the_redo_window_rows_pinned_faces_cross_the_wire():
    """Every face the redo window pins is a message on decoder_to_decoder.

    Skoric 2209.08552 lines 1038-1040 sends the artificial defects block
    to block, and Bombin's Fig. 14
    (lines 2256-2259) routes them through the boundary condition data
    store between decoder modules. A pinned face that crossed nothing
    would be free in the model.
    """
    machine = _gate_machine(declared_run.REDO_WINDOW)
    result = machine.run()
    shape = machine.windows.window_manager.strong_redecode.shape
    assert type(shape) is strong_window_shapes.RedoWindow
    strong_windows = _strong_window_keys(machine)
    pinned = _pinned_boundary_transfers(result)
    pinned_windows = {(1, window_id) for window_id, _bits in pinned}
    pinned_bits = {payload_bits for _window_id, payload_bits in pinned}
    assert pinned_windows == strong_windows
    assert len(pinned) == len(strong_windows)
    # a d=3 bulk layer carries d*d-1 = 8 detectors, one bit each under
    # the dense row
    assert pinned_bits == {8}


def test_the_double_window_row_reads_exactly_the_rounds_it_commits():
    """Toshio 2510.25222 lines 1248-1250, as the paper states it.

    W1 of a d=3 run commits 4-6, so the double-window region commits
    r_com + 2 r_buf = 9 rounds, 4-12. Pinning both faces needs no
    buffer of context on either side, so the row reads the 9 rounds it
    commits.
    """
    machine = fabric.switching_machine(
        rounds=15,
        escalated_windows={1},
        strong_window=declared_run.DOUBLE_WINDOW,
        round_microseconds=4.0,
    )
    requests = _heard_requests(machine)
    machine.run()
    strong = _strong_job(requests, 1)
    assert (strong.window.start_round, strong.window.buffer_hi) == (4, 12)
    assert strong.round_count == 9
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 4), "weak"),
        ((1, 1), "strong"),
    ]


def test_the_double_window_row_pins_its_near_and_its_far_face():
    """One message per pinned face, both on decoder_to_decoder.

    The near face is the window before the strong region, the far face
    the window that restarts the weak chain after it, the boundary the
    row waits for (Toshio 2510.25222 lines 1253-1259). Skoric 2209.08552
    lines 1038-1040 sends each face's defects block to block, so two
    faces are two messages.
    """
    machine = fabric.switching_machine(
        rounds=15,
        escalated_windows={1},
        strong_window=declared_run.DOUBLE_WINDOW,
        round_microseconds=4.0,
    )
    result = machine.run()
    pinned = _pinned_boundary_transfers(result)
    pinned_windows = [window_id for window_id, _payload_bits in pinned]
    assert pinned_windows == [1, 1]
    sources = _pin_sources(result)
    # W1's own dependency, and the window that restarts the chain past
    # the strong region 4-12
    assert sources == [0, 4]


def _pin_sources(result) -> list:
    """The window that produced each pinned face's boundary, in send order."""
    sources = []
    for transfer in result.link_traffic["transfers"]:
        if transfer["path"] != "decoder_to_decoder":
            continue
        attribution = transfer["attribution"]
        if not _is_pinned_face(attribution):
            continue
        source_window_id = attribution["relation"]["request_key"]["window_id"]
        sources.append(source_window_id)
    return sources


def _pin_sources_into(result, window_id: int) -> list:
    """The window that produced each face pinned into that strong window."""
    sources = []
    for transfer in result.link_traffic["transfers"]:
        if transfer["path"] != "decoder_to_decoder":
            continue
        attribution = transfer["attribution"]
        if attribution["window_id"] != window_id:
            continue
        if not _is_pinned_face(attribution):
            continue
        source_window_id = attribution["relation"]["request_key"]["window_id"]
        sources.append(source_window_id)
    return sources


def test_the_double_window_row_at_the_operations_end_has_no_far_pin():
    """Tan 2209.09219 lines 953-955: the last window's faces are closed.

    A terminal strong region has no later window to pin on, so it waits
    for the terminal data and reads to its own last committed round.
    """
    machine = fabric.switching_machine(
        rounds=9,
        escalated_windows={2},
        strong_window=declared_run.DOUBLE_WINDOW,
    )
    requests = _heard_requests(machine)
    result = machine.run()
    submitted = fabric.log_lines_containing(
        machine, "terminal data complete -> strong window submitted"
    )
    assert len(submitted) == 1
    strong = _strong_job(requests, 2)
    assert (strong.window.start_round, strong.window.buffer_hi) == (7, 9)
    sources = _pin_sources(result)
    # only the near face: W2's own dependency
    assert sources == [1]


def test_a_region_at_a_back_to_back_seam_reads_the_rounds_it_commits():
    """A window whose dependency was absorbed still has a near boundary.

    W1's strong region commits 4-12 and absorbs W2 and W3, so the
    restart window W4 keeps no dependency: no neighbour commits the
    round before it. W4's own weak decode owns the faults crossing that
    seam and the region 4-12 is pinned on them, so when W4 escalates in
    turn, its region commits 13-21 and reads those nine rounds and
    nothing behind them (Toshio 2510.25222 lines 1248-1250).
    """
    machine = fabric.switching_machine(
        rounds=30,
        escalated_windows={1, 4},
        strong_window=declared_run.DOUBLE_WINDOW,
        round_microseconds=4.0,
    )
    requests = _heard_requests(machine)
    machine.run()
    first_region = _strong_job(requests, 1)
    first_window = first_region.window
    assert (first_window.start_round, first_window.buffer_hi) == (4, 12)
    seam_region = _strong_job(requests, 4)
    seam_window = seam_region.window
    assert (seam_window.start_round, seam_window.buffer_hi) == (13, 21)


def test_a_region_at_a_back_to_back_seam_pins_its_near_face_on_itself():
    """The near face's commit is the escalated window's own weak one.

    Each pinned face is one message on decoder_to_decoder: W4, whose
    region sits at the back-to-back seam, pins its own weak commit and
    W7, the window that restarts the chain after it.
    """
    machine = fabric.switching_machine(
        rounds=30,
        escalated_windows={1, 4},
        strong_window=declared_run.DOUBLE_WINDOW,
        round_microseconds=4.0,
    )
    result = machine.run()
    seam_window_id = 4
    restart_window_id = 7
    sources = _pin_sources_into(result, seam_window_id)
    sources_in_order = sorted(sources)
    assert sources_in_order == [seam_window_id, restart_window_id]


def test_the_gate_card_on_the_double_window_runs_through():
    """The gate's own switching card on the double window builds that row.

    It runs the point through with no strong window left pending.
    """
    machine = _gate_machine(declared_run.DOUBLE_WINDOW)
    result = machine.run()
    shape = machine.windows.window_manager.strong_redecode.shape
    assert type(shape) is strong_window_shapes.DoubleWindow
    statuses = _run_statuses(result)
    assert statuses == [(1, "logical_observables")]
    assert not machine.windows.window_manager.strong_redecode.has_pending()


def test_a_re_reading_restart_window_owns_the_faults_the_far_pin_carries():
    """Fig. 12 step 5: the restart window reads the region's last block.

    Toshio 2510.25222 draws the restarted weak window with its buffer
    over the strong region's last block, and the region is decoded once
    the weak decoder has fixed both of its ends (lines 1248-1250). At
    re-read 1 on the gate's card W3's region commits 10-18, and the
    restart window W6 reads 16-24: that last buffer region as its own
    past context. The region reads nothing past 18, so W6 owns the
    faults crossing 18 to 19 and the region's far face is pinned on its
    commit.
    """
    machine = _gate_double_window_machine(3, 3, 40.0, 5.0, 2, 1)
    result = machine.run()
    resliced = fabric.log_lines_containing(machine, "re-sliced")
    sources = _pin_sources_into(result, 3)
    assert _run_statuses(result) == [(1, "logical_observables")]
    assert (
        "restart window (1, 6) re-sliced across strong window edge 18 "
        "(reads rounds 16-24; crossing faults owned by restart_window)"
    ) in resliced[0]
    assert 6 in sources


def _strong_window_keys(machine) -> set:
    """The windows that escalated and pinned a face on their neighbour.

    The operation's first window pins nothing, so it is not among them.
    """
    keys = set()
    for window_key, tier in fabric.frame_tiers(machine):
        if tier != "strong":
            continue
        if window_key[1] == 0:
            continue
        keys.add(window_key)
    return keys


class _StoredRounds:
    """A retention that answers with the rounds the window reads."""

    def strong_window_input(self, builder, window) -> tuple:
        """The stored rounds, whatever the builder and the window are."""
        del builder
        del window
        return ("the stored rounds",)


class _RecordingCourier:
    """A courier that records the faces a row asked it to pin."""

    def __init__(self) -> None:
        self.pinned = []

    def pin_strong_face(
        self, source_key, destination, model, operation, request_key
    ):
        """Note the face the row pinned, with what the fold reads."""
        self.pinned.append(
            (source_key, destination, model, operation, request_key)
        )


def _folding_shape(retention, courier):
    """A shape with only the ports the fold reads bound.

    The builder is the one this retention hands back untouched, so the
    fold's own two components are all the test binds for real.
    """
    shape = strong_window_shapes.RedoWindow(None)
    shape.retention = retention
    shape.courier = courier
    shape.builder = _UnusedBuilder()
    return shape


class _UnusedBuilder:
    """The job builder this retention stub never asks anything of."""


def test_a_row_that_pins_a_face_folds_the_boundary_it_declares():
    """The input adaptation is asked for once per declared face.

    A row that pins a face carries its neighbour's committed correction
    into the strong job's input, which is Bombin et al. 2303.04846's
    input adaptation (lines 775-788): the courier ships the committed
    boundary to this window and folds it into the window's boundary
    state, against the strong window's own model. A row that folds none
    reads the stored rounds and asks the courier for nothing.
    """
    retention = _StoredRounds()
    courier = _RecordingCourier()
    shape = _folding_shape(retention, courier)
    window = object()
    model = object()
    operation = object()
    request_key = object()
    payloads = strong_window_shapes.strong_job_payloads(
        shape,
        window,
        model,
        operation,
        request_key,
        strong_window_shapes.FOLDS_NO_BOUNDARY,
    )
    assert payloads == ("the stored rounds",)
    assert courier.pinned == []
    payloads = strong_window_shapes.strong_job_payloads(
        shape, window, model, operation, request_key, ((1, 1),)
    )
    assert payloads == ("the stored rounds",)
    assert courier.pinned == [((1, 1), window, model, operation, request_key)]

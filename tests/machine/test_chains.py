"""A chain of three pools on declared ticks: each timeline by hand.

The declared switching machine (tests/escalation/declared_fabric.py)
grows a third pool on the host, deep, behind the strong pool: the weak
pool escalates to the strong one, which escalates on to deep. Every
hop of the chain's own is priced at its role's declared latency, so
strong_to_deep rides as weak_decoder_to_strong_decoder does (3 us),
deep_buffer_to_deep as strong_buffer_to_strong_decoder (6 us),
deep_to_frame as strong_decoder_to_frame (4 us) and deep_to_weak as
strong_decoder_to_weak_decoder (4 us). The deep pool decodes in 50 us.

Each pool declares which windows it doubts, so the three runs are the
three shapes of a chain: nothing escalates, a window escalates once and
is kept by the strong pool, and a window escalates twice. The first
two are the switching machine's own timelines
(tests/escalation/test_switching_mode.py); the third adds the deep leg.
"""

import dataclasses

import decsim.config as config
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.escalation.deadlines as deadlines
import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.records.decoding as decoding_records
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph
import decsim.topology.graph_layout as graph_layout
import decsim.topology.slotted as slotted
import tests.declared_run as declared_run
import tests.escalation.declared_fabric as fabric
import tests.topology.test_graph_layout as layout_tests

DEEP_MICROSECONDS = 50.0
DEEP_POOL = "deep"


class DeclaredTwoSolves:
    """A declared gap read off one solve per forced logical class.

    The strong decoder declares each solve's gap, and the signal hands
    on the first solve's: the complementary gap is one number per
    window, from the lightest correction in each class (Toshio et al.
    2510.25222 Sec. II C).
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The declared two-solve signal, as the switching slot names it."""

        name = "declared_two_solves"

        def build(self, weak_algorithm, threshold_nats):
            """The signal; it reads neither the decoder nor the threshold."""
            del weak_algorithm
            del threshold_nats
            return DeclaredTwoSolves()

    source = declared_run.DECLARED_CONFIDENCE_SOURCE
    fault_model_requirement = None
    decoder_evidence_requirement = frozenset()
    evidence_refusal = "the declared signal needs no evidence"
    forced_logical_classes = (0, 1)

    def compute(self, solves):
        """The first solve's declared soft output, computed in no time."""
        soft_output = solves[0].soft_output
        return decoding_records.SoftOutputComputation(soft_output)


@dataclasses.dataclass(frozen=True)
class DeclaredHops:
    """The declared card; a chain's own hop at its role's declared latency.

    A test's fabric rule: a slotted hop is the card's path as written, and
    a hop of the graph's own is the card's latency for the slotted hop
    whose role it carries (fabrics.Route.reference).
    """

    card: object

    def link_for(self, route: fabrics.Route) -> object:
        """The card's path, or a timed link at the reference's latency."""
        names = self.card.path_names()
        if route.path in names:
            path = self.card.path_settings(route.path)
            return fabrics.FixedPath(path)
        reference = self.card.path_settings(route.reference)
        channel = reference.channel
        latency_ticks = channel.propagation_latency_ticks
        latency_microseconds = config.ticks_to_microseconds(latency_ticks)
        excludes = reference.excludes_receiver_processing
        return fabrics.TimedLink(
            latency_microseconds=latency_microseconds,
            excludes_receiver_processing=excludes,
        )


def chain_run(
    weak_doubts: set, strong_doubts: set, *, record: bool = False
) -> machine_module.Machine:
    """The declared chain, run: each pool escalates the windows it doubts.

    record keeps the switching records, and the decode records they read.
    """
    declared = chain_graph(weak_doubts, strong_doubts, record=record)
    machine = chain_built(declared)
    machine.run()
    return machine


def chain_built(declared: graph.Machine) -> machine_module.Machine:
    """A declared chain built on seed 0, not yet run."""
    layout = layout_tests.laid_out(declared)
    entry_settings = graph_layout.entry_settings(declared)
    return machine_module.Machine.build(entry_settings, 0, layout=layout)


def chain_graph(
    weak_doubts: set,
    strong_doubts: set,
    *,
    record: bool = False,
    entry_window=declared_run.REDO_WINDOW,
    rounds: int = 9,
    deep_speculates: bool = False,
    entry_speculates: bool = False,
    deep_confidence=None,
    deep_deadline=None,
) -> graph.Machine:
    """The declared chain: each pool escalates the windows it doubts.

    The entry edge lays its strong windows out by entry_window; the deep
    edge redoes the strong revision. Each edge that speculates starts
    its target's decode with its source's job. deep_confidence replaces
    the deep edge's declared one-solve signal, and deep_deadline gives
    the deep edge a deadline.
    """
    settings = fabric.switching_settings(
        rounds=rounds,
        escalated_windows=weak_doubts,
        strong_window=entry_window,
        run_both_at_once=entry_speculates,
        record=record,
    )
    base = slotted.graph_of(settings)
    strong_microseconds = fabric.DECLARED_MICROSECONDS["strong"]
    is_doubted = fabric.escalate_only(strong_doubts)
    strong_algorithm = declared_run.DeclaredConfidenceDecoder.Settings(
        strong_microseconds, is_doubted
    )
    with_doubts = base.with_settings("strong", algorithm=strong_algorithm)
    deep_changes = {"run_both_at_once": deep_speculates}
    if deep_confidence is not None:
        deep_changes["confidence"] = deep_confidence
    if deep_deadline is not None:
        deep_changes["deadline"] = deep_deadline
    chain = _with_a_deep_pool(with_doubts, deep_changes)
    card_fabric = DeclaredHops(settings.links)
    return dataclasses.replace(chain, fabric=card_fabric)


def test_a_chain_that_doubts_nothing_runs_as_the_weak_pool_alone():
    """W0's weak decode ends at 30 us and rides 2 us home, 1 us to commit."""
    machine = chain_run(set(), set())

    first = _first_record(machine)

    assert first.window_key == (1, 0)
    assert first.tier == "weak"
    assert first.accepted_ticks == _ticks(32.0)
    assert first.committed_ticks == _ticks(33.0)


def test_a_window_the_strong_pool_keeps_follows_the_serial_timeline():
    """The deep edge's verdict is synchronous, so nothing moves.

    The weak verdict at 30 us carries W0 up (3 us), its input crosses
    to the strong unit (6 us) at 39 us, the strong decode ends at 69 us,
    and the strong pool keeps it: the answer lands 4 us later, 73 us,
    and commits at 74 us. W1's held boundary crosses back (4 us) and its
    input (5 us), so W1 starts at 78 us.
    """
    machine = chain_run({0}, set())

    first = _first_record(machine)
    released_start = _log_tick(machine, "START DECODE mem1 W1")

    assert first.tier == "strong"
    assert first.accepted_ticks == _ticks(73.0)
    assert first.committed_ticks == _ticks(74.0)
    assert released_start == _ticks(78.0)


def test_a_window_doubted_twice_is_answered_by_the_deep_pool():
    """The strong result at 69 us escalates again, from the strong store.

    W0's rounds landed in the strong syndrome buffer at 33 us and are
    held there for the deep edge, so they leave it at 69 us over
    strong_to_deep (3 us), land in the deep store at 72 us, and cross
    to the deep unit (6 us): the deep decode starts at 78 us and ends at
    128 us. Its answer rides deep_to_frame (4 us) to 132 us and commits
    at 133 us. The strong result was provisional, so no boundary left
    at 69 us: W1's held boundary leaves the deep unit at 128 us, crosses
    deep_to_weak (4 us) and W1's input (5 us), so W1 starts at 137 us.
    """
    machine = chain_run({0}, {0})

    first = _first_record(machine)
    deep_start = _log_tick(machine, "START DECODE deep(mem1 W0)")
    released_start = _log_tick(machine, "START DECODE mem1 W1")

    assert first.tier == DEEP_POOL
    assert deep_start == _ticks(78.0)
    assert first.accepted_ticks == _ticks(132.0)
    assert first.committed_ticks == _ticks(133.0)
    assert released_start == _ticks(137.0)


def test_a_deep_decode_started_with_the_strong_one_is_selected():
    """The deep edge starts beside the strong job, 36 us sooner.

    W0's rounds land in the strong store at 33 us and the strong job
    leaves its hold; the deep edge plans its decode then and carries the
    rounds on over strong_to_deep (3 us) to the deep store at 36 us,
    then to the deep unit (6 us): the deep decode starts at 42 us and
    ends at 92 us. The strong result at 69 us is doubted and selects
    it. The answer rides deep_to_frame (4 us) to 96 us and commits at
    97 us; W1's held boundary leaves at 92 us, crosses deep_to_weak
    (4 us) and W1's input (5 us), so W1 starts at 101 us.
    """
    declared = chain_graph({0}, {0}, deep_speculates=True)
    machine = chain_built(declared)
    requests = declared_run.EndedRequests()
    requests.attach(machine)
    machine.run()

    first = _first_record(machine)
    deep_start = _log_tick(machine, "START DECODE deep(mem1 W0)")
    released_start = _log_tick(machine, "START DECODE mem1 W1")
    outcomes = _outcomes_of(requests, DEEP_POOL)

    assert first.tier == DEEP_POOL
    assert deep_start == _ticks(42.0)
    assert first.accepted_ticks == _ticks(96.0)
    assert first.committed_ticks == _ticks(97.0)
    assert released_start == _ticks(101.0)
    assert outcomes == ["STRONG_FORWARDED_FOR_DELIVERY"]


def test_a_kept_strong_result_cancels_the_deep_decode_beside_it():
    """The serial keep's timeline; the deep decode ends at 69 us."""
    declared = chain_graph({0}, set(), deep_speculates=True)
    machine = chain_built(declared)
    requests = declared_run.EndedRequests()
    requests.attach(machine)
    machine.run()

    first = _first_record(machine)
    released_start = _log_tick(machine, "START DECODE mem1 W1")
    outcomes = _outcomes_of(requests, DEEP_POOL)

    assert first.tier == "strong"
    assert first.accepted_ticks == _ticks(73.0)
    assert first.committed_ticks == _ticks(74.0)
    assert released_start == _ticks(78.0)
    assert outcomes == ["STRONG_CANCELLED_DURING_SERVICE"]


def test_a_kept_weak_result_cancels_every_decode_started_beside_it():
    """A keep cancels the strong decode and the deep one it started."""
    declared = chain_graph(
        set(), set(), deep_speculates=True, entry_speculates=True
    )
    machine = chain_built(declared)
    requests = declared_run.EndedRequests()
    requests.attach(machine)
    machine.run()

    strong = _outcomes_of(requests, "strong")
    deep = _outcomes_of(requests, DEEP_POOL)
    snapshot = machine.control.pauli_frame.snapshot()
    tiers = {record.tier for record in snapshot.records}

    assert tiers == {"weak"}
    assert strong == ["STRONG_CANCELLED_DURING_SERVICE"] * 3
    assert deep == ["STRONG_CANCELLED_DURING_SERVICE"] * 3


def test_a_deep_edge_on_two_strong_solves_waits_for_both():
    """The strong pool solves W0 once per class, one after the other.

    Both inputs reach the one strong unit at 39 us. Class 0 decodes
    39-69 us and class 1 69-99 us, so the deep edge's gap joins at
    99 us. The rounds then ride strong_to_deep (3 us) and the deep
    input (6 us): the deep decode runs 108-158 us, its answer lands at
    162 us over deep_to_frame (4 us) and commits at 163 us. W1's
    boundary leaves at 158 us, crosses deep_to_weak (4 us) and W1's
    input (5 us), so W1 starts at 167 us.
    """
    machine = _two_solve_chain({0}, {0}, speculates=False)

    first = _first_record(machine)
    join = _log_tick(machine, "GAP JOIN strong(mem1 W0)")
    deep_start = _log_tick(machine, "START DECODE deep(mem1 W0)")
    released_start = _log_tick(machine, "START DECODE mem1 W1")

    assert first.tier == DEEP_POOL
    assert join == _ticks(99.0)
    assert deep_start == _ticks(108.0)
    assert first.accepted_ticks == _ticks(162.0)
    assert first.committed_ticks == _ticks(163.0)
    assert released_start == _ticks(167.0)


def test_a_strong_answer_kept_on_two_solves_leaves_at_the_join():
    """Kept at 99 us: the frame at 103 us, a commit at 104 us, W1 at 108."""
    machine = _two_solve_chain({0}, set(), speculates=False)

    first = _first_record(machine)
    released_start = _log_tick(machine, "START DECODE mem1 W1")

    assert first.tier == "strong"
    assert first.accepted_ticks == _ticks(103.0)
    assert first.committed_ticks == _ticks(104.0)
    assert released_start == _ticks(108.0)


def test_a_deep_decode_beside_two_strong_solves_waits_for_their_join():
    """Started with the strong job at 42 us, done at 92, selected at 99.

    The selection rides strong_to_deep (3 us) to 102 us, and the answer
    deep_to_frame (4 us) to 106 us; it commits at 107 us and W1 starts
    at 111 us.
    """
    machine = _two_solve_chain({0}, {0}, speculates=True)

    first = _first_record(machine)
    deep_start = _log_tick(machine, "START DECODE deep(mem1 W0)")
    released_start = _log_tick(machine, "START DECODE mem1 W1")

    assert deep_start == _ticks(42.0)
    assert first.accepted_ticks == _ticks(106.0)
    assert first.committed_ticks == _ticks(107.0)
    assert released_start == _ticks(111.0)


def test_a_late_strong_result_is_answered_by_the_deep_rescue():
    """The deep edge's deadline runs from the strong job's arrival.

    The strong job arrives at 33 us, so its deadline is 43 us. The
    rescue starts then: the rounds held in the strong store ride
    strong_to_deep (3 us) and the deep input (6 us), so the deep decode
    runs 52-102 us. The strong result at 69 us would be kept, but it is
    late, so the deep edge escalates and selects the rescue. The answer
    lands at 106 us over deep_to_frame (4 us) and commits at 107 us;
    W1's boundary leaves at 102 us, crosses deep_to_weak (4 us) and W1's
    input (5 us), so W1 starts at 111 us.
    """
    deadline = deadlines.Deadline.Settings(slack_microseconds=10.0)
    declared = chain_graph({0}, set(), deep_deadline=deadline)
    machine = chain_built(declared)
    machine.run()

    first = _first_record(machine)
    rescue = _log_tick(machine, "strong(mem1 W0): deadline trigger")
    deep_start = _log_tick(machine, "START DECODE deep(mem1 W0)")
    released_start = _log_tick(machine, "START DECODE mem1 W1")

    assert rescue == _ticks(43.0)
    assert deep_start == _ticks(52.0)
    assert first.tier == DEEP_POOL
    assert first.accepted_ticks == _ticks(106.0)
    assert first.committed_ticks == _ticks(107.0)
    assert released_start == _ticks(111.0)


def test_every_store_of_a_chain_settles_empty():
    """Every hold the three stores placed ends with the last commit."""
    machine = chain_run({0, 1, 2}, {0, 2})

    left = [len(parts.store.round_by_key) for parts in machine.readout.stores]

    assert left == [0, 0, 0]


def test_every_pools_decodes_are_recorded():
    """The run's records hear the deep pool beside the first two."""
    machine = chain_run({0}, {0}, record=True)

    requests = machine.observation.decode_records.requests
    pools = {record.request_key.pool.word for record in requests}

    assert pools == {"weak", "strong", DEEP_POOL}


def test_a_deep_redo_decodes_the_strong_revision_on_its_own_terms():
    """One decoder at both depths gives one answer.

    The strong pool's double window over W1 commits the rounds it
    absorbed from W2 and pins its far face on the restart window. The
    deep redo reads that region, not W1's planner window, pins the same
    faces and decodes on the error model the strong pool decoded on.
    """
    declared = chain_graph(
        {1}, {1}, entry_window=declared_run.DOUBLE_WINDOW, rounds=15
    )
    machine = chain_built(declared)
    requests = declared_run.EndedRequests()
    requests.attach(machine)
    machine.run()

    strong_job = _job_of(requests, "strong")
    deep_job = _job_of(requests, DEEP_POOL)

    assert _bounds(deep_job) == _bounds(strong_job)
    assert deep_job.pinned_faces == strong_job.pinned_faces
    assert deep_job.detector_error_model is strong_job.detector_error_model


def _with_a_deep_pool(machine: graph.Machine, changes: dict) -> graph.Machine:
    """The strong pool escalating on to a deep pool on the host.

    changes are the deep edge's switching fields that differ from the
    declared ones.
    """
    deep_algorithm = decoders.PresetLatencyDecoder.Settings(DEEP_MICROSECONDS)
    deep_pool = decoder_settings.DecoderPoolSettings(
        algorithm=deep_algorithm, engine=declared_run.DECLARED_ENGINE
    )
    deep_switching = declared_run.declared_switching(**changes)
    return machines.with_escalation(
        machine, machines.STRONG_POOL, DEEP_POOL, deep_pool, deep_switching
    )


def _job_of(requests: declared_run.EndedRequests, pool_name: str):
    """The one job a pool decoded for an escalated window."""
    jobs = []
    for ended in requests.ended:
        if ended.job.request_key.pool.word == pool_name:
            jobs.append(ended.job)
    (job,) = jobs
    return job


def _two_solve_chain(weak_doubts, strong_doubts, speculates: bool):
    """The declared chain whose deep edge reads two strong solves, run."""
    signal = DeclaredTwoSolves.Settings()
    declared = chain_graph(
        weak_doubts,
        strong_doubts,
        deep_speculates=speculates,
        deep_confidence=signal,
    )
    machine = chain_built(declared)
    machine.run()
    return machine


def _outcomes_of(requests: declared_run.EndedRequests, pool_name: str):
    """The outcome names of one pool's requests, in the order they ended."""
    names = []
    for ended in requests.ended:
        if ended.job.request_key.pool.word == pool_name:
            names.append(ended.outcome.name)
    return names


def _bounds(job) -> tuple:
    window = job.window
    return (
        window.buffer_lo,
        window.commit_lo,
        window.commit_hi,
        window.buffer_hi,
    )


def _first_record(machine: machine_module.Machine):
    snapshot = machine.control.pauli_frame.snapshot()
    return snapshot.records[0]


def _log_tick(machine: machine_module.Machine, text: str) -> int:
    log_lines = machine.observation.log.lines
    return declared_run.log_tick(log_lines, text)


def _ticks(microseconds: float) -> int:
    return config.microseconds_to_ticks(microseconds)

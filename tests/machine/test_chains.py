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
import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph
import decsim.topology.graph_layout as graph_layout
import decsim.topology.legacy as legacy
import tests.declared_run as declared_run
import tests.escalation.declared_fabric as fabric
import tests.topology.test_graph_layout as layout_tests

DEEP_MICROSECONDS = 50.0
DEEP_POOL = "deep"


@dataclasses.dataclass(frozen=True)
class DeclaredHops:
    """The declared card; a chain's own hop at its role's declared latency.

    A test's fabric rule: a legacy hop is the card's path as written, and
    a hop of the graph's own is the card's latency for the legacy hop
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
    layout = layout_tests.laid_out(declared)
    entry_settings = graph_layout.entry_settings(declared)
    machine = machine_module.Machine.build(entry_settings, 0, layout=layout)
    machine.run()
    return machine


def chain_graph(
    weak_doubts: set, strong_doubts: set, *, record: bool = False
) -> graph.Machine:
    """The declared chain: each pool escalates the windows it doubts."""
    settings = fabric.switching_settings(
        rounds=9, escalated_windows=weak_doubts, record=record
    )
    base = legacy.graph_of(settings)
    strong_microseconds = fabric.DECLARED_MICROSECONDS["strong"]
    is_doubted = fabric.escalate_only(strong_doubts)
    strong_algorithm = declared_run.DeclaredConfidenceDecoder.Settings(
        strong_microseconds, is_doubted
    )
    with_doubts = base.with_settings("strong", algorithm=strong_algorithm)
    chain = _with_a_deep_pool(with_doubts)
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


def _with_a_deep_pool(machine: graph.Machine) -> graph.Machine:
    """The strong pool escalating on to a deep pool on the host."""
    deep_algorithm = decoders.PresetLatencyDecoder.Settings(DEEP_MICROSECONDS)
    deep_pool = decoder_settings.DecoderPoolSettings(
        algorithm=deep_algorithm, engine=declared_run.DECLARED_ENGINE
    )
    deep_switching = declared_run.declared_switching()
    return machines.with_escalation(
        machine, machines.STRONG_POOL, DEEP_POOL, deep_pool, deep_switching
    )


def _first_record(machine: machine_module.Machine):
    snapshot = machine.control.pauli_frame.snapshot()
    return snapshot.records[0]


def _log_tick(machine: machine_module.Machine, text: str) -> int:
    log_lines = machine.observation.log.lines
    return declared_run.log_tick(log_lines, text)


def _ticks(microseconds: float) -> int:
    return config.microseconds_to_ticks(microseconds)

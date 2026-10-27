"""The Decoder port's laws, on a fake row and on every shipped row.

sinter's abstract class with defaults is the shape
(.pydeps/sinter/_decoding/_decoding_decoder_class.py), and the fake rows
pin decoder.py's defaults against it.

Every row that decodes a fault model answers Microsoft deq's standard
decoder test problems (https://github.com/microsoft/qdk-ec, commit
50f3bff4, MIT license): the seven hypergraphs and their syndromes of
deq/deq_runtime/src/decoder/test_problems.rs, judged by the rule of
test_harness.rs and blackbox_util.rs is_parity_factor, under which a
decode passes when the XOR of its chosen edges' vertex sets is the
syndrome, a one-vertex edge flipping its one vertex. A row that matches
on a graph takes the six problems whose edges touch at most two
vertices; deq's single_hyperedge_3 is for the rows that read physical
hyperedges.
"""

import importlib.util

import numpy
import pytest
import scipy.sparse

import decsim.config as config
import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.decoders.belief_propagation_osd.decoder as bposd
import decsim.decoders.decoder as decoder_module
import decsim.decoders.decoders as decoders
import decsim.decoders.dispatch_steps.decoder as dispatch_steps
import decsim.decoders.measured_table.decoder as measured_table
import decsim.decoders.minimum_weight_perfect_matching.decoder as matching
import decsim.decoders.relay_belief_propagation.decoder as relay
import decsim.decoders.tesseract.decoder as tesseract
import decsim.decoders.union_find.cycle_count as cycle_count
import decsim.decoders.union_find.decoder as union_find
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models
import decsim.records.seeds as seed_records
import decsim.seeding as seeding
from tests.decoders import windows

MEASURED_NS = 2500
GRAPHLIKE = fault_models.FaultRepresentation.GRAPHLIKE
PHYSICAL = fault_models.FaultRepresentation.PHYSICAL
SEED = 7
# deq gives every edge of every problem this probability
DEQ_EDGE_PROBABILITY = 0.1
RELAY_BP_SPECIFICATION = importlib.util.find_spec("relay_bp")
LACKS_RELAY_BP = RELAY_BP_SPECIFICATION is None
TESSERACT_SPECIFICATION = importlib.util.find_spec("tesseract_decoder")
LACKS_TESSERACT = TESSERACT_SPECIFICATION is None
# both are the bb-decoders extra, which a test install leaves out
NEEDS_RELAY_BP = pytest.mark.skipif(
    LACKS_RELAY_BP, reason="could not import 'relay_bp'"
)
NEEDS_TESSERACT = pytest.mark.skipif(
    LACKS_TESSERACT, reason="could not import 'tesseract_decoder'"
)
HOST_TIME = cycle_count.HostMeasuredTime()
UNION_FIND = union_find.UnionFindDecoder.Settings(timing=HOST_TIME)
PYMATCHING = matching.PyMatchingDecoder.Settings()
UNWEIGHTED_PYMATCHING = matching.UnweightedPyMatchingDecoder.Settings()
BELIEF_MATCHING = belief_matching.BeliefMatchingDecoder.Settings()
BPOSD = bposd.BeliefPropagationOsdDecoder.Settings()
RELAY_BP = relay.RelayBeliefPropagationDecoder.Settings()
TESSERACT = tesseract.TesseractDecoder.Settings()
MEASURED_TABLE = measured_table.MeasuredTableDecoder.Settings()
DISPATCH_STEPS = dispatch_steps.DispatchStepsDecoder.Settings()
# the rows whose selected faults are columns of a matching graph
GRAPH_ROWS = (
    pytest.param(UNION_FIND, id="union_find"),
    pytest.param(PYMATCHING, id="pymatching"),
    pytest.param(UNWEIGHTED_PYMATCHING, id="unweighted_pymatching"),
    pytest.param(BELIEF_MATCHING, id="belief_matching"),
)
# the rows whose selected faults are physical hyperedges; the two strong
# rows answer with the relay_bp row's decode
HYPERGRAPH_ROWS = (
    pytest.param(BPOSD, id="bposd"),
    pytest.param(RELAY_BP, id="relay_bp", marks=NEEDS_RELAY_BP),
    pytest.param(TESSERACT, id="tesseract", marks=NEEDS_TESSERACT),
    pytest.param(MEASURED_TABLE, id="measured_table", marks=NEEDS_RELAY_BP),
    pytest.param(DISPATCH_STEPS, id="dispatch_steps", marks=NEEDS_RELAY_BP),
)
# deq's problems as (vertex count, edges, syndrome), each edge a vertex set
# and each syndrome its defects; the id is deq's problem/case
DEQ_GRAPH_PROBLEMS = (
    pytest.param(0, (), (), id="empty_graph/zero"),
    pytest.param(3, (), (), id="no_edges/zero"),
    pytest.param(1, ((0,),), (), id="single_vertex_boundary/zero"),
    pytest.param(1, ((0,),), (0,), id="single_vertex_boundary/vertex_0"),
    pytest.param(2, ((0, 1),), (), id="simple_edge/zero"),
    pytest.param(2, ((0, 1),), (0, 1), id="simple_edge/both"),
    pytest.param(3, ((0, 1), (1, 2)), (), id="line_chain_3/zero"),
    pytest.param(3, ((0, 1), (1, 2)), (0, 1), id="line_chain_3/left"),
    pytest.param(3, ((0, 1), (1, 2)), (1, 2), id="line_chain_3/right"),
    pytest.param(3, ((0, 1), (1, 2)), (0, 2), id="line_chain_3/ends"),
    pytest.param(3, ((0, 1), (1, 2), (0, 2)), (), id="triangle/zero"),
    pytest.param(3, ((0, 1), (1, 2), (0, 2)), (0, 1), id="triangle/v01"),
    pytest.param(3, ((0, 1), (1, 2), (0, 2)), (1, 2), id="triangle/v12"),
    pytest.param(3, ((0, 1), (1, 2), (0, 2)), (0, 2), id="triangle/v02"),
)
DEQ_HYPEREDGE_PROBLEMS = (
    pytest.param(3, ((0, 1, 2),), (), id="single_hyperedge_3/zero"),
    pytest.param(3, ((0, 1, 2),), (0, 1, 2), id="single_hyperedge_3/all"),
)
DEQ_PROBLEMS = DEQ_GRAPH_PROBLEMS + DEQ_HYPEREDGE_PROBLEMS


class FixedRow(decoder_module.DecoderBase):
    """A priced row: three ticks, one fixed observable."""

    def latency(self, job):
        del job
        return 3

    def decode(self, job):
        return decoding_records.DecodeResult(
            job.stream_id, job.window_id, logical_observables=(1,)
        )


class MeasuredRow(decoder_module.DecoderBase):
    """A measured row: 2.5 microseconds on the host clock."""

    def latency(self, job):
        del job
        raise NotImplementedError

    def occupancy(self, job):
        del job
        return None

    def decode(self, job):
        return decoding_records.DecodeResult(job.stream_id, job.window_id)

    def decode_timed(self, job):
        result = self.decode(job)
        return result, MEASURED_NS


class EmptyWindowRow(decoder_module.WindowDecoderBase):
    """A window row that never gets a model in these tests."""

    def compile(self, faults, model):
        del faults
        del model
        raise AssertionError("no model, nothing to compile")

    def decode_window(self, backend, model, faults, syndrome):
        del backend
        del model
        del faults
        del syndrome
        raise AssertionError("no model, nothing to decode")


class NoFaultRow(decoder_module.WindowDecoderBase):
    """A window row that selects no fault, whatever events it reads."""

    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED

    def compile(self, faults, model):
        del faults
        del model

    def decode_window(self, backend, model, faults, syndrome):
        del backend
        del model
        del syndrome
        fault_count = faults.check.shape[1]
        selected = numpy.zeros(fault_count, dtype=numpy.uint8)
        return decoding_records.WindowDecode(selected)


def _job(**fields) -> decoding_records.DecodeJob:
    return decoding_records.DecodeJob(
        stream_id=1, window_id=0, round_count=2, label="W0", **fields
    )


def _started(row, job):
    engine = engine_module.Engine()
    delivered = []

    def on_result(result):
        delivered.append((engine.now, result))

    row.start(job, engine, on_result)
    engine.run()
    return delivered


def _seeded_row(settings):
    """The row its settings build, bound to the run seed as a run binds it."""
    row = settings.build()
    segment = seed_records.RunSeedPathSegment("field", "decoder")
    root = ((segment,), row)
    seeding.bind_run_seed(SEED, [root])
    return row


def _deq_window(vertex_count: int, edges: tuple):
    """One window holding deq's hypergraph, owning every edge.

    Both representations hold the same columns, linked one to one, so a
    row reads deq's edges whichever it asks for; a three-vertex edge in
    the graphlike view is never handed to a row that matches on a graph.
    """
    graphlike = _deq_faults(GRAPHLIKE, vertex_count, edges)
    physical = _deq_faults(PHYSICAL, vertex_count, edges)
    edge_count = len(edges)
    link = numpy.identity(edge_count, dtype=numpy.uint8)
    vertices = range(vertex_count)
    detector_ids = tuple(vertices)
    defect_positions = {}
    for detector_id in detector_ids:
        defect_positions[detector_id] = (0, detector_id)
    return fault_models.WindowErrorModel(
        detector_ids=detector_ids,
        detector_coordinates=None,
        defect_positions=defect_positions,
        first_commit_round=0,
        graphlike_faults=graphlike,
        physical_faults=physical,
        physical_to_graphlike_detector_projection=link,
    )


def _deq_faults(representation, vertex_count: int, edges: tuple):
    """Owned fault columns of deq's edges, the vertices each flips.

    No column flips a logical observable, as no deq edge carries one.
    """
    edge_count = len(edges)
    check = numpy.zeros((vertex_count, edge_count), dtype=numpy.uint8)
    for column, edge in enumerate(edges):
        rows = list(edge)
        check[rows, column] = 1
    priors = [DEQ_EDGE_PROBABILITY] * edge_count
    observables = numpy.zeros((0, edge_count), dtype=numpy.uint8)
    owned = numpy.ones(edge_count, dtype=bool)
    columns = range(edge_count)
    source_fault_ids = tuple(columns)
    flips_by_column = enumerate(edges)
    boundary_flips = dict(flips_by_column)
    return fault_models.PlacedFaultModel(
        representation=representation,
        check=check,
        priors=priors,
        observables=observables,
        owned=owned,
        source_fault_ids=source_fault_ids,
        boundary_flips=boundary_flips,
    )


def _deq_syndrome(vertex_count: int, defects: tuple) -> numpy.ndarray:
    syndrome = numpy.zeros(vertex_count, dtype=numpy.uint8)
    rows = list(defects)
    syndrome[rows] = 1
    return syndrome


def _flipped_vertices(edges: tuple, selected_faults) -> tuple:
    """The vertices an odd number of the selected edges touch, in order.

    deq's is_parity_factor (blackbox_util.rs): each chosen edge toggles
    every vertex it holds, so a one-vertex edge flips its one vertex.
    """
    flipped = set()
    chosen = numpy.flatnonzero(selected_faults)
    for fault_index in chosen:
        edge = edges[fault_index]
        flipped.symmetric_difference_update(edge)
    ordered = sorted(flipped)
    return tuple(ordered)


def test_start_delivers_the_result_after_latency_ticks():
    job = _job()
    row = FixedRow()
    delivered = _started(row, job)
    assert len(delivered) == 1
    tick, result = delivered[0]
    assert tick == 3
    assert result.logical_observables == (1,)


def test_a_job_without_a_window_delivers_none():
    job = _job(on_done=lambda: None)
    row = FixedRow()
    delivered = _started(row, job)
    assert delivered == [(3, None)]


def test_a_measured_row_delivers_after_the_measured_ticks():
    job = _job()
    row = MeasuredRow()
    delivered = _started(row, job)
    assert len(delivered) == 1
    tick, result = delivered[0]
    assert tick == config.microseconds_to_ticks(2.5)
    assert result.window_id == 0


def test_occupancy_is_the_latency_of_a_priced_row():
    job = _job()
    row = FixedRow()
    assert row.occupancy(job) == 3


def test_occupancy_is_none_for_a_measured_row():
    job = _job()
    measured = MeasuredRow()
    assert measured.occupancy(job) is None
    window_row = EmptyWindowRow(latency_model=None)
    assert window_row.occupancy(job) is None


def test_a_window_row_without_a_latency_model_has_no_latency():
    job = _job()
    row = EmptyWindowRow(latency_model=None)
    with pytest.raises(NotImplementedError, match="a decoder measured"):
        row.latency(job)


def test_cancel_on_a_plain_row_changes_nothing():
    row = FixedRow()
    job = _job()
    row.cancel(job)
    delivered = _started(row, job)
    _tick, result = delivered[0]
    assert result.logical_observables == (1,)


def test_a_window_row_without_a_model_gives_the_empty_result():
    latency_model = decoders.PresetLatencyDecoder(1.0)
    row = EmptyWindowRow(latency_model=latency_model)
    job = _job()
    result = row.decode(job)
    assert result.correction is None
    assert result.logical_observables is None


def test_a_committed_fault_reaching_behind_the_window_is_a_crossing_commit():
    """The seam commit a region ending there and its own redo are pinned on.

    A window that owns the faults crossing behind its commit region
    carries that part of its correction on its own, because the residual
    XORs every committed fault's detectors together and the crossing
    part cannot be taken out of it again (Toshio et al. 2510.25222 lines
    1248-1250).
    """
    check = scipy.sparse.csc_matrix([[1, 0], [1, 1]], dtype=numpy.uint8)
    observables = scipy.sparse.csc_matrix([[1, 1]], dtype=numpy.uint8)
    placed = fault_models.PlacedFaultModel(
        representation=fault_models.FaultRepresentation.GRAPHLIKE,
        check=check,
        priors=[0.1, 0.2],
        observables=observables,
        owned=[True, True],
        source_fault_ids=[4, 9],
        boundary_flips={0: [0, 2], 1: [1]},
    )
    model = fault_models.WindowErrorModel(
        detector_ids=(0, 1),
        detector_coordinates=None,
        # detector 2 sits on round 5, the round before the commit region
        defect_positions={0: (6, 0), 1: (6, 1), 2: (5, 0)},
        first_commit_round=6,
        graphlike_faults=placed,
        physical_faults=None,
    )
    job = _job()
    result = decoder_module.result_from_selected_faults(
        job, model, placed, [1, 1]
    )
    assert result.logical_observables == (0,)
    assert result.boundary_data.detector_ids == (0, 1, 2)
    assert result.crossing_commit.residual.detector_ids == (0, 2)
    assert result.crossing_commit.logical_observables == (1,)


def test_a_window_result_carries_the_detection_events_its_decode_read():
    """A confidence signal reads the window's events off its result."""
    circuit = windows.memory_circuit(3, 3, 0.05)
    requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    model = windows.whole_circuit_window(circuit, 3, requirement)
    detection_events, _observables = windows.sampled_shots(circuit, 1, 5)
    shot = detection_events[0]
    job = windows.job_for(model, shot)

    row = NoFaultRow()
    result = row.decode(job)

    expected = windows.row_syndrome(model, shot)
    assert expected.any()
    assert numpy.array_equal(result.detection_events, expected)
    assert not result.detection_events.flags.writeable


def test_a_preset_latency_takes_its_later_latency_from_its_window_on():
    settings = decoders.PresetLatencyDecoder.Settings(
        2.0, later_latency_microseconds=3.0, from_window=4
    )
    row = settings.build()
    before = decoding_records.DecodeJob(stream_id=1, window_id=3, round_count=2)
    at = decoding_records.DecodeJob(stream_id=1, window_id=4, round_count=2)

    latencies = (row.latency(before), row.latency(at))

    assert latencies == (
        config.microseconds_to_ticks(2.0),
        config.microseconds_to_ticks(3.0),
    )


def test_a_later_latency_without_its_window_is_refused():
    with pytest.raises(ValueError, match="go together"):
        decoders.PresetLatencyDecoder.Settings(
            2.0, later_latency_microseconds=3.0
        )


@pytest.mark.parametrize("settings", GRAPH_ROWS)
@pytest.mark.parametrize("vertex_count, edges, defects", DEQ_GRAPH_PROBLEMS)
def test_the_faults_a_graph_row_selects_reproduce_each_deq_syndrome(
    settings, vertex_count, edges, defects
):
    row = _seeded_row(settings)
    model = _deq_window(vertex_count, edges)
    syndrome = _deq_syndrome(vertex_count, defects)
    job = windows.job_for(model, syndrome)

    result = row.decode(job)

    edge_count = len(edges)
    flipped = _flipped_vertices(edges, result.selected_faults)
    assert result.selected_faults.shape == (edge_count,)
    assert flipped == defects


@pytest.mark.parametrize("settings", HYPERGRAPH_ROWS)
@pytest.mark.parametrize("vertex_count, edges, defects", DEQ_PROBLEMS)
def test_the_faults_a_hypergraph_row_selects_reproduce_each_deq_syndrome(
    settings, vertex_count, edges, defects
):
    row = _seeded_row(settings)
    model = _deq_window(vertex_count, edges)
    syndrome = _deq_syndrome(vertex_count, defects)
    job = windows.job_for(model, syndrome)

    result = row.decode(job)

    edge_count = len(edges)
    flipped = _flipped_vertices(edges, result.selected_faults)
    assert result.selected_faults.shape == (edge_count,)
    assert flipped == defects

"""decsim's Union-Find as a sinter decoder.

The adapter reads the graphlike faults with the machine's catalog
(detector_error_model/stim_fault_catalog.py), builds the union_find
row's weighted graph and decodes every shot with the same compiled
growth and peeling, so an offline run answers as the row does. It
refuses what the row cannot decode: a fault that flips an observable and
no detector, and an undecomposed hyperedge, which sinter hands on when
decomposition fails (sinter/_collection/_collection_worker_state.py:28-33).
"""

import numpy
import scipy.sparse
import sinter
import stim

import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.detector_error_model.fault_identity_validation as fault_identities
import decsim.detector_error_model.stim_fault_catalog as stim_fault_catalog
import decsim.records.decoder_evidence as evidence_records
import decsim.records.fault_model_contracts as fault_models


class UnionFindDecoder(sinter.Decoder):
    """The factory sinter pickles to each worker.

    sinter compiles it once per task. weight_step is checked where it
    enters, as the union_find row checks its own
    (decoders/union_find/decoder.py).
    """

    def __init__(
        self, weight_step: float = evidence_records.DEFAULT_WEIGHT_STEP
    ) -> None:
        self.weight_step = evidence_records.normalized_weight_step(weight_step)

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> "CompiledUnionFindDecoder":
        """The graph of the model; sinter names the keyword dem."""
        faults = _whole_model_faults(dem)
        graph = window_decoder.graph_from_model(
            faults,
            location="sinter's detector error model",
            weight_step=self.weight_step,
        )
        return CompiledUnionFindDecoder(graph)


class CompiledUnionFindDecoder(sinter.CompiledDecoder):
    """One task's graph, decoding bit-packed shots one at a time."""

    def __init__(self, graph: evidence_records.UnionFindGraph) -> None:
        self.graph = graph

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: numpy.ndarray
    ) -> numpy.ndarray:
        """Each shot's predicted observable flips, bit packed as sinter's."""
        events = numpy.unpackbits(
            bit_packed_detection_event_data,
            axis=1,
            count=self.graph.detector_count,
            bitorder="little",
        )
        shot_count = events.shape[0]
        shape = (shot_count, self.graph.logical_observable_count)
        predictions = numpy.zeros(shape, dtype=numpy.uint8)
        for shot, syndrome in enumerate(events):
            evidence = window_decoder.decode_graph(self.graph, syndrome)
            predictions[shot] = evidence.logical_observables
        return numpy.packbits(predictions, axis=1, bitorder="little")


def _whole_model_faults(
    detector_error_model: stim.DetectorErrorModel,
) -> fault_models.PlacedFaultModel:
    """The model as one window that holds every detector and owns every fault.

    The machine's whole-circuit window (windows/schemes/naive_online.py)
    places the same catalog with its rows in detector order and its
    columns in catalog order, which this keeps.
    """
    detector_sets, observable_sets, priors = (
        stim_fault_catalog.detector_error_model_to_faults(detector_error_model)
    )
    _refuse_hyperedges(detector_sets, observable_sets)
    check = _incidence(detector_sets, detector_error_model.num_detectors)
    observables = _incidence(
        observable_sets, detector_error_model.num_observables
    )
    fault_count = len(priors)
    owned = numpy.ones(fault_count, dtype=bool)
    fault_ids = range(fault_count)
    return fault_models.PlacedFaultModel(
        representation=fault_models.FaultRepresentation.GRAPHLIKE,
        check=check,
        priors=priors,
        observables=observables,
        owned=owned,
        source_fault_ids=tuple(fault_ids),
        boundary_flips={},
    )


def _refuse_hyperedges(detector_sets: list, observable_sets: list) -> None:
    """Every fault graphlike, by the check the machine's catalog makes."""
    faults = zip(detector_sets, observable_sets, strict=True)
    for fault_index, (detectors, observables) in enumerate(faults):
        fault_identities.validate_graphlike_fault(
            detectors,
            observables,
            location=f"fault {fault_index} of sinter's detector error model",
        )


def _incidence(member_sets: list, row_count: int) -> scipy.sparse.csc_matrix:
    """Rows by faults, a one where fault j's set holds row i."""
    rows = []
    columns = []
    for column, members in enumerate(member_sets):
        for member in members:
            rows.append(member)
            columns.append(column)
    ones = numpy.ones(len(rows), dtype=numpy.uint8)
    shape = (row_count, len(member_sets))
    return scipy.sparse.csc_matrix((ones, (rows, columns)), shape=shape)

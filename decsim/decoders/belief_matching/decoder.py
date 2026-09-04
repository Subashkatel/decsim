"""The belief-matching adapter: the strong tier's decoder on one window.

Toshio et al. 2510.25222 (tmp/papers) run belief matching as the
accurate decoder invoked on demand; the algorithm is the window
decoder's, here wrapped for the machine.
"""

import time
from typing import Optional

import numpy

import decsim.decoders.belief_matching.window_decoder as window_decoder
import decsim.decoders.window_decode_results as window_decode_results
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.message as message
import decsim.ports as ports


class BeliefMatchingDecoder:
    """Decode one hyperedge-bearing window with belief matching.

    Simulated latency comes from a latency model, or, with
    ``latency_model=None``, from the measured wall clock of the
    BP-plus-matching call itself (software decoder on this host, the
    PyMatchingDecoder pattern). Wide state (7 attributes) recorded for
    the port commit, which folds the window decoder and the warm-up set
    into the base class.
    """

    fault_model_requirement = fault_models.LINKED_FAULT_MODELS_REQUIRED

    def __init__(
        self,
        latency_model: Optional[ports.Decoder] = None,
        max_iterations: int = 30,
        belief_propagation_method: str = "product_sum",
    ):
        self.latency_model = latency_model
        self.measures_wall_clock = latency_model is None
        self.last_decode_ns: Optional[int] = None
        self._warmed_models: set = set()
        self.max_iterations = max_iterations
        self.belief_propagation_method = belief_propagation_method
        self.window_decoder = window_decoder.belief_matching_window_decoder(
            max_iterations=max_iterations,
            belief_propagation_method=belief_propagation_method,
        )

    def run_seed_children(self) -> tuple:
        """The latency model that controls simulated service time."""
        path = (message.RunSeedPathSegment("field", "latency_model"),)
        child = message.RunSeedChild(path, self.latency_model)
        return (child,)

    def latency(self, job: message.DecodeJob) -> int:
        """Timing comes from the wrapped latency model."""
        if self.measures_wall_clock:
            raise RuntimeError(
                "measured wall-clock timing needs the DecoderEngine: it "
                "decodes first and charges the measured time"
            )
        return self.latency_model.latency(job)

    def decode(self, job: message.DecodeJob) -> message.DecodeResult:
        """Run windowed belief matching on the job's window model."""
        model = job.dem
        if model is None:
            return message.DecodeResult(job.op_id, job.window_id)
        faults = model.require_faults(
            fault_models.FaultRepresentation.GRAPHLIKE
        )
        model.require_faults(fault_models.FaultRepresentation.PHYSICAL)
        if model.physical_to_graphlike_detector_projection is None:
            raise ValueError(
                f"{job.label}: BeliefMatchingDecoder needs the physical-to-"
                "graphlike link"
            )
        syndrome = window_decode_results.payload_syndrome(job)
        window_decode_results.check_syndrome_size(job, syndrome, faults)
        self._warm_up(model, syndrome)
        selected, decode_status = self._timed_window_decode(
            model, syndrome, faults
        )
        return window_decode_results.result_from_selected_faults(
            job, model, faults, selected, decode_status=decode_status
        )

    def _warm_up(self, model, syndrome) -> None:
        """Build the model's BP decoder before the first timed call.

        Measured mode times the BP-plus-matching call only, never the
        one-time window-model construction (validation, BP build), the
        same contract as PyMatchingDecoder's warm-up.
        """
        if not self.measures_wall_clock:
            return
        model_identity = id(model)
        if model_identity in self._warmed_models:
            return
        empty_syndrome = numpy.zeros_like(syndrome)
        self.window_decoder(model, empty_syndrome)
        self._warmed_models.add(model_identity)

    def _timed_window_decode(self, model, syndrome, faults) -> tuple:
        """(selected faults, status) of one timed call.

        PyMatching raises on odd parity in a boundaryless component (see
        mwpm); that case is an empty correction marked invalid.
        """
        started_ns = time.perf_counter_ns()
        try:
            selected = self.window_decoder(model, syndrome)
            decode_status = None
        except ValueError as error:
            if "perfect matching" not in str(error):
                raise
            fault_count = faults.check.shape[1]
            selected = numpy.zeros(fault_count, dtype=numpy.uint8)
            decode_status = (
                window_decode_results.BackendDecodeStatus.INVALID_CORRECTION
            )
        finally:
            finished_ns = time.perf_counter_ns()
            self.last_decode_ns = finished_ns - started_ns
        return selected, decode_status

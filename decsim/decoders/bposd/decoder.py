"""The BP-OSD adapter: qLDPC's default decoder as a tier of its own."""

import decsim.decoders.bposd.window_decoder as window_decoder
import decsim.decoders.window_decode_results as window_decode_results
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.message as message
import decsim.ports as ports


class BPOSDDecoder:
    """Decode one window with BP-OSD and report simulated latency separately.

    Wide state (7 attributes) recorded for the port commit, which folds
    the window decoder into the base class.
    """

    fault_model_requirement = fault_models.PHYSICAL_FAULT_MODEL_REQUIRED

    def __init__(
        self,
        latency_model: ports.Decoder,
        max_iterations: int = 2,
        osd_order: int = 0,
        belief_propagation_method: str = "product_sum",
        schedule: str = "serial",
        osd_method: str = "osd_cs",
    ):
        self.latency_model = latency_model
        self.max_iterations = max_iterations
        self.osd_order = osd_order
        self.belief_propagation_method = belief_propagation_method
        self.schedule = schedule
        self.osd_method = osd_method
        self.window_decoder = window_decoder.bposd_window_decoder(
            max_iterations=max_iterations,
            osd_order=osd_order,
            belief_propagation_method=belief_propagation_method,
            schedule=schedule,
            osd_method=osd_method,
        )

    def run_seed_children(self) -> tuple:
        """The latency model that controls simulated service time."""
        path = (message.RunSeedPathSegment("field", "latency_model"),)
        child = message.RunSeedChild(path, self.latency_model)
        return (child,)

    def latency(self, job: message.DecodeJob) -> int:
        """Timing comes from the wrapped latency model."""
        return self.latency_model.latency(job)

    def decode(self, job: message.DecodeJob) -> message.DecodeResult:
        """Run windowed BP-OSD on the job's window model."""
        model = job.dem
        if model is None:
            return message.DecodeResult(job.op_id, job.window_id)
        faults = model.require_faults(fault_models.FaultRepresentation.PHYSICAL)
        syndrome = window_decode_results.payload_syndrome(job)
        window_decode_results.check_syndrome_size(job, syndrome, faults)
        selected = self.window_decoder(model, syndrome)
        return window_decode_results.result_from_selected_faults(
            job, model, faults, selected
        )

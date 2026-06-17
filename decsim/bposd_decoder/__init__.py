"""BP-OSD (belief propagation + ordered-statistics decoding) for the harness, backed by the
`ldpc` library. The inner decoder for BB / qLDPC windows, whose faults may flip > 2 detectors
(build the window models with decompose_errors=False; matching does not apply).

decsim/decsim's OWN adapter around `ldpc.BpOsdDecoder`. Mirrors decsim.mwpm_decoder's two faces:

  * ``BPOSDDecoder`` (decoder.py) -- the runtime harness Decoder: real windowed BP-OSD over
    job.dem with commit-region ownership + artificial-defect handoff (latency(job)/decode(job)
    protocol), runnable under any windowing scheme for the surface DEMs the cluster builds.
  * ``bposd_window_decoder`` (window_decoder.py) -- the inner ``decode_window(model, syndrome)``
    callable for the offline ``decode_windowed`` reference / windowed studies (the path to use
    for full-DEM qLDPC BP-OSD, decompose_errors=False).

The CODE-AGNOSTIC windowing engine they build on (``WindowErrorModel``, ``build_window_error_models``,
``decode_windowed``, ``detector_error_model_to_faults``) is shared with the matching / belief-matching
inner decoders, so it stays in ``decsim.adapters.window_error_models`` -- it is not BP-OSD-specific.

Requires the optional ``ldpc`` dependency.
"""
from .decoder import BPOSDDecoder
from .window_decoder import bposd_window_decoder

__all__ = ["BPOSDDecoder", "bposd_window_decoder"]

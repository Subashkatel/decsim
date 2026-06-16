"""MWPM (minimum-weight perfect matching) decoding for the harness, backed by PyMatching.

This is qecsim/decsim's OWN adapter code around the `pymatching` library -- NOT vendored
third-party code (unlike decsim.uf_decoder). It gathers the two pymatching-specific pieces that
were previously split across decsim.adapters:

  * ``PyMatchingDecoder`` (decoder.py) -- the runtime harness Decoder: per-window DEM matching with
    commit-region ownership and artificial-defect handoff (latency(job)/decode(job) protocol).
  * ``matching_window_decoder`` (window_decoder.py) -- the inner ``decode_window(model, syndrome)``
    callable for the offline ``decode_windowed`` reference.

The CODE-AGNOSTIC windowing engine they build on (``WindowErrorModel``, ``build_window_error_models``,
``decode_windowed``, ``detector_error_model_to_faults``) is shared with BP-OSD and other inner
decoders, so it stays in ``decsim.adapters.window_error_models`` -- it is not pymatching-specific.

Requires the optional ``pymatching`` dependency:  pip install decsim[pymatching]
"""
from .decoder import PyMatchingDecoder
from .window_decoder import matching_window_decoder

__all__ = ["PyMatchingDecoder", "matching_window_decoder"]

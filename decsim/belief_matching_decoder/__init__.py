"""Belief-matching decoding for the harness (Higgott & Gidney, arXiv:2203.04948): belief
propagation on the UNDECOMPOSED hypergraph, then MWPM on the BP-reweighted matching graph.
This is the *strong decoder* of the decoder-switching paper (arXiv:2510.25222).

decsim/decsim's OWN adapter around that algorithm, using the `ldpc` library for the BP pass
and `pymatching` for the MWPM pass (NOT the `beliefmatching` package, which returns observables
rather than the per-fault correction the windowed commit needs). Mirrors decsim.mwpm_decoder's
two faces:

  * ``BeliefMatchingDecoder`` (decoder.py) -- the runtime harness Decoder: real windowed
    belief-matching over job.dem with commit-region ownership + artificial-defect handoff
    (latency(job)/decode(job) protocol), runnable under any windowing scheme.
  * ``belief_matching_window_decoder`` (window_decoder.py) -- the inner
    ``decode_window(model, syndrome)`` callable for the offline ``decode_windowed`` reference
    and windowed-decoding studies. Returns the EDGE selection, so the shared windowing engine's
    commit-region ownership + artificial-defect handoff are unchanged.

The CODE-AGNOSTIC windowing engine they build on (``WindowErrorModel`` incl. the belief-matching
hyperedge fields, ``build_window_error_models(belief_matching=True)``, ``decode_windowed``,
``detector_error_model_to_faults_bm``) is shared with the matching / BP-OSD inner decoders, so it
stays in ``decsim.adapters.window_error_models`` -- it is not belief-matching-specific.

Requires the optional ``pymatching`` + ``ldpc`` dependencies.
"""
from .decoder import BeliefMatchingDecoder
from .window_decoder import belief_matching_window_decoder

__all__ = ["BeliefMatchingDecoder", "belief_matching_window_decoder"]

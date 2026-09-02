"""Slices a circuit's Stim detector error model into per-window decoder inputs.

The measured bits change every shot; the sliced error models are built
once per operation and shared across shots. The window decoders (matching,
belief propagation with ordered statistics, belief matching, union find,
Tesseract) consume them; decsim/qpu/stim_device.py builds them per
operation. detector_formation is the other half of the same reading of a
circuit: it turns raw measurement packets into the detection events those
windows decode.

Modules, each importing only the ones above it:

    fault_model_contracts       what a decoder is handed: representations,
                                requirements, placed faults, window models
    fault_identity_validation   identities reduced modulo two; matrix checks
    detector_chronology         which round each detector belongs to
    detector_formation          raw bits to detection events, per round
    stim_fault_catalog          the whole-circuit fault catalog from Stim
    window_placement            one window's rows, columns and ownership
    window_slicer               the catalog sliced window by window
    window_ownership_dag        one owner per fault from a dependency graph
    window_protocol_policy      what a plan must look like per protocol
    window_model_builders       the entry points that slice a plan

This file holds the docstring and nothing else. Every consumer imports the
module that owns the name it wants, so reading the contract never loads
the Stim parser or the slicer.

The fault representations and the window protocols are closed sets: the
two FaultRepresentation members and the two WindowProtocol members are
built in, and a third of either needs an edit to fault_model_contracts or
window_protocol_policy.
"""

"""Slice a global Stim detector error model into per-window decoder inputs.

The measured bits change every shot; the sliced error models are compile-time
data shared across shots.  Consumed by the window decoders (MWPM / BP+OSD /
belief matching) and built per op by adapters/stim_device.py.

Modules, in strict one-way import order (a module never imports its own or a
higher level):

L0  fault_model_contracts      decoder-facing vocabulary, requirement
                               declaration and the immutable window products
L0  fault_identity_validation  parity reduction and decoder-input matrix and
                               domain validators
L0  detector_chronology        emitted rounds, within-round positions, row
                               coordinates
L1  stim_dem_catalog           Stim canonicalisation and global fault-catalog
                               compilation
L1  window_placement           per-window geometry, ownership policy, array
                               placement and the window-local link projection
L2  window_slicer              the slicing engine binding catalogs, chronology
                               and placement into WindowErrorModel products
L3  window_ownership_dag       one fault owner per window from the dependency
                               DAG
L3  window_protocol_policy     window-protocol and temporal-boundary policy
L4  window_model_builders      the caller-facing plan and single-window
                               builders

This ``__init__`` is a docstring and nothing else: zero imports, zero
assignments, zero ``__all__``.  Nothing is re-exported here, so every consumer
imports the submodule that owns the symbol it wants and no consumer drags the
Stim parser or the slicer in with a contracts-only import.

The fault-domain and window-protocol seams are CLOSED, not pluggable: the two
FaultRepresentation members and the two accepted WindowProtocol members are
built in, and adding a third of either requires editing fault_model_contracts
and window_protocol_policy respectively.  This package is not an extensible
fault-domain framework.
"""

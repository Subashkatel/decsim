"""Slices a circuit's Stim detector error model into per-window decoder inputs.

The sliced error models are built once per operation by
decsim/qpu/stim_device.py and shared across shots by the window decoders;
detector_formation is the other half of the same reading of a circuit,
turning raw measurement packets into the detection events those windows
decode. This file holds the docstring and nothing else: every consumer
imports the module that owns the name it wants.
"""

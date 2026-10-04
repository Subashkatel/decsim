"""Slices a circuit's Stim detector error model into per-window decoder inputs.

The window models are built once per operation (qpu/stim_device.py) and
shared across shots; detector_formation turns raw measurement packets
into the detection events those windows decode.
"""

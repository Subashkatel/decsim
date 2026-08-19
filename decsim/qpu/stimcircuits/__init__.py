"""Stim circuit generators for surface, toric, and repetition codes.

Vendored third-party code, not decsim's own authorship:
  Copyright 2022 Oscar Higgott. Licensed under the Apache License, Version 2.0 (see LICENSE).
  Source: https://github.com/oscarhiggott/stimcircuits.
  Full provenance and local packaging notes are in NOTICE.md.

Exposes ``generate_circuit(code_task, distance=..., rounds=..., ...)`` returning a
``stim.Circuit`` for tasks such as "surface_code:rotated_memory_x",
"toric_code:unrotated_memory_x", or "repetition_code:memory". stim circuits are a clean,
well-tested substrate, so decsim uses them directly for code/noise generation.

Requires the optional ``stim`` dependency.
"""
from .surface_code import generate_circuit
from .noise import NoiseModel

__all__ = ["generate_circuit", "NoiseModel"]

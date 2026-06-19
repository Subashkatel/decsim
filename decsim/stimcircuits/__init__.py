"""Stim circuit generators for surface / toric / repetition codes.

VENDORED third-party code -- NOT decsim's own authorship:
  Copyright 2022 Oscar Higgott. Licensed under the Apache License, Version 2.0 (see LICENSE).
  Source: https://github.com/oscarhiggott/stimcircuits  (not on PyPI, so vendored).
  surface_code.py is upstream-unchanged (its Apache header is preserved); only this __init__
  is ours (relative import). Full provenance + modifications in NOTICE.md.

Exposes ``generate_circuit(code_task, distance=..., rounds=..., ...)`` returning a
``stim.Circuit`` for tasks such as "surface_code:rotated_memory_x",
"toric_code:unrotated_memory_x", or "repetition_code:memory". stim circuits are a clean,
well-tested substrate, so decsim uses them directly for code/noise generation.

Requires the optional ``stim`` dependency:  pip install decsim[stim]
"""
from .surface_code import generate_circuit
from .noise import NoiseModel

__all__ = ["generate_circuit", "NoiseModel"]

"""Stim circuit generators vendored from Oscar Higgott's stimcircuits.

Copyright 2022 Oscar Higgott, Apache License 2.0; the license and the
provenance notes are LICENSE and NOTICE.md in this folder. surface_code.py
is the vendored generator; noise.py holds decsim's noise presets on top
of it. Nothing in decsim calls either: every caller builds its circuit
with stim.Circuit.generated.
"""

# Third-party code: `stimcircuits`

This subpackage is **vendored third-party code**, not part of qecsim's own authorship.

- **Copyright** 2022 Oscar Higgott
- **License** Apache License, Version 2.0 — full text in [LICENSE](LICENSE); SPDX: `Apache-2.0`
- **Source** https://github.com/oscarhiggott/stimcircuits
- **Why vendored** it is not published on PyPI, so it cannot be pinned as a normal
  dependency. We copy it here to make stim circuit generation a first-class, offline part of
  qecsim (stim circuits are clean and well-tested; we use them directly for code/noise setup).
- **Modifications**
  - `surface_code.py` — unchanged from upstream; its Apache-2.0 file header is preserved.
  - `__init__.py` — written by us; uses a relative import (`from .surface_code import
    generate_circuit`) appropriate to its place inside the qecsim package.

Requires the optional `stim` dependency (`pip install qecsim[stim]`).

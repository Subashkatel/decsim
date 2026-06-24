# Third-party code: `stimcircuits`

This subpackage is **vendored third-party code**, not part of decsim's own authorship.

- **Copyright** 2022 Oscar Higgott
- **License** Apache License, Version 2.0. Full text in [LICENSE](LICENSE). SPDX: `Apache-2.0`
- **Source** https://github.com/oscarhiggott/stimcircuits
- **Why vendored** it is not published on PyPI, so it cannot be pinned as a normal
  dependency. We copy it here to make stim circuit generation a first-class, offline part of
  decsim (stim circuits are clean and well-tested; we use them directly for code/noise setup).
- **Modifications**
  - `surface_code.py` keeps the upstream algorithm and Apache-2.0 header. decsim adds a short
    module docstring only.
  - `__init__.py` is written by us; uses a relative import (`from .surface_code import
    generate_circuit`) appropriate to its place inside the decsim package.
  - `noise.py` is written by us; it carries decsim's Stim noise presets.

Requires the optional `stim` dependency (`pip install decsim[stim]`).

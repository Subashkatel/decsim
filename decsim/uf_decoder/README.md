# `decsim.uf_decoder` — ported third-party decoder

This subpackage is **ported third-party code, NOT decsim's own.** It is the base
**Union-Find + peeling decoder** from the *Coset Ensemble Decoder*, copied here so decsim can drive
the real decoder offline.

- **Origin:** github.com/IMSeonL/coset-ensemble-decoder — arXiv:2606.11076 (Coset Ensemble Decoder).
- **Decode output is byte-identical to the original** (verified: 2100 shots over 5 `(d, p)` cases
  incl. high-noise, 0 mismatches).

## Files

| File | Source | Role |
|---|---|---|
| `software/uf_original.py` | upstream `software/uf_original.py` | the UF clustering + peeling algorithm |
| `tools/post_precessing.py` | upstream `tools/post_precessing.py` | `edge_to_qubit_index` (peeling edge → qubit column) |
| `code_structure.py` | excerpt of upstream `uf_decoder.py` | the `CodeStructure` input class the decoder requires |
| `codes.py` | verbatim from upstream `codes.py` | repetition + toric builders (scipy-only); plus `planar_code_extract`/`rotated_code_extract`, which lazily import **Tuckett's `qecsim`** (the separate PyPI package). Toric/repetition need no Tuckett; planar/rotated need `pip install qecsim`. |


> Planar/rotated construction calls **Tuckett's `qecsim`** — a *different* PyPI package. This project
> was renamed `qecsim` → `decsim` precisely so the two coexist: `import decsim` (this project) vs
> `import qecsim` (Tuckett's, a normal dependency). Nothing of Tuckett's is copied into this repo.

## Using it from the harness

The decsim DES adapter is **not** here — it lives with the other decoders in
`decsim/decoders.py` as `UnionFindDecoder` (ours). It wraps this package's `uf_original` +
`CodeStructure` and implements the harness `latency(job)` / `decode(job)` protocol. For toric codes
it is fully self-contained:

```python
from decsim.decoders import UnionFindDecoder
dec = UnionFindDecoder.for_toric(L, circuit, latency_model, channel="x")  # builds matrices+remap from decsim
```


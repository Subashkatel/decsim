# `decsim.soft_output` — soft-output metrics for decoder switching

A soft output `g` is a per-window confidence (Toshio et al. 2510.25222): a non-negative
score, smaller = less confident. Two interchangeable metrics live behind one seam.

## The seam
`decsim.protocols.SoftOutputMetric` (a `typing.Protocol`):
```python
class SoftOutputMetric(Protocol):
    name: str
    def evaluate(self, syndrome) -> SoftOutput: ...   # -> (logical_value, gap, w_min, w_comp)
```
`SoftOutput` is in `decsim.message`. Build a metric from a stim DEM (`from_dem`) or a decsim
`WindowErrorModel` (`from_window_model`), then call `evaluate(syndrome)` per shot.

## Metrics (same interface, swap one for the other)
- **`ComplementaryGapMetric`** (`complementary.py`) — `g_comp = |w_comp - w_min|` for MWPM:
  decode normally to `w_min`, force the opposite logical class to `w_comp`. Reproduces the
  paper's P(error|g) and γ(g_th)-vs-d.
- **`ClusterGapMetric`** (`cluster.py`) — the cluster gap for Union-Find (Meister et al.
  2405.07433, Def. 9 / Alg. 2): UF clusters → quotient graph (within-cluster edges = 0) →
  shortest logical operator between the two inequivalent boundaries (parity-doubled on the
  observable). The grown-ball fill (crediting the outward frontier) makes the signature
  inequality `g_cluster ≤ g_comp` hold on ≥99% of shots; `evaluate(robust=True)` (the
  duality-gap correction) holds it in the UF-suboptimal regime (higher rate / larger d).
  Pinned by `tests/test_soft_output_cluster.py`. (Hard `logical_value` needs
  `faults_matrix=model.obs` in `from_window_model`.)

## Landing it in the data path
`SoftOutputDecoder(base, metric_cls)` (`decoder.py`) wraps any `decsim.protocols.Decoder`
and sets `DecodeResult.soft_output = g` on the same window graph the hard decoder uses, so
the `decsim.switching.Switching` policy (keep weak iff `g ≥ g_th`) runs on the confidence:
```python
decoder = SoftOutputDecoder(PyMatchingDecoder(latency), ComplementaryGapMetric)
```

## Note on graphs
- The data-path metric uses `from_window_model` (decsim's merged-fault canonical graph,
  consistent with `PyMatchingDecoder`'s hard decode).
- The paper-reproduction uses `from_dem` (decomposed, matching the reference). They differ
  by ~0.8% of predictions; both are valid complementary gaps.
- The complementary-gap augment needs a **graphlike-observable** DEM (`decompose_errors=True`);
  `qlx.dem_of` / `emit_decoder_params` may place a 2-detector edge on the observable — use the
  circuit's stim-decomposed DEM there.

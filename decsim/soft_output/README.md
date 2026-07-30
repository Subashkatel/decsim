# `decsim.soft_output` — soft-output metrics for decoder switching

A soft output `g` is a per-window confidence (Toshio et al. 2510.25222v1): a
non-negative score, smaller = less confident. The confidence mechanism is paired
with the weak decoder that produces the hard result.

## The seam
`decsim.protocols.SoftOutputMetric` (a `typing.Protocol`):
```python
class SoftOutputMetric(Protocol):
    name: str
    def evaluate(self, syndrome) -> SoftOutput: ...   # -> (gap, source, optional weights)
```
`SoftOutput` is in `decsim.message`. Build a metric from a stim DEM (`from_dem`) or a decsim
`WindowErrorModel` (`from_window_model`), then call `evaluate(syndrome)` per shot.

## Decoder/confidence pairs

- **`ComplementaryGapMetric`** (`complementary.py`) — `g_comp = |w_comp - w_min|` for MWPM:
  decode normally to `w_min`, force the opposite logical class to `w_comp`. Reproduces the
  paper's P(error|g) and γ(g_th)-vs-d.
- **`UnionFindClusterGapDecoder`** (`cluster.py`) — confidence wrapper for
  `decsim.union_find_decoder.UnionFindDecoder`. It quotients the weighted edge
  intervals returned by that exact hard decode and searches for the shortest
  odd-logical closed walk. The hard decode follows Delfosse and Nickerson
  arXiv:1709.06218v3; confidence follows Meister et al.
  arXiv:2405.07433v2, Definition 9 and Algorithm 2.

Hard-only Union-Find is a normal decoder endpoint:
```python
from decsim.union_find_decoder import UnionFindDecoder
```
The paper cluster-gap pairing is explicit:
```python
from decsim.soft_output import UnionFindClusterGapDecoder
from decsim.union_find_decoder import UnionFindDecoder

decoder = UnionFindClusterGapDecoder(UnionFindDecoder(latency))
```

The hard decoder accepts graphlike/stringlike faults, uses prior-weighted global
fair growth, and supports every logical-observable row. Cluster gap requires
exactly one nonzero logical row and reports decibels. Its exact likelihood-ratio
interpretation applies only to the uniform repetition-code setting of Meister
et al., Theorem 10. Surface-code gap is confidence, not a calibrated failure
probability or a general Union-Find likelihood bound. Host execution time is not
simulated decoder latency, and this Python implementation makes no claim to the
papers' complexity bounds.

`SoftOutput` contains confidence only. Its immutable `source` identifies the
method, cluster origin, growth schedule, gap units, correction variant, and
references required to interpret its threshold. The hard logical prediction
has one owner: `DecodeResult.logical_observables`.

## Landing it in the data path
`SoftOutputDecoder(base, metric_cls)` (`decoder.py`) wraps an MWPM-family
`decsim.protocols.Decoder` with a configured metric builder. The builder
declares one `source`, one `fault_model_requirement`, and
`from_window_model(model)`. `SoftOutputDecoder` sets
`DecodeResult.soft_output` to typed confidence from the same placed window
model the hard decoder uses. The A2 behavior-child path remains
`field("metric_cls")`.

`decsim.switching.Switching` keeps the weak result iff `gap ≥ threshold`,
after requiring the confidence source to equal the threshold source exactly:
```python
decoder = SoftOutputDecoder(
    PyMatchingDecoder(latency),
    ComplementaryGapMetricFactory(),
)
strategy = Switching(
    confidence_threshold=8.0,
    expected_source=COMPLEMENTARY_GAP_SOURCE,
)
```

## Note on graphs
- The data-path metric uses `from_window_model` (decsim's merged-fault canonical graph,
  consistent with `PyMatchingDecoder`'s hard decode).
- The paper-reproduction uses `from_dem` (decomposed, matching the reference). They differ
  by ~0.8% of predictions; both are valid complementary gaps.
- The complementary-gap augment needs a **graphlike-observable** DEM (`decompose_errors=True`);
  `qlx.dem_of` / `emit_decoder_params` may place a 2-detector edge on the observable — use the
  circuit's stim-decomposed DEM there.

# tests/data — frozen test-data corpus

Pre-generated, version-controlled inputs so tests run on **fixed data** instead of
sampling stim live each run. Live sampling (with a seed) is only reproducible *within* a
stim version; freezing the shots makes decode tests deterministic across versions and
faster.

## Files

| file | what it is | used by |
| --- | --- | --- |
| `bb72_12_6_p003_r10.stim` | [[72,12,6]] bivariate-bicycle memory circuit, QUITS-built, frozen | `test_window_error_models.py` (BP-OSD windowed) |
| `rsc-d{3,5}-*.stim` | rotated surface-code memory circuits (frozen noise model) | `test_golden_decoding.py` |
| `rsc-d{3,5}-*.shots.npz` | N frozen detection-event samples: `dets` (N×n_det uint8) + `obs` (N×n_obs uint8) | `test_golden_decoding.py` |
| `golden_decoding.json` | per scenario: exact failure counts our decoders (global MWPM, windowed MWPM, windowed belief-matching) produce on the frozen shots, + the library versions | `test_golden_decoding.py` |
| `make_fixtures.py` | the run-once generator (documents exactly how every fixture was made) | — |

## Regenerating

The goldens are tied to the `stim` / `pymatching` / `ldpc` versions recorded in
`golden_decoding.json`. After a deliberate change (new scenario, or a library bump that
legitimately shifts a few decodes), rebaseline with:

```bash
python tests/data/make_fixtures.py
```

Add scenarios by appending to `SCENARIOS` in `make_fixtures.py` and re-running.

## Coverage today / to grow

Covered: rotated surface code d=3,5 under circuit-level depolarizing noise; global vs
windowed MWPM vs windowed belief-matching; the Skoric/Tan windowed==global anchor; the
qLDPC (BB) windowed BP-OSD path (bb72). Natural extensions: more distances/noise rates,
toric + Union-Find decodes, multi-operation / feedback circuits, and frozen **golden engine
traces** (a canonical end-to-end DES run pinned for byte-level regression).

# Anchor comparison against published numbers

## 1. Software decoder time, microseconds per shot (memory_x, p = 0.001, rounds = d)

| d | published (M1 Max, decode_batch) | this host (decode_batch, same method) | host/M1 | decsim decode() per window (median) | decsim/host |
|---|---|---|---|---|---|
| 5 | 0.388 | 0.312 | 0.80 | 39.2 | 125.6 |
| 7 | 1.055 | 1.046 | 0.99 | 50.2 | 48.0 |
| 9 | 2.315 | 2.460 | 1.06 | 77.0 | 31.3 |
| 13 | 6.966 | 7.646 | 1.10 | 169.2 | 22.1 |
| 17 | 16.006 | 17.969 | 1.12 | 322.9 | 18.0 |

host/M1 is the CPU-speed ratio and should be roughly constant across d; decsim/host is the cost of one Python decode() call per window over the batched C++ path, which is why the simulator charges a MODELED algorithm latency (LILLIPUT card or a measured software figure), never its own wall clock.

## 2. Logical error rate, windowed loop vs whole-circuit PyMatching (memory_z)

| d | rounds | p | pipeline LER (shots) | reference LER (100k shots) |
|---|---|---|---|---|
| 3 | 9 | 0.01 | 0.138 (400) | 0.1500 |
| 5 | 15 | 0.01 | 0.210 (200) | 0.2207 |

The pipeline decodes sliding windows (commit d, buffer d) serially with real boundary handoff, so its rate may sit slightly above the whole-circuit reference; a rate below it, or outside the binomial interval of the pipeline shot count, would indicate a data-path defect.

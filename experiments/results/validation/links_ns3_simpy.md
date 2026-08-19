# Gate 6: decsim links vs ns-3 point-to-point and a SimPy store-and-forward model

Per transfer: (serializer start, serializer end, delivery) in ticks (1 us = 1,000,000). ns-3 rule: FIFO device queue, txTime = bits / bps, receive at txTime + delay (point-to-point-net-device.cc:226-286, data-rate.cc:226-231, point-to-point-channel.cc:78-97). SimPy: one Resource per channel, hold bits / bandwidth, then the propagation delay.

## one channel, unbounded bandwidth, 0.5 us propagation

| send tick | bits | decsim | ns-3 | SimPy | equal |
|---|---|---|---|---|---|
| 0 | 24 | (0, 0, 500000) | (0, 0, 500000) | (0, 0, 500000) | yes |
| 100 | 24 | (100, 100, 500100) | (100, 100, 500100) | (100, 100, 500100) | yes |
| 150 | 96 | (150, 150, 500150) | (150, 150, 500150) | (150, 150, 500150) | yes |
| 150 | 8 | (150, 150, 500150) | (150, 150, 500150) | (150, 150, 500150) | yes |

## one channel, 24 bits/us, bursty sends queue FIFO

| send tick | bits | decsim | ns-3 | SimPy | equal |
|---|---|---|---|---|---|
| 0 | 24 | (0, 1000000, 1150000) | (0, 1000000, 1150000) | (0, 1000000, 1150000) | yes |
| 100000 | 24 | (1000000, 2000000, 2150000) | (1000000, 2000000, 2150000) | (1000000, 2000000, 2150000) | yes |
| 200000 | 24 | (2000000, 3000000, 3150000) | (2000000, 3000000, 3150000) | (2000000, 3000000, 3150000) | yes |
| 5000000 | 48 | (5000000, 7000000, 7150000) | (5000000, 7000000, 7150000) | (5000000, 7000000, 7150000) | yes |
| 5100000 | 12 | (7000000, 7500000, 7650000) | (7000000, 7500000, 7650000) | (7000000, 7500000, 7650000) | yes |

## two paths (WDO, DO) on one 100 bits/us channel, one FIFO

| send tick | bits | decsim | ns-3 | SimPy | equal |
|---|---|---|---|---|---|
| 0 | 500 | (0, 5000000, 6000000) | (0, 5000000, 6000000) | (0, 5000000, 6000000) | yes |
| 1000000 | 500 | (5000000, 10000000, 11000000) | (5000000, 10000000, 11000000) | (5000000, 10000000, 11000000) | yes |
| 2000000 | 200 | (10000000, 12000000, 13000000) | (10000000, 12000000, 13000000) | (10000000, 12000000, 13000000) | yes |
| 2100000 | 900 | (12000000, 21000000, 22000000) | (12000000, 21000000, 22000000) | (12000000, 21000000, 22000000) | yes |
| 2200000 | 100 | (21000000, 22000000, 23000000) | (21000000, 22000000, 23000000) | (21000000, 22000000, 23000000) | yes |

## reference fabric, random fixed schedule on QC and CWD

| path | sends | decsim = ns-3 = SimPy |
|---|---|---|
| qc | 12 | yes |
| cwd | 12 | yes |

Verdict: PASS. decsim's Link is ns-3's point-to-point link with an unbounded drop-tail queue and no inter-frame gap: one serializer per channel, first come first served, bits / bandwidth, then a fixed propagation delay; two semantic paths on one channel share that serializer, and the per-path counters reconcile with the channel's. What decsim adds on top is the attribution ledger (whose transfer this was, on which path, for which window or round), which ns-3 has no counterpart for.

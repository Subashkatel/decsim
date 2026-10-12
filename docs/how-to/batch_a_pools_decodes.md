[decsim docs](../README.md) › [How-to guides](README.md)

# How to batch a pool's decodes

A pool an edge escalates to can serve the re-decodes waiting for a unit
as one batch. Its batching row decodes each member with the pool's
algorithm, gives each member its own result, and prices the batch by a
law (`decsim/decoders/batching.py`).

## 1. Give the pool a batching row

```python
import dataclasses

import decsim.confidence.complementary as complementary
import decsim.decoders.batching as batching
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.presets.link_cards as link_cards
import decsim.presets.machines as machines

base = machines.weak_baseline(5, 0.004, 1.0)
decode_each = batching.DecodeEach.Settings(law=batching.Slowest.Settings())
strong_pool = dataclasses.replace(
    machines.STRONG_BASELINE_POOL, batching=decode_each
)
switching = escalation_settings.SwitchingSettings(
    confidence=complementary.ComplementaryGap.Settings(),
    threshold=threshold_sources.FixedThreshold.Settings(threshold_decibels=20.0),
)
fabric = link_cards.strong_side(base.fabric)
machine = machines.with_strong_tier(base, strong_pool, switching, fabric)
```

The strong pool is belief matching on the host, timed by the host's
clock, behind the redo window.

Or from the command line:

```bash
decsim run run.py --set strong.batching.type=decode_each --set strong.batching.law.type=slowest
```

Only a pool an edge escalates to batches; the row on any other pool is
refused.

## 2. Choose the law

| Row | The batch takes | Source |
|---|---|---|
| `sum` (default) | the members' times added | Toshio 2510.25222 lines 1096-1101 (a linear decoding time); 2604.12841 lines 9968-9972 (runtime near linear in batch size on one GPU) |
| `slowest` | the slowest member's time | Lottery BP 2605.00038 line 3561 (a batch "must synchronize to its slowest sample"); 2605.30765 line 3788 (cudaq-qec's synchronous batch) |

A member's time is the row's latency, or the host's clock for a row
measured there, as an ensemble's members are timed. The sources time
batches of shots; a batch here is of windows, the blocks 2604.12841
batches on one GPU.

## 3. Know how a batch forms

- When a unit frees, every waiting re-decode that owes no boundary
  joins one batch; a batch of one is the plain decode. There is no size
  limit and no wait for more.
- A member cancelled while its batch waits leaves it. One cancelled
  once its batch holds a unit is not decoded if the batch has not
  started, and its result is dropped if it has. Either way the rest
  decode.
- The threshold hears each member's own result, so an online threshold
  audits each window.

## 4. Know what is refused or not built

- `decoder_manager.bulk_strong` is the timing-only batch, kept as it
  was: members get empty results. It is refused beside a batching row.
- Refused, not implemented yet ([the refusals](build_each_switching_pattern.md#what-is-refused-for-now)): a batching row with
  `run_both_at_once`; with an absorbing strong window
  (`double_window`); over a strong backend row (`measured_table`,
  `dispatch_steps`), whose device prices one region per submit and
  states no batch time. Experiment 1 has both a double window and
  such a row, so the example above starts from the weak baseline.
- Not built: a measured batch table (2604.12841 Fig. 18 measures a
  network no row runs); batch size and wait limits; a fetch priced per
  word of each member's rounds (a batch's stages price its summed
  rounds).

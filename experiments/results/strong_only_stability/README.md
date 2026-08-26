# Strong-only stability sweep (first pass)

90 rounds, strong decode 5.0 us against a 3.0 us window stride,
timing-only runs. Raw per-window data in stability_results.json;
generator tmp/strong-prototype/ (ex_stability.py in the session jobs
folder, figures from stability_results.json).

- `sliding_units_blind_divergence`: sliding completion times diverge from
  the stream; the 4-unit curve lies exactly on the 1-unit curve (serial
  boundary chain, units cannot help).
- `tan_units_close_gap`: settled-frontier curves; lag behind the stream
  171 / 69 / 15 us at 1 / 2 / 4 units.
- `memory_cryo_vs_room`: SB1 peak rounds per configuration (57/57/67/50/21);
  cryo Buffer 0 measured at <= 1 round everywhere.

## Input staging delta (2026-08-26, depth-1 ping-pong, same 90-round load)

| config | depth 0 last done | depth 1 last done | saved | SB1 peak 0 -> 1 |
|---|---|---|---|---|
| tan u1 | 394.15 us | 286.15 us | 108.00 us | 67 -> 59 |
| tan u2 | 205.15 us | 152.15 us |  53.00 us | 50 -> 36 |
| tan u4 | 113.65 us | 104.65 us |   9.00 us | 21 -> 13 |

Hand-checked: with depth 1 the completion-gap set on one saturated unit is
exactly {5.0 us} = max(csd, decode), the reference-model prediction
(../strong_only_hand_check/, staging section). The price is per-unit SRAM
for two windows, charged visibly by decoder memory.

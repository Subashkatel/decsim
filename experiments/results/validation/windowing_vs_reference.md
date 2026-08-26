# Gate 9: windowed and unwindowed loop vs whole-circuit PyMatching, same shots

Circuit: surface_code:rotated_memory_z d=3, 60 rounds, p=0.005, 200 shots, identical detection events to all three.

| decoder | logical failures | LER | shots disagreeing with PyMatching |
|---|---|---|---|
| sliding | 52 | 0.2600 | 0 |
| naive_online | 52 | 0.2600 | 0 |
| pymatching | 52 | 0.2600 |  |

No-window timing rule, shot 0: 1 window; dispatched at 61.5 us = last round complete at 61.5 us; decode done 62.772 us (rule 62.772); frame commit 63.776 us (rule 63.776). Shots violating the rule: 0/200.

Windowed timing against SWIPER: Gate 8 (experiments/results/validation/loop_swiper.md).

Verdict: PASS (no-window predictions identical to PyMatching on every shot; windowed disagreements are tie-breaks between equal-weight matchings).

# Gate 10: timing invariance of the functional outcome

Circuit: surface_code:rotated_memory_z d=3, 60 rounds, p=0.005, 50 seeds x 8 timing points (round periods (1.0, 0.9, 0.5, 0.1) us x algorithm cards (0.028, 0.28) us).

Compared per seed across all timing points: sampled detection events, observable truth, windowed prediction, whole-circuit PyMatching prediction, logical failure.

- functional comparisons: 400, violations: 0
- seeds with a logical failure (check 5 exercised): 16/50
- seeds whose last frame commit tick did not move between 1.0 and 0.1 us rounds: 0

Verdict: PASS (timing knobs move timestamps only, never the decoded answer).

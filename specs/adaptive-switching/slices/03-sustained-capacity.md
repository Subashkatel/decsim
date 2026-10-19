# 03 Capacity and backlog

## Question

What escalation fraction can the declared hardware sustain, and does the
event simulator reproduce the paper's backlog behavior under the paper's own
assumptions?

## Why a long horizon

The paper measures LER per shot but backlog over 1e4 non-Clifford gates
(Figs. 11, 14, 21). Collect mode builds a new `Machine` every shot, so any
backlog is wiped after 10d rounds; at d = 13 a shot has about 10 windows and
most shots at a 1–5% escalation rate contain none. Capacity and latency are
therefore measured on one long machine in `decsim stream`.

## Deliverable and ownership

Stream-mode measurements fit for capacity studies, an exact queue oracle for
ordinary redo, and a reproduction of the paper's sliding-window backlog
result. Observations are added at their owning component, not in a second
pipeline.

## Work

1. **Fix stream mode** (`experiments/stream_command.py`,
   `stream_measures.py`): throughput over the measurement interval, not the
   whole run including warmup; lag sampled over time; per-window reaction
   percentiles (p50, p99, max), not only count, mean and max; several seeds
   per configuration; per-window escalation records.
2. **Exact oracle for ordinary redo.** With held boundaries the chain is one
   server fed every commit stride. Window k's finish time obeys a Lindley
   recursion: `F_k = max(F_{k-1}, A_k) + S_k`, where `A_k` is when its data
   and boundary are available and `S_k` is its total service (weak decode,
   gap walk, and on escalation the forward hop, strong decode and return).
   Check simulator timestamps against the recursion on recorded service
   sequences to the tick.
3. **Weak-only capacity.** Sweep weak service around the paper's Eq. 16,
   `τ_weak (r_com + r_buf) ≤ τ_gen r_com`; lag is bounded below it and grows
   above it.
4. **Paper reproduction, App. C.** Timing-only source, linear supplied service
   (τ per round), synthetic i.i.d. escalation at rate γ per window, the
   paper's parameters (d = 21, T_comm,weak = τ_gen, T_comm,strong = τ_strong =
   10 τ_gen). Compare the stability boundary with the paper's Fig. 21. This
   needs a Bernoulli verdict (new; `verdicts.py` has only GapThreshold, Status,
   Agreement, Always, Never), seeded per run.
5. **Real decodes.** Weak-only and fixed-20-dB switching with the slice 02
   table. Report lag, reaction percentiles, busy fraction per unit, strong
   queue depth, and the capacity fraction q\* found by sweeping γ.
6. Horizon: 1e4, 1e5, 1e6 rounds; separate warmup, measurement and drain.

## Gate and evidence

The recursion matches the simulator exactly. The weak-only boundary matches
Eq. 16. The App. C reproduction shows the same divergence trend against γ
and τ_weak/τ_gen as Fig. 21; differences are explained by stated modeling
differences (we time windows, they time gate intervals). q\* is reported with
its hardware assumptions and repeated over seeds.

## Scope boundary

Offered load can exceed 100%; one unit's busy fraction cannot. An empty
decoder queue alone does not prove capacity, since backlog can sit upstream.
Finite runs do not prove worst-case bounds. No adaptive threshold yet.

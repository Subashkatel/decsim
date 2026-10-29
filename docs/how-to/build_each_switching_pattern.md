[decsim docs](../README.md) › [How-to guides](README.md)

# How to build each switching pattern

Each pattern is a part or a rule the machine already has. This page
gives each one's setting, its source, and the guide with the details.
The patterns combine: a chain can speculate at one edge and keep a
deadline at the next. A pattern no source describes is not built, and
the last section lists what is refused for now.

| Pattern | What to set | Source | Guide |
|---|---|---|---|
| Escalate on doubt | a switching part: a confidence row and a verdict | Toshio 2510.25222 Sec. III A | [two tiers](../tutorials/two_tiers.md) |
| A chain of any length | `machines.with_escalation`, once per pool | | [a chain of decoders](build_a_chain_of_decoders.md) |
| Speculative | `switching.run_both_at_once` | Toshio Sec. III A, Step 1 | below |
| A backup on a deadline | `switching.deadline` | DART-Q 2605.09142 | below |
| A backup on a failure | `switching.verdict.type=status` | BP+OSD runs OSD when BP does not converge | [a chain of decoders](build_a_chain_of_decoders.md) |
| An ensemble or a vote | the `ensemble` decoder row | Harmony 2401.12434; VibeLSD 2508.15743 | [an ensemble](decode_a_window_with_an_ensemble.md) |
| A race | `ensemble` with `first_converged(count=1)` | VibeLSD Sec. III.4 | [an ensemble](decode_a_window_with_an_ensemble.md) |
| A disagreement trigger | `ensemble`, `consensus` and `agreement` | Harmony Sec. V | [an ensemble](decode_a_window_with_an_ensemble.md) |
| A decoder per patch | a `router` with `by_patch` | VQD 2406.17995v2 Sec. 4.1 and 5 | [several pools](route_windows_to_several_pools.md) |
| One pool shared by patches | `decoder_manager.scheduler` | VQD Sec. 4.2 | [a shared pool](share_a_pool_among_patches.md) |
| Batches | `<pool>.batching` | Toshio lines 1096-1101; Lottery BP 2605.00038 | [batching](batch_a_pools_decodes.md) |
| A rule of your own | any rule slot | | [a record of your own](add_a_store_card_signal_policy_or_scheme.md) |

## Escalate on doubt

A switching part decides on each result of its source pool. The
confidence row computes a gap; the verdict keeps a result at or above
the threshold and sends the rest on:

```bash
decsim run run.py --set switching.confidence.type=cluster_gap --set switching.threshold.threshold=20dB
```

The confidence rows are `cluster_gap`, `extra_cluster_gap`,
`complementary_gap`, `consensus` and `none`. The verdicts are
`gap_threshold` (the default), `status`, `agreement`, `always` and
`never`. `switching.strong_window` says what the strong pool decodes:
`redo_window`, the window again, or `double_window`, Toshio's double
window (Sec. III C, Fig. 12).

## Speculative

```bash
decsim run run.py --set switching.run_both_at_once=true
```

The target pool starts on every window when the source pool does. A
kept result cancels the target's decode (`cancel_downstream`, the
default) or lets it run out and drops its answer
(`--set switching.cancel_rule.type=no_cancel`). A cancel goes on down a
chain. Which revision the frame takes is `switching.replace_rule`:
`latest`, the default, or `best_status`.

## A backup on a deadline

```bash
decsim run run.py --set switching.deadline.type=deadline --set switching.deadline.slack=10us
```

A source job's deadline is its arrival plus the slack. The trigger
starts the target decode early, a rescue: `slack`, the default, when the
slack left reaches its threshold (0 us, the deadline itself), and
`backlog` at arrival when the source queue holds `queued_jobs`
(`--set switching.deadline.trigger.type=backlog`). A source result that
comes late is unavailable by default, so the verdict escalates;
`switching.deadline.late_source_result=revision` lets it meet the
verdict. The edge counts `rescues` and `misses`.

## What is refused for now

Each of these raises "not implemented yet", with its reason and a pointer
to this table, when the machine is built:

| Refused | Why |
|---|---|
| `double_window` on an edge past the entry pool | Toshio defines it over the entry pool's sliding chain; a pool past it decodes only what it is sent. A `redo_window` there redoes the double region. |
| `double_window` beside a router | it absorbs neighbouring windows, which a router may send to two pools |
| more than one router | |
| a VQD scheduler over a stream | a stream's critical point is a segment's end, which the scheduler does not read |
| likelihood pooling over members of two fault representations | Harmony prices a matching's hyperedge errors by an edge decomposition |
| an ensemble member, or a pool's batching, over a strong backend row | the device prices each decode as it runs it, and an ensemble or a batch does not take its time from there yet |
| a pool's batching with `run_both_at_once` or `double_window` | a batch of every window's speculative decode, or of deferred windows, is not built |
| `decsim stream` over a workload other than decoded Stim memory patches | a stream repeats the one REPEAT block of each patch's Stim memory circuit |
| `decsim stream` with a source other than the Stim device | the flip stream samples the circuit as the Stim device does |
| `decsim stream` with the trace, the log, or the data movement or window outcome records | each keeps something for every round or window, and a stream keeps nothing that grows with its length |
| `decsim stream` with a pool's batching or `bulk_strong` | a merged batch files its stages under its own key, which the per-window measures do not read |
| `double_window` over a stream whose operations emit detector data | the stream's windows end at each such operation, and a strong region does not |
| a logical failure counted inside a stream, other than at a checkpoint | no source defines one inside an unbounded stream; a checkpoint is the outcome had the stream ended there |
| checkpoints or answer checks beside an escalate edge past the entry pool | that edge's source pool joins its solves where no checkpoint reads them |
| checkpoints or answer checks beside the double window | a strong region answers for rounds past its escalated window and pins its far face on the restart window's commit, so a checkpoint inside it has no answer a stream ending there would keep |
| checkpoints or answer checks beside a decoder or a verdict that draws from the run's seed | a checkpoint decides its last windows as the machine does, and a draw there would change the run |
| parallel windows on a stream that a segment cuts, such as a feedback stream (refused when the segment binds) | a segment's result is a sum of whole windows, and the parallel windows are laid across a stream with no segments |
| parallel windows on a live or a finite Stim stream, not the flip stream (refused at the first B window) | those sources make each window's model after the windows before it, and a B window also waits for the window after it |

Not built, because no source describes it: one window's results merged
across separate pools (a race or a vote across pools); a router rule
that reads the run's state (the least loaded pool, a backlog, a round);
a measured batch table; a limit on a batch's size or wait.

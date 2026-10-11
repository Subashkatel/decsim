[decsim docs](../README.md) › [How-to guides](README.md)

# How to build a chain of decoders

A chain is decoder pools joined by escalate edges. The entry pool
decodes every window. Every other pool is the target of one switching
part, which decides on each result of its source pool and sends the
windows it escalates on to its target. A window is revised once for
each edge it crosses.

## 1. Add a pool behind one you have

```python
import decsim
import decsim.confidence.no_signal as no_signal
import decsim.decoders.belief_propagation_osd.decoder as bposd
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.escalation.verdicts as verdicts
import decsim.presets.experiment_one as experiment_one
import decsim.presets.machines as machines

two_pools = experiment_one.switching(5, 0.004)
osd = bposd.BeliefPropagationOsdDecoder.Settings(
    max_iterations=30, osd_order=10
)
deep_pool = decoder_settings.DecoderPoolSettings(algorithm=osd)
deep_switching = escalation_settings.SwitchingSettings(
    confidence=no_signal.NoSignal.Settings(),
    threshold=threshold_sources.FixedThreshold.Settings(threshold_decibels=20.0),
    verdict=verdicts.Status.Settings(),
)
chain = machines.with_escalation(
    two_pools, machines.STRONG_POOL, "deep", deep_pool, deep_switching
)
task = decsim.Task("uf_relay_osd", chain, collection=collection)
```

`with_escalation` adds three parts named from the new pool's name,
`deep_buffer`, `deep` and `deep_switching`, and their wires. The new
pool reads its own store. An escalated window's rounds come to it from
the store its source pool reads, over the hop `strong_to_deep`, and its
answer goes straight to the frame. The parts sit on the host unless
you pass `site`. Call `with_escalation` again with `"deep"` as the
source to add a fourth pool.

Adding a pool past the entry moves no other part's seed. A chain whose
new pool is never asked runs as the machine without it, shot for shot.

## 2. Choose what each edge decides on

A switching part's `verdict` decides on each result of its source pool
(`decsim/escalation/verdicts.py`):

- `gap_threshold`, the default: keep a result whose confidence gap is
  at or above the part's `threshold`.
- `status`: escalate a result its backend did not report as succeeded,
  as BP+OSD runs OSD only when BP does not converge.
- `always` and `never`: escalate or keep every window.
- `agreement`: escalate an ensemble's answer its members do not agree
  on, beside `confidence=Consensus.Settings()`
  ([decode a window with an ensemble](decode_a_window_with_an_ensemble.md)).

A verdict that reads no gap takes `confidence=NoSignal.Settings()`, so
its source pool computes no soft output; its `threshold` is set but
unread. A gap verdict on no confidence is refused when the machine is
built.

`replace_rule` says which revision the frame takes: `latest`, the
default, or `best_status`. `cancel_rule` says what a keep does to the
target's decode of the window: `cancel_downstream`, the default, or
`no_cancel` (`decsim/windows/revision_rules.py`). A keep does the same
to every decode further down the chain that was started for the window.

`run_both_at_once=True` on an edge starts its target's decode with its
source's job instead of after the verdict. On an edge past the entry,
that is when the source pool's job is queued. The verdict then selects
the running decode or cancels it.

A confidence of two solves, `complementary_gap`, on an edge past the
entry has its source pool decode each window it is asked for once per
logical class. The two solves read one input and may run on two units
at once. The verdict reads the gap between their weights, and the
lighter solve is the answer (Toshio et al. 2510.25222 Sec. II C). The
source pool's decoder must find the lightest correction inside a fixed
class; a union-find pool is refused, and `cluster_gap` is its signal.

## 3. Give an edge a deadline

`deadline` on an edge gives each of its source pool's jobs a deadline
and a rescue (`decsim/escalation/deadlines.py`). It follows DART-Q
(Bhave, Choudhury and Basu 2605.09142):

- The deadline is the job's arrival at the source pool plus
  `slack_microseconds`, the paper's slack budget.
- A result that reaches the verdict after it is a miss. By default it
  is unavailable, as the paper treats a late output, so the edge
  escalates it. `late_source_result: revision` lets it meet the
  verdict instead.
- The `trigger` starts the rescue, the target's decode before the
  verdict; the verdict selects or cancels it as it would a decode
  started with `run_both_at_once`.
  - `slack(threshold_microseconds=s)` starts it when the slack left
    reaches s, the default 0 being the deadline itself.
  - `backlog(queued_jobs=k)` starts it at arrival when the source
    pool already holds k waiting jobs.
  - The paper's third trigger, a decoder that hit its iteration cap,
    is the `status` verdict on a pool whose decoder has one.

```
decsim run run.py --only uf_relay_osd \
    --set switching.deadline.type=deadline \
    --set switching.deadline.slack=10us \
    --set switching.deadline.trigger.type=backlog
```

`<edge>.deadline` counts `rescues` and `misses`. The paper finds a
rescue helps only while its trigger stays selective (its Table I).
A deadline is refused beside `run_both_at_once`, which starts the
target with every job, and beside an absorbing strong window, which
starts it late. Not built: earliest-deadline-first queues, a cutoff in
microseconds, and a rescue into the source pool's own queue. The
timelines are worked by hand in `tests/escalation/test_deadlines.py`.

## 4. Set and read it

```
decsim run run.py --only uf_relay_osd --set deep.unit_count=2
```

Each frame record's `tier` is the pool that answered the window,
`deep` here. Every pool's statistics sit at its part's name
(`deep.queue`, `deep.units`). The summary's weak and strong columns
read the pools named `weak` and `strong`; read any other pool from its
statistics.

## What a chain does not do yet

An edge that does not leave the entry pool redoes the region its
source pool decoded, on the same faces and the same error model. A
strong window that absorbs weak windows (`DoubleWindow`) on such an
edge is refused with "not implemented yet (stage 5)": Toshio et al.
define it over a sliding weak chain's neighbouring windows, which a pool
past the entry does not decode. The timelines a
chain follows are worked by hand in `tests/machine/test_chains.py`.

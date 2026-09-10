# Two tiers

A decoder can be fast or it can be accurate. The two-tier idea is to
have both by running the fast one all the time and calling the accurate
one only where the fast one says it is unsure. This page says what that
means, what it costs, and how decsim models each piece of it.

## The problem

A decoder that cannot keep up with the machine it is decoding for is not
slow, it is broken. Rounds arrive at a fixed rate; if decoding a round's
worth of data takes longer than a round, the undecoded backlog grows
without bound, and the computation cannot proceed past the point where
it needs a decoded answer. That is the **backlog problem**.

Accuracy pushes the other way: the decoders that make the fewest
mistakes are the ones that take the longest. Toshio et al. call this the
fundamental trade-off between accuracy and speed, and **decoder
switching** is their answer to it (arXiv:2510.25222 Sec. III).

## The two tiers

decsim calls the two decoders the **weak tier** and the **strong tier**,
which are Toshio's own words: "we refer to these paired decoders as
'weak decoder' and 'strong decoder'" (`2510.25222.txt` lines 590-598,
Sec. III A; every paper line on these pages is into the sandbox's
`tmp/papers/txt/` extraction, the one the code's own docstrings cite). The weak decoder is fast and reports a **soft output**; the
strong decoder is accurate and has a relatively high latency.

Three rows of `ESCALATIONS` say which of the two arrangements a run is:

| Row | What runs |
| --- | --- |
| `weak_baseline` | every window once on the weak tier, and every result kept. There is no second tier. |
| `strong_only` | every window once on the strong tier. |
| `switching` | the weak tier first, and a window whose confidence falls below its threshold is decoded again on the strong tier. |

A policy decides and is told. It builds no job: the window side plans
and submits the strong re-decode and the decoder manager owns the units,
the hold-or-deliver decision and the cancellation
(`decsim/escalation/policies.py`, which cites gem5's conditional
predictor for the shape).

## The confidence

Escalating usefully needs the weak decoder to know when it is unsure.
**Soft information** is the name for that: an analog number quantifying
how reliable the decoder's own estimate is, rather than a hard decision
about the most likely logical error (Toshio Sec. II B,
`2510.25222.txt` lines 386-396). decsim's name for the number is the
`SoftOutput`, and `CONFIDENCE_SIGNALS` has two rows for how to get it.

**`complementary_gap`.** Decode the window twice, each solve pinned to
one logical class, and subtract the two weights. If the two hypotheses
are nearly as likely as each other, the decoder had almost no reason to
prefer the one it chose. Toshio Sec. II B and Fig. 3(a,b) define it, and
Gidney, Newman, Brooks and Jones do too (arXiv:2312.04522, Sec.
"Complementary gaps"). It is the exact reference metric, and it costs
two decodes per window.

**`cluster_gap`.** Decode the window once and walk the clustering the
decode already did. Toshio's own device computes its soft output this
way rather than by running two matchings, so this is what a real-time
system would run (Toshio Fig. 3(c,d), citing Meister arXiv:2405.07433,
Algorithm 2).

The pairing is checked at load. A signal needs particular evidence from
the decode, and a decoder row that cannot produce that evidence is
refused by name rather than running and reporting a meaningless number.
`complementary_gap` needs a decoder that minimises weight inside a
logical class, which is minimum-weight perfect matching today;
`cluster_gap` needs a cluster-based decoder, which is union find today.

Gaps and thresholds are carried in natural-log weight, the unit the
decoder compares in. The yaml lets you write the paper's decibels and
converts once, at load.

## The threshold

`THRESHOLD_SOURCES` says where the number the confidence is compared
against comes from.

| Row | Where the threshold comes from |
| --- | --- |
| `fixed` | one constant, the paper's `g_th` (Toshio Sec. III A, step 3) |
| `table` | a table of values, resolved by the front to a fixed threshold per sweep point, so at run time it is the fixed row |
| `online` | it starts at the fixed value and adapts across a sweep point's shots |

The `online` row is the interesting one. It runs two loops: a rate
tracker that pins the fraction of windows escalated at a target, and an
audit lane that strong-decodes a random sample of the windows it kept,
to learn whether that target is safe. It is one instance per sweep
point, shared by every shot of that point, because it learns across
shots.

## The strong window's shape

When a window is escalated, what exactly does the strong decoder decode?
Not just the escalated window: a re-decode that saw only the escalated
commit region would face the same artificial boundaries the weak decoder
faced, and would have little reason to do better.

`STRONG_WINDOW_SHAPES` has four rows.

| Row | The rounds it reads |
| --- | --- |
| `two_sided_context` | the escalated window's commit region with one buffer of raw context on each side. decsim's own geometry, not the paper's, built the moment it is asked for. |
| `forward` | Toshio Sec. III C and Fig. 12: it starts at the escalated commit and extends forward, absorbing the weak windows it covers. |
| `near_seam_pinned` | the context row's commit region with its past face pinned on the neighbour's committed correction. |
| `forward_seam_pinned` | the forward row's extent with both faces pinned. |

Toshio's own assumption for the size is `r_strong = r_com + 2 r_buf`
("In this paper, we assume that rstrong = rcom + 2rbuf",
`2510.25222.txt` line 1250).

**Pinning a face** means reading the neighbour's already committed
correction and folding it into the strong window's input rather than
re-deriving it. That is Bombin et al.'s input adaptation
(arXiv:2303.04846): the input to the later decoding task is the syndrome
of the errors plus the corrections already committed. It is the
`BoundaryCourier` port, and the courier lives in the windows package.

A fifth shape suggests itself and is deliberately not a row: both faces
pinned and absorbing nothing. It waits for the window after it, which
waits for its own strong result, and a serial sliding chain deadlocks.
What would make it a row is a windowing scheme whose windows do not
commit in one serial chain, and the row would read that off a fact the
scheme declares rather than off the scheme's class.

## Restart

The weak tier does not stop while the strong tier works. When the strong
window absorbs the weak windows it covers, the weak chain has to resume
somewhere, and where it resumes is the **restart window**
(`PotentialRestart`). How far back into the strong region that restarted
weak decode reads is `restart_reread_buffer_regions`, which defaults to
the paper's value.

This is also why the two stores exist. Buffer 0 streams to the weak
tier round by round as the rounds arrive. Buffer 1 keeps the same rounds
for a strong re-decode that may be asked for later, in bulk, once its
boundaries are known. A round may not be dropped from either store while
any consumer still holds it, and `PotentialStrong` and
`PotentialRestart` are exactly the tokens that say a re-decode or a
restart might still need it.

## What it costs

Everything above has a price, and decsim's point is to charge all of it:

- the second decode, if the confidence signal needs one;
- the signal's own computation, charged on the weak unit that produced
  the evidence;
- the escalation's own hop, `weak_decoder_to_strong_decoder`, which
  carries only the selection;
- the strong region's transfer out of Buffer 1;
- the store capacity the holds occupy while a re-decode might still be
  asked for;
- and the strong decode itself.

A study of switching is a study of whether the accuracy bought is worth
that list. `configs/seam_pinned_switching.yaml` is a worked point, and
`docs/tutorials/two_tiers.md` runs one.

## Read next

- `docs/tutorials/two_tiers.md`: a switching run, with the trace of an
  escalation.
- `docs/explanation/windows_and_boundaries.md`: what a face and a seam
  are.
- `docs/reference/tables.md`: the four tables named above, with rows.

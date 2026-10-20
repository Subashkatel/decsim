# Region windows: decoding spacetime cut by patch, not by operation

Design note for slice 08. Status: accepted by the owner (2026-10-19),
with the four proposals at the end.

## Why

decsim ties every decoding window to one operation's rounds. A
lattice-surgery program merges several patches, and the papers that
decode such programs cut the decoding spacetime into regions of one patch
by at most d rounds and build windows from them: SWIPER (2412.05115,
swiper/window_builder.py SpacetimeRegion), Triage (2605.04459, a slice is
one d×d patch over d rounds), Lin et al. (2403.01353, d×d tiles), and the
general, proved form, Bombín et al.'s modular decoding (2303.04846 Sec.
IV, VI). Without that cut decsim decodes a merge as one job and cannot be
checked number for number against SWIPER's sliding and parallel windows.

## The common law (the referents agree on this)

- Region: one patch × a run of rounds; commit regions partition the
  decoded rounds (Bombín Eq. 2).
- Window: commit regions plus buffer regions; past commits enter only as
  boundary conditions (Bombín Sec. IV-D).
- Edges: windows whose commit regions share a face are ordered; the
  earlier owns the buffer across that face. The order is the scheme's
  choice (Bombín L720-723; SWIPER Sec. 3; Tan; Lin).
- Ready: every round of its regions arrived and every predecessor
  committed (Bombín L884-887).
- Release: an operation's outcome is usable once every window committing
  its rounds has committed (Bombín Eq. 4 and L1295-1297).

Buffer width is where they differ: Bombín b ≥ d (proved), SWIPER one
whole neighbouring region, Tan and Lin about d/2. It is the scheme's
setting.

## The design

decsim already has the unit: the decode stream (records/program.py
decode_identity). Rounds, arrivals, windows, the ledger and a job's size
are keyed by it, and a standalone operation is its own stream.

1. Patch streams. In a lattice-surgery run each patch's life, from its
   first decoded operation to its DISCARD, is one decode stream; every
   round the patch measures, operation or idle, is the stream's next
   round. An operation on k patches is a segment of k streams. A new
   controller row, PatchStreams, implements the controller's existing
   Streams seam beside NoFeedbackStreams and FeedbackStreams.
2. A region is a RoundSpan(stream_id, first_round, last_round). A window
   keeps its home-stream interval as now and gains spatial_reads, the
   spans it reads on other streams (a merge's spatial buffers).
3. The windowing scheme is the one plug-in: it cuts regions, orients
   edges and picks buffers. WindowingScheme.plan_operation becomes
   lay_windows over the region graph; today's sliding, parallel, sandwich
   and naive schemes are the one-stream case, and SWIPER's sliding and
   parallel windows are one class each.
4. An operation on patch streams is released when each of its streams
   has committed its rounds of the operation (the existing segment
   release, over several streams).
5. Volume and latency need nothing new: a job's round count is its
   distinct (stream, round) pairs, which is SWIPER's volume, and the
   engine's per-round cost makes latency linear in it.

Memory runs are the special case: one stream per operation, no spatial
reads, PatchStreams never built. Their results and the 19 regression
digests stay identical.

## Code-agnostic by construction

The windows must serve qLDPC codes as well as the surface code, so
nothing in the core reads surface-code geometry:

- A patch is decsim's opaque id for one code block: a surface patch, a
  bivariate-bicycle block (qpu/code_geometry.py's two cards), or any
  later card. A patch stream is that block's rounds.
- Region length and buffer width come from the code card
  (commit_rounds, buffer_rounds) or the scheme's settings, never from a
  literal d. SWIPER's d-round regions are the surface card's commit
  rounds.
- Which streams a window reads in space comes from the operation: the
  blocks a joint measurement couples, as the program states them. Grid
  coordinates, merge faces and SWIPER's coordinate rule for orienting a
  spatial edge are the lattice-surgery frontend's and SWIPER's scheme's
  data, read only there; a qLDPC surgery program states its couplings
  (its ancilla blocks among them) the same way and uses another scheme.
- A job's size is per stream: each stream's rounds weighed by its own
  card's graph size per round (spatial_nodes), so blocks of different
  codes price correctly; the surface-only run reduces to SWIPER's
  patch-round volume.

## Stages, each committed with the full suite and the digests green

1. Rename the window identity from operation to stream (values unchanged).
2. Frontend: read merge faces and DISCARD; T-injection rounds are not
   decoded (swiper/window_builder.py); tested against SWIPER's reading.
3. Patch streams: PatchStreams, one decode stream per patch life, the
   timing-only device emitting per stream, release over several streams.
4. The port change and the region planner: lay_windows, RoundSpan,
   spatial reads, and SWIPER's sliding scheme, tested against SWIPER's
   own window builder on toffoli (same regions, volumes and edges).
5. The referent run: SWIPER-SIM and decsim on the same schedules, each
   conditional S's reaction time compared, within a stated tolerance.
6. Fold the dynamic-stream growth into the sliding scheme, so one law
   lays every window.
7. SWIPER's parallel windows.
8. Move multi-operation runs onto patch streams and delete the old
   successor-overflow reads, so one model of "what follows an operation"
   remains (question 2).

### Stage 3 as built, and what it leaves for stage 5

The lives match SWIPER's device at seed 0 on toffoli (36) and msd_15to1
(65): the same patch, the same instructions, the same instruction rounds
(tests/frontends/test_patch_lives.py). Each stream's rounds arrive once,
1 to n, and a two-patch merge releases after both streams commit
(tests/controller/test_patch_streams.py). Three differences remain,
each measured in stage 5 rather than guessed at:

- Idle rounds. On toffoli 10 of the 36 lives idle longer in decsim, by
  1 to 62 rounds: SWIPER starts a patch's first instruction as late as
  its successors allow (device_manager.py
  _predict_instruction_start_time), decsim as early as its predecessors
  allow.
- A skipped conditional S still sends each of its blocked zero-round
  instructions down the control path, and the patch idles a round on
  each with no live stream; SWIPER finishes them in the round they are
  released and never activates the patch.
- The windows are today's sliding ones, cut at every segment's ends;
  SWIPER's d-round regions come with stage 4.

A decoder slower than the rounds arrive falls ever further behind with
no bound, and the run never drains (one unit at 5 µs a window on
toffoli); the tests run one that keeps up.

Not needed for timing-only runs: window models, per-patch real syndromes
of a merge round, escalation over patch streams (refused at build until
the strong decoder is in play on lattice surgery).

## The owner's decisions

1. Release rule. Bombín releases an outcome once the windows committing
   it have committed; SWIPER also waits for the windows of every earlier
   instruction. Proposed: Bombín's, with stage 5 measuring the gap.
2. Stage 8, one model of what follows an operation: proposed yes, after
   stage 5 validates.
3. A discarded patch: the QPU keeps idling it today. Proposed: a DISCARD
   ends the patch's life and its rounds.
4. Rename scope: proposed the whole identity, decode jobs and request
   keys included (STYLE rule 5: a rename moves every caller).

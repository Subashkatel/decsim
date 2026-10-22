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
- Release: an operation's outcome is usable once every task that can flip
  it has committed: Eq. 4's relevant tasks T(M) of its membrane, which
  "continues towards the past and depends on the history of the two
  qubits" and branches through earlier merges (Bombín Sec. I.A, Fig. 1,
  L1295-1297). Stage 5 below says how the rows bound that set.

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

### Stage 4 as built

A patch-stream run is windowed by a region scheme, the SWIPER sliding
row (windows/schemes/swiper_sliding.py), over the region layout
(windows/region_layout.py). The layout cuts each stream into regions of
at most the card's commit rounds, cut only at a coupled segment's edges
and at the seal; a window commits one region, reads the next region of
its stream, and reads each coupled partner's region the scheme orients
it to read, the partner's window waiting for it. Two port additions
carry it: WindowingScheme.lays_regions with the RegionScheme port
(orient_coupling), and WindowInput.couple_segments, which PatchStreams
calls with a merge's per-patch spans and faces. Readiness waits for the
spatial reads' rounds, retention holds them and a job's size counts
them, as SWIPER's volume does.

Checked two ways, number for number:

- The layout fed SWIPER's own seed-0 rounds gives SWIPER's sliding
  windows and edges: toffoli 102 windows and 105 edges, msd_15to1 149
  and 168 (tests/windows/test_region_layout.py, against
  tools/swiper_referent.py's json).
- decsim's run, its rounds fed to SWIPER's WindowBuilder and
  SlidingWindowManager, gives decsim's windows and edges: T gate 8 and
  7, toffoli 144 and 147, msd_15to1 172 and 191 (run by hand at this
  stage; the in-tree test checks a run's windows against the layout of
  its own rounds, tests/controller/test_patch_streams.py).

Left as they are, for stage 5 or later:

- lay_windows does not yet replace plan_operation: a static operation
  under the SWIPER row is its one-stream case, and stage 6 folds the
  stride-and-cut growth into the schemes.
- SWIPER cuts where merge rounds meet other rounds, so two merges back
  to back on a patch can share a region there; here each coupled
  segment is cut on its own. They agree when merges last whole regions,
  as on both schedules.
- decsim knows a merge's cut when the merge starts; SWIPER learns it
  from the merge's first round, so the window before a merge may be
  ready a round earlier here.
- A schedule read at distance d sets its rounds only; the QPU card's
  distance sets the region length, so a run sets qpu.distance to the
  same d. Nothing checks the two agree yet.

### Stage 5 as built

The release rule is a row (windows/release_rules.py, set by
WindowSettings.release_rule, None choosing by the run in
release_rule_for):

- OwnStreams: a blocker waits for each of its streams committed from
  round 1 through its rounds, its membrane's history along its own
  patches. The default off lattice surgery, where it is the whole
  history.
- DecodedAncestors: SWIPER's (decoder_manager.py
  get_incomplete_instruction_indices) and Triage's causal cone
  (2605.04459 Sec. II-B); a blocker also waits for every ancestor reached
  through operations that decode rounds. The default on a lattice-surgery
  run. It reads each ancestor's stream segments, which only patch
  streams bind for every operation, so build refuses it on any other run.

tools/swiper_reaction.py runs SWIPER-SIM on toffoli and msd_15to1 at
d = 7, sliding windows, no speculation, unlimited decoders, every
window decoded in 1, 5 or 10 rounds, seed 0, and writes each
conditioned instruction's wait (tests/data/swiper_*_d7_reaction.json).
decsim runs the same on SWIPER's machine (a round a microsecond, free
hops, 200 units, PresetLatencyDecoder) with DecodedAncestors; its
GateStall.stall_ticks is SWIPER's wait
(tests/controller/test_swiper_reaction.py):

- toffoli: every wait equal at all three latencies, including the
  10-round run whose backlog stretches it to 840 rounds.
- msd_15to1: every wait SWIPER's or one round less. SWIPER cuts a
  region at a merge's edge only once a round from beyond it arrives
  (window_builder.py build_windows), one round after decsim, which
  knows the merge when it binds; msd's half-distance merges cut
  mid-stride. toffoli's merges end on strides, so it shows no gap.

The referent is SWIPER started as soon as possible, as decsim's
controller starts operations (the tool's "asap" runs, a subclass of its
DeviceManager). SWIPER's own start rule, fresh patches as late as their
successors allow, is also written ("alap"): it changes msd_15to1 not at
all and two toffoli waits (instruction 74 at 5 rounds, 12 against 15;
instruction 67 at 10 rounds, 231 against 262). It is a device choice,
not a decoding one, so decsim does not copy it; that is also stage 3's
idle-round difference.

The two rules, decsim against SWIPER-ASAP:

| schedule, latency | DecodedAncestors | OwnStreams |
|---|---|---|
| toffoli, 1 / 5 / 10 | equal | equal but two waits a round short at 5 |
| msd_15to1, 1 | within 1 | 5 against 5 to 8 |
| msd_15to1, 5 | within 1 | 9 to 48 against 33 to 49 |
| msd_15to1, 10 | within 1 | 17 to 113 against 68 to 114 |

On msd_15to1 OwnStreams releases up to 4× sooner: its later merges read
patches whose earlier merge partners are still being decoded.

What the literature requires, each source read in full: every source that says which decodes a feed-forward outcome waits for
asks for its whole past.

| Source | Says | Rule |
|---|---|---|
| Bombín 2303.04846 | the membrane "continues towards the past"; "each logical membrane ... must be fully decoded" | membrane |
| SWIPER 2412.05115 Sec. 5.1 | "fully decoded (and verified) the instruction history up to and including the blocking operation" | ancestors |
| Triage 2605.04459 Sec. II-B | "all undecoded historical slices ... of any logical qubit ... correlated with the target through a chain of multi-qubit operations" | ancestors |
| Terhal 1302.3428 Sec. VI | "best to know the logical Pauli frame of the data qubit and the ancilla qubit" before going past the T gate | history |
| Skoric 2209.08552 Sec. I | response time from "the last round for which the full syndrome history has been decoded" | history |
| Chamberland 2109.02746 Sec. III | data errors on the patches "prior to the merge" flip its outcome | history |
| Sahay 2408.01393 | the partner "can only begin after the target is decoded" | partners |
| Zhou 2406.17653 | "all accessible syndrome information up to a logical measurement" | history |

Riverlane's deltakit 0.9 decodes whole shots offline, with no windows or
feed-forward, so it does not answer this.

The exact set is the membrane's, and tools/membrane_check.py measures it
with Stim. Three rotated patches start in logical Bell pairs with
noiseless references; A merges with C, splits, then merges with B, and
the script asks, over GF(2), whether a membrane of the A-B outcome that
reads nothing after its round avoids every single fault in C's past. At
d = 3 and 5, with the A-B seam either way:

| earlier A-C merge | C's past | A's past (control) |
|---|---|---|
| same logical type as A-B | not needed | needed |
| other logical type | needed | needed |
| same type, its outcome a Pauli frame update on A | needed | needed |

So DecodedAncestors over-waits only where an earlier merge on a patch
has the later one's type and its outcome sets nothing in the frame the
later one reads. The exact rule needs the program's frame logic, which
SWIPER's schedule format does not carry; it is a possible refinement.

## The owner's decisions

1. Release rule. Decided after stage 5: DecodedAncestors on a
   lattice-surgery run, OwnStreams on any other; a setting overrides
   either. A merge's outcome is read in a Pauli frame that earlier merges
   with other patches set, which the literature and the Stim check agree
   on.
2. Stage 8, one model of what follows an operation: proposed yes, after
   stage 5 validates.
3. A discarded patch: the QPU keeps idling it today. Proposed: a DISCARD
   ends the patch's life and its rounds.
4. Rename scope: proposed the whole identity, decode jobs and request
   keys included (STYLE rule 5: a rename moves every caller).

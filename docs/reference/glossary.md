[decsim docs](../README.md) › [Reference](README.md)

# Glossary

decsim's names are full words (`STYLE.md` rule 2) and the literature's
names are letters and acronyms. This page is the map, both ways: what
the code calls a thing, what the papers call it, and where in the papers
to read it. Every reference below was opened before it was written down,
and every decsim name below exists in the tree. A line number into a
paper is into the plain-text extraction of that paper, the one the
code's own docstrings cite. Those text files are the maintainers' and
are not in this tree; the section and figure numbers beside them are the
part any reader can follow.

## The words this code means precisely

- **round**: one cycle of syndrome measurement on an operation's patch group.
  The round key is `(operation_id, round_index)`, and rounds are numbered from 1.
- **syndrome**: the classical bits one round measures. They are parity
  checks on the encoded qubits, not the qubits' state.
- **detection event**: a check whose value changed from the round
  before. A decoder reads detection events, not raw checks.
- **window**: the slice of rounds one decode covers.
- **tick**: the engine's integer time unit. One microsecond is a million
  ticks (`decsim/config.py`, `TICKS_PER_MICROSECOND`), and every cost in
  the machine is an integer number of them.
- **operation**: one logical operation of the workload. It is the unit a
  workload is written in and the unit the QPU runs.
- **window key, request, service**: the window is
  `(operation_id, window_index)`; the decode asked for it is a
  `DecoderRequestKey`, which carries the tier; the run of that decode on
  a unit is a `DecoderServiceKey`.
- **move, copy, reference**: whether a transfer leaves the bits behind,
  ends with both sides holding them, or hands over an object both sides
  read. The traffic ledger books every hop as one of the three
  (`decsim/observe/data_movement.py`).
- **reaction time**: the time from a round's measurement to the
  instruction that acts on its correction. It is what a run measures.
- **tier**: which of the two decoders ran. `weak` is the fast one and
  `strong` the accurate, slower one. The escalation says which tier
  decodes the planned windows: `weak_baseline` and `switching` start
  every window on the weak tier, and `strong_only` sends every window to
  the strong tier.

## decsim's own words

These are not from the papers. The pages use them everywhere.

- **table**, **row**: a table is a dictionary of the names a yaml may
  write for one pluggable component; a row is one such name and the
  class the machine builds for it (`decsim/tables.py`).
- **port**: a small Protocol in `decsim/ports.py` naming the methods one
  component needs from a neighbour. The part that holds a component
  binds its ports.
- **part**: one of the six records a machine is built from, in
  `decsim/build/` (`Qpu`, `Control`, `Readout`, `Windows`, `Decoders`)
  and the links. A part builds its components and wires them to one
  another; `Machine.assemble` wires the parts together.
- **seat**: a place on a round's path where its detection events may
  form, named in `detection_events.formed_at`.
- **card**: a set of stated numbers used in place of a measurement. A
  decoder card is a `kind` that is a number, the decode's time in
  microseconds. A link card is a latency and, if bounded, a bandwidth
  for one hop. A code card is the numbers the machine needs from a code.
- **hop**: one priced link between two components; the link paths below
  are the hops' names.
- **unit**: one decoding engine inside a tier's chip: a memory that
  holds a job's rounds plus the compute that decodes them. `units` is
  how many identical engines the chip holds, gem5's `FUDesc.count`. They
  form the tier's pool and share the tier's links. decsim models one
  chip a tier, so the chip count is not a key.
- **unit memory**: the input memory of one decoder unit, sized in bits
  by `<tier>_decoder.unit_memory.bits`. A window's rounds are copied
  into it before the unit decodes them and freed when the decode ends.
- **job**: one decode of one window, asked of a tier's pool.
- **stage (a job)**: to put a job's rounds on a unit before the job may
  compute, so the move overlaps the wait. A staged job stays on that
  unit (`decsim/decoders/decoder_pool.py`).
- **park**: the wait between a job's input landing in a unit and its
  compute starting. `dep_block` is the part spent waiting for a boundary
  or an escalation message, and `compute_wait` the part spent waiting
  for the unit's compute.
- **service**: one decode's fetch, algorithm and release, without
  anything it waited for.
- **residence**: one round's or one window's stay in a store or a unit's
  memory, with when it became readable and why it was freed.
- **load**: the service time per window divided by the time between
  windows arriving. Above 1 the decoder cannot keep up.
- **escalation**: sending a window the weak tier was unsure of to the
  strong tier. The **verdict** is that decision, taken on a finished
  weak decode: keep its correction, or escalate. The **selection** is
  the message that names the escalated window to the strong side; it
  is the request's 64-bit name and nothing else. The region and the
  strong answer carry the same name in front of their bits.
- **absorb**: a strong window absorbs a weak window when it decodes the
  same rounds again and replaces that window's answer, so the weak
  window is never decoded on its own.
- **face**: one end of a window, where it meets the window beside it.
  **Pinning a face** folds the neighbour's committed correction into the
  window's input.
- **seam**: the layer of rounds where two neighbouring windows meet.
- **forced-class job**: one decode of a window made to return its
  correction in one of the two logical classes. The complementary gap
  subtracts the weights of the two.
- **the plan**: the windows the windowing scheme lays out for each
  operation before the run starts, with the holds they place on the
  stores (`decsim/frontends/planner.py`).
- **latency point**: one named span of a window's path that the run
  folder reports as its own columns (`decsim/experiments/measure.py`,
  `POINTS`).
- **piece**: a run of consecutive seeds of one sweep point, run by one
  process and saved whole as one folder of the experiment folder, sized
  by `collection.piece_rounds` (`decsim/experiments/pieces.py`).
- **batch**: one written plan of pieces, `batches/<k>/plan.csv`, and
  the Slurm arrays that run it; `decsim run --slurm` plans the next
  batch from what the pieces say (`decsim/experiments/plan_command.py`).
- **scored shot**, **unscored shot**: a shot is scored when every
  decode it committed, provisional or final, got a correction from its
  decoder's backend, and unscored when a backend produced none; sinter
  calls an unscored shot a discard. It is counted apart and never as a
  failure (`is_scored` and `unscored_reason` in `shots.csv`).
- **trace source**: one named event a component fires and listeners
  hear, which is how observation reaches a component without a port
  (`decsim/trace_source.py`).
- **narrator**: the engine's line-by-line log of a run
  (`decsim/observe/log_writers.py`).
- **referee**: a second decoder that decodes every window again and
  counts the disagreements, turned on by
  `observation.check_windows_with`.
- **buffer0 in a column name**: the weak syndrome buffer. The csv column
  names kept an older spelling.
- **gate**, **gate point**, **golden**: the gate is a frozen set of runs
  whose results and logs are hashed, kept outside this tree, and every
  change is run against it. A gate point is one of those runs. The
  golden is the recorded hashes.

## The decoding regions

In the papers' symbols, `r_com` is the commit region's round count,
`r_buf` the buffer region's, and `r_strong` the strong region's.

| decsim | The papers | Where |
| --- | --- | --- |
| commit region, `commit_round_count` | `r_com` in Toshio, `n_com` in Skoric: the part of a window whose correction is taken as final | Toshio, arXiv:2510.25222, Sec. III C; Skoric, arXiv:2209.08552, Sec. I B (2209.08552.txt lines 194-199) |
| buffer region, `buffer_round_count` | `r_buf` in Toshio, `n_buf` in Skoric: the lookahead rounds a window reads but does not commit | Toshio Sec. III C; Skoric Sec. I B (2209.08552.txt lines 194-199) |
| window | one step of the overlapping recovery method: `commit + buffer` rounds | Dennis, Kitaev, Landahl and Preskill, arXiv:quant-ph/0110143, "Overlapping recovery method" and Fig. 13; Skoric Sec. I B, which cites it |
| strong region, `StrongWindowShape` | `r_strong`, what the strong decoder re-decodes. Toshio assumes `r_strong = r_com + 2 r_buf` | Toshio Sec. III C, Fig. 12 (2510.25222.txt lines 1248-1251) |
| restart window, `PotentialRestart` | the window the weak decoder restarts on after an escalation | Toshio Sec. III C, Fig. 12 (2510.25222.txt lines 1232-1251) |
| re-read width, `restart_reread_buffer_regions` | how far back into the strong region the restarted weak decode reads | Toshio Sec. III C |
| boundary, `boundary_in`, artificial defects | the previous window's correction folded into this one. qLDPC calls it `net_error`, cuda-q QEC calls it `syndrome_mods` | Skoric Sec. I B, which calls them artificial defects (2209.08552.txt lines 269-281) |
| seam window, sandwich schedule | Tan's type-2 window, the block between two independent type-1 windows | Tan, arXiv:2209.09219, supplementary material, the sandwich decoder section, p.14 of the arXiv pdf, and Fig. S4(b) |

## The two tiers and the confidence

| decsim | The papers | Where |
| --- | --- | --- |
| weak tier, strong tier | the fast soft-output decoder and the accurate, high-latency one it escalates to. Toshio's own words are "weak decoder" and "strong decoder" | Toshio Sec. III A, "Protocol" (2510.25222.txt lines 590-598) |
| confidence, `SoftOutput` | soft information: an analog number saying how much the decoder trusts its own answer, rather than the answer itself | Toshio Sec. II B, "Soft information in decoding problem" (2510.25222.txt lines 386-396) |
| `complementary_gap` row of `CONFIDENCE_SIGNALS` | the complementary gap: decode the window twice, each solve pinned to one logical class, and subtract the two weights | Toshio Sec. II B and Fig. 3(a,b) (2510.25222.txt lines 436-457); Gidney, Newman, Brooks and Jones, arXiv:2312.04522, Sec. "Complementary gaps" |
| `cluster_gap` row of `CONFIDENCE_SIGNALS` | the cluster gap: decode the window once and walk the clustering that decode already did | Toshio Sec. II B and Fig. 3(c,d), which cites it as ref. 47; Meister, arXiv:2405.07433, Algorithm 2 |
| `extra_cluster_gap` row of `CONFIDENCE_SIGNALS` | the extra-cluster gap: decode the window once, then grow every cluster on until the two boundaries join or the threshold's worth of growth is spent | Kishi, Toshio, Fujisaki, Oshima, Sato and Fujii, arXiv:2602.03336, Algorithm 1 and Theorems 1 and 2 |
| threshold, `gap_threshold_db` in the yaml and `threshold_decibels` on a threshold record | `g_th`, the value of the soft output below which a window is escalated. The yaml and the record are in decibels and a gap is compared in natural-log weight, the record's `threshold_nats`: nats are decibels times ln(10) over 10 (`decsim/escalation/threshold_sources.py`, `decibels_to_nats`) | Toshio Sec. III A, step 3 |
| osd | ordered statistics decoding, the post-processing step after belief propagation in BP-OSD. decsim calls the `ldpc` package's `BpOsdDecoder` | `decsim/decoders/belief_propagation_osd/decoder.py`, which names `ldpc`'s own `osd.hpp` and `stimbposd`'s `bp_osd.py` |

## The two stores

The controller writes every packed round into the store its tier reads.
A switching run's strong syndrome buffer is filled by the escalation,
which carries the strong window's rounds up from the weak syndrome
buffer. These two names are decsim's own; this table cites no paper.

| decsim | In other words | What it is for |
| --- | --- | --- |
| weak syndrome buffer, `SyndromeBuffer` in `decsim/syndrome_buffer/syndrome_buffer.py` | the streamed decoder buffer | what the weak tier reads, round by round, as it arrives |
| strong syndrome buffer, a `SyndromeBuffer`; its receiving end is `StrongSyndromeRoundReceiver` in `decsim/syndrome_buffer/strong_syndrome_round_receiver.py`, seen by the sender through the port of that name in `decsim/ports.py` | the strong side's copy of the rounds | what a strong re-decode reads, in bulk, once its boundaries are known |
| hold, `DecoderInputHold`, `PotentialStrong`, `PotentialRestart` | the reason a round may not be dropped yet | one token per consumer that still needs the round |

## The strong side's parts

The strong side is four named components and a pool of decoders. Each
is a field of a part in `decsim/build/` or a pool inside one.

| The part | decsim | What it does |
| --- | --- | --- |
| strong syndrome buffer | the readout part's `strong_syndrome_buffer`, a `SyndromeBuffer`, with `strong_syndrome_round_receiver` as its landing and `strong_output` as its read | holds the rounds an escalation carried up until the strong decode has read them |
| ledger of pending regions | the switching part's `pending_strong_windows`, `PendingStrongWindows` in `decsim/escalation/pending_strong_windows.py`, reached by the strong redecode's `pending` port | which held strong windows wait on which weak commits and stored rounds; a window leaves when its conditions fire |
| strong window manager | the switching part's `shape`, one row of `STRONG_WINDOW_SHAPES` in `decsim/escalation/strong_window_shapes.py`, with its `regions` as its geometry and `strong_redecode` as the side that submits | cuts an escalated window's strong region, names what releases it, builds its job |
| strong decoder manager | the decoders part's `strong_decoder_manager`, a second `DecoderManager` over the strong pool alone, with its own ready queue, staging and outcomes; its `strong_requests` is the ledger it shares with the chip's `decoder_manager` | gives each strong job a free strong unit and returns its result; halts a request the weak result made unnecessary |
| the strong decoders, `strong_decoder.units` of them | the `strong_decoder` row's units, a `Decoder` behind the port of that name | decode a window accurately and slowly |

## The link paths

Every hop is booked under one path name, the `LinkPath` values in
`decsim/records/transfers.py`. [The data path, hop by hop](../explanation/data_path.md) walks
them in order; this is the name list.

| decsim | The hop |
| --- | --- |
| `qpu_to_controller` | the readout electronics to the control workstation |
| `controller_to_weak_buffer` | the packed round into the weak syndrome buffer |
| `controller_to_strong_buffer` | a strong-only run's round into the strong syndrome buffer |
| `weak_buffer_to_weak_decoder` | a window's rounds into a weak unit's memory |
| `strong_buffer_to_strong_decoder` | the strong region into the strong unit |
| `weak_decoder_to_strong_decoder` | the escalation's selection, then the strong window's rounds |
| `decoder_to_decoder` | one committed window's boundary to the next window |
| `weak_decoder_to_frame`, `strong_decoder_to_frame` | a correction to the Pauli frame |
| `frame_to_controller` | the conditional release |
| `controller_to_qpu` | the instruction, and its pulse cost |

## Read next

- `decsim/records/`: the frozen records these names belong to.
- [Architecture](../explanation/architecture.md): where each of them sits.
- [The ports](ports.md): the handoffs between them.

# Glossary

decsim's names are full words (`STYLE.md` rule 2) and the literature's
names are letters and acronyms. This page is the map, both ways: what
the code calls a thing, what the papers call it, and where in the papers
to read it. Every reference below was opened before it was written down,
and every decsim name below exists in the tree. A line number into a
paper is into the research sandbox's `tmp/papers/txt/<id>.txt`, the
extraction the code's own docstrings cite.

## The words this code means precisely

- **round**: one cycle of syndrome measurement on one patch. The round
  key is `(operation_id, round_index)`, and rounds are numbered from 1.
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
- **tier**: which of the two decoders ran. `weak` is the fast one that
  decodes every window; `strong` is the accurate, slower one.

## The decoding regions

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
| threshold, `gap_threshold_nats` | `g_th`, the value of the soft output below which a window is escalated | Toshio Sec. III A, step 3 |
| osd | ordered statistics decoding, the post-processing step after belief propagation in BP-OSD. decsim calls the `ldpc` package's `BpOsdDecoder` | `decsim/decoders/belief_propagation_osd/decoder.py`, which names `ldpc`'s own `osd.hpp` and `stimbposd`'s `bp_osd.py` |

## The two stores

The controller writes every packed round to both stores at once, when
both are built. The papers' figures call them Buffer 0 and Buffer 1.

| decsim | The papers | What it is for |
| --- | --- | --- |
| syndrome buffer 0, `RoundStore` in `decsim/syndrome_buffer/round_store.py` | Buffer 0, the streamed decoder buffer | what the weak tier reads, round by round, as it arrives |
| syndrome buffer 1, `StrongRoundStore` in `decsim/syndrome_buffer/strong_round_writer.py` | Buffer 1, the room-side store | what a strong re-decode reads, in bulk, once its boundaries are known |
| hold, `DecoderInputHold`, `PotentialStrong`, `PotentialRestart` | the reason a round may not be dropped yet | one token per consumer that still needs the round |

## The link paths

Every hop is booked under one path name, the `LinkPath` values in
`decsim/records/transfers.py`. `docs/explanation/data_path.md` walks
them in order; this is the name list.

| decsim | The hop |
| --- | --- |
| `qpu_to_controller` | the readout electronics to the control workstation |
| `controller_to_weak_buffer` | the packed round into Buffer 0 |
| `controller_to_strong_buffer` | the same round into Buffer 1 |
| `weak_buffer_to_weak_decoder` | a window's rounds into a weak unit's memory |
| `strong_buffer_to_strong_decoder` | the strong region into the strong unit |
| `weak_decoder_to_strong_decoder` | the escalation's selection, and no payload |
| `decoder_to_decoder` | one committed window's boundary to the next window |
| `weak_decoder_to_frame`, `strong_decoder_to_frame` | a correction to the Pauli frame |
| `frame_to_controller` | the conditional release |
| `controller_to_qpu` | the instruction, and its pulse cost |

## Read next

- `decsim/records/`: the frozen records these names belong to.
- `docs/explanation/architecture.md`: where each of them sits.
- `docs/reference/ports.md`: the handoffs between them.

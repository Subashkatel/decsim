# Glossary

decsim's names are full words (`REWRITE.md` rule 2), and the papers'
names are letters and acronyms. This page is the map, both ways: what
the code calls a thing, what the literature calls it, and where in the
literature to read it.

## The decoding regions

| decsim | The papers | Where |
| --- | --- | --- |
| commit region, `commit_round_count` | `r_com`, the committed region of a window | Toshio, arXiv:2510.25222, Sec. III C; Skoric, arXiv:2209.08552, Sec. I B |
| buffer region, `buffer_round_count` | `r_buf`, the buffer or lookahead region | Toshio Sec. III C; Skoric Sec. I B |
| window | one round of overlapping recovery, `commit + buffer` rounds | Skoric Sec. I B; Dennis, arXiv:quant-ph/0110143 |
| strong region, `StrongWindowShape` | `r_strong = r_com + 2 r_buf`, what the strong decoder redecodes | Toshio Sec. III C |
| restart window, `PotentialRestart` | the window the weak decoder restarts on after an escalation | Toshio Sec. III C, Fig. 12 |
| re-read width, `restart_reread_buffer_regions` | how far back into the strong region the restarted weak decode reads | Toshio Sec. III C |
| boundary, `boundary_in`, artificial defects | the previous window's correction folded into this one; `net_error` in qLDPC, `syndrome_mods` in cuda-q QEC | Skoric Sec. I B |
| seam window, sandwich schedule | the type-2 block between two parallel cores | Tan, arXiv:2209.09219, supplement S8 |
| confidence, `SoftOutput` | the complementary gap `g_comp`, the weight difference between the decoded class and its complement, against the threshold `g_th` | Toshio Sec. III A; Gidney, arXiv:2312.04522 |
| weak tier, strong tier | the lightweight decoder that runs constantly and the accurate one it escalates to | Toshio Sec. II |
| osd | ordered statistics decoding, the post-processing step of belief propagation in BP-OSD | Panteleev, arXiv:1904.02703; ldpc's `BpOsdDecoder` |

## The two stores

The controller writes every packed round to both stores at once. They
are called Buffer 0 and Buffer 1 in the papers' figures.

| decsim | The papers | What it is for |
| --- | --- | --- |
| syndrome buffer 0, `round_store.py`, `RoundStore` | Buffer 0, the streamed decoder buffer | what the weak tier reads, round by round, as it arrives |
| syndrome buffer 1, `strong_round_writer.py`, `StrongRoundStore` | Buffer 1, the room-side store | what a strong redecode reads, in bulk, once its boundaries are known |
| hold, `DecoderInputHold`, `PotentialStrong`, `PotentialRestart` | the reason a round may not be dropped yet | one token per consumer that still needs the round |

## The link paths

Every hop is booked under one path name (`decsim/records/transfers.py`).
The abbreviations the earlier code used are gone.

| decsim | Was | The hop |
| --- | --- | --- |
| `qpu_to_controller` | | readout electronics to the control workstation |
| `controller_to_weak_buffer` | `cwb`, `c2b` | the packed round into Buffer 0 |
| `controller_to_strong_buffer` | `csb`, `copy_out` | the same round into Buffer 1 |
| `weak_buffer_to_weak_decoder` | `wbd`, `cwd` | a window's rounds into a weak unit's input slot |
| `strong_buffer_to_strong_decoder` | `sbd`, `csd` | the strong region into the strong unit |
| `weak_decoder_to_strong_decoder` | | the escalation's selection, no payload |
| `decoder_to_decoder` | | one committed window's boundary to the next window |
| `weak_decoder_to_frame`, `strong_decoder_to_frame` | | a correction to the Pauli frame |
| `frame_to_controller` | | the conditional release |
| `controller_to_qpu` | | the instruction, and its pulse cost |

## The renamed modules

Rule 2 spelled the acronyms out. If you are coming from the earlier
tree or from a paper's notation, this is where each thing went.

| Now | Was |
| --- | --- |
| `decsim/records/` (one module per record family) | `decsim/message.py` |
| `decsim/machine.py`, `decsim/front/experiment.py` | `decsim/run_spec.py`, `decsim/run_configuration.py` |
| `decsim/decoders/minimum_weight_perfect_matching/decoder.py` | `decsim/decoders/mwpm/window_decoder.py` |
| `decsim/decoders/belief_propagation_osd/decoder.py` | `decsim/decoders/bposd/window_decoder.py` |
| `decsim/ports.py` | `decsim/protocols.py` |
| `decsim/detector_error_model/` | `dem` in the module and file names |
| `decsim/detector_error_model/stim_fault_catalog.py` | `stim_dem_catalog.py` |
| `decsim/syndrome_buffer/round_store.py`, `strong_round_writer.py` | `syndrome_buffer.py`, `syndrome_buffer_1.py` |
| `decsim/controller/round_assembly.py`, `round_writes.py`, `round_transmission.py` | `controller/syndrome_packing.py` |
| `decsim/escalation/` | `decoders/weak_strong_switching.py` |
| `decsim/links/fabric.py`, `channel.py` | `links/links.py` |
| `decsim/observe/link_traffic.py` | `links/link_traffic_report.py` |
| `decsim/decoders/relay_belief_propagation/` | `decoders/relay_bp/` |
| `decsim/decoders/belief_matching/decoder.py` | `belief_matching/window_decoder.py` |
| `decsim/decoders/staged_decoder.py` | `decoders/decoder_engine.py` |
| `decsim/escalation/strong_redecode.py`, `strong_window_shapes.py` | `decoders/strong_escalation.py` |
| `decsim/decoders/decode_outcomes.py`, `backend_outcome.py` | `decoders/window_decode_results.py` |
| `decsim/windows/window_planner.py`, `round_tracker.py` | `windows/dynamic_windows.py`, dissolved into them |

One acronym is only half gone. The modules and the files spell it
out, and the code has not caught up: `DecodeJob.dem`, the field every
plug-in decoder reads, and the other `dem` names beside it still carry
it. A rule 2 surface commit for them is owed.

## Words this code means precisely

- **round**: one QEC cycle on one patch. The round key is
  `(operation_id, round_index)`, and rounds are numbered from 1.
- **tick**: the engine's integer time unit. One microsecond is a
  million ticks, and every cost in the machine is an integer of them.
- **operation**: one logical operation of the workload, the unit a
  workload is written in and the unit the QPU runs.
- **window key, request, service**: the window `(operation_id,
  window_id)`, the decode asked for it (`DecoderRequestKey`, which
  carries the tier), and the run of that decode on a unit
  (`DecoderServiceKey`).
- **move, copy, reference**: whether a hop leaves the bits behind, ends
  with both sides holding them, or hands over an object both sides read.
  The traffic ledger books each hop as one of the three.
- **reaction time**: the time from a round's measurement to the
  instruction that acts on its correction, which is what a run measures.

## Read next

- `decsim/records/`: the frozen values these names belong to.
- `docs/architecture.md`: where each of them sits in the pipeline.

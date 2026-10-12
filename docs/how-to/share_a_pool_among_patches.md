[decsim docs](../README.md) › [How-to guides](README.md)

# How to share a pool among patches

When a pool has fewer units than the patches that send it windows, its
scheduler chooses which waiting window a free unit takes next
(`decsim/decoders/schedulers.py`). FIFO, the default, takes the oldest.
The other three rows are VQD's policies (Maurya and Tannu 2406.17995v2
Sec. 4.2), checked against the authors' simulator
(github.com/satvikmaurya/decoder-resources, `scripts/sim.py`,
`Sim.schedule`).

## 1. Choose the rule

```bash
decsim run run.py --set decoder_manager.scheduler.type=round_robin
```

| Row | VQD | The rest go to |
|---|---|---|
| `FifoScheduler` | | the oldest job |
| `longest_undecoded` | MLS, Sec. 4.2.3 | the patch with the most rounds waiting |
| `round_robin` | RR, Sec. 4.2.2 | the next patch in turn; a served patch goes to the back |
| `most_critical` | MFD, Sec. 4.2.1 | the patch with the most critical decodes not yet served |

The three VQD rows take the waiting jobs in three tiers, the older job
first within a tier:

1. Every job on a patch with a waiting critical job. A critical job
   belongs to an operation that another operation is `blocked_by`, a
   decode a non-Clifford gate waits on (Sec. 2.4). The whole backlog of
   its patch goes with it, since the gate needs every syndrome
   generated up to that point.
2. Every job on a patch with a waiting job over several patches.
   `sim.py` gives such a measurement priority 5, under a critical
   one's 1000; the paper's text does not mention it.
3. The rest, by the row's policy.

## 2. Know what decsim reads differently

- VQD plans one slice of d rounds at a time, at compile time. A row
  here orders the queue each time a unit frees, as Triage's steady mode
  does (2605.04459 Sec. III-B1).
- A scheduler reads the planned operations once (`read_program`) and
  hears of each job its pool dispatches (`served`). A row of your own
  implements both, and `pop`.
- A program with a stream (a protected region) is refused, not
  implemented yet ([the refusals](build_each_switching_pattern.md#what-is-refused-for-now)). A stream's critical point is a segment's
  end, which the scheduler does not read.
- Triage's EDF and MDF are not built. They rank slices of a constraint
  graph, whose neighbours may not decode at once, by a compiled
  deadline. decsim has neither the graph nor the deadline.

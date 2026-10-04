[decsim docs](../README.md) › [How-to guides](README.md)

# How to add a store, a link card, a signal, a policy or a scheme

You want a machine to use a weak syndrome buffer, a link card, a
confidence signal, an escalation policy or a windowing scheme of your
own. Each one is a
record you put in a field of the machine's settings. A decoder is the
same kind of job, with a page of its own:
[How to add a decoder backend](add_a_decoder_backend.md).

## The steps they share

1. **Write the record.** A frozen dataclass with the members its field
   asks for. The field's type is a Protocol named for the record, such
   as `SyndromeBufferSettings` or `ConfidenceSettings`, and
   [the parts page](../reference/parts.md) lists its members. No
   registration line is needed: any record with those members fits.
2. **Write the component.** The record's `build` returns it, and it
   fills the port the field's component fills (`decsim/ports.py`).
3. **Put the record in the field** with `dataclasses.replace`, as the
   decoder how-to does for `weak_decoder`.
4. **Name it.** A short form in a class, a folder or a record's `name`
   needs a row in [the glossary](../reference/glossary.md). A row
   under `decsim/` also joins its package docstring's list of rows,
   where it has one.
5. **Run `python tools/docs_map.py`** if the record lives under
   `decsim/`, so the parts page lists it; `tests/test_docs.py` fails
   until it does.
6. **Test the component alone,** beside the shipped ones. Build it on a
   fresh `decsim.engine.Engine()`, call its port's methods and run the
   engine. Time is in ticks, a million to the microsecond
   (`config.TICKS_PER_MICROSECOND`, `config.microseconds_to_ticks`). A
   cost of n cycles on a `config.Clock` starts at the clock's edge at
   or after now, so from mid-cycle it ends n periods after that edge,
   as gem5's `clockEdge` does.

## A weak syndrome buffer

- **Field:** `weak_syndrome_buffer`, typed `ports.SyndromeBufferSettings`.
- **Record:** `bits` (the bound, None for none), `clock` (None is the
  machine's), `prices_read_bits` and `build(engine)`. Set
  `prices_read_bits` to `True` when your store prices a read's bits
  itself; the build then refuses a rate on `weak_buffer_to_weak_decoder`,
  which would charge the same bits again.
- **Component:** inherit `SyndromeBuffer`
  (`decsim/syndrome_buffer/syndrome_buffer.py`) and override
  `book_write(round_key, bits)` and `book_read(round_keys)`. Each
  returns the tick its access completes. `book_write` is called once
  per round as it lands, and the round is published at the tick it
  returns. `book_read` is called once per decode job, with every round
  the job reads, at the tick the job asks. A round is always written
  before it is read.
- **Trace:** fire `self.trace.access_served` on each access to get the
  store's `port <i>` lanes ([the run folder](../reference/run_folder.md),
  the trace row).
- **Worked example:** `decsim/syndrome_buffer/ported_syndrome_buffer.py`.
  **Tests:** `tests/syndrome_buffer/`.
- **Where the time lands:** a read's time is in the `dep_block`
  latency point. A write's time is in no per-stage point: it delays
  the round's publication, so it shows in the `qpu_*` totals and not
  in the `buffer0_*` ones.

## A link card

A card is numbers, not a class: a `FabricSettings` record, one
`PathSettings` per hop, put in the machine's `links` field.

- **Start** from `link_profiles.logical_reference_profile()`
  (`decsim/links/link_profiles.py`) and replace the hops you have
  numbers for. To time a strong-side hop with no weak decoder in the
  way, start the machine from
  `decsim.settings.strong_decoder_baseline(3, 0.001, 1.0)` (distance,
  physical error probability, round period in microseconds), the strong
  decoder alone.
- **One hop in cycles of a clock:** `link_profiles.path_card(links,
  "weak_decoder_to_strong_decoder", clock=..., latency_cycles=...,
  bits_per_cycle=..., source=...)` returns the hop's `PathSettings`;
  `bits_per_cycle=None` is an unbounded wire.
- **One hop's latency alone:** `link_profiles.with_path_latency(links,
  path_name, latency_microseconds)` returns the whole card.
- **Two hops on one wire** share a `ChannelSettings.name`; a rate is
  bits per microsecond, a `fractions.Fraction`.
- **Write the paper's numbers in your run file**, each with its source
  in a comment (STYLE.md rule 2), not in decsim.
- **Check the card** by pricing the paper's own transfer on it:
  `decsim.links.fabric.LinkFabric(card, engine).expected_delay_ticks(
  path, bits, engine.now)` is what one send costs on an idle wire
  (`tests/links/test_fabric.py`). `tests/links/test_link_profiles.py`
  checks the measured cards against their papers' transfers.

## A confidence signal

- **Field:** `switching.confidence`, typed `ConfidenceSettings`.
- **Record:** `name` and `build(weak_algorithm, threshold_nats)`, which
  builds the signal from the weak decoder's record and the threshold.
- **Component:** fills `ports.ConfidenceSignal`, with five
  attributes:
  - `source`: `SoftOutputSource(method=...)`, the signal's name;
  - `fault_model_requirement`: what a window model must offer,
    `NO_FAULT_MODEL_REQUIRED` for nothing
    (`decsim/detector_error_model/fault_model_contracts.py`);
  - `decoder_evidence_requirement`: what the decode must show,
    `NO_DECODER_EVIDENCE` for nothing (`decsim/records/decoding.py`);
  - `evidence_refusal`: the sentence the build prints when the weak
    decoder cannot show it;
  - `forced_logical_classes`: the classes each window is decoded in,
    one decode each, `()` for one ordinary decode.
- **`compute(solves)`** gets the window's `DecodeResult`s, one per
  forced class, or one, and returns
  `SoftOutputComputation(SoftOutput(gap, source), ticks)`, the ticks
  being the signal's own computation.
- **A result carries:**
  - `detection_events`: a read-only uint8 array, one 0 or 1 per
    detector row of the window's model, the bits the decode read.
    Every decoder built on `WindowDecoderBase` fills it.
  - `cluster_evidence`, from a decoder that grows clusters.
  - `forced_class_weight`, from a forced solve.
- **The gap's unit is nats,** a natural-log weight, and never negative.
  The policy keeps a window whose gap is at or above the threshold;
  `math.inf` is certain, and a soft output of `None` escalates.
- **A signal where higher is worse,** such as a count of detection
  events, reports a gap that falls as the count rises and stops at
  zero. A gap of zero escalates only when the threshold is above zero.
- **The weak pool's solves:** `linear_decoder_pool`'s
  `solves_per_window` splits a window's decode time over the solves,
  so give it `len(forced_logical_classes)`, or 1 when that is empty.
- **Evidence:** name what your signal reads in
  `decoder_evidence_requirement`; the build refuses a weak decoder that
  does not declare it, with the decoder's own reason or else your
  `evidence_refusal`.
- **Worked example:** `decsim/confidence/cluster.py`.
  **Tests:** `tests/confidence/`.

## An escalation policy

- **Field:** `switching`, a `SwitchingSettings`. Subclass it as a
  frozen dataclass, add your own fields, and override
  `build_policy(threshold)` to return your policy.
- **Component:** fills `ports.EscalationPolicy`. The port's docstrings
  say when each method is called: a verdict once per weak window, and
  the strong answer once per strong decode.
- **Counts your policy keeps** are not columns of the results. Count
  them from `window_confidence.csv` or from the trace's `verdict`
  events instead.
- Under the default redo window, one stream has at most one strong
  decode in flight ([Two tiers](../tutorials/two_tiers.md)).
- **Worked example:** `Switching` in `decsim/escalation/policies.py`.
  **Tests:** `tests/escalation/`.

## A windowing scheme

- **Field:** `windows.scheme`, typed `SchemeSettings`
  (`decsim/windows/settings.py`).
- **Record:** `name`, `commit_rounds` and `buffer_rounds` (None is the
  code distance) and `build(terminal_policy)`. Check the sizes with
  `window_data.check_window_sizes(self)` in `__post_init__`, and add
  your row to `SCHEME_ROWS` in `tests/windows/test_settings.py`.
- **Component:** fills `ports.WindowingScheme`. Set its three flags;
  the port's docstring says what each one means. `data_complete` says
  when a window has its rounds; `window_data.sliding_data_complete` is
  the rule every shipped row uses.
- **The plan:** `plan_operation` returns an `OperationWindowPlan`
  (`decsim/records/windows.py`). Rounds count from 1.
  - `windows`: one `WindowGeometry` per window, `buffer_lo` to
    `buffer_hi` read, `commit_lo` to `commit_hi` committed.
  - `internal_dependencies`: `(earlier, later)` window indices; the
    later window waits for the earlier one.
  - `entry_window_indices` wait for the operation before;
    `exit_window_indices` are what the operation after waits for.
  - `windowed` False decodes the operation as one batch, and
    `batch_preceding_idle_rounds` folds the idle rounds before it into
    that batch. Only `naive_online` sets them so.
  - `protocol`: `WindowProtocol.GENERIC`. The other member is Tan's
    sandwich alone: one-layer seams, graphlike decoders only.
- **Every plan must hold,** or the build refuses it:
  - the commit regions, in plan order, cover the rounds with no gap or
    overlap;
  - a window with `closed_temporal_boundaries` is the later end of a
    dependency and cuts no fault.
- **Which window decodes a fault:** with dependencies, the shallowest
  window whose commit rounds the fault touches
  (`decsim/detector_error_model/window_ownership_dag.py`). So a seam
  decodes only its own rounds.
- **On a machine:**

  ```python
  windows = dataclasses.replace(base.windows, scheme=MyScheme.Settings())
  machine = dataclasses.replace(base, windows=windows)
  ```

- **Worked example:** `decsim/windows/schemes/sandwich.py`.
  **Tests:** `tests/windows/schemes/test_sandwich.py`.

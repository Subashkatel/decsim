from __future__ import annotations
 
from typing import Callable, Optional, TYPE_CHECKING
 
from .config import fmt
from .engine import Engine
from .message import Operation, SyndromePayload, DecodeJob, DecodeResult, Window, WindowPlan
 
from .codes import SurfaceCodeModel
from .links import LinkModel
from .schemes import SlidingWindowScheme
from .planner import WindowPlanner, FixedRounds
from .layouts import UniformLayout
from .decoders import CodeRouter
from .schedulers import EnqueueTimeDeadline
 
if TYPE_CHECKING:                      # type-only; these collaborators are handed to __init__
    from .protocols import (Decoder, Scheduler, Controller, Orchestrator,
                            CodeModel, DecodingScheme, LayoutModel, RoundsPolicy,
                            DecoderRouter, DeadlinePolicy)
    from .switching import Switching
# ========================================================================================
# CLUSTER
# This module defines the decoder cluster -- the paper's decoder cluster with its
# "efficient workload manager that queues the decoding jobs and communicates dependencies
# between them", where the dependencies are "boundaries of committed decoding regions from
# prior decoding windows" (arXiv:2511.10633 Sec III) -- the main piece of the decoding side.
# It receives the WindowPlan from the orchestrator's planner ahead of time (load_execution_plan),
# runs the queue against it, exchanges committed window boundaries between decoders (the t_dd hop),
# and DELIVERS each finished operation result to the orchestrator (the t_do hop); the
# ORCHESTRATOR then owns sending any conditional instruction back to the chip.
# The cluster never reaches into the chip or factory (dependency inversion); it only fires the
# lifecycle callback `on_workload_complete` the wiring installs.
# ========================================================================================

class DecoderCluster:
    """
    Owns three things:
      - per-operation window buffers (rounds accumulate here until complete),
      - the BLOCKED set (window complete but a dependency unmet -> not runnable),
      - the READY QUEUE + a pool of decoder units (runnable jobs wait for a free unit).
        The ready queue is THE queue of syndromes/jobs waiting to be decoded.
    """

    def __init__(self, engine: Engine, decoder: Decoder, scheduler: Scheduler,
                 controller: Controller, orchestrator: Orchestrator,
                 num_units: int, code_distance: Optional[int] = None,
                 rounds_per_op: int = 11, *, code: Optional[CodeModel] = None,
                 scheme: Optional[DecodingScheme] = None,
                 layout: Optional[LayoutModel] = None,
                 decoders: Optional[dict] = None,
                 rounds_policy: Optional[RoundsPolicy] = None,
                 router: Optional["DecoderRouter"] = None,
                 deadline_policy: Optional["DeadlinePolicy"] = None,
                 links: Optional[LinkModel] = None,
                 unit_pools: Optional[dict] = None,
                 switching: Optional["Switching"] = None):
        """Set up the cluster: decoder units, ready queue, syndrome buffer, bookkeeping.
        `controller` is accepted as part of the construction contract (the make_cluster
        hook signature) but no longer read -- link prices come from `links`."""
        self.engine = engine
        self.decoder = decoder
        # G1 -- HETEROGENEOUS PER-CODE DECODING. `decoder` is the DEFAULT decoder, used for any
        # job whose code is not in `decoders`. `decoders` is an optional {code_name: Decoder}
        # map so each code can be decoded by its own algorithm (surface by MWPM, BB by Relay-BP,
        # ...). A job carries its code on DecodeJob.code; the ROUTER picks the right one.
        # With no map (the common single-code case) every job routes to `decoder`, so behaviour
        # is byte-identical to before this seam existed.
        self.decoders = dict(decoders) if decoders else {}
        # ROUTING seam: a DecoderRouter picks the decoder PER JOB. The default CodeRouter
        # reproduces the per-code map above; a custom router can route by job.hint/attempt
        # instead -- the seam decoder switching needs (arXiv:2510.25222), and the way to mix
        # decoding devices with different speeds (FPGA/ASIC/GPU latency models) in one cluster.
        self.router = router if router is not None \
            else CodeRouter(default=decoder, by_code=self.decoders)
        # DEADLINE seam: assigns each window job its deadline at enqueue time. The default
        # (enqueue time) makes EarliestDeadlineScheduler behave like FIFO, as before; pass
        # ReactionPathDeadline to prioritize windows whose result gates a waiting non-Clifford
        # gate (the reaction path of arXiv:2511.10633).
        self.deadline_policy = deadline_policy if deadline_policy is not None \
            else EnqueueTimeDeadline()
        self.scheduler = scheduler
        self.orchestrator = orchestrator
        # LINKS: the fabric price list for the two hops the cluster pays (dd boundary
        # messages, do result delivery). The wiring passes the SAME LinkModel it gave
        # the controller, so the fabric has one source of truth; the default (a flat
        # Table-2 LinkModel) covers direct construction in tests.
        self.links = links if links is not None else LinkModel()
        # DECODER UNITS. By default one anonymous pool of `num_units` interchangeable
        # units (the original behavior). `unit_pools` (e.g. {"default": 4, "strong": 1})
        # splits the cluster into NAMED pools: each pool owns its units AND its own
        # ready queue, and a job runs on the pool its hint names ("default" otherwise).
        # A slow strong-decoder job can then neither occupy a weak unit nor make ready
        # weak windows queue behind it -- the device split of arXiv:2510.25222 Fig 1
        # (weak = FPGA/ASIC, strong = CPU/GPU). When unit_pools is given it defines ALL
        # the units and num_units is ignored.
        if unit_pools is None:
            unit_pools = {"default": num_units}
        if "default" not in unit_pools:
            raise ValueError(f'unit_pools must include a "default" pool '
                             f'(got {sorted(unit_pools)})')
        for pool_name, units in unit_pools.items():
            if units < 1:
                raise ValueError(f"pool {pool_name!r} needs at least 1 unit (got {units})")
        self.unit_totals = dict(unit_pools)          # pool -> units it owns
        self.pool_free = dict(unit_pools)            # pool -> units free right now
        self.num_units = self.unit_totals["default"]   # back-compat (metrics read it)
        # The QEC code is now a swappable model. Passing code_distance=d (back-compatible)
        # constructs a surface code of that distance, reproducing the original behavior.
        if code is None and layout is None:
            if code_distance is None:
                raise ValueError("provide code=<CodeModel>, layout=<LayoutModel>, "
                                 "or code_distance=<int>")
            code = SurfaceCodeModel(d=code_distance)
        # The LAYOUT seam: which code each patch uses. A single code (or code_distance) means a
        # UniformLayout -- one zone -- which reproduces the original single-code behavior exactly.
        # A heterogeneous QPU passes layout=ZonedLayout(...) instead.
        if layout is None:
            self.layout = UniformLayout(code)
        else:
            self.layout = layout
        # A representative code for the summary log and the window-timing defaults below. For a
        # uniform layout this IS the one code, so the logged plan is unchanged.
        self.code = code if code is not None else self.layout.codes()[0]
        # The window/commit policy is a swappable scheme; the default is fixed sliding window.
        self.scheme = scheme if scheme is not None else SlidingWindowScheme()
        self.d = self.code.distance
        self.commit = self.code.commit_rounds()     # commit region (d rounds for surface code)
        self.buffer = self.code.buffer_rounds()     # buffer/lookahead (d rounds for surface code)
        # ROUNDS seam (D1 fix). rounds-per-operation is a swappable policy. A plain rounds_per_op
        # int is wrapped as FixedRounds (the original behaviour, byte-identical regression); pass
        # rounds_policy=CodeRounds() for per-code rounds so high-distance zones' buffers fit.
        self.rounds_policy = rounds_policy if rounds_policy is not None \
            else FixedRounds(rounds_per_op)
        # per-op bookkeeping
        self.ops: dict[int, Operation] = {}
        self.rounds_arrived: dict[int, int] = {}   # highest round number arrived for op
        self.memory_rounds: dict[int, int] = {}    # idle memory rounds (gated successors)
        # REAL syndrome buffer: arriving payloads are retained here (the decoder-cluster RAM
        # of arXiv:2511.10633 Sec III) until their window decodes, then assembled into the
        # decode job. Keyed [op_id][round_index][patch_id] -> SyndromePayload: a round may
        # arrive as several per-patch FRAGMENTS (SyndromePayload.n_fragments) when the
        # device emits per-patch payloads; the single-payload default stores one fragment
        # per round, identical to before. In timing-only mode the payloads carry no bits,
        # but the plumbing is identical, so a real decoder needs no cluster change -- only
        # a real DeviceModel + Decoder.
        self.payload_store: dict[int, dict[int, dict]] = {}
        # syndrome-RAM accounting: a running count kept in step with payload_store
        # (+1 when a payload is stored, -an op's payloads when its store is freed),
        # so the high-water mark costs O(1) per round instead of recounting the store.
        self.payloads_held = 0                     # payloads retained right now
        self.peak_payloads = 0                     # most ever retained at once (storage high-water)
        # Free each round's syndrome data from memory as soon as every window that needs it has
        # finished decoding (paper: arXiv:2511.10633 Sec VI.B). A single round is needed by a few
        # windows: the window that decodes it, the window just before it (which looks at it as
        # look-ahead), and -- across operations -- a previous operation whose look-ahead spilled
        # into it. _round_refs counts, for each round, how many of those windows haven't finished
        # yet; when that count hits zero (in _on_decode_done) we drop the round. This keeps
        # peak_payloads at the real amount of data held at once, instead of keeping a whole
        # operation's data until the operation completely finishes.
        self._round_refs: dict[tuple, int] = {}    # (operation, round) -> windows still needing it
        self._read_sets: dict[tuple, list] = {}    # window -> the (operation, round)s it reads
 
        # the sliding-window plan (built once all ops are loaded)
        self.windows: dict[tuple, Window] = {}
        self.op_windows: dict[int, list] = {}
        self.nwin: dict[int, int] = {}
        self.successors: dict[int, list] = {}
        self.committed_windows: set = set()
        self._committed_per_op: dict[int, int] = {}   # committed-window count per op
        self._gating_ops: set[int] = set()            # ops whose result gates a successor
        self.op_results: dict[int, int] = {}      # accumulated logical value per op (real decoders)
        self.window_models: dict = {}             # (op_id, k) -> WindowErrorModel, built at plan load
        #                                           for ops that carry a stim circuit (real decoding)
        self.total_windows = 0
        self._windows_built = False
        self._plan_spatial = None                 # per-op decode sizes from the loaded plan
        self.ready: list[DecodeJob] = []          # THE ready queue (= the default pool's)
        self.pool_ready: dict[str, list] = {p: [] for p in self.unit_totals
                                            if p != "default"}   # other pools' queues
        self.queue_log: list[tuple[int, int]] = []
        # DECODER SWITCHING (a Switching object from switching.py, or None for no switching).
        # When set, the weak `decoder` is paired with a strong one (reached through the router via
        # hint="strong"), and the cluster runs the strong decoder on the windows the weak decoder is
        # unsure about. None = no switching, so every switching branch below is skipped.
        self.switching = switching
        self.strong_needed = 0        # windows the weak decoder was unsure about (the strong decoder was needed)
        self.strong_cancelled = 0     # strong re-decodes cancelled early ("run both at once", weak turned out confident)
        self._running_strong_decodes: dict[tuple, DecodeJob] = {}   # window key -> its strong re-decode job
        # Dependency INVERSION (no back-reference to the chip): the cluster -- the paper's
        # "workload manager" (arXiv:2511.10633 Sec III) -- never reaches into the chip or factory.
        # It queues decoding jobs, exchanges committed window boundaries (the t_dd dependencies),
        # and DELIVERS each finished result to the orchestrator (the t_do hop). It does NOT own
        # the conditional return path: dispatching a decision back to the chip is the
        # ORCHESTRATOR's job (orchestrator.integrate -> controller -> chip). The cluster keeps one
        # lifecycle sink the wiring fills in:
        #   on_workload_complete()  -- fire when the last window has committed (e.g. factory.shutdown)
        self.on_workload_complete = None          # Optional[Callable[[], None]]
 
    def register_op(self, op: Operation) -> None:
        """Start tracking a new operation (arrived rounds, payload buffer, etc.).
        Idempotent: re-registering an already-known op refreshes the reference but keeps its
        runtime state, so the orchestrator can register ops before the chip loads them."""
        if op.id not in self.ops:
            self.rounds_arrived[op.id] = 0
            self.memory_rounds[op.id] = 0
            self.payload_store[op.id] = {}
        self.ops[op.id] = op
        if op.gated_by is not None:
            # the GATING op's decode result releases this op: its windows sit on the
            # reaction path (used by the DeadlinePolicy to prioritize them).
            self._gating_ops.add(op.gated_by)

    @property
    def free_units(self) -> int:
        """Free units in the DEFAULT pool (back-compat; pool_free has every pool)."""
        return self.pool_free["default"]

    def _pool_for(self, job: DecodeJob) -> str:
        """The unit pool a job runs on: its hint when that names a pool, else default."""
        return job.hint if job.hint in self.unit_totals else "default"

    def _queue_for(self, pool: str) -> list:
        """A pool's ready queue (self.ready IS the default pool's queue)."""
        return self.ready if pool == "default" else self.pool_ready[pool]

    def _queued_total(self) -> int:
        """Jobs waiting across ALL pools' queues -- what queue_log records (identical to
        len(self.ready) when no extra pools exist)."""
        return len(self.ready) + sum(len(q) for q in self.pool_ready.values())

    @staticmethod
    def _pool_tag(pool: str) -> str:
        """Log prefix naming the pool -- empty for the default pool, so single-pool
        traces are unchanged."""
        return "" if pool == "default" else f"{pool} "

    def _decoder_for(self, job: DecodeJob) -> "Decoder":
        """Pick the decoder for a job via the ROUTER. The default CodeRouter routes by
        job.code (G1) and falls back to the default `decoder`; a custom DecoderRouter can
        route by job.hint/attempt instead (decoder switching, heterogeneous device pools)."""
        return self.router.route(job)
 
    def rounds_for(self, op: Operation) -> int:
        """Rounds this operation runs for, via the ROUNDS policy under the op's own code (D1).
        The planner, this cluster (window completion / payload assembly), and the chip all read
        this, so they agree on each operation's temporal length even across heterogeneous codes."""
        return self.rounds_policy.rounds_for(op, self.layout.code_for_op(op))
 
    def _spatial_nodes(self, op: Operation) -> int:
        # The decode-job size is part of the plan the orchestrator hands over; if a plan is
        # loaded, read it from there. Otherwise (no plan yet) fall back to the layout directly.
        # A uniform layout reproduces the single code's aggregate spatial_nodes(num_patches).
        """Decoding-graph size for an operation (from the loaded plan, else the layout)."""
        if self._plan_spatial is not None and op.id in self._plan_spatial:
            return self._plan_spatial[op.id]
        return self.layout.spatial_nodes_for(op)
 
    def load_execution_plan(self, plan: WindowPlan) -> None:
        """Receive the pre-computed WindowPlan from the orchestrator's planner and install it
        (arXiv:2511.10633 Sec III: the plan is communicated to the cluster AHEAD OF TIME). The
        cluster does NOT compute windows in this path -- it just holds the plan and runs the
        queue against it at runtime. Called once, before syndromes flow."""
        if self._windows_built:
            return
        self._windows_built = True
        self.windows = plan.windows
        self.nwin = plan.nwin
        self.op_windows = plan.op_windows
        self.successors = plan.successors
        self._plan_spatial = plan.spatial_nodes
        self.total_windows = plan.total_windows
        # wire dependents into the cluster's view (the plan already filled them on the Windows)
        wpo = plan.nwin.get(next(iter(self.ops), 0), 0)
        rpo = plan.summary.get('rounds_per_op', '?')
        # the scheme reports its OWN strategy (ground truth); the trace never infers
        # "is this windowed?" from the window count. NaiveOnlineScheme => global decode.
        label = getattr(self.scheme, "scheme_label", type(self.scheme).__name__)
        # describe the ACTUAL first-window geometry from the plan, NOT the code's nominal
        # commit/buffer: a global scheme ignores those (its single window spans the whole op
        # with no buffer), so printing commit=d/buffer=d would falsely imply a size-2d window.
        first_op = next(iter(self.ops), None)
        w0 = self.windows.get((first_op, 0)) if (first_op is not None and self.windows) else None
        njobs = plan.total_windows
        jobs = f"{njobs} decode job" + ("" if njobs == 1 else "s")
        if not self._windowed:
            struct = f"each {rpo}-round operation decoded as one batch"
        elif w0 is not None:
            cs, bf = w0.commit_hi - w0.commit_lo + 1, max(0, w0.buffer_hi - w0.commit_hi)
            struct = (f"each {rpo}-round operation split into {wpo} window" + ("" if wpo == 1 else "s")
                      + f" of {cs} commit + {bf} buffer ({cs + bf} rounds each)")
        else:
            struct = f"{wpo} window(s) per operation"
        # two short lines: the strategy, then the job structure in plain words.
        self.engine.log("DecoderClstr", f"received execution plan: d={self.d}, scheme={label}")
        self.engine.log("DecoderClstr", f"  -> {jobs}: {struct}")
        # Switching can require a minimum window size (the weak decoder must keep pace with its
        # commit region). Check it once, loudly, when the plan loads.
        if self.switching is not None:
            self.switching.check_window_size(self.commit, self.buffer)
        self._build_window_error_models()
        self._build_round_refcounts()

    def _build_window_error_models(self) -> None:
        """For every op that carries a stim circuit, slice its detector error model into
        per-window WindowErrorModels and file them by window key -- compile-time data
        (zero simulated ticks), attached to each window's DecodeJob as job.dem so a real
        decoder has its decoding problem (docs/DESIGN-real-window-decoding.md, phase R2).

        Round convention shared with StimDevice: chip round r holds stim layer t = r - 1,
        layers past the chip's last round fold into the last round (round = min(t+1, R)),
        so a window's concatenated payload bits equal its model's rows exactly.

        Scope (documented limits, loud where it matters): windows with a LEADING buffer
        (the parallel A/B scheme) are skipped -- their jobs keep dem=None and decode as
        timing-only, as before. Idle-round growth (the naive scheme's
        prepend_idle_rounds) happens after planning; those ops are skipped too since
        their window geometry no longer matches the circuit."""
        # A belief-matching (or other BP-reweighting) decoder needs the per-window hyperedge
        # graph too; it advertises that via needs_hyperedges. Build the extra matrices iff some
        # routed decoder wants them -- otherwise the slice is byte-identical to before.
        want_bm = getattr(self.decoder, "needs_hyperedges", False) or any(
            getattr(d, "needs_hyperedges", False) for d in self.decoders.values())
        for op_id, op in self.ops.items():
            if op.circuit is None:
                continue
            keys = [(op_id, k) for k in self.op_windows.get(op_id, [])]
            wins = [self.windows[key] for key in keys]
            if not wins or any(w.buffer_lo not in (None, w.commit_lo) for w in wins):
                continue                       # A/B leading buffers: timing-only for now
            from .adapters.window_error_models import build_window_error_models
            R = self.rounds_for(op)
            coords = op.circuit.get_detector_coordinates()
            folded = {det: min(int(c[-1]) + 1, R) for det, c in coords.items()}
            plan = [(w.commit_lo, w.commit_hi, min(w.buffer_hi, R)) for w in wins]
            models = build_window_error_models(op.circuit, plan,
                                               detector_rounds=folded,
                                               belief_matching=want_bm)
            for key, model in zip(keys, models):
                self.window_models[key] = model
            n = len(models)
            self.engine.log("DecoderClstr",
                            f"{op.name}: built {n} decode error model(s) "
                            f"({sum(m.check.shape[1] for m in models)} fault columns)")
 
    def _build_round_refcounts(self) -> None:
        """Work out which rounds each window reads, then count how many windows read each round --
        so we can free a round from memory the moment the last window that needs it has decoded
        (paper: arXiv:2511.10633 Sec VI.B, "discard ... as soon as the associated decoding tasks
        are complete").

        For each window we list the exact rounds it reads (the same ones _assemble_payloads hands
        to the decoder): its own commit + look-ahead rounds, and -- if its look-ahead runs past the
        end of this operation -- the first few rounds of the next operation. We then tally, per
        round, how many windows read it. Because neighbouring windows share rounds (one window's
        look-ahead is the next window's commit region) and a previous operation can spill over,
        freeing only when the tally reaches zero is correct no matter what order windows finish."""
        self._round_refs = {}
        self._read_sets = {}
        for key, w in self.windows.items():
            R_op = self.rounds_for(self.ops[w.op_id])
            reads = [(w.op_id, r) for r in range(w.start_round, min(w.buffer_hi, R_op) + 1)]
            overflow = w.buffer_hi - R_op
            if overflow > 0:
                for s in self.successors.get(w.op_id, []):
                    reads += [(s, r) for r in range(1, overflow + 1)]
            self._read_sets[key] = reads
            for rk in reads:
                self._round_refs[rk] = self._round_refs.get(rk, 0) + 1

    def build_windows(self) -> None:
        """Back-compat entry: if no execution plan has been loaded yet, build one in place
        (using this cluster's scheme + layout) and install it.
 
        The PREFERRED path is for the orchestrator's WindowPlanner to compute the plan and call
        load_execution_plan() ahead of time -- the planning then lives in the planner, not here
        (see the ROLE MAP in the Section 3 banner). This shim keeps direct chip.load() callers
        working and produces an identical plan, so behavior is unchanged."""
        if self._windows_built:
            return
        planner = WindowPlanner(self.scheme, self.layout, self.rounds_policy)
        self.load_execution_plan(planner.plan(list(self.ops.values())))
 
    # ---- a syndrome round has arrived from the controller -------------------
    def on_syndrome_arrival(self, payload: SyndromePayload) -> None:
        """A syndrome round arrived: buffer it and re-check which windows can now decode."""
        op = self.ops[payload.operation_id]
        # retain the payload in the cluster's syndrome buffer until its window decodes.
        # A round counts as ARRIVED only once all its fragments are in. With the standard
        # ModularController this is trivially true on the last delivery (the CONTROLLER
        # aggregates a round's fragments and ships them as one atomic packet, per
        # arXiv:2511.10633 Sec III.1); the count here is the receiving-side completeness
        # BACKSTOP for custom controllers that forward fragments individually -- a window
        # must never be assembled from half a round.
        op_store = self.payload_store.get(op.id)
        if op_store is None:
            raise RuntimeError(
                f"round {payload.round_index} of {op.name} arrived after the op's last "
                f"window committed and its syndrome RAM was freed -- the device emitted "
                f"more rounds than the execution plan expects")
        fragments = op_store.setdefault(payload.round_index, {})
        if payload.patch_id not in fragments:      # a re-delivered fragment is not new storage
            self.payloads_held += 1
        fragments[payload.patch_id] = payload
        if len(fragments) >= payload.n_fragments:
            self.rounds_arrived[op.id] = max(self.rounds_arrived[op.id],
                                             payload.round_index)
        self.peak_payloads = max(self.peak_payloads, self.payloads_held)
        self.engine.log("DecoderClstr",
                        f"round {payload.round_index} of {op.name} arrived "
                        f"(op now has rounds 1..{self.rounds_arrived[op.id]})")
        for k in range(self.nwin[op.id]):
            self._check_window((op.id, k))
        # this op's early rounds also feed the BUFFER overflow of its predecessors' last windows
        for pred_id in op.predecessors:
            for k in range(self.nwin[pred_id]):
                self._check_window((pred_id, k))
 
    def prepend_idle_rounds(self, op_id: int, n_rounds: int) -> None:
        """The op's patch idled `n_rounds` before the op began (waiting for a decode).
        Under a scheme that batches per feedback-to-feedback SEGMENT (NaiveOnlineScheme,
        flag batches_idle_rounds_into_next_op), those rounds join the op's FIRST window,
        so its decode covers idle + op rounds -- the r_i segment of arXiv:2510.25222
        Eq. 5, the record Terhal's backlog argument requires processed before the next
        feedback. Continuously-windowed schemes ignore this (their idle decoding
        overlaps the wait; Phase 2 of docs/DESIGN-idle-stream-windows.md). TIMING-level:
        the decode job grows by n_rounds; the idle payloads themselves are not retained
        (they arrive via on_memory_round, which keeps only a count)."""
        if n_rounds <= 0 or not getattr(self.scheme, "batches_idle_rounds_into_next_op",
                                        False):
            return
        w = self.windows[(op_id, 0)]
        w.n_rounds += n_rounds
        self.engine.log("DecoderClstr",
                        f"{self.ops[op_id].name} W0 absorbs {n_rounds} idle rounds: "
                        f"its batch decode now covers {w.n_rounds} rounds (the "
                        f"feedback-to-feedback segment)")

    def on_memory_round(self, op_id: int) -> None:
        """An idle/memory round arrived for a waiting patch (fills window buffers)."""
        self.memory_rounds[op_id] += 1
        self.engine.log("DecoderClstr",
                        f"memory round for {self.ops[op_id].name} "
                        f"(idle buffer rounds: {self.memory_rounds[op_id]})")
        for k in range(self.nwin[op_id]):
            self._check_window((op_id, k))
 
    @staticmethod
    def _xor_mask(prev, mask) -> list:
        """Elementwise XOR of two bit sequences (zero-padded to the longer one)."""
        a = [int(b) for b in prev] if prev is not None else []
        b = [int(b) for b in mask]
        if len(a) < len(b):
            a += [0] * (len(b) - len(a))
        for i, bit in enumerate(b):
            a[i] ^= bit
        return a

    def _apply_boundary(self, w: Window, payload: SyndromePayload,
                        round_key: Optional[int] = None) -> SyndromePayload:
        """Fold this window's received ARTIFICIAL DEFECTS into one payload: the next
        window "includes the artificial defects along with the unresolved defects from
        the buffer region" (arXiv:2209.08552 Sec I.B). The XOR happens on a COPY -- the
        shared payload store is never mutated, since overlapping windows read the same
        rounds. A timing-only payload (bits=None) becomes the defect mask itself.
        `round_key` overrides the lookup round (used for successor-overflow payloads,
        whose own round numbering starts at 1 but whose defects are keyed past this
        op's last round)."""
        r = payload.round_index if round_key is None else round_key
        mask = w.boundary_in.get((r, payload.patch_id), w.boundary_in.get(r))
        if mask is None:
            return payload
        from dataclasses import replace
        bits = [int(m) for m in mask] if payload.bits is None \
            else self._xor_mask(payload.bits, mask)
        return replace(payload, bits=bits)

    def _assemble_payloads(self, w: Window) -> list:
        """Collect retained payloads for this window's commit+buffer rounds (plus the
        successor-operation overflow rounds), so a real decoder receives the actual data --
        with any received artificial defects XORed in (per-round, patch-sorted fragments).
        Timing-only payloads carry no bits, but the assembly path is identical."""
        op_store = self.payload_store.get(w.op_id, {})
        R_op = self.rounds_for(self.ops[w.op_id])      # this operation's own length (D1)
        hi = min(w.buffer_hi, R_op)
        out = []
        for r in range(w.start_round, hi + 1):
            if r in op_store:
                out += [self._apply_boundary(w, op_store[r][p])
                        for p in sorted(op_store[r])]
        overflow = w.buffer_hi - R_op
        if overflow > 0:
            for s in self.successors.get(w.op_id, []):
                succ = self.payload_store.get(s, {})
                for r in range(1, overflow + 1):
                    if r in succ:
                        # defects aimed at these rounds are keyed past this op's end
                        out += [self._apply_boundary(w, succ[r][p], round_key=R_op + r)
                                for p in sorted(succ[r])]
        return out
 
    def _window_data_complete(self, w: Window) -> bool:
        """Whether a window has all the rounds it needs (delegates to the scheme)."""
        op = self.ops[w.op_id]
        succ_rounds = max((self.rounds_arrived[s] for s in self.successors[w.op_id]),
                          default=0)
        return self.scheme.data_complete(
            w, rounds_arrived=self.rounds_arrived[w.op_id], successor_rounds=succ_rounds,
            memory_rounds=self.memory_rounds[w.op_id], n_rounds=self.rounds_for(op),
            has_successor=op.has_successor, op=op, layout=self.layout)
 
    @property
    def _windowed(self) -> bool:
        """True if the scheme decodes in sliding windows; False for a batch/global scheme
        (NaiveOnlineScheme). The scheme is the single source of truth -- the trace never infers
        windowing from window counts."""
        return getattr(self.scheme, "windowed", True)

    def _job_desc(self, w, op) -> str:
        """Scheme-appropriate identity for a decode job, used at EVERY decode-lifecycle log
        line so the trace stays consistent. Windowing scheme -> 'name W0 [commit 1-7]';
        batch/global scheme -> 'name [whole op, rounds 1-3]' (no window index, no commit
        vocabulary -- it decodes the whole operation in one shot)."""
        if self._windowed:
            return f"{op.name} W{w.k} [commit {w.commit_lo}-{w.commit_hi}]"
        return f"{op.name} [whole op, rounds {w.commit_lo}-{w.commit_hi}]"

    def _check_window(self, key: tuple) -> None:
        """If a window has its data and its dependencies, build its job and enqueue it."""
        w = self.windows[key]
        if w.queued or w.committed:
            return
        if w.t_first_round is None and self.rounds_arrived[w.op_id] >= w.start_round:
            w.t_first_round = self.engine.now            # its first round just arrived
        if not self._window_data_complete(w):
            return
        if w.t_data_complete is None:
            w.t_data_complete = self.engine.now          # commit+buffer rounds all present
        op = self.ops[w.op_id]
        if w.deps_remaining > 0:
            if not w.blocked_logged:
                w.blocked_logged = True
                self.engine.log("DecoderClstr",
                                f"{self._job_desc(w, op)} has all its data, but is WAITING for "
                                f"the boundary from {w.deps_remaining} predecessor window(s)")
            return
        w.t_queued = self.engine.now                     # data AND deps met -> ready queue
        # the DEADLINE policy decides this job's urgency; windows of an op whose result
        # gates a waiting non-Clifford gate are on the reaction path (arXiv:2511.10633).
        deadline = self.deadline_policy.deadline(
            op, w, self.engine.now, on_reaction_path=(op.id in self._gating_ops))
        job = DecodeJob(op_id=w.op_id, window_id=w.k, n_rounds=w.n_rounds,
                        ready_time=self.engine.now, deadline=deadline,
                        spatial_nodes=self._spatial_nodes(op),
                        payloads=self._assemble_payloads(w),
                        dem=self.window_models.get(key),         # this window's decoding problem (R2)
                        code=self.layout.code_for_op(op).name,   # G1: route to this code's decoder
                        window=w,                                # commit geometry for boundary defects
                        label=self._job_desc(w, op))             # scheme-aware (window vs batch)
        pool = self._pool_for(job)
        queue = self._queue_for(pool)
        self.scheduler.insert(queue, job)
        w.queued = True
        self.engine.log("DecoderClstr",
                        f"{self._job_desc(w, op)} READY -> enqueue "
                        f"({self._pool_tag(pool)}ready-queue length = {len(queue)})")
        self.queue_log.append((self.engine.now, self._queued_total()))
        # "Run both at once" switching: start the strong decoder on this same window right now,
        # alongside the weak one. If the weak answer later turns out confident we cancel it
        # (_cancel_strong_decode); otherwise it finishes and its result is ready sooner than if we
        # had waited for the weak verdict first.
        if self.switching is not None and self.switching.run_both_at_once:
            self._start_strong_decode(w, in_parallel=True)
        self._try_dispatch()
 
    def submit_decode(self, n_rounds: int, on_done: Callable[[], None],
                      label: str = "external", deadline: Optional[int] = None,
                      code: Optional[str] = None,
                      spatial_nodes: Optional[int] = None,
                      hint: Optional[str] = None) -> None:
        """DecodeService entry point: submit a self-contained decode job that competes for
        the SAME decoder units as the core (used by the magic state factory and by the
        chip's idle-round memory decoding). `code` (optional) routes the job to that code's
        decoder; None uses the cluster's default decoder (G1). `spatial_nodes` (optional)
        sizes the decoding graph for latency models; None lets the decoder use its default.
        `hint` (optional) routes the job to a custom DecoderRouter and, when it names one
        of the cluster's unit pools, to that pool's units and queue."""
        job = DecodeJob(op_id=-1, window_id=0, n_rounds=n_rounds,
                        ready_time=self.engine.now,
                        deadline=self.engine.now if deadline is None else deadline,
                        on_done=on_done, label=label, code=code,
                        spatial_nodes=spatial_nodes, hint=hint)
        self.scheduler.insert(self._queue_for(self._pool_for(job)), job)
        self.queue_log.append((self.engine.now, self._queued_total()))
        self._try_dispatch()
 
    def _start_strong_decode(self, window: Window, in_parallel: bool = False) -> None:
        """Queue a strong re-decode of `window`. It is a side job: it occupies a strong decoder
        unit and adds to the strong decoder's queue, but it does NOT commit the window -- the weak
        decoder already committed it, so the weak stream keeps flowing. We remember the job (in
        _running_strong_decodes) so we can cancel it later.

        in_parallel=True  ("run both at once"): queue it right away, on the same data the weak
                          decoder got, with no extra hand-off cost.
        in_parallel=False (serial): the data has to travel from the weak decoder to the strong one
                          first, so we queue it after the weak->strong link delay (links.ws)."""
        op = self.ops[window.op_id]
        key = (window.op_id, window.k)
        n_rounds = self.switching.calculate_strong_redo_rounds(window)
        label = f"strong({op.name} W{window.k})"

        def queue_it():
            job = DecodeJob(op_id=-1, window_id=0, n_rounds=n_rounds,
                            ready_time=self.engine.now, deadline=self.engine.now,
                            on_done=lambda: None, label=label, hint="strong",
                            spatial_nodes=self._spatial_nodes(op), strong_decode_for=key)
            self._running_strong_decodes[key] = job
            self.scheduler.insert(self._queue_for(self._pool_for(job)), job)
            self.queue_log.append((self.engine.now, self._queued_total()))
            self._try_dispatch()

        if in_parallel:
            queue_it()
        else:
            self.engine.log("DecoderClstr",
                            f"weak decoder unsure about {self._job_desc(window, op)} -> hand "
                            f"{n_rounds} rounds to the strong decoder (after the weak->strong link)")
            self.engine.schedule(self.links.ws.cost(), queue_it,
                                 label=f"weak->strong handoff {label}")

    def _cancel_strong_decode(self, key: tuple) -> None:
        """Cancel the strong re-decode of window `key`, if one is still running. Used in "run both
        at once" mode when the weak decoder turns out confident, so the strong result is no longer
        needed. If the strong job is still waiting in the queue, just remove it. If it is already
        running on a unit, free that unit right now and mark the job cancelled, so when its
        already-scheduled "finished" event fires later it does nothing. (In serial mode there is
        no strong job to cancel here, so this does nothing.)"""
        job = self._running_strong_decodes.pop(key, None)
        if job is None:
            return
        if job.pool is None:                          # still waiting in the queue: just remove it
            queue = self._queue_for(self._pool_for(job))
            if job in queue:
                queue.remove(job)
        else:                                         # already running: free its unit now
            job.cancelled = True
            self.pool_free[job.pool] += 1
            self._try_dispatch()
        self.strong_cancelled += 1

    def _try_dispatch(self) -> None:
        """While a pool has a free unit and a ready job, dispatch. Pools are independent:
        a busy strong pool never blocks the default queue (and vice versa)."""
        for pool in self.unit_totals:
            queue = self._queue_for(pool)
            while self.pool_free[pool] > 0 and queue:
                job = self.scheduler.pop(queue)
                job.pool = pool                              # so done() frees THIS pool
                self.pool_free[pool] -= 1
                if job.op_id >= 0:                           # window job: stamp dispatch
                    self.windows[(job.op_id, job.window_id)].t_dispatch = self.engine.now
                lat = self._decoder_for(job).latency(job)    # routed decoder (G1 / per-job)
                waited = self.engine.now - job.ready_time
                self.engine.log("DecoderClstr",
                                f"START DECODE {job.label} (waited {fmt(waited).strip()} in queue, "
                                f"{self._pool_tag(pool)}units free now {self.pool_free[pool]})")
                self.queue_log.append((self.engine.now, self._queued_total()))
                self.engine.schedule(lat, lambda j=job: self._on_decode_done(j),
                                     label=f"decode_done({job.label})")
 
    def _on_decode_done(self, job: DecodeJob) -> None:
        """Commit a finished window, hand its boundary to the next window, report the op result on its last window."""
        if job.cancelled:                            # a strong re-decode that was cancelled early:
            return                                   # its unit was already freed, so do nothing here
        self.pool_free[job.pool] += 1
        if job.strong_decode_for is not None:        # a strong re-decode finished -> forget it
            self._running_strong_decodes.pop(job.strong_decode_for, None)
        if job.on_done is not None:                  # factory / idle / strong re-decode
            self.engine.log("DecoderClstr",
                            f"DECODE DONE {job.label} ({self._pool_tag(job.pool)}units "
                            f"free now {self.pool_free[job.pool]})")
            job.on_done()
            self._try_dispatch()
            return
        # ---- an operation window ----
        key = (job.op_id, job.window_id)
        w = self.windows[key]
        op_id = job.op_id
        op = self.ops[op_id]
        # Decode FIRST, before committing. The result is needed twice: its soft output
        # decides whether this window ALSO goes to the strong decoder (double-window
        # switching), and its boundary defects feed the dependent windows below. decode()
        # emits no engine events, so running it ahead of the commit log leaves the trace
        # order unchanged for every scheme. For a timing-only stub the result carries no
        # logical value and the orchestrator falls back to its toy outcome.
        res = self._decoder_for(job).decode(job)             # routed decoder (G1 / per-job)
        # DECODER SWITCHING (switching.py). This weak window commits either way below -- the weak
        # stream never stalls. We only decide here what to do about the STRONG decoder.
        if self.switching is not None and job.attempt == 0:
            if self.switching.keep_weak_result(res):
                # weak answer is good enough. In "run both at once" mode a strong re-decode of this
                # window is already running, so cancel it; in serial mode there is none.
                self._cancel_strong_decode(key)
            else:
                # weak answer is shaky -> we need the strong decoder for this window.
                self.strong_needed += 1
                if not self.switching.run_both_at_once:
                    self._start_strong_decode(w)     # serial: start it now (data travels over ws)
                # "run both at once": its strong re-decode is already running, so just let it finish.
        w.committed = True
        w.t_done = self.engine.now
        self.committed_windows.add(key)
        self._committed_per_op[op_id] = self._committed_per_op.get(op_id, 0) + 1
        self.engine.log("DecoderClstr",
                        f"DECODE DONE {self._job_desc(w, op)} -- "
                        f"{'committed' if self._windowed else 'decoded'} "
                        f"({self._pool_tag(job.pool)}units free now {self.pool_free[job.pool]})")
        if res is not None and res.logical_value is not None:
            self.op_results[op_id] = self.op_results.get(op_id, 0) ^ int(res.logical_value)
        defects = res.boundary_defects if res is not None else None
        # A committed window passes its boundary information to the next window's decoder:
        # the artificial defects created where committed correction chains crossed out of
        # the commit region (arXiv:2209.08552 Fig. 2). This is a decoder->decoder exchange:
        # SENT now, ARRIVES one t_dd hop later. Timing-only decoders send no data (None) --
        # the hop still clears the dependency, exactly as before.
        for dep_key in w.dependents:
            dst = self.ops[dep_key[0]]
            self.engine.log("DecoderClstr",
                            f"decoder->decoder SEND: {op.name} W{w.k} -> {dst.name} "
                            f"W{dep_key[1]}  (boundary/artificial defects, arrives in t_dd)")
            self.engine.schedule(self.links.dd.cost(),
                                 lambda dk=dep_key, sn=op.name, sk=w.k, so=op_id, bd=defects:
                                     self._receive_boundary(dk, sn, sk, so, bd),
                                 label=f"defects {op.name}W{w.k}->{dst.name}W{dep_key[1]}")
        # This window just finished, so drop every round it read that no other unfinished window
        # still needs (paper: arXiv:2511.10633 Sec VI.B). This is safe: the decoder already took
        # its copy of the data when the job was built, and the hand-off to the next window carries
        # the *decoded* boundary, not the raw rounds -- so freeing here can never leave a
        # not-yet-decoded window short of data (its rounds keep a non-zero count and stay). This is
        # what keeps peak_payloads equal to the data actually held at once.
        for round_key in self._read_sets.get(key, ()):
            self._round_refs[round_key] = self._round_refs.get(round_key, 0) - 1
            if self._round_refs[round_key] <= 0:                 # no window needs this round anymore
                round_op, round_no = round_key
                frags = self.payload_store.get(round_op, {}).pop(round_no, None)
                if frags is not None:
                    self.payloads_held -= len(frags)
        # the operation's logical outcome is known when ALL its windows have committed ->
        # orchestrator. (Not "when window k = nwin-1 commits": only a sequential chain
        # guarantees that window finishes last; the parallel A/B scheme's windows commit
        # out of order under contention.) For the sequential default this fires at the
        # exact same event as before.
        if self._committed_per_op[op_id] == self.nwin[op_id]:
            self.engine.schedule(self.links.do.cost(),
                                 lambda: self._deliver_to_orchestrator(op),
                                 label=f"result->orch({op.name})")
            # This operation is fully decoded. Its rounds were already freed one by one above, so
            # here we just remove the (now empty) operation from the store. That also means a stray
            # round arriving after the operation finished gets caught and reported, rather than
            # silently kept. The sum is a safety net (normally zero) in case any round was missed.
            freed = self.payload_store.pop(op_id, None)
            if freed:
                self.payloads_held -= sum(len(frags) for frags in freed.values())
        self._try_dispatch()
        # WORKLOAD COMPLETE: the last window has committed. The cluster does not know about
        # factories or chips -- it just fires the lifecycle callback the wiring installed.
        if len(self.committed_windows) == self.total_windows and self.on_workload_complete is not None:
            self.on_workload_complete()
 
    def _store_boundary(self, w: Window, src_op_id: int, defects: Optional[dict]) -> None:
        """File a committed window's artificial defects on the dependent window, keyed in
        the DEPENDENT op's round numbering: a cross-op boundary (predecessor's exit window
        -> successor's entry window) shifts rounds by the predecessor's length, so defects
        past its last round land on the successor's first rounds. Masks for the same node
        XOR-merge (defects are mod-2 syndrome flips)."""
        if not defects:
            return
        shift = 0 if src_op_id == w.op_id else -self.rounds_for(self.ops[src_op_id])
        for key, mask in defects.items():
            r, patch = key if isinstance(key, tuple) else (key, None)
            r += shift
            if r < 1:
                continue                       # refers to the predecessor's own rounds
            dst_key = (r, patch) if patch is not None else r
            w.boundary_in[dst_key] = self._xor_mask(w.boundary_in.get(dst_key), mask)

    def _receive_boundary(self, key: tuple, src_name: str, src_k: int,
                          src_op_id: int, defects: Optional[dict] = None) -> None:
        """A neighbor window's boundary arrived; file its artificial defects (if the
        decoder produced any), clear one dependency and re-check."""
        w = self.windows[key]
        op = self.ops[w.op_id]
        self._store_boundary(w, src_op_id, defects)
        w.deps_remaining -= 1
        still = f"; still waiting on {w.deps_remaining}" if w.deps_remaining > 0 else ""
        self.engine.log("DecoderClstr",
                        f"decoder->decoder RECV: {op.name} W{w.k} <- {src_name} W{src_k}  "
                        f"(boundary arrived after t_dd){still}")
        self._check_window(key)
 
    def _deliver_to_orchestrator(self, op: Operation) -> None:
        # The decoder -> orchestrator hop (t_do, ~1 us) is paid once per OPERATION -- when
        # its last window commits -- Clifford or not: the orchestrator always needs the
        # result to update its frames. Carry the operation's real decoded logical value
        # (None if the decoder was a timing-only stub, in which case the orchestrator uses
        # its toy outcome).
        """Hand the op's decoded result to the orchestrator (the t_do hop) and stop. What happens
        next -- frame update, and DISPATCH of any conditional decision back to the chip
        (orchestrator -> controller -> chip) -- is the orchestrator's job, not the cluster's."""
        result = DecodeResult(op.id, self.nwin[op.id] - 1,
                              logical_value=self.op_results.get(op.id))
        self.orchestrator.integrate(op, result)
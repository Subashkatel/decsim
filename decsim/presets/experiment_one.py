"""Experiment 1's machine: a union-find weak tier, Relay-BP-5 on the host.

A union-find weak tier on Helios's cycle law beside a Relay-BP-5 strong
tier at its measured A100 time, escalating windows whose cluster gap is
under 20 dB into double windows (Toshio et al. 2510.25222 Sec. III C),
100 rounds a shot. Every latency is held at one sourced value, the
source beside each, so later experiments move one at a time.

    weak_alone(d, p)   no escalation: union-find keeps every window
    strong_alone(d, p, pool)  every window decoded by a strong pool
    switching(d, p)    union-find first, a window under 20 dB redone by
                       Relay-BP-5

The weak loop is a leaf-to-root fiber network (2603.16203): a round goes
up in 157 ns (lines 895-897) and the correction comes back in 155 + 9 ns
(lines 903-904), on four lanes of 38.79 bits a 250 MHz cycle (lines
971-975), and the decoder reads its whole frame in one cycle (lines
668-670), an estimate. The strong side is Backline's CPU echo over RoCE
v2 (link_cards.ROCE_V2_CPU_FABRIC); every other hop keeps the reference
card.
"""

import dataclasses

import decsim.confidence.cluster as cluster
import decsim.config as config
import decsim.controller.settings as controller_settings
import decsim.decoders.measured_table.decoder as measured_table
import decsim.decoders.settings as decoder_settings
import decsim.decoders.union_find.cycle_count as cycle_count
import decsim.decoders.union_find.decoder as union_find
import decsim.detector_error_model.settings as detection_event_settings
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.observe.settings as observe_settings
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.presets.clocks as clocks
import decsim.presets.link_cards as link_cards
import decsim.presets.machines as machines
import decsim.producers as producers
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph
import decsim.windows.settings as window_settings

ROUND_PERIOD_MICROSECONDS = 1.0
ROUNDS_PER_SHOT = 100
# a chosen cluster-gap threshold, not a calibrated one: the paper's
# 20 dB is on the complementary gap
THRESHOLD_DECIBELS = 20.0
FIBER_SOURCE = "Experiment 1's leaf-to-root fiber loop (2603.16203)"
# The uplink's latency leaves out the root's own 20 ns of work (lines
# 897-899), so it times the wire alone; the other hops' numbers hold
# their receiver's.
UPLINK = fabrics.ClockedLink(
    clock=clocks.FRIDGE_CLOCK,
    latency_cycles=40,
    bits_per_cycle=38.79,
    lane_count=4,
    source=FIBER_SOURCE,
)
FRAME_READ = fabrics.ClockedLink(
    clock=clocks.FRIDGE_CLOCK,
    latency_cycles=1,
    excludes_receiver_processing=False,
    source=FIBER_SOURCE,
)
DOWNLINK = fabrics.ClockedLink(
    clock=clocks.FRIDGE_CLOCK,
    latency_cycles=41,
    bits_per_cycle=38.79,
    lane_count=4,
    excludes_receiver_processing=False,
    source=FIBER_SOURCE,
)
# The controller's write into the weak store, its read, and the weak
# tier's correction ride the fiber loop; the strong side, Backline's
# cable. The fabric prices the machine's every hop, so a run that drops
# the weak pool keeps Experiment 1's card.
FABRIC = fabrics.RoleLinks(
    write_written=UPLINK,
    read_written=FRAME_READ,
    answer_fridge=DOWNLINK,
    base=link_cards.ROCE_V2_CPU_FABRIC,
)


def weak_alone(
    distance: int, physical_error_probability: float
) -> graph.Machine:
    """The machine with no escalation: union-find keeps every window.

    Readout, packing, event forming, the frame and the union-find unit
    are sourced values, each written below with its source.
    """
    stim_source = stim_device.StimDevice.Settings()
    qpu = qpu_settings.QpuSettings(
        source=stim_source,
        round_period_microseconds=ROUND_PERIOD_MICROSECONDS,
        distance=distance,
    )
    # the store's costs are paid once in the windows' decision
    weak_syndrome_buffer = syndrome_buffer_module.SyndromeBufferSettings(
        clock=clocks.FRIDGE_CLOCK
    )
    # no manager on Helios's path (2603.16203 lines 669-670), one region
    # a decode (2510.25222 lines 1255-1258)
    decoder_manager = decoder_settings.DecoderManagerSettings(
        clock=clocks.FRIDGE_CLOCK
    )
    # made at the task, so it follows an assigned qpu.distance
    workload = producers.MemorySettings(
        physical_error_probability, rounds=ROUNDS_PER_SHOT
    )
    # both machines lay out the same last window, the lookahead tail
    sliding_windows = window_plan()
    windows = dataclasses.replace(sliding_windows, terminal_policy="lookahead")
    observation = observe_settings.ObservationSettings(
        record_switching_windows=True, backlog_trace=True
    )
    # the frame update takes 4 ns, one cycle of the chip clock
    # (2605.04892 Table I, lines 1051 and 1063)
    pauli_frame = pauli_frame_module.PauliFrameConfig(
        write_cycles=1, clock=clocks.FRIDGE_CLOCK
    )
    machine = machines.base(
        qpu,
        controller(),
        detection_events(),
        windows,
        decoder_manager,
        workload,
        pauli_frame,
        clock=clocks.FRIDGE_CLOCK,
        observation=observation,
        fabric=FABRIC,
    )
    return machines.with_pool(
        machine, weak_decoder(), store=weak_syndrome_buffer
    )


def strong_alone(
    distance: int,
    physical_error_probability: float,
    strong_pool: decoder_settings.DecoderPoolSettings,
) -> graph.Machine:
    """The same machine with every window decoded by a strong pool."""
    machine = weak_alone(distance, physical_error_probability)
    return machines.strong_alone(machine, strong_pool)


def switching(
    distance: int, physical_error_probability: float
) -> graph.Machine:
    """Weak union-find first, a window under 20 dB redone by Relay-BP-5.

    The double window re-reads one buffer region on restart (2510.25222
    Fig. 12), and its strong decode starts only after the weak verdict
    (lines 632-640).
    """
    machine = weak_alone(distance, physical_error_probability)
    strong_decoder = strong_decoder_pool()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=THRESHOLD_DECIBELS
    )
    double_window = strong_window_shapes.DoubleWindow.Settings()
    # the cluster gap's walk priced at 1.0 us on the weak unit, a chosen
    # value (2602.03336 lines 17-19); the verdict is one compare and one
    # state change, a cycle each of the chip clock, estimates
    confidence = cluster.ClusterGap.Settings(walk_microseconds=1.0)
    switching_slot = escalation_settings.SwitchingSettings(
        confidence=confidence,
        threshold=threshold,
        clock=clocks.FRIDGE_CLOCK,
        threshold_cycles=1,
        switch_cycles=1,
        strong_window=double_window,
    )
    return machines.with_strong_tier(machine, strong_decoder, switching_slot)


def strong_decoder_pool() -> decoder_settings.DecoderPoolSettings:
    """Relay-BP-5 priced on the line measured on a whole A100.

    The keys are the Relay-BP paper's surface code Relay-BP-5 (Mueller
    et al. 2506.01779 lines 307, 332, 343), X and Z together, and the
    time is the line measured at those keys on the whole A100-SXM4-80GB
    (decsim/decoders/measured_table/measurements.py, Slurm job
    14676845). The measured time holds the syndrome's copies to and
    from the device, so the engine charges no stage, on the chip clock.
    The boundary folds into a copy of the rounds, as CUDA-Q QEC keeps
    the raw rounds and applies the syndrome mods when it assembles a
    window (cudaqx sliding_window.cpp:287-293).
    """
    algorithm = measured_table.MeasuredTableSettings(
        gamma0=0.35,
        gamma_interval=(-0.254, 0.985),
        relay_set_count=600,
        converged_solution_count=5,
        device="a100",
        partition="whole",
        bases="together",
    )
    engine = decoder_settings.EngineSettings(
        clock=clocks.FRIDGE_CLOCK,
        fetch_cycles_per_round=0,
        fetch_cycles_per_job=0,
        release_cycles_per_job=0,
        release_cycles_per_round=0,
    )
    return decoder_settings.DecoderPoolSettings(
        algorithm=algorithm, engine=engine
    )


def weak_decoder() -> decoder_settings.DecoderPoolSettings:
    """One union-find unit on a cycle law traced from an FPGA design.

    The law's delay is 3 cycles (MAXIMUM_DELAY,
    Helios_single_FPGA_core.v:76 at github.com/yale-paragon/
    Helios_scalable_QEC 2dda998, the design behind 2301.08419v2); an
    element per vertex costs no cycle per edge (2301.08419 lines
    764-767), and the graph sits in registers, so there is no setup
    (lines 912-913). The clock is the design's 100 MHz synthesis target
    (2406.08491 line 1230). The step that turns a log-odds weight into
    growth ticks is 0.1, far finer than the design's integer weights
    from 2 to wmax (2406.08491 lines 1616-1620), for accuracy: at 0.1
    the decoder fails as often as the weighted union-find of mwpf
    (Wu et al. 2508.04969, after Delfosse et al. 2103.08049) on the
    same shots at d = 5 to 9, where a step of 0.5 fails 3 percent more
    and Helios's own rule (software_code/main.c lines 51-54) 36 to 45
    percent more at d = 7 and 9. The engine takes one cycle
    a decode for the header that starts it (control_node_single_FPGA.v
    lines 131-142), then reads one whole round a cycle from the unit's
    memory, as Helios's loading stage takes a round a cycle (lines
    170-182); the bytes that fill that memory are the off-chip link,
    priced on controller_to_weak_buffer. The fetch runs after the
    window's last round has landed, a bound above designs that load
    while rounds arrive (DecoNet 2504.11805 lines 961-974). The
    correction leaves in one 25 ns message
    (2603.16203 line 902), rounded up to 3 cycles. The boundary folds
    into the unit's own copy of the rounds (2301.08419 lines 632-640).
    """
    clock = config.Clock.from_megahertz(100.0)
    timing = cycle_count.CycleCount(
        clock=clock,
        delay_cycles=3,
        cycles_per_edge=0.0,
        setup_cycles=0,
        setup_cycles_per_vertex=0,
        setup_cycles_per_edge=0,
    )
    algorithm = union_find.UnionFindDecoder.Settings(
        weight_step=0.1, timing=timing
    )
    engine = decoder_settings.EngineSettings(
        clock=clock,
        fetch_cycles_per_round=1,
        fetch_cycles_per_job=1,
        release_cycles_per_job=3,
        release_cycles_per_round=0,
    )
    return decoder_settings.DecoderPoolSettings(
        algorithm=algorithm, engine=engine, copies_boundary_fold=False
    )


def controller() -> controller_settings.ControllerSettings:
    """A leaf controller that packs a round in 8 cycles and issues in 8.

    The syndrome aggregator packs a round in 29 ns (2603.16203 lines
    894-895), rounded up to cycles of the 250 MHz machine clock. That
    paper gives no issue count, so the issue is QubiC's pipeline from
    the decision to the pulse trigger (2404.15260 lines 173-191), 8
    cycles counted with gem5's MinorCPU stage delays, an estimate.
    Readout to bits sits in the readout hop (2605.04892 lines 1053-1055).
    """
    return controller_settings.ControllerSettings(
        readout_to_bits_cycles=0,
        packing_cycles_per_round=8,
        decision_to_pulse_cycles=8,
    )


def detection_events() -> detection_event_settings.DetectionEventSettings:
    """Events formed at the controller, pipelined: 5 cycles, then 1 a round.

    Syndrome preprocessing fully pipelined in registers takes 20 ns
    (2605.04892 lines 1273-1275), 5 cycles of the 250 MHz chip clock
    (line 1063). That paper forms the events on the decoding FPGA (lines
    199-200); this machine forms them at the controller.
    """
    return detection_event_settings.DetectionEventSettings(
        clock=clocks.FRIDGE_CLOCK,
        latency_cycles=5,
        cycles_per_round=1,
    )


def window_plan() -> window_settings.WindowSettings:
    """Sliding windows of the code distance, issued in 20 ns.

    A root controller issues each decode request in 20 ns (2603.16203
    lines 897-899), 5 cycles of the 250 MHz chip clock (2605.04892 line
    1063).
    """
    return window_settings.WindowSettings(
        clock=clocks.FRIDGE_CLOCK, decision_cycles=5
    )

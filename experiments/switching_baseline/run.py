"""Experiment 1, the switching baseline, and its paired weak-alone run.

A union-find weak tier on Helios's cycle law beside a Relay-BP-5 strong
tier at its measured A100 time, escalating windows whose cluster gap is
under 20 dB into double windows (Toshio et al. 2510.25222 Sec. III C),
at d = 5 to 13 and six physical error rates, 100 rounds a shot. The
weak-alone points run the same machine with no escalation, every window
kept by the weak tier; a shot's samples depend on its seed alone, so
each pairs with the switching shot of its seed. Every latency is held
at one sourced value, the source beside each preset, so later
experiments move one at a time.

A point stops at 100 failures or five million shots. A switching piece
is 50 shots: the slowest point, d = 13 at p = 0.005, is estimated at up
to 600 s a shot (a measured 32 s at d = 9, p = 0.003, about three times
the region's detectors, twice the escalations), so a piece is at most
about 8 hours, a third of a 24-hour task. A weak-alone piece is 2,000
shots, since weak-alone shots took 0.17 to 0.41 s each.

Usage
-----

```
decsim run experiments/switching_baseline/run.py
```
"""

import dataclasses

import decsim
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
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.observe.settings as observe_settings
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.settings as machine_settings
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import decsim.windows.settings as window_settings

NAME = "switching_baseline"
DISTANCES = (5, 7, 9, 11, 13)
PHYSICAL_ERROR_PROBABILITIES = (0.0005, 0.001, 0.002, 0.003, 0.004, 0.005)
ROUND_PERIOD_MICROSECONDS = 1.0
ROUNDS_PER_SHOT = 100
# a chosen cluster-gap threshold, not a calibrated point: the paper's
# 20 dB is on the complementary gap
THRESHOLD_DECIBELS = 20.0
# the three strong-side legs that cross the cable to the host
STRONG_CABLE_LEGS = (
    "controller_to_strong_buffer",
    "weak_decoder_to_strong_decoder",
    "strong_decoder_to_frame",
)
EXPERIMENT_ONE_LINKS_SOURCE = "experiments/switching_baseline/run.py"
SWITCHING_COLLECTION = decsim.CollectionSettings(
    max_failures=100, max_shots=5_000_000, min_shots=0, piece_rounds=5000
)
WEAK_ALONE_COLLECTION = decsim.CollectionSettings(
    max_failures=100, max_shots=5_000_000, min_shots=0, piece_rounds=200_000
)


def switching(
    distance: int, physical_error_probability: float
) -> machine_settings.MachineSettings:
    """Weak union-find first, a window under 20 dB redone by Relay-BP-5.

    The double window re-reads one buffer region on restart (2510.25222
    Fig. 12), and its strong decode starts only after the weak verdict
    (lines 632-640).
    """
    machine = weak_alone(distance, physical_error_probability)
    strong_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=measured_table.A100_RELAY_BP_5,
        engine=decoder_settings.MEASURED_TIME_ENGINE,
    )
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
        clock=machine_settings.FRIDGE_CLOCK,
        threshold_cycles=1,
        switch_cycles=1,
        strong_window=double_window,
    )
    windows = window_settings.switching_windows(machine.windows, double_window)
    return dataclasses.replace(
        machine,
        windows=windows,
        strong_decoder=strong_decoder,
        switching=switching_slot,
    )


def weak_alone(
    distance: int, physical_error_probability: float
) -> machine_settings.MachineSettings:
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
    links = experiment_one_links()
    # the store's costs are paid once in the windows' decision
    weak_syndrome_buffer = syndrome_buffer_module.SyndromeBufferSettings(
        clock=machine_settings.FRIDGE_CLOCK
    )
    # no manager on Helios's path (2603.16203 lines 669-670), one region
    # a decode (2510.25222 lines 1255-1258)
    decoder_manager = decoder_settings.DecoderManagerSettings(
        clock=machine_settings.FRIDGE_CLOCK
    )
    workload = machine_settings.memory_workload(
        distance, physical_error_probability, ROUNDS_PER_SHOT
    )
    # both machines lay out the same last window, the lookahead tail
    windows = dataclasses.replace(window_plan(), terminal_policy="lookahead")
    observation = observe_settings.ObservationSettings(
        record_switching_windows=True, backlog_trace=True
    )
    # the frame update takes 4 ns, one cycle of the chip clock
    # (2605.04892 Table I, lines 1051 and 1063)
    pauli_frame = pauli_frame_module.PauliFrameConfig(
        write_cycles=1, clock=machine_settings.FRIDGE_CLOCK
    )
    return decsim.MachineSettings(
        clock=machine_settings.FRIDGE_CLOCK,
        qpu=qpu,
        controller=controller(),
        detection_events=detection_events(),
        links=links,
        weak_syndrome_buffer=weak_syndrome_buffer,
        windows=windows,
        weak_decoder=weak_decoder(),
        decoder_manager=decoder_manager,
        pauli_frame=pauli_frame,
        workload=workload,
        observation=observation,
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
    growth ticks is an estimate, as the design's weights are integers
    from 2 to wmax (2406.08491 lines 1616-1620). The engine reads a
    header byte, then a round a cycle (control_node_single_FPGA.v:137
    and 170-182), and the correction leaves in one 25 ns message
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
        weight_step=0.5, timing=timing
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
        clock=machine_settings.FRIDGE_CLOCK,
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
        clock=machine_settings.FRIDGE_CLOCK, decision_cycles=5
    )


def experiment_one_links() -> link_settings.FabricSettings:
    """A measured RoCE v2 round trip on the strong side, fiber on the weak.

    Each strong leg that crosses the 100 Gb/s cable is half the 2.305 us
    median CPU echo (2609.09270 Table III, line 1627) less the 16-byte
    echo's time on that cable (line 1611), and the host's poll of its
    own memory costs nothing. The weak loop is a leaf-to-root fiber
    network (2603.16203): a round goes up in 157 ns (lines 895-897) and
    the correction comes back in 155 + 9 ns (lines 903-904), on four
    lanes of 38.79 bits a 250 MHz cycle (lines 971-975), and the decoder
    reads its whole frame in one cycle (lines 668-670), an estimate.
    Every other hop keeps the reference card.
    """
    links = link_profiles.logical_reference_profile()
    leg_microseconds = 2.305 / 2 - 128 / 100_000
    for path_name in STRONG_CABLE_LEGS:
        links = link_profiles.with_path_latency(
            links, path_name, leg_microseconds
        )
    # the uplink's latency leaves out the root's own 20 ns of work
    # (lines 897-899); the other hops' numbers hold their receiver's
    uplink = _fridge_path(links, "controller_to_weak_buffer", 40, 38.79)
    frame_read = _fridge_path(links, "weak_buffer_to_weak_decoder", 1, None)
    downlink = _fridge_path(links, "weak_decoder_to_frame", 41, 38.79)
    poll = _fridge_path(links, "strong_buffer_to_strong_decoder", 0, None)
    return dataclasses.replace(
        links,
        controller_to_weak_buffer=uplink,
        weak_buffer_to_weak_decoder=_with_receiver(frame_read),
        weak_decoder_to_frame=_with_receiver(downlink),
        strong_buffer_to_strong_decoder=_with_receiver(poll),
        profile_name="roce_v2_cpu with the risc_q weak loop",
    )


def _fridge_path(links, path_name, latency_cycles, bits_per_lane_cycle):
    """One path in cycles of the chip clock, four lanes when it has a rate."""
    lane_count = 1 if bits_per_lane_cycle is None else 4
    return link_profiles.path_card(
        links,
        path_name,
        clock=machine_settings.FRIDGE_CLOCK,
        latency_cycles=latency_cycles,
        bits_per_cycle=bits_per_lane_cycle,
        source=EXPERIMENT_ONE_LINKS_SOURCE,
        lane_count=lane_count,
    )


def _with_receiver(path: link_settings.PathSettings):
    """The path whose latency holds its receiver's processing too."""
    return dataclasses.replace(path, excludes_receiver_processing=False)


def switching_baseline_points() -> list:
    """Every switching point, then every weak-alone point.

    Each set runs rate by distance, the distance fastest, as the yaml's
    sweep did.
    """
    grid = decsim.grid(
        physical_error_probability=PHYSICAL_ERROR_PROBABILITIES,
        distance=DISTANCES,
    )
    switching_points = []
    weak_alone_points = []
    for values in grid:
        probability = values["physical_error_probability"]
        distance = values["distance"]
        metadata = _metadata(distance, probability)
        cell_name = f"d{distance}_p{probability}"
        switching_machine = switching(distance, probability)
        switching_point = decsim.Point(
            f"switching_{cell_name}",
            switching_machine,
            metadata,
            SWITCHING_COLLECTION,
        )
        switching_points.append(switching_point)
        weak_alone_machine = weak_alone(distance, probability)
        weak_alone_point = decsim.Point(
            f"weak_alone_{cell_name}",
            weak_alone_machine,
            metadata,
            WEAK_ALONE_COLLECTION,
        )
        weak_alone_points.append(weak_alone_point)
    return switching_points + weak_alone_points


def _metadata(distance: int, physical_error_probability: float) -> dict:
    """The point's swept cells as the yaml named them, so its id is theirs."""
    return {
        "workload.arguments.physical_error_probability": (
            physical_error_probability
        ),
        "qpu.distance": distance,
        "qpu.round_period_microseconds": ROUND_PERIOD_MICROSECONDS,
    }


points = switching_baseline_points()
experiment = decsim.Experiment(NAME, points)

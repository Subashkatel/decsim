"""The switching study's one machine: union-find, then belief matching.

    machine(5, physical_error_probability=0.003, rounds_per_distance=10)

Union-find in the fridge, belief matching on a GPU host, switching
between them at 20 dB.

Every number is in constants.yaml beside this file, each with its
source; every part and wire is built here on decsim's own parts
(decsim/presets/machines.py). A rotated surface code Z memory under
Deltakit's SD6 at p (Gidney et al. 2108.10457 Tables 1 and 2), its data
prepared and read out without noise as Toshio et al. 2510.25222 state
(Sec. IV). Its union-find runs on Helios's cycle count (2301.08419) on
the fiber loop of 2603.16203; belief matching runs at a stated GPU time
behind Backline's measured RoCE v2 round trip (2609.09270). A window
whose cluster gap is under the threshold is redone by belief matching
on the same rounds (RedoWindow, escalation/strong_window_shapes.py), so
belief matching decodes the windows it decodes alone; the paper's double
window (2510.25222 Sec. III C) would hand it a different window. Its
answer's boundary is held until final, so the next window starts from
belief matching's correction. The four run files beside this one each
change this machine by part name.
"""

import pathlib

import decsim.controller.settings as controller_settings
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.settings as detection_event_settings
import decsim.escalation.settings as escalation_settings
import decsim.frontends.deltakit as deltakit
import decsim.observe.settings as observe_settings
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.presets.machines as machines
import decsim.producers as producers
import decsim.qpu.settings as qpu_settings
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import decsim.topology.constants as constants_module
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph
import decsim.topology.machine_file as machine_file
import decsim.windows.settings as window_settings

HERE = pathlib.Path(__file__)
CONSTANTS_PATH = HERE.with_name("constants.yaml")
constants = constants_module.load(CONSTANTS_PATH)
record = machine_file.record
# Deltakit's SD6 at each memory's p, unchanged (frontends/deltakit.py)
SD6 = deltakit.SD6Noise()


def machine(
    distance: int,
    physical_error_probability: float,
    rounds_per_distance: int,
) -> graph.Machine:
    """The whole switching machine at one d, p and shot length."""
    weak = union_find_machine(
        distance, physical_error_probability, rounds_per_distance
    )
    edge = switching()
    strong = belief_matching_pool()
    return machines.with_strong_tier(weak, strong, edge)


def union_find_machine(
    distance: int,
    physical_error_probability: float,
    rounds_per_distance: int,
) -> graph.Machine:
    """The machine's parts held once, and union-find in the fridge."""
    chip = constants.clocks.chip
    machine_clock = machine_file.clock_of(chip)
    qpu = record(
        qpu_settings.QpuSettings,
        constants.qpu,
        source={"type": "stim_device"},
        distance=distance,
    )
    controller = record(
        controller_settings.ControllerSettings,
        constants.controller,
        machine_clock=machine_clock,
    )
    events = record(
        detection_event_settings.DetectionEventSettings,
        constants.detection_events,
        clock=chip,
    )
    windows = record(
        window_settings.WindowSettings, constants.windows, clock=chip
    )
    manager = record(decoder_settings.DecoderManagerSettings, clock=chip)
    frame = record(
        pauli_frame_module.PauliFrameConfig, constants.frame, clock=chip
    )
    workload = producers.MemorySettings(
        physical_error_probability,
        rounds_per_distance=rounds_per_distance,
        noise=SD6,
        noiseless_time_boundaries=True,
    )
    observation = record(
        observe_settings.ObservationSettings, constants.observation
    )
    links = fabric()
    base = machines.base(
        qpu,
        controller,
        events,
        windows,
        manager,
        workload,
        frame,
        clock=machine_clock,
        observation=observation,
        fabric=links,
    )
    store = record(syndrome_buffer_module.SyndromeBufferSettings, clock=chip)
    pool = union_find_pool()
    return machines.with_pool(base, pool, store=store)


def union_find_pool() -> decoder_settings.DecoderPoolSettings:
    """One union-find unit on Helios's cycle count, on a fridge FPGA."""
    settings = constants.union_find
    clock = constants.clocks.union_find
    timing = {"type": "CycleCount", "clock": clock, **settings.cycle_count}
    algorithm = {"type": "union_find", "timing": timing, **settings.algorithm}
    engine = {"type": "Engine", "clock": clock, **settings.engine}
    return record(
        decoder_settings.DecoderPoolSettings,
        settings.pool,
        algorithm=algorithm,
        engine=engine,
    )


def belief_matching_pool() -> decoder_settings.DecoderPoolSettings:
    """One GPU running belief matching at a stated time per BP iteration."""
    settings = constants.belief_matching
    timing = {"type": "IterationTime", **settings.timing}
    algorithm = {
        "type": "belief_matching",
        "timing": timing,
        **settings.algorithm,
    }
    clock = constants.clocks.chip
    engine = {"type": "Engine", "clock": clock, **settings.engine}
    return record(
        decoder_settings.DecoderPoolSettings,
        settings.pool,
        algorithm=algorithm,
        engine=engine,
    )


def switching() -> escalation_settings.SwitchingSettings:
    """Belief matching redoes a window whose cluster gap is under 20 dB."""
    settings = constants.escalation
    confidence = {"type": "cluster_gap", **settings.confidence}
    threshold = {"type": "FixedThreshold", **settings.threshold}
    redo_window = {"type": "redo_window"}
    return record(
        escalation_settings.SwitchingSettings,
        settings.edge,
        confidence=confidence,
        threshold=threshold,
        strong_window=redo_window,
        clock=constants.clocks.chip,
    )


def fabric() -> fabrics.RoleLinks:
    """The fiber loop in the fridge rack; Backline's RoCE v2 to the GPU."""
    loop = constants.fiber_loop
    chip = constants.clocks.chip
    links = {}
    for name in ("up", "read", "down"):
        links[name] = record(
            fabrics.ClockedLink, loop[name], clock=chip, source=loop.source
        )
    gpu = gpu_fabric()
    return fabrics.RoleLinks(
        write_written=links["up"],
        read_written=links["read"],
        answer_fridge=links["down"],
        base=gpu,
    )


def gpu_fabric() -> fabrics.FridgeHostFabric:
    """The cable to the GPU host, and the GPU's poll of its own memory."""
    gpu = constants.gpu_link
    cable = record(fabrics.RoundTripLeg, gpu.cable, source=gpu.source)
    host_local = record(fabrics.TimedLink, gpu.host_local, source=gpu.source)
    return fabrics.FridgeHostFabric(cable=cable, host_local=host_local)

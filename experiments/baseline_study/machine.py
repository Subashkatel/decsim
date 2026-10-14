"""The baseline study's machine: a weak decoder, then a strong one.

    machine("union_find", "ising", "redo_window", 9, 0.003)

The weak decoder, union-find or minimum-weight perfect matching, sits in
the fridge and decodes every window; a window whose gap is under the
threshold goes to the strong decoder on a host, belief matching, NVIDIA's
Ising pre-decoder with PyMatching, or correlated matching. Union-find
reports the cluster gap (Meister et al. 2405.07433), matching the
complementary gap (Gidney et al. 2312.04522), as Toshio et al.
2510.25222 pair them. Every number is in constants.yaml beside this
file, each with its source. A rotated surface code Z memory under
Deltakit's SD6 at p (Gidney et al. 2108.10457), its data prepared and
read out without noise (2510.25222 Sec. IV), 10d rounds in windows of 2d
committing d. The redo window re-decodes the escalated window; the
double window is the paper's r_com + 2 r_buf rounds (Sec. III C). The
run files take parts away by name for each decoder alone.
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
    weak: str,
    strong: str,
    strong_window: str,
    distance: int,
    physical_error_probability: float,
) -> graph.Machine:
    """The whole switching machine for one pair at one d and p.

    weak is union_find or minimum_weight_perfect_matching; strong is
    belief_matching, ising or correlated_matching; strong_window is
    redo_window or double_window.
    """
    weak_side = weak_machine(weak, distance, physical_error_probability)
    edge = switching(weak, strong_window)
    pool = STRONG_POOLS[strong]
    strong_pool = pool()
    return machines.with_strong_tier(weak_side, strong_pool, edge)


def weak_machine(
    weak: str, distance: int, physical_error_probability: float
) -> graph.Machine:
    """The machine's parts held once, and the weak decoder in the fridge."""
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
        rounds_per_distance=constants.rounds_per_distance,
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
    weak_pool = WEAK_POOLS[weak]
    pool = weak_pool()
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


def matching_pool() -> decoder_settings.DecoderPoolSettings:
    """One matching unit in the fridge at Micro Blossom's latency."""
    settings = constants.minimum_weight_perfect_matching
    algorithm = {"type": "PyMatchingDecoder", **settings.algorithm}
    clock = constants.clocks.chip
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


def ising_pool() -> decoder_settings.DecoderPoolSettings:
    """One unit running NVIDIA's pre-decoder, then PyMatching (ising_row.py).

    The row is named by its import path, so this folder is on the path
    of every process that builds it.
    """
    settings = constants.ising
    algorithm = {
        "type": "ising_row.IsingMatchingDecoder.Settings",
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


def correlated_matching_pool() -> decoder_settings.DecoderPoolSettings:
    """One host running PyMatching's correlated matching, timed on the host."""
    settings = constants.correlated_matching
    # no hardware number exists, so the decode is timed on the host CPU
    algorithm = {"type": "correlated_pymatching"}
    clock = constants.clocks.chip
    engine = {"type": "Engine", "clock": clock, **settings.engine}
    return record(
        decoder_settings.DecoderPoolSettings,
        settings.pool,
        algorithm=algorithm,
        engine=engine,
    )


def tesseract_pool() -> decoder_settings.DecoderPoolSettings:
    """One Tesseract unit, the reference decoder, on the paper's profile."""
    settings = constants.tesseract
    algorithm = {"type": "tesseract", **settings.algorithm}
    clock = constants.clocks.chip
    engine = {"type": "Engine", "clock": clock, **settings.engine}
    return record(
        decoder_settings.DecoderPoolSettings,
        settings.pool,
        algorithm=algorithm,
        engine=engine,
    )

WEAK_POOLS = {
    "union_find": union_find_pool,
    "minimum_weight_perfect_matching": matching_pool,
}
STRONG_POOLS = {
    "belief_matching": belief_matching_pool,
    "ising": ising_pool,
    "correlated_matching": correlated_matching_pool,
}


def switching(
    weak: str, strong_window: str
) -> escalation_settings.SwitchingSettings:
    """The strong decoder re-decodes a window whose gap is under 20 dB.

    Union-find reports the cluster gap, matching the complementary gap.
    """
    settings = constants.escalation
    confidence = weak_gap(weak)
    threshold = {"type": "FixedThreshold", **settings.threshold}
    shape = {"type": strong_window}
    return record(
        escalation_settings.SwitchingSettings,
        settings.edge,
        confidence=confidence,
        threshold=threshold,
        strong_window=shape,
        clock=constants.clocks.chip,
    )


def weak_gap(weak: str) -> dict:
    """The gap the weak decoder reports, as a confidence slot reads it."""
    # the cluster gap's walk is charged on union-find's own unit; the
    # second forced solve is a second job of the matching unit
    # (2510.25222 Sec. II C)
    if weak == "union_find":
        settings = constants.escalation
        return {"type": "cluster_gap", **settings.confidence}
    return {"type": "complementary_gap"}


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

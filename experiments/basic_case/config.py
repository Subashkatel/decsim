"""The basic case's machines: union-find, a strong decoder, or switching.

    machine("weak_alone", 5, 0.002)
    machine("strong_alone", 5, 0.002)
    machine("switching", 5, 0.002, strong_link="nvqlink_gpu")
    machine("strong_alone", 9, 0.003, strong_link="roce_cpu",
            strong_pool=correlated_matching_pool("60.3us"))

Every number is in constants.yaml beside this file, each with its
source; the setup is specs/adaptive-switching/parameters.md. The three
runs share one machine but for the decoding: union-find keeps every
window, the strong decoder decodes every window, or union-find first and
the strong decoder redoing, on the same window, each window whose
cluster gap is under the threshold, the next window waiting for its
answer (strong_window_shapes.RedoWindow, Toshio et al. 2510.25222
App. C). The strong decoder is belief matching on a GPU unless a run
gives another pool. Each run reads its shots from the point's saved
dataset (make_datasets.py), so the data never changes between runs.
"""

import dataclasses
import pathlib
from typing import Any, Optional

import decsim.controller.settings as controller_settings
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.settings as detection_event_settings
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.frontends.deltakit as deltakit
import decsim.observe.settings as observe_settings
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.presets.link_cards as link_cards
import decsim.presets.machines as machines
import decsim.producers as producers
import decsim.qpu.settings as qpu_settings
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import decsim.topology.constants as constants_module
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph
import decsim.topology.machine_file as machine_file
import decsim.windows.boundary_policies as boundary_policies
import decsim.windows.settings as window_settings

HERE = pathlib.Path(__file__)
CONSTANTS_PATH = HERE.with_name("constants.yaml")
constants = constants_module.load(CONSTANTS_PATH)
record = machine_file.record
# the cable to the host: Backline's measured RoCE v2 echo, 4.5 us a round
# trip to a GPU or 2.305 us to a CPU, or NVQLink's 3.839 us to a GPU
# (presets/link_cards.py)
STRONG_LINKS = {
    "roce_gpu": link_cards.ROCE_V2_GPU_FABRIC,
    "roce_cpu": link_cards.ROCE_V2_CPU_FABRIC,
    "nvqlink_gpu": link_cards.NVQLINK_GPU_FABRIC,
}
CODE_TASK = "surface_code:rotated_memory_z"
# the points every run file covers (parameters.md)
DECODINGS = ("weak_alone", "strong_alone", "switching")
DISTANCES = (5, 7, 9, 11)
PHYSICAL_ERROR_PROBABILITIES = (0.002, 0.003, 0.005)
# exactly a million shots a run, no early stop; the datasets hold as many
SHOTS = 1_000_000
# one Slurm task's dataset chunk: about 33 minutes at d = 11
CHUNK_SHOTS = 50_000


@dataclasses.dataclass(frozen=True)
class Scheme:
    """How a switching machine handles a window it escalates.

    strong_window is a strong window shape's Settings record, and
    strong_units the units the strong pool holds.
    """

    strong_window: Any
    strong_units: int = 1


REDO = strong_window_shapes.RedoWindow.Settings()
DOUBLE = strong_window_shapes.DoubleWindow.Settings()
CHECKED = boundary_policies.Checked.Settings()
CHECKED_REDO = strong_window_shapes.RedoWindow.Settings(boundary_policy=CHECKED)
SCHEMES = {
    "sliding": Scheme(REDO),
    "double": Scheme(DOUBLE),
    "speculation": Scheme(CHECKED_REDO),
}


def machine(
    decoding: str,
    distance: int,
    physical_error_probability: float,
    strong_link: str = "roce_gpu",
    strong_pool: Optional[decoder_settings.DecoderPoolSettings] = None,
) -> graph.Machine:
    """One run at one point: weak_alone, strong_alone or switching.

    strong_pool is the strong decoder; None is belief matching on a GPU.
    """
    weak = weak_alone(distance, physical_error_probability, strong_link)
    if decoding == "weak_alone":
        return weak
    if strong_pool is None:
        strong_pool = belief_matching_pool()
    if decoding == "strong_alone":
        return machines.strong_alone(weak, strong_pool)
    return switching(weak, strong_pool)


def memory(physical_error_probability: float) -> producers.MemorySettings:
    """The memory every run decodes: SD6, data prepared and read cleanly."""
    sd6 = deltakit.SD6Noise()
    return producers.MemorySettings(
        physical_error_probability,
        code_task=CODE_TASK,
        rounds_per_distance=constants.rounds_per_distance,
        noise=sd6,
        noiseless_time_boundaries=True,
    )


def dataset_folder(distance: int, physical_error_probability: float) -> str:
    """Where the point's saved shots are (make_datasets.py)."""
    folder = pathlib.Path(constants.datasets.folder)
    point = folder / f"d{distance}_p{physical_error_probability}"
    return str(point)


def weak_alone(
    distance: int, physical_error_probability: float, strong_link: str
) -> graph.Machine:
    """Union-find keeps every window; the parts on the chip's clock."""
    chip = constants.clocks.chip
    folder = dataset_folder(distance, physical_error_probability)
    source = {"type": "dataset_stim", "dataset_folder": folder}
    qpu = record(
        qpu_settings.QpuSettings,
        constants.qpu,
        source=source,
        distance=distance,
    )
    workload = memory(physical_error_probability)
    machine_clock = machine_file.clock_of(chip)
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
    store = record(syndrome_buffer_module.SyndromeBufferSettings, clock=chip)
    observation = record(
        observe_settings.ObservationSettings, constants.observation
    )
    link_fabric = fabric(strong_link)
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
        fabric=link_fabric,
    )
    weak_pool = union_find_pool()
    return machines.with_pool(base, weak_pool, store=store)


def switching(
    weak: graph.Machine, strong_pool: decoder_settings.DecoderPoolSettings
) -> graph.Machine:
    """A window whose cluster gap is under the threshold, redone on the host."""
    settings = constants.escalation
    confidence = {"type": "cluster_gap", **settings.confidence}
    threshold = {"type": "FixedThreshold", **settings.threshold}
    edge = record(
        escalation_settings.SwitchingSettings,
        settings.edge,
        confidence=confidence,
        threshold=threshold,
        clock=constants.clocks.chip,
    )
    return machines.with_strong_tier(weak, strong_pool, edge)


def with_scheme(switching: Any, scheme: Scheme) -> Any:
    """The switching machine with its escalated windows of this scheme.

    The windows take the boundary row the strong window gives, as
    machines.with_strong_tier lays it.
    """
    shape = scheme.strong_window
    shaped = switching.with_settings(machines.SWITCHING, strong_window=shape)
    windows = shaped.held(machines.WINDOWS)
    rowed = window_settings.switching_windows(windows, shape)
    shaped = shaped.with_held(machines.WINDOWS, rowed)
    return shaped.with_settings(
        machines.STRONG_POOL, unit_count=scheme.strong_units
    )


def union_find_pool() -> decoder_settings.DecoderPoolSettings:
    """Union-find on Helios's cycle law, on an FPGA in the fridge rack."""
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
    """Belief matching on one GPU, priced by its stated time per iteration."""
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
        decoder_settings.DecoderPoolSettings, algorithm=algorithm, engine=engine
    )


def correlated_matching_pool(
    decode_time: str,
) -> decoder_settings.DecoderPoolSettings:
    """PyMatching's correlated matching on one host core, at a stated time.

    decode_time is one decode's time, written with its unit ("60.3us").
    The host link's measured round trip holds the loading and the
    unloading, so the engine's stages cost nothing.
    """
    algorithm = {"type": "correlated_pymatching", "preset_latency": decode_time}
    engine = {
        "type": "Engine",
        "clock": constants.clocks.chip,
        "fetch_per_job": "0 cycles",
        "fetch_per_round": "0 cycles",
        "release_per_job": "0 cycles",
        "release_per_round": "0 cycles",
    }
    return record(
        decoder_settings.DecoderPoolSettings, algorithm=algorithm, engine=engine
    )


def fabric(strong_link: str) -> fabrics.RoleLinks:
    """The fiber loop in the fridge rack; the cable to the GPU host."""
    loop = constants.fiber_loop
    chip = constants.clocks.chip
    links = {}
    for name in ("up", "read", "down"):
        links[name] = record(
            fabrics.ClockedLink, loop[name], clock=chip, source=loop.source
        )
    return fabrics.RoleLinks(
        write_written=links["up"],
        read_written=links["read"],
        answer_fridge=links["down"],
        base=STRONG_LINKS[strong_link],
    )

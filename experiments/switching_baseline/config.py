"""Experiment 1's machine from its constants: union-find, then Relay-BP-5.

    machine("switching", distance=11, physical_error_probability=0.003)
    machine("weak_alone", 11, 0.003)

Every number is in constants.yaml beside this file, each with its
source; every choice of part is written here, on decsim's own parts and
wires (decsim/presets/machines.py). The machine is
decsim/presets/experiment_one.py's at every cell
(tests/test_three_files_example.py).
"""

import pathlib

import decsim.controller.settings as controller_settings
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.settings as detection_event_settings
import decsim.escalation.settings as escalation_settings
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
import decsim.windows.settings as window_settings

HERE = pathlib.Path(__file__)
CONSTANTS_PATH = HERE.with_name("constants.yaml")
constants = constants_module.load(CONSTANTS_PATH)
record = machine_file.record


def machine(
    decoding: str, distance: int, physical_error_probability: float
) -> graph.Machine:
    """Experiment 1 at one cell: switching, or union-find alone."""
    weak = weak_alone(distance, physical_error_probability)
    if decoding == "weak_alone":
        return weak
    return switching(weak)


def weak_alone(
    distance: int, physical_error_probability: float
) -> graph.Machine:
    """Union-find keeps every window; the parts on the chip's clock."""
    chip = constants.clocks.chip
    qpu = record(
        qpu_settings.QpuSettings,
        constants.qpu,
        source={"type": "stim_device"},
        distance=distance,
    )
    workload = producers.MemorySettings(
        physical_error_probability, rounds=constants.rounds_per_shot
    )
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
    machine = machines.base(
        qpu,
        controller,
        events,
        windows,
        manager,
        workload,
        frame,
        clock=machine_clock,
        observation=observation,
        fabric=fabric(),
    )
    return machines.with_pool(machine, union_find_pool(), store=store)


def strong_alone(
    distance: int,
    physical_error_probability: float,
    strong_pool: decoder_settings.DecoderPoolSettings,
) -> graph.Machine:
    """Every window decoded by a strong pool on the host."""
    weak = weak_alone(distance, physical_error_probability)
    return machines.strong_alone(weak, strong_pool)


def switching(weak: graph.Machine) -> graph.Machine:
    """A window whose cluster gap is under the threshold, redone on the host."""
    settings = constants.escalation
    confidence = {"type": "cluster_gap", **settings.confidence}
    threshold = {"type": "FixedThreshold", **settings.threshold}
    double_window = {"type": "double_window"}
    edge = record(
        escalation_settings.SwitchingSettings,
        settings.edge,
        confidence=confidence,
        threshold=threshold,
        strong_window=double_window,
        clock=constants.clocks.chip,
    )
    strong = relay_bp_pool()
    return machines.with_strong_tier(weak, strong, edge)


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


def relay_bp_pool() -> decoder_settings.DecoderPoolSettings:
    """Relay-BP-5 priced on its line measured on a whole A100."""
    settings = constants.relay_bp
    algorithm = {"type": "measured_table", **settings.algorithm}
    clock = constants.clocks.chip
    engine = {"type": "Engine", "clock": clock, **settings.engine}
    return record(
        decoder_settings.DecoderPoolSettings, algorithm=algorithm, engine=engine
    )


def fabric() -> fabrics.RoleLinks:
    """The fiber loop in the fridge rack; Backline's RoCE v2 to the host."""
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
        base=link_cards.ROCE_V2_CPU_FABRIC,
    )

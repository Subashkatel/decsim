"""The whole machine's settings: one record per part.

The aggregate object model of a run, kept apart from the build script
that reads it. gem5 draws the same line: a component's parameters are
its own Python class (src/python/m5/SimObject.py:204-205) and the
configuration script in configs/ reads them
(configs/deprecated/example/se.py), never the other way round. Each
part's record is defined by the package that owns it.
"""

import dataclasses
from typing import Optional, Union

import decsim.config as config
import decsim.controller.policies as idle_policies
import decsim.controller.settings as controller_settings
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.settings as detection_event_settings
import decsim.escalation.settings as escalation_settings
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.observe.settings as observe_settings
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.ports as ports
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.qpu.settings as qpu_settings
import decsim.records.windows as window_records
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import decsim.windows.routers as routers
import decsim.windows.settings as window_settings


@dataclasses.dataclass(frozen=True)
class MachineSettings:
    """One settings record per part of the machine.

    Every field has a default: a timing-only run of distance-three
    surface code patches with no decoder. clock is the machine's clock,
    the one every part that names none of its own counts its cycles on.

    The decode side is three slots, each None when the run has no such
    part: weak_decoder alone, strong_decoder alone, or both with
    switching, which decodes weak first and escalates a window to the
    strong decoder. None of the three plans no decoding. router, None
    by default, sends each window to one of several entry pools; only a
    graph of its own shape holds more than one (windows/routers.py).

    workload is a workload lowered in Python or a maker row
    (decsim/producers.py), which at_task lowers against the qpu, so the
    task's id and every shot see the lowered workload. A workload made
    at a distance of its own fixes the qpu's: set_in_the_run_file names
    the paths an assignment by path (decsim/parameters/tree.py) may not
    change, since it would leave the circuit behind, a distance-5
    machine running a distance-3 circuit.
    """

    clock: Optional[config.Clock] = None
    qpu: qpu_settings.QpuSettings = qpu_settings.QpuSettings()
    controller: controller_settings.ControllerSettings = (
        controller_settings.ControllerSettings()
    )
    idle_policy: Union[
        idle_policies.IgnoreSettings, idle_policies.SeparateDecodeJobsSettings
    ] = idle_policies.SeparateDecodeJobsSettings()
    detection_events: detection_event_settings.DetectionEventSettings = (
        detection_event_settings.DetectionEventSettings()
    )
    links: link_settings.FabricSettings = (
        link_profiles.logical_reference_profile()
    )
    weak_syndrome_buffer: ports.SyndromeBufferSettings = (
        syndrome_buffer_module.SyndromeBufferSettings()
    )
    strong_syndrome_buffer: syndrome_buffer_module.SyndromeBufferSettings = (
        syndrome_buffer_module.SyndromeBufferSettings()
    )
    windows: window_settings.WindowSettings = window_settings.WindowSettings()
    weak_decoder: Optional[decoder_settings.DecoderPoolSettings] = None
    strong_decoder: Optional[decoder_settings.DecoderPoolSettings] = None
    decoder_manager: decoder_settings.DecoderManagerSettings = (
        decoder_settings.DecoderManagerSettings()
    )
    switching: Optional[escalation_settings.SwitchingSettings] = None
    pauli_frame: Optional[pauli_frame_module.PauliFrameConfig] = None
    workload: Union[
        workload_settings.WorkloadSettings, ports.WorkloadMakerSettings
    ] = workload_settings.WorkloadSettings()
    magic_state_factory: qpu_settings.FactorySettings = (
        magic_state_factories.InfiniteFactory.Settings()
    )
    observation: observe_settings.ObservationSettings = (
        observe_settings.ObservationSettings()
    )
    router: Optional[routers.Router.Settings] = config.added_field(None)

    def __post_init__(self) -> None:
        _check_decode_slots(self)

    def decoder_settings_for(
        self, tier: str
    ) -> Optional[decoder_settings.DecoderPoolSettings]:
        """The pool of one decoder tier, by its tier word, weak or strong."""
        if tier == "weak":
            return self.weak_decoder
        assert tier == "strong", f"no decoder tier named {tier!r}"
        return self.strong_decoder

    @property
    def window_tier(self) -> window_records.PoolId:
        """The tier that decodes the windows: weak if set, else strong."""
        if self.weak_decoder is None and self.strong_decoder is not None:
            return window_records.STRONG_POOL_ID
        return window_records.WEAK_POOL_ID

    @property
    def escalation_kind(self) -> str:
        """The word a trace names these slots' run by."""
        if self.switching is not None:
            return "switching"
        if self.window_tier == window_records.STRONG_POOL_ID:
            return "strong_only"
        return "weak_baseline"

    def task_facts(self) -> dict:
        """The task's facts a threshold reads, None where none is stated.

        The names are threshold_sources.TASK_FACTS; a window size of
        None is the distance.
        """
        workload_record = self.workload.workload_record
        probability = None
        if workload_record is not None:
            probability = workload_record.physical_error_probability
        scheme = self.windows.scheme
        return {
            "distance": self.qpu.distance,
            "physical_error_probability": probability,
            "round_period_microseconds": self.qpu.round_period_microseconds,
            "commit_rounds": scheme.commit_rounds,
            "buffer_rounds": scheme.buffer_rounds,
        }

    def at_task(self) -> "MachineSettings":
        """The settings with their workload lowered and their threshold read.

        A maker row is lowered against the qpu, then a calibration
        table's row is read at the task's facts, once per task; reading
        settings already read returns them unchanged.
        """
        settings = self._with_workload_lowered()
        switching = settings.switching
        if switching is None:
            return settings
        facts = settings.task_facts()
        threshold = switching.threshold.at_task(facts)
        if threshold is switching.threshold:
            return settings
        resolved = dataclasses.replace(switching, threshold=threshold)
        return dataclasses.replace(settings, switching=resolved)

    def set_in_the_run_file(self) -> tuple:
        """The paths the workload fixes, which an assignment may not set."""
        return set_in_the_run_file_with(self.workload)

    def _with_workload_lowered(self) -> "MachineSettings":
        workload = self.workload.lowered_for(self.qpu)
        if workload is self.workload:
            return self
        return dataclasses.replace(self, workload=workload)


def set_in_the_run_file_with(workload: object) -> tuple:
    """The machine paths a workload fixes: the distance, its lowered fields.

    A workload made at a distance of its own fixes qpu.distance; a
    lowered one fixes its lowered fields too. A maker row that follows
    the qpu fixes nothing.
    """
    paths = []
    if workload.fixes_the_distance:
        paths.append("qpu.distance")
    lowered_fields = getattr(workload, "fields_set_in_the_run_file", ())
    for field in lowered_fields:
        paths.append(f"workload.{field}")
    return tuple(paths)


def _check_decode_slots(settings: MachineSettings) -> None:
    """Without switching, one decoder; a second would run on ignored."""
    has_weak = settings.weak_decoder is not None
    has_strong = settings.strong_decoder is not None
    has_two_decoders = has_weak and has_strong
    if settings.switching is not None or not has_two_decoders:
        return
    raise ValueError(
        "weak_decoder and strong_decoder are both set and switching is "
        "not; a run with no switching decodes its windows on one decoder, "
        "so set switching or drop one decoder"
    )

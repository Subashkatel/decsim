"""The whole machine's settings: one record per yaml section.

The aggregate object model of a run, kept apart from the build script
that reads it. gem5 draws the same line: a component's parameters are
its own Python class (src/python/m5/SimObject.py:204-205) and the
configuration script in configs/ reads them
(configs/deprecated/example/se.py), never the other way round. Each
section's record is built by the package that owns it, and `row`
resolves a section's `kind` against that package's plug-in table.
"""

import dataclasses
from collections.abc import Mapping
from typing import Optional, Union

import decsim.burst_detectors.settings as burst_detector_settings
import decsim.config as config
import decsim.controller.settings as controller_settings
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.settings as detection_event_settings
import decsim.escalation.settings as escalation_settings
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.observe.settings as observe_settings
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.qpu.settings as qpu_settings
import decsim.syndrome_buffer.ported_syndrome_buffer as ported_syndrome_buffer
import decsim.syndrome_buffer.settings as syndrome_buffer_settings
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import decsim.windows.settings as window_settings

# The yaml sections, in the order MachineSettings reads them. Each
# section's own package owns its settings record and its plug-in table;
# the root looks a kind up in that table once and refuses one that is
# not a row, naming the rows (decsim/tables.py).
SECTIONS = (
    "clocks",
    "qpu",
    "controller",
    "idle_policy",
    "detection_events",
    "links",
    "weak_syndrome_buffer",
    "strong_syndrome_buffer",
    "windows",
    "weak_decoder",
    "strong_decoder",
    "decoder_manager",
    "escalation",
    "burst_detector",
    "pauli_frame",
    "workload",
    "magic_state_factory",
    "observation",
)
# The sections from_mapping reads without a default; the rest fall back
# to their settings record's defaults when the yaml leaves them out.
REQUIRED_SECTIONS = (
    "clocks",
    "qpu",
    "controller",
    "links",
    "weak_syndrome_buffer",
    "strong_syndrome_buffer",
    "windows",
    "pauli_frame",
    "workload",
)


@dataclasses.dataclass(frozen=True)
class MachineSettings:
    """One settings record per yaml section, plus the Python-only knobs.

    Every field has a default, so a Python caller names only what
    differs from a timing-only run of three-qubit surface code patches
    with no decoder at all. links is the fabric card; the reference card
    prices propagation only. clock is the machine's clock, the one every
    part that names none of its own counts its cycles on.
    """

    clocks: config.ClockSettings = config.ClockSettings()
    clock: Optional[config.Clock] = None
    qpu: qpu_settings.QpuSettings = qpu_settings.QpuSettings()
    controller: controller_settings.ControllerSettings = (
        controller_settings.ControllerSettings()
    )
    idle_policy: controller_settings.IdlePolicySettings = (
        controller_settings.IdlePolicySettings()
    )
    detection_events: detection_event_settings.DetectionEventSettings = (
        detection_event_settings.DetectionEventSettings()
    )
    links: link_settings.FabricSettings = (
        link_profiles.logical_reference_profile()
    )
    weak_syndrome_buffer: Union[
        syndrome_buffer_module.SyndromeBufferSettings,
        ported_syndrome_buffer.PortedSyndromeBufferSettings,
    ] = syndrome_buffer_module.SyndromeBufferSettings()
    strong_syndrome_buffer: syndrome_buffer_module.SyndromeBufferSettings = (
        syndrome_buffer_module.SyndromeBufferSettings()
    )
    windows: window_settings.WindowSettings = window_settings.WindowSettings()
    weak_decoder: decoder_settings.DecoderSettings = (
        decoder_settings.DecoderSettings()
    )
    strong_decoder: decoder_settings.DecoderSettings = (
        decoder_settings.DecoderSettings()
    )
    decoder_manager: decoder_settings.DecoderManagerSettings = (
        decoder_settings.DecoderManagerSettings()
    )
    escalation: escalation_settings.EscalationSettings = (
        escalation_settings.EscalationSettings()
    )
    burst_detector: burst_detector_settings.BurstDetectorSettings = (
        burst_detector_settings.BurstDetectorSettings()
    )
    pauli_frame: Optional[pauli_frame_module.PauliFrameConfig] = None
    workload: workload_settings.WorkloadSettings = (
        workload_settings.WorkloadSettings()
    )
    magic_state_factory: qpu_settings.FactorySettings = (
        qpu_settings.FactorySettings()
    )
    observation: observe_settings.ObservationSettings = (
        observe_settings.ObservationSettings()
    )

    def decoder_settings_for(
        self, tier: str
    ) -> decoder_settings.DecoderSettings:
        """The card of one decoder tier, named the way the yaml names it."""
        if tier == "weak":
            return self.weak_decoder
        assert tier == "strong", f"no decoder tier named {tier!r}"
        return self.strong_decoder

    @classmethod
    def from_mapping(
        cls, sections: Mapping, *, name: str, section_folders: Mapping
    ) -> "MachineSettings":
        """One yaml's sections, each handed to the package that owns it.

        name labels the links card in the run's description; section_folders
        maps a section to the folder of the yaml that wrote it, which the
        escalation section's relative table path and the workload's
        relative files resolve against.
        """
        unknown = set(sections) - set(SECTIONS)
        if unknown:
            listed = sorted(unknown)
            raise ValueError(
                f"the yaml has no section {listed}; the sections are "
                f"{list(SECTIONS)}"
            )
        _check_section_shapes(sections)
        clocks = config.ClockSettings.from_yaml(sections["clocks"])
        idle_policy_section = sections.get("idle_policy", {})
        detection_events_section = sections.get("detection_events", {})
        escalation_section = sections.get("escalation", {})
        decoder_manager_section = sections.get("decoder_manager", {})
        observation_section = sections.get("observation", {})
        factory_section = sections.get("magic_state_factory", {})
        qpu = qpu_settings.QpuSettings.from_yaml(sections["qpu"])
        controller = controller_settings.ControllerSettings.from_yaml(
            sections["controller"], clocks
        )
        idle_policy = controller_settings.IdlePolicySettings.from_yaml(
            idle_policy_section
        )
        event_settings = detection_event_settings.DetectionEventSettings
        detection_events = event_settings.from_yaml(
            detection_events_section, clocks
        )
        links = link_profiles.from_yaml(sections["links"], clocks, name)
        weak_syndrome_buffer = syndrome_buffer_settings.from_yaml(
            sections["weak_syndrome_buffer"], "weak_syndrome_buffer", clocks
        )
        strong_section = sections["strong_syndrome_buffer"]
        syndrome_buffer_settings.check_strong_section_charges_nothing(
            strong_section
        )
        strong_syndrome_buffer = syndrome_buffer_settings.from_yaml(
            strong_section, "strong_syndrome_buffer", clocks
        )
        windows = window_settings.WindowSettings.from_yaml(
            sections["windows"], clocks, controller.clock
        )
        weak_decoder = _tier_settings(sections, "weak_decoder", clocks)
        strong_decoder = _tier_settings(sections, "strong_decoder", clocks)
        decoder_manager = decoder_settings.DecoderManagerSettings.from_yaml(
            decoder_manager_section, clocks
        )
        escalation_folder = section_folders.get("escalation")
        escalation = escalation_settings.EscalationSettings.from_yaml(
            escalation_section, clocks, escalation_folder, controller.clock
        )
        burst_detector_section = sections.get("burst_detector", {})
        burst_detector = (
            burst_detector_settings.BurstDetectorSettings.from_yaml(
                burst_detector_section, clocks
            )
        )
        pauli_frame = pauli_frame_module.PauliFrameConfig.from_yaml(
            sections["pauli_frame"], clocks
        )
        workload_folder = section_folders.get("workload")
        workload = workload_settings.WorkloadSettings.from_yaml(
            sections["workload"], workload_folder
        )
        magic_state_factory = qpu_settings.FactorySettings.from_yaml(
            factory_section
        )
        observation = observe_settings.ObservationSettings.from_yaml(
            observation_section
        )
        return cls(
            clocks=clocks,
            clock=controller.clock,
            qpu=qpu,
            controller=controller,
            idle_policy=idle_policy,
            detection_events=detection_events,
            links=links,
            weak_syndrome_buffer=weak_syndrome_buffer,
            strong_syndrome_buffer=strong_syndrome_buffer,
            windows=windows,
            weak_decoder=weak_decoder,
            strong_decoder=strong_decoder,
            decoder_manager=decoder_manager,
            escalation=escalation,
            burst_detector=burst_detector,
            pauli_frame=pauli_frame,
            workload=workload,
            magic_state_factory=magic_state_factory,
            observation=observation,
        )


def _check_section_shapes(sections: Mapping) -> None:
    """Every section the machine reads is there, and each holds its keys."""
    missing = set(REQUIRED_SECTIONS) - set(sections)
    if missing:
        listed = sorted(missing)
        raise ValueError(
            f"the yaml needs the sections {listed}; configs/reference.yaml "
            "holds every section with its keys"
        )
    for name, section in sections.items():
        if not isinstance(section, Mapping):
            sentence = _not_a_mapping_sentence(name, section)
            raise ValueError(sentence)


def _not_a_mapping_sentence(name: str, section) -> str:
    """The refusal of a section written as one value, with its mapping form.

    A section written as a bare word is a row's name, so the sentence
    shows that word under the section's kind key.
    """
    sentence = (
        f"the yaml section {name} holds {section!r}; a section is a "
        "mapping of its keys, as in configs/reference.yaml"
    )
    if not isinstance(section, str):
        return sentence
    return f"{sentence}, so write {name}: {{kind: {section}}}"


def _tier_settings(
    sections: Mapping, tier: str, clocks: config.ClockSettings
) -> decoder_settings.DecoderSettings:
    """A tier's section, or no decoder when the yaml leaves it out."""
    if tier not in sections:
        return decoder_settings.DecoderSettings()
    return decoder_settings.DecoderSettings.from_yaml(
        sections[tier], clocks, tier
    )

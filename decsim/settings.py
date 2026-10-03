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
import decsim.confidence.signals as confidence_signals
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
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.qpu.settings as qpu_settings
import decsim.records.windows as window_records
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
# Where a point's yaml writes each fact a threshold row reads at the
# point (threshold_sources.POINT_FACTS): the code distance, the physical
# error probability the workload's maker takes, the round period, and
# the window's commit and buffer rounds.
POINT_FACT_PATHS = {
    "distance": ("qpu", "distance"),
    "physical_error_probability": (
        "workload",
        "arguments",
        "physical_error_probability",
    ),
    "round_period_microseconds": ("qpu", "round_period_microseconds"),
    "commit_rounds": ("windows", "commit_rounds"),
    "buffer_rounds": ("windows", "buffer_rounds"),
}


@dataclasses.dataclass(frozen=True)
class MachineSettings:
    """One settings record per yaml section, plus the Python-only knobs.

    Every field has a default, so a Python caller names only what
    differs from a timing-only run of three-qubit surface code patches
    with no decoder at all. links is the fabric card; the reference card
    prices propagation only. clock is the machine's clock, the one every
    part that names none of its own counts its cycles on.

    The decode side is three slots, each None when the run has no such
    part: weak_decoder alone decodes every window once on the weak
    decoder, strong_decoder alone on the strong one, and both with
    switching decode weak first and escalate a window to the strong
    decoder; none of the three is a run that plans no decoding. Two
    decoders without switching, or switching without both, is refused.
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
    weak_syndrome_buffer: Union[
        syndrome_buffer_module.SyndromeBufferSettings,
        ported_syndrome_buffer.PortedSyndromeBufferSettings,
    ] = syndrome_buffer_module.SyndromeBufferSettings()
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
    workload: workload_settings.WorkloadSettings = (
        workload_settings.WorkloadSettings()
    )
    magic_state_factory: qpu_settings.FactorySettings = (
        magic_state_factories.InfiniteFactory.Settings()
    )
    observation: observe_settings.ObservationSettings = (
        observe_settings.ObservationSettings()
    )

    def __post_init__(self) -> None:
        _check_decode_slots(self)

    def decoder_settings_for(
        self, tier: str
    ) -> Optional[decoder_settings.DecoderPoolSettings]:
        """The card of one decoder tier, named the way the yaml names it."""
        if tier == "weak":
            return self.weak_decoder
        assert tier == "strong", f"no decoder tier named {tier!r}"
        return self.strong_decoder

    @property
    def window_tier(self) -> window_records.DecoderTier:
        """The tier that decodes the plan's windows: weak if set, else strong.

        A run with no decoder plans no decoding, and its windows, if any,
        are the weak tier's.
        """
        if self.weak_decoder is None and self.strong_decoder is not None:
            return window_records.DecoderTier.STRONG
        return window_records.DecoderTier.WEAK

    @property
    def escalation_kind(self) -> str:
        """The yaml's escalation.kind for these slots, which a trace prints.

        The trace's process name and the plots that read it name a run
        by this word (escalation/settings.py ESCALATION_KINDS).
        """
        if self.switching is not None:
            return "switching"
        if self.window_tier is window_records.DecoderTier.STRONG:
            return "strong_only"
        return "weak_baseline"

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
        idle_policy = controller_settings.idle_policy_from_yaml(
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
        strong_syndrome_buffer = syndrome_buffer_settings.from_yaml(
            strong_section, "strong_syndrome_buffer", clocks
        )
        weak_decoder = _tier_settings(sections, "weak_decoder", clocks)
        strong_decoder = _tier_settings(sections, "strong_decoder", clocks)
        decoder_manager = decoder_settings.DecoderManagerSettings.from_yaml(
            decoder_manager_section, clocks
        )
        escalation_folder = section_folders.get("escalation")
        facts = point_facts(sections)
        switching = escalation_settings.SwitchingSettings.from_yaml(
            escalation_section,
            clocks,
            escalation_folder,
            confidence_signals.confidence_settings,
            facts,
        )
        windows = window_settings.WindowSettings.from_yaml(
            sections["windows"], clocks, switching
        )
        kind = escalation_settings.escalation_kind(escalation_section)
        kept_sections = escalation_settings.ESCALATION_KINDS[kind]
        if "weak_decoder" not in kept_sections:
            weak_decoder = None
        if "strong_decoder" not in kept_sections:
            strong_decoder = None
        burst_detector_section = sections.get("burst_detector", {})
        burst_detector = burst_detector_settings.detector_from_yaml(
            burst_detector_section, clocks
        )
        burst_detector_settings.refuse_a_detector_without_switching(
            burst_detector_section, switching
        )
        if switching is not None:
            switching = dataclasses.replace(
                switching, burst_detector=burst_detector
            )
        pauli_frame = pauli_frame_module.PauliFrameConfig.from_yaml(
            sections["pauli_frame"], clocks
        )
        workload_folder = section_folders.get("workload")
        workload = workload_settings.WorkloadSettings.from_yaml(
            sections["workload"], workload_folder
        )
        magic_state_factory = qpu_settings.factory_from_yaml(factory_section)
        observation = observe_settings.ObservationSettings.from_yaml(
            observation_section
        )
        window_check = observe_settings.window_check_from_yaml(
            observation_section
        )
        if weak_decoder is not None:
            weak_decoder = _checked(weak_decoder, window_check)
        elif strong_decoder is not None:
            strong_decoder = _checked(strong_decoder, window_check)
        return cls(
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
            switching=switching,
            pauli_frame=pauli_frame,
            workload=workload,
            magic_state_factory=magic_state_factory,
            observation=observation,
        )


def point_facts(sections: Mapping) -> dict:
    """A point's facts by name, from its resolved sections; None if absent."""
    facts = {}
    for name, path in POINT_FACT_PATHS.items():
        facts[name] = _value_at(sections, path)
    return facts


def _value_at(sections: Mapping, path: tuple) -> object:
    """The value a path of keys reaches, or None where a key is missing."""
    value = sections
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _checked(
    pool: decoder_settings.DecoderPoolSettings, window_check
) -> decoder_settings.DecoderPoolSettings:
    """The pool of the tier that decodes the windows, with its referee.

    The referee wraps that tier's decoder only, the one window_tier names;
    window_check None leaves the pool as it is.
    """
    if window_check is None:
        return pool
    checked = window_check.Settings(inner=pool.algorithm)
    return dataclasses.replace(pool, algorithm=checked)


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
) -> Optional[decoder_settings.DecoderPoolSettings]:
    """A tier's section, or no decoder when the yaml leaves it out."""
    if tier not in sections:
        return None
    return decoder_settings.DecoderPoolSettings.from_yaml(
        sections[tier], clocks, tier
    )


def _check_decode_slots(settings: MachineSettings) -> None:
    """Switching fills all three slots; without it one decoder at most."""
    has_weak = settings.weak_decoder is not None
    has_strong = settings.strong_decoder is not None
    if settings.switching is None:
        if has_weak and has_strong:
            raise ValueError(
                "weak_decoder and strong_decoder are both set and switching "
                "is not; a run with no switching decodes its windows on one "
                "decoder, so set switching or drop one decoder"
            )
        return
    if not has_strong:
        raise ValueError(
            "switching is set and strong_decoder is not; switching "
            "escalates a window to the strong decoder, so set strong_decoder"
        )
    if not has_weak:
        raise ValueError(
            "switching is set and weak_decoder is not; switching decodes "
            "every window on the weak decoder first, so set weak_decoder"
        )

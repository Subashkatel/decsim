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
from typing import Optional

import decsim.config as config
import decsim.controller.settings as controller_settings
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.observe.settings as observe_settings
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.qpu.settings as qpu_settings
import decsim.syndrome_buffer.settings as round_store_settings
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
    "links",
    "round_store",
    "strong_round_store",
    "windows",
    "weak_decoder",
    "strong_decoder",
    "decoder_manager",
    "escalation",
    "pauli_frame",
    "workload",
    "observation",
)


@dataclasses.dataclass(frozen=True)
class MachineSettings:
    """One settings record per yaml section, plus the Python-only knobs.

    Every field has a default, so a Python caller names only what
    differs from a timing-only run of three-qubit surface code patches
    with no decoder at all. links is the fabric card; the reference card
    prices propagation only. magic_state_factory has no yaml key today.
    """

    clocks: config.ClockSettings = config.ClockSettings()
    qpu: qpu_settings.QpuSettings = qpu_settings.QpuSettings()
    controller: controller_settings.ControllerSettings = (
        controller_settings.ControllerSettings()
    )
    idle_policy: controller_settings.IdlePolicySettings = (
        controller_settings.IdlePolicySettings()
    )
    links: link_settings.FabricSettings = (
        link_profiles.logical_reference_profile()
    )
    round_store: round_store_settings.RoundStoreSettings = (
        round_store_settings.RoundStoreSettings()
    )
    strong_round_store: round_store_settings.RoundStoreSettings = (
        round_store_settings.RoundStoreSettings()
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

    @classmethod
    def from_mapping(
        cls, sections: Mapping, *, name: str, base_directory
    ) -> "MachineSettings":
        """One yaml's sections, each handed to the package that owns it.

        name labels the links card in the traffic ledger; base_directory
        resolves the escalation section's relative table path.
        """
        unknown = set(sections) - set(SECTIONS)
        if unknown:
            listed = sorted(unknown)
            raise ValueError(
                f"the yaml has no section {listed}; the sections are "
                f"{list(SECTIONS)}"
            )
        clocks = config.ClockSettings.from_yaml(sections["clocks"])
        idle_policy = controller_settings.IdlePolicySettings()
        if "idle_policy" in sections:
            idle_policy = controller_settings.IdlePolicySettings(
                kind=sections["idle_policy"]
            )
        escalation_section = sections.get("escalation", {})
        decoder_manager_section = sections.get("decoder_manager", {})
        observation_section = sections.get("observation", {})
        qpu = qpu_settings.QpuSettings.from_yaml(sections["qpu"])
        controller = controller_settings.ControllerSettings.from_yaml(
            sections["controller"], clocks
        )
        links = link_profiles.from_yaml(sections["links"], clocks, name)
        round_store = round_store_settings.RoundStoreSettings.from_yaml(
            sections["round_store"]
        )
        strong_round_store = round_store_settings.RoundStoreSettings.from_yaml(
            sections["strong_round_store"]
        )
        windows = window_settings.WindowSettings.from_yaml(sections["windows"])
        weak_decoder = _tier_settings(sections, "weak_decoder", clocks)
        strong_decoder = _tier_settings(sections, "strong_decoder", clocks)
        decoder_manager = decoder_settings.DecoderManagerSettings.from_yaml(
            decoder_manager_section, clocks
        )
        escalation = escalation_settings.EscalationSettings.from_yaml(
            escalation_section, base_directory
        )
        pauli_frame = pauli_frame_module.PauliFrameConfig.from_yaml(
            sections["pauli_frame"], clocks
        )
        workload = workload_settings.WorkloadSettings.from_yaml(
            sections["workload"]
        )
        observation = observe_settings.ObservationSettings.from_yaml(
            observation_section
        )
        return cls(
            clocks=clocks,
            qpu=qpu,
            controller=controller,
            idle_policy=idle_policy,
            links=links,
            round_store=round_store,
            strong_round_store=strong_round_store,
            windows=windows,
            weak_decoder=weak_decoder,
            strong_decoder=strong_decoder,
            decoder_manager=decoder_manager,
            escalation=escalation,
            pauli_frame=pauli_frame,
            workload=workload,
            observation=observation,
        )


def _tier_settings(
    sections: Mapping, tier: str, clocks: config.ClockSettings
) -> decoder_settings.DecoderSettings:
    """A tier's section, or no decoder when the yaml leaves it out."""
    if tier not in sections:
        return decoder_settings.DecoderSettings()
    return decoder_settings.DecoderSettings.from_yaml(
        sections[tier], clocks, tier
    )

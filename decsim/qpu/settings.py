"""The QPU's settings, and the magic state factory that feeds it.

The QPU has a cycle, a code card and a syndrome source; the factory
supplies its non-Clifford operations with states.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any, Optional

import decsim.config as config
import decsim.ports as ports
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.layouts as layouts
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.qpu.stim_device as stim_device
import decsim.qpu.streaming_stim_device as streaming_stim_device
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.tables as tables

# qpu.kind names one of these rows: the device that emits the readout.
SYNDROME_SOURCES = {
    "stim_device": stim_device.StimDevice,
    "timing_only": syndrome_devices.TimingOnlyDevice,
    "syndrome_bits": syndrome_devices.SyndromeBitDevice,
    "recorded_stim": stim_device.RecordedStimDevice,
    "streaming_stim": streaming_stim_device.StreamingStimDevice,
    "burst_stim": stim_device.BurstStimDevice,
}
# qpu.code_card names one of these rows: the code card the run prices.
# CUDA-Q QEC builds a code by name the same way
# (libs/qec/include/cudaq/qec/code.h:257, get_code(name, options)).
CODE_CARDS = {
    "rotated_surface": code_geometry.SurfaceCodeModel,
    "bivariate_bicycle": code_geometry.BivariateBicycleCodeModel,
}
# The key the magic_state_factory section reads for itself; any other key
# is the row's own (its Settings).
FACTORY_KEYS = ("kind",)
# The keys the qpu section reads for itself; any other key is the source
# row's or the code card row's own (its Settings, decsim/tables.py
# row_settings). The sweep sets the distance and the round period, so
# neither is a key here.
QPU_KEYS = ("kind", "code_card")
# magic_state_factory.kind names one of these rows: what supplies the T
# states an operation consumes.
MAGIC_STATE_FACTORIES = {
    "infinite": magic_state_factories.InfiniteFactory,
    "distillation": magic_state_factories.DistillationFactory,
    "multi_level": magic_state_factories.MultiLevelDistillationFactory,
}


@dataclasses.dataclass(frozen=True)
class QpuSettings:
    """The QPU: its round period, its code card and its syndrome source.

    Table rows for the source (SYNDROME_SOURCES, above): stim_device (Stim
    samples the operation's circuit), timing_only (payloads of the
    code's size with no values), syndrome_bits (seeded random bits
    shaped like the code's syndrome), recorded_stim (a released
    experiment's measurements replayed), streaming_stim (repeated Stim
    fragments executed as the controller requests rounds), burst_stim
    (stim_device sampling each shot with one error burst the decoders are
    not told of, its keys the row's own Settings). The round
    period is the device's physical cadence, a quantum-device number,
    not a classical clock's cycles: Google 921 ns (2207.06431) and
    1.1 us (2408.13687), Krinner 1.1 us (2112.03708), Yang 1.25 us
    (2605.04892). The code card is the row code_card names (CODE_CARDS,
    above: rotated_surface, the default, after Stim's generated
    surface_code:rotated_memory_z; bivariate_bicycle, Bravyi et al.
    2308.07915) at the sweep's distance, with the windows section's
    commit and buffer sizes; a Python-built code, layout, device or
    error-model provider is used as it is. The card provisions the links
    and sizes the circuit-less sources' rounds; a circuit source's
    payloads carry the circuit's own widths, so a card and a circuit at
    different distances run links provisioned for the wrong code.
    row_settings and code_card_settings are the source row's and the
    card row's own Settings, read from the section's keys outside
    QPU_KEYS, or None for a row that declares none.
    """

    kind: str = "timing_only"
    code_card: str = "rotated_surface"
    round_period_microseconds: float = 1.1
    distance: Optional[int] = None
    code: Optional[ports.CodeModel] = None
    layout: Optional[layouts.LayoutModel] = None
    device: Optional[ports.SyndromeSource] = None
    error_model_provider: Optional[Any] = None
    # the source row's own Settings record, opaque to the section
    row_settings: Optional[Any] = None
    # the code card row's own Settings record, opaque to the section
    code_card_settings: Optional[Any] = None

    def __post_init__(self) -> None:
        config.check_duration(
            "round_period_microseconds", self.round_period_microseconds
        )

    @classmethod
    def from_yaml(cls, section: Mapping) -> "QpuSettings":
        """The `qpu` section: the source kind; the sweep sets the rest.

        A distance or a period written here, which the sweep would
        silently replace, is refused with every other key the section
        and its row do not declare. The kind is looked up here, where the
        yaml enters, so a misspelt source is refused before `decsim show`
        prints it or a run folder exists.
        """
        kind = section.get("kind")
        source_row = tables.row(SYNDROME_SOURCES, "qpu.kind", kind)
        code_card = section.get("code_card", "rotated_surface")
        card_row = tables.row(CODE_CARDS, "qpu.code_card", code_card)
        source_keys = tables.row_keys(source_row)
        card_keys = tables.row_keys(card_row)
        keys_beside_the_source = QPU_KEYS + card_keys
        keys_beside_the_card = QPU_KEYS + source_keys
        row_settings = tables.row_settings(
            source_row, "qpu", section, keys_beside_the_source
        )
        code_card_settings = tables.row_settings(
            card_row, "qpu", section, keys_beside_the_card
        )
        return cls(
            kind=kind,
            code_card=code_card,
            row_settings=row_settings,
            code_card_settings=code_card_settings,
        )

    def build_code(
        self,
        *,
        commit_rounds_override: Optional[int],
        buffer_rounds_override: Optional[int],
    ) -> tuple:
        """The code card and its layout: the built ones, or the named card.

        At most one of distance, code and layout is given. The two
        overrides are the window sizes the yaml declares, which size the
        named card's commit and buffer regions; None leaves each at the
        card's own. A distance of None is the card's own default.
        """
        self._check_one_code_source()
        if self.layout is not None:
            code = _the_layouts_one_code(self.layout)
            return code, self.layout
        code = self.code
        if code is None:
            code = self._named_card(
                commit_rounds_override, buffer_rounds_override
            )
        return code, layouts.UniformLayout(code)

    def _named_card(
        self,
        commit_rounds_override: Optional[int],
        buffer_rounds_override: Optional[int],
    ) -> ports.CodeModel:
        """The code_card row, at the sweep's distance and the yaml's windows."""
        card_row = tables.row(CODE_CARDS, "qpu.code_card", self.code_card)
        arguments = {
            "commit_rounds_override": commit_rounds_override,
            "buffer_rounds_override": buffer_rounds_override,
        }
        if self.distance is not None:
            arguments["distance"] = self.distance
        if self.code_card_settings is not None:
            arguments["settings"] = self.code_card_settings
        return card_row(**arguments)

    def _check_one_code_source(self) -> None:
        """Refuse settings that name the code more than one way."""
        given = []
        if self.distance is not None:
            given.append("distance")
        if self.code is not None:
            given.append("code")
        if self.layout is not None:
            given.append("layout")
        if len(given) > 1:
            listed = ", ".join(given)
            raise ValueError(f"multiple code sources supplied: {listed}")


def _the_layouts_one_code(layout: layouts.LayoutModel):
    """The one code a layout declares; a layout of several is refused."""
    declared_codes = layout.codes()
    codes = list(declared_codes)
    if len(codes) != 1:
        raise ValueError(
            f"layout must declare exactly one code (got {len(codes)})"
        )
    return codes[0]


@dataclasses.dataclass(frozen=True)
class FactorySettings:
    """The magic state factory: where non-Clifford operations get states.

    Table rows (MAGIC_STATE_FACTORIES, above): infinite (a state is
    always in stock), distillation (Litinski's 15-to-1 stage,
    1905.06903),
    multi_level (Silva's chain of levels, 2411.04270). row_settings is
    the row's own Settings, read from the section's keys beside kind, or
    None for a row that declares none; it rides to the row inside the
    one FactoryCollaborators record, beside the engine and the round
    ticks the root supplies (magic_state_factories.py).
    """

    kind: str = "infinite"
    # the row's own Settings record, opaque to the section
    row_settings: Optional[Any] = None

    @classmethod
    def from_yaml(cls, section: Mapping) -> "FactorySettings":
        """The `magic_state_factory` section: a kind and the row's keys."""
        kind = section.get("kind", "infinite")
        row = tables.row(
            MAGIC_STATE_FACTORIES, "magic_state_factory.kind", kind
        )
        row_settings = tables.row_settings(
            row, "magic_state_factory", section, FACTORY_KEYS
        )
        return cls(kind=kind, row_settings=row_settings)

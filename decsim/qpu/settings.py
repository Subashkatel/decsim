"""The QPU's settings, and the magic state factory that feeds it.

The QPU has a cycle, a code card and a syndrome source; the factory
supplies its non-Clifford operations with states.
"""

import dataclasses
from collections.abc import Mapping
from typing import Optional, Union

import decsim.config as config
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.layouts as layouts
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.qpu.stim_device as stim_device
import decsim.qpu.streaming_stim_device as streaming_stim_device
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.tables as tables

# qpu.kind names one of these rows: the device that emits the readout,
# built from the row's Settings record.
SYNDROME_SOURCES = {
    "stim_device": stim_device.StimDevice,
    "timing_only": syndrome_devices.TimingOnlyDevice,
    "syndrome_bits": syndrome_devices.SyndromeBitDevice,
    "recorded_stim": stim_device.RecordedStimDevice,
    "streaming_stim": streaming_stim_device.StreamingStimDevice,
}
# qpu.code_card names one of these rows: the code card the run prices,
# built from the row's Settings record.
# CUDA-Q QEC builds a code by name the same way
# (libs/qec/include/cudaq/qec/code.h:257, get_code(name, options)).
CODE_CARDS = {
    "rotated_surface": code_geometry.SurfaceCodeModel,
    "bivariate_bicycle": code_geometry.BivariateBicycleCodeModel,
}
# The key the magic_state_factory section reads for itself; any other key
# is a field of the record its kind names.
FACTORY_KEYS = ("kind",)
# The keys the qpu section reads for itself; any other key is a field of
# the source row's or the code card row's Settings record.
QPU_KEYS = ("kind", "code_card", "distance", "round_period_microseconds")
# The keys above that are the section's fields, each left at the field's
# default when the yaml does not write it.
QPU_VALUE_KEYS = ("distance", "round_period_microseconds")
# magic_state_factory.kind names one of these records, each of which
# builds what supplies the T states an operation consumes: infinite (a
# state is always in stock), distillation (Litinski's 15-to-1 stage,
# 1905.06903), multi_level (Silva's chain of levels, 2411.04270).
MAGIC_STATE_FACTORIES = {
    "infinite": magic_state_factories.InfiniteFactory.Settings,
    "distillation": magic_state_factories.DistillationFactory.Settings,
    "multi_level": (
        magic_state_factories.MultiLevelDistillationFactory.Settings
    ),
}

# the settings record of whichever factory the run has
FactorySettings = Union[
    magic_state_factories.InfiniteFactory.Settings,
    magic_state_factories.DistillationFactory.Settings,
    magic_state_factories.MultiLevelDistillationFactory.Settings,
]
# the settings record of whichever syndrome source the run has
SourceSettings = Union[
    stim_device.StimDevice.Settings,
    syndrome_devices.TimingOnlyDevice.Settings,
    syndrome_devices.SyndromeBitDevice.Settings,
    stim_device.RecordedStimDevice.Settings,
    streaming_stim_device.StreamingStimDevice.Settings,
    stim_device.BurstStimDevice.Settings,
]
# the settings record of whichever code card the run prices
CodeCardSettings = Union[
    code_geometry.SurfaceCodeModel.Settings,
    code_geometry.BivariateBicycleCodeModel.Settings,
]


@dataclasses.dataclass(frozen=True)
class QpuSettings:
    """The QPU: its syndrome source, its code card and its round period.

    source is a source row's Settings record (SYNDROME_SOURCES, above),
    which builds the device over the run's card and the workload's
    circuits: stim_device (Stim samples the operation's circuit),
    timing_only (payloads of the code's size with no values),
    syndrome_bits (seeded random bits shaped like the code's syndrome),
    recorded_stim (a released experiment's measurements replayed),
    streaming_stim (repeated Stim fragments executed as the controller
    requests rounds), and BurstStimDevice's record (stim_device sampling
    each shot with one error burst the decoders are not told of).
    code_card is a card row's Settings record (CODE_CARDS, above:
    rotated_surface, the default, after Stim's generated
    surface_code:rotated_memory_z; bivariate_bicycle, Bravyi et al.
    2308.07915), which builds the card
    at distance, None being the card's own, with the windows record's
    commit and buffer sizes. layout builds which code every patch runs
    on, the uniform layout by default. error_model_provider is a circuit
    source's record (stim_device, streaming_stim and their kin, each its
    own window model source) whose models the decoders read in place of
    the source's own; None is the source's own. The round period is the
    device's physical cadence, a quantum-device number, not a classical
    clock's cycles: Google 921 ns (2207.06431) and 1.1 us (2408.13687),
    Krinner 1.1 us (2112.03708), Yang 1.25 us (2605.04892). The card
    provisions the links and sizes the circuit-less sources' rounds; a
    circuit source's payloads carry the circuit's own widths, so a card
    and a circuit at different distances run links provisioned for the
    wrong code.
    """

    source: SourceSettings = syndrome_devices.TimingOnlyDevice.Settings()
    code_card: CodeCardSettings = code_geometry.SurfaceCodeModel.Settings()
    round_period_microseconds: float = 1.1
    distance: Optional[int] = None
    layout: layouts.UniformLayout.Settings = layouts.UniformLayout.Settings()
    error_model_provider: Optional[SourceSettings] = None

    def __post_init__(self) -> None:
        config.check_duration(
            "round_period_microseconds", self.round_period_microseconds
        )

    @classmethod
    def from_yaml(cls, section: Mapping) -> "QpuSettings":
        """The `qpu` section: the source, the code card, distance and period.

        A key the section and its rows do not declare is refused. The
        rows are looked up here, where the yaml enters, so a misspelt
        source is refused before `decsim show` prints it or a run folder
        exists.
        """
        kind = section.get("kind")
        source_row = tables.row(SYNDROME_SOURCES, "qpu.kind", kind)
        card_name = section.get("code_card", "rotated_surface")
        card_row = tables.row(CODE_CARDS, "qpu.code_card", card_name)
        source_keys = tables.row_keys(source_row)
        card_keys = tables.row_keys(card_row)
        known_keys = QPU_KEYS + card_keys + source_keys
        tables.refuse_unknown_keys("qpu", section, known_keys)
        source = _row_record(source_row, section, source_keys)
        code_card = _row_record(card_row, section, card_keys)
        values = {}
        for key in QPU_VALUE_KEYS:
            if key in section:
                values[key] = section[key]
        return cls(source=source, code_card=code_card, **values)

    def build_code(
        self,
        *,
        commit_rounds_override: Optional[int],
        buffer_rounds_override: Optional[int],
    ) -> tuple:
        """The run's code and the layout over it.

        The two overrides are the window sizes the windows record
        declares, which size the card's commit and buffer regions; None
        leaves each at the card's own. The run's code is the one the
        layout declares, which the uniform layout takes from the card.
        """
        card = self.code_card.build(
            self.distance, commit_rounds_override, buffer_rounds_override
        )
        layout = self.layout.build(card)
        code = _the_layouts_one_code(layout)
        return code, layout


def factory_from_yaml(section: Mapping) -> FactorySettings:
    """The `magic_state_factory` section: its kind's record, from its keys."""
    kind = section.get("kind", "infinite")
    record = tables.row(MAGIC_STATE_FACTORIES, "magic_state_factory.kind", kind)
    own_section = tables.record_fields(
        record, "magic_state_factory", section, FACTORY_KEYS
    )
    return record.from_yaml(own_section)


def _row_record(row, section: Mapping, keys: tuple):
    """The row's Settings record, from the section's keys of its own.

    A section that writes none of them takes the record's defaults.
    """
    own_section = {}
    for key in keys:
        if key in section:
            own_section[key] = section[key]
    if not own_section:
        return row.Settings()
    return row.Settings.from_yaml(own_section)


def _the_layouts_one_code(layout: layouts.LayoutModel):
    """The one code a layout declares; a layout of several is refused."""
    declared_codes = layout.codes()
    codes = list(declared_codes)
    if len(codes) != 1:
        raise ValueError(
            f"layout must declare exactly one code (got {len(codes)})"
        )
    return codes[0]

"""The QPU's settings, and the magic state factory that feeds it."""

import dataclasses
from typing import Optional, Union

import decsim.config as config
import decsim.ports as ports
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.layouts as layouts
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.qpu.stim_device as stim_device
import decsim.qpu.streaming_stim_device as streaming_stim_device
import decsim.qpu.syndrome_devices as syndrome_devices

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
    """The QPU's settings.

    source is a source row's Settings record, which builds the device over
    the run's card and the workload's circuits. code_card is a card row's
    record, built at distance (None is the card's own) with the windows
    record's commit and buffer sizes. layout maps patches to codes, uniform
    by default. error_model_provider is a circuit source whose window
    models the decoders read in place of the source's own; None is the
    source's own.

    The round period is the device's physical cadence, not a classical
    clock's cycles: Google 921 ns (2207.06431) and 1.1 us (2408.13687),
    Krinner 1.1 us (2112.03708), Yang 1.25 us (2605.04892). The card
    provisions the links and sizes circuit-less rounds, while a circuit
    source's payloads carry the circuit's widths, so a card and a circuit at
    different distances run links provisioned for the wrong code.
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

    def build_code(
        self,
        *,
        commit_rounds_override: Optional[int],
        buffer_rounds_override: Optional[int],
    ) -> tuple:
        """The run's code and the layout over it.

        The overrides are the windows record's commit and buffer sizes; None
        leaves each at the card's own.
        """
        card = self.code_card.build(
            self.distance, commit_rounds_override, buffer_rounds_override
        )
        layout = self.layout.build(card)
        code = _the_layouts_one_code(layout)
        return code, layout


def _the_layouts_one_code(layout: ports.LayoutModel):
    """The one code a layout declares; a layout of several is refused."""
    declared_codes = layout.codes()
    codes = list(declared_codes)
    if len(codes) != 1:
        raise ValueError(
            f"layout must declare exactly one code (got {len(codes)})"
        )
    return codes[0]

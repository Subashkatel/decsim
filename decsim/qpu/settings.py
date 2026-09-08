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
import decsim.qpu.syndrome_devices as syndrome_devices

# qpu.kind names one of these rows: the device that emits the readout.
SYNDROME_SOURCES = {
    "stim_device": stim_device.StimDevice,
    "timing_only": syndrome_devices.TimingOnlyDevice,
    "syndrome_bits": syndrome_devices.SyndromeBitDevice,
    "recorded_stim": stim_device.RecordedStimDevice,
}
# qpu.magic_state_factory names one of these rows: what supplies the
# T states an operation consumes.
MAGIC_STATE_FACTORIES = {
    "infinite": magic_state_factories.InfiniteFactory,
    "distillation": magic_state_factories.DistillationFactory,
    "multi_level": magic_state_factories.MultiLevelDistillationFactory,
}


@dataclasses.dataclass(frozen=True)
class QpuSettings:
    """The QPU: its round period, its code card and its syndrome source.

    Table rows for the source (SYNDROME_SOURCES, above): stim_device (Stim
    samples the operation's circuit), timing_only (bitless payloads,
    timing alone), syndrome_bits (seeded random bits shaped like the
    code's syndrome), recorded_stim (a released experiment's
    measurements replayed). The round period is the device's physical
    cadence, a quantum-device number, not a classical clock's cycles:
    Google 921 ns (2207.06431) and 1.1 us (2408.13687), Krinner 1.1 us
    (2112.03708), Yang 1.25 us (2605.04892). The code card is a rotated
    surface code of the distance, with the windows section's commit and
    buffer sizes; a Python-built code, layout, device or error-model
    provider is used as it is. The arguments are the source row's
    keyword arguments (a seed, a recorded shot's measurements).
    """

    kind: str = "timing_only"
    round_period_microseconds: float = 1.1
    distance: Optional[int] = None
    code: Optional[code_geometry.CodeModel] = None
    layout: Optional[layouts.LayoutModel] = None
    device: Optional[ports.SyndromeSource] = None
    error_model_provider: Optional[Any] = None
    arguments: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        config.check_duration(
            "round_period_microseconds", self.round_period_microseconds
        )

    @classmethod
    def from_yaml(cls, section: Mapping) -> "QpuSettings":
        """The `qpu` section: the source kind; the sweep sets the rest."""
        return cls(kind=section["kind"])

    def build_code(
        self,
        *,
        commit_rounds_override: Optional[int],
        buffer_rounds_override: Optional[int],
    ) -> tuple:
        """The code card and its layout: the built ones, or one surface code.

        At most one of distance, code and layout is given. The two
        overrides are the window sizes the yaml declares, which size the
        default surface code's commit and buffer regions; None leaves
        each at the code distance.
        """
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
        if self.layout is not None:
            declared_codes = self.layout.codes()
            codes = list(declared_codes)
            if len(codes) != 1:
                raise ValueError(
                    f"layout must declare exactly one code (got {len(codes)})"
                )
            return codes[0], self.layout
        code = self.code
        if code is None:
            distance = self.distance
            if distance is None:
                distance = 3
            code = code_geometry.SurfaceCodeModel(
                distance=distance,
                commit_rounds_override=commit_rounds_override,
                buffer_rounds_override=buffer_rounds_override,
            )
        return code, layouts.UniformLayout(code)


@dataclasses.dataclass(frozen=True)
class FactorySettings:
    """The magic state factory: where non-Clifford operations get states.

    Table rows (MAGIC_STATE_FACTORIES, above): infinite (a state is
    always in stock), distillation (Litinski's 15-to-1 stage,
    1905.06903),
    multi_level (Silva's chain of levels, 2411.04270). The arguments are
    the row's keyword arguments, everything but the engine, the decode
    service and the round ticks the root supplies
    (magic_state_factories.py names them). No yaml key today.
    """

    kind: str = "infinite"
    arguments: Mapping[str, Any] = dataclasses.field(default_factory=dict)

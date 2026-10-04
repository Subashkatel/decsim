"""The qpu part: the device on its round clock, its readouts, the factory.

The device reads one round out per cycle of a clock whose period the
run plan sized, from the syndrome source the plan built for the code;
the magic state factory beside it supplies the non-Clifford operations.
"""

import dataclasses

import decsim.build.plan as plan_build
import decsim.config as config
import decsim.engine as engine_module
import decsim.ports as ports
import decsim.qpu.cycle_clock as cycle_clock
import decsim.qpu.settings as qpu_settings


@dataclasses.dataclass(frozen=True)
class Qpu:
    """The quantum side of the machine, as the controller meets it."""

    # the source the plan built for its code card, whichever row it is
    syndrome_source: ports.SyndromeSource
    device: cycle_clock.QPUDevice
    factory: ports.MagicStateFactory

    @classmethod
    def build(
        cls,
        factory_settings: qpu_settings.FactorySettings,
        engine: engine_module.Engine,
        plan: plan_build.Plan,
    ) -> "Qpu":
        """The device on the plan's round clock, and the factory."""
        round_clock = config.Clock(plan.round_ticks)
        device = cycle_clock.QPUDevice(engine, round_clock, plan.code)
        device.syndrome_source = plan.device
        factory = factory_settings.build(engine, plan.round_ticks)
        return cls(
            syndrome_source=plan.device,
            device=device,
            factory=factory,
        )

    def connect(
        self,
        readout_receiver: ports.ReadoutReceiver,
        runtime: ports.OperationRuntime,
        idle_rounds: ports.IdleRoundReceiver,
        decode_queue: ports.DecodeQueue,
    ) -> None:
        """Send the readouts to the controller, and a card's decodes on."""
        self.device.readout_receiver = readout_receiver
        self.device.runtime = runtime
        self.device.idle_rounds = idle_rounds
        self.factory.decode_queue = decode_queue

    def seed_roots(self) -> tuple:
        """This part's stochastic owners, each by the name its seed hashes."""
        return (("factory", self.factory),)

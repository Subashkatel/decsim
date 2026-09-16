"""What a seat's builder reads: the run's fixtures, and the seats so far.

A seat builder takes one Parts record and returns its seat. The record
sits here, below the builders in the package order, so a builder can
name the type it reads (STYLE.md rule 10).
"""

import dataclasses
from typing import Any

import decsim.build.decoders as decoder_build
import decsim.build.plan as plan_build
import decsim.engine as engine_module
import decsim.ports as ports
import decsim.settings as machine_settings


@dataclasses.dataclass(frozen=True)
class Parts:
    """What a seat's builder reads: the run's fixtures, and the seats so far.

    The settings and the engine are every seat's input; the plan, the
    decoder pool and the escalation policy are the records the root
    compiles from them; the detection event placement is the one seat the
    pool is compiled from, so the root builds it with them. seats holds
    the rows built so far, and only the magic state factory's row and
    the primary output's row read it.
    """

    settings: machine_settings.MachineSettings
    engine: engine_module.Engine
    plan: plan_build.Plan
    escalation_policy: Any
    pool: decoder_build.DecoderPool
    detection_events: ports.DetectionEventPlacement
    seats: dict = dataclasses.field(default_factory=dict)

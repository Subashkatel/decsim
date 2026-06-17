# Public API for the decsim package. Re-exports the names examples and external
# callers use, so they can do `from decsim import build_and_run, ...` instead of
# reaching into individual submodules.

from .config import SimConfig, us, fmt
from .wiring import build_and_run
from .frontends.circuit import (CircuitFrontend, SurgeryIRFrontend,
                                three_cnot_circuit, cnot_plus_two_t_circuit,
                                independent_t_circuit,
                                three_cnot_six_qubits_circuit)
from .decoders import (PresetLatencyDecoder, PerRoundDecoder, RelayBPDecoder,
                      SwitchingDecoder, SwitchingRouter, SampledSoftOutputDecoder,
                      switch_probability_per_round)
from .factories import InfiniteFactory, DistillationFactory
from .schemes import ParallelWindowScheme
from .switching import Switching
from .schedulers import EarliestDeadlineScheduler, ReactionPathDeadline
from .metrics import (DecoderUtilization, ReadyQueueStats,
                      WindowLatencyBreakdown, MagicStateLatency,
                      StrongDecoderBacklog)

__all__ = [
    "SimConfig", "us", "fmt",
    "build_and_run",
    "CircuitFrontend", "SurgeryIRFrontend",
    "three_cnot_circuit", "cnot_plus_two_t_circuit", "independent_t_circuit",
    "three_cnot_six_qubits_circuit",
    "PresetLatencyDecoder", "PerRoundDecoder", "RelayBPDecoder", "SwitchingDecoder",
    "SwitchingRouter", "SampledSoftOutputDecoder", "switch_probability_per_round",
    "InfiniteFactory", "DistillationFactory",
    "ParallelWindowScheme",
    "Switching",
    "EarliestDeadlineScheduler", "ReactionPathDeadline",
    "DecoderUtilization", "ReadyQueueStats",
    "WindowLatencyBreakdown", "MagicStateLatency", "StrongDecoderBacklog",
]

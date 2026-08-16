"""Public import surface for the decsim package.

Compose a RunSpec (decsim.run_spec) and drive it with simulate().
"""

from .decoders import (PresetLatencyDecoder, PerRoundDecoder,
                       FunctionLatencyDecoder,
                       SwitchingDecoder, SwitchingRouter, SampledConfidenceDecoder,
                       switch_probability_per_round)
from .factories import (InfiniteFactory, DistillationFactory,
                        MultiLevelDistillationFactory)
from .frontends.circuit import (CircuitFrontend, SurgeryIRFrontend,
                                three_cnot_circuit, cnot_plus_two_t_circuit,
                                independent_t_circuit,
                                three_cnot_six_qubits_circuit)
from .metrics import (DecoderInputStoreOccupancy, DecoderUtilization,
                      ReadyQueueStats,
                      WindowLatencyBreakdown, MagicStateLatency,
                      StrongDecoderBacklog, ConditionalReactionTime)
from .schemes import ParallelWindowScheme
from .run_spec import RunSpec, simulate
from .switching import Switching
from .config import TimingConfig, fmt, us

__all__ = [
    "RunSpec", "TimingConfig", "simulate",
    "us", "fmt",
    "CircuitFrontend", "SurgeryIRFrontend",
    "three_cnot_circuit", "cnot_plus_two_t_circuit", "independent_t_circuit",
    "three_cnot_six_qubits_circuit",
    "PresetLatencyDecoder", "PerRoundDecoder", "FunctionLatencyDecoder",
    "SwitchingDecoder",
    "SwitchingRouter", "SampledConfidenceDecoder", "switch_probability_per_round",
    "InfiniteFactory", "DistillationFactory", "MultiLevelDistillationFactory",
    "ParallelWindowScheme",
    "Switching",
    "DecoderInputStoreOccupancy", "DecoderUtilization", "ReadyQueueStats",
    "WindowLatencyBreakdown", "MagicStateLatency", "StrongDecoderBacklog",
    "ConditionalReactionTime",
]

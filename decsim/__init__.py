"""Public import surface for the decsim package.

Compose a RunSpec (decsim.run_spec) and drive it with simulate().
"""

from .decoders.decoders import (PresetLatencyDecoder, PerRoundDecoder,
                       FunctionLatencyDecoder,
                       SwitchingRouter, SampledConfidenceDecoder,
                       switch_probability_per_round)
from .qpu.magic_state_factories import (InfiniteFactory, DistillationFactory,
                        MultiLevelDistillationFactory)
from .frontends.circuit_frontend import (CircuitFrontend, SurgeryIRFrontend,
                                three_cnot_circuit, cnot_plus_two_t_circuit,
                                independent_t_circuit,
                                three_cnot_six_qubits_circuit)
from .observe.metrics import (DecoderMemoryOccupancy, DecoderUtilization,
                      ReadyQueueStats,
                      WindowLatencyBreakdown, MagicStateLatency,
                      StrongDecoderBacklog, ConditionalReactionTime)
from .windows.windowing_schemes import ParallelWindowScheme
from .run_spec import RunSpec, simulate
from .decoders.weak_strong_switching import Switching
from .config import TimingConfig, fmt, us

__all__ = [
    "RunSpec", "TimingConfig", "simulate",
    "us", "fmt",
    "CircuitFrontend", "SurgeryIRFrontend",
    "three_cnot_circuit", "cnot_plus_two_t_circuit", "independent_t_circuit",
    "three_cnot_six_qubits_circuit",
    "PresetLatencyDecoder", "PerRoundDecoder", "FunctionLatencyDecoder",
    "SwitchingRouter", "SampledConfidenceDecoder", "switch_probability_per_round",
    "InfiniteFactory", "DistillationFactory", "MultiLevelDistillationFactory",
    "ParallelWindowScheme",
    "Switching",
    "DecoderMemoryOccupancy", "DecoderUtilization", "ReadyQueueStats",
    "WindowLatencyBreakdown", "MagicStateLatency", "StrongDecoderBacklog",
    "ConditionalReactionTime",
]

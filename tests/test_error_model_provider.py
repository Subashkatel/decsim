"""Replacement tests for the decoder-facing error-model seam."""
from decsim import RunSpec
from decsim.decoders import PresetLatencyDecoder
from decsim.devices import TimingOnlyDevice
from decsim.message import Operation
from decsim.planner import FixedRounds
from decsim.protocols import ErrorModelProvider, SyndromeDevice


class MinimalErrorModelProvider:
    """From-scratch provider with no QPU sampling or cadence methods."""
    def __init__(self): self.window_calls = 0
    def register_dynamic_stream(self, stream_op, round_count, *,
                                fault_model_requirement): return None
    def validate_stream_length(self, stream_op, stream_round_count): return None
    def window_models_for_operation(self, op, windows, round_count, *,
                                    fault_model_requirement,
                                    fault_exclusion_ranges, window_protocol):
        self.window_calls += 1
        return [None for _ in windows]
    def window_model_for_stream(self, stream_id, window, *, is_last): return None
    def strong_window_model_for_operation(self, op, window, round_count, *,
                                          fault_model_requirement,
                                          exclude_faults_touching=None): return None


def test_from_scratch_error_provider_is_independent_of_qpu_device():
    provider = MinimalErrorModelProvider()
    device = TimingOnlyDevice()
    assert isinstance(provider, ErrorModelProvider)
    assert not isinstance(provider, SyndromeDevice)
    completed = RunSpec(
        ops=[Operation(0, "memory", (0,))], d=3,
        rounds_policy=FixedRounds(3), decoder=PresetLatencyDecoder(0),
        device=device, error_model_provider=provider,
    ).build(verbose=False)
    assert completed.qpu.model is device
    assert completed.window_manager.error_model_provider is provider
    assert provider.window_calls == 1


def test_error_model_and_qpu_ownership_do_not_cross():
    from pathlib import Path
    root = Path(__file__).parents[1] / "decsim"
    qpu_text = (root / "qpu.py").read_text()
    manager_text = (root / "window_manager.py").read_text()
    assert "error_model_provider" not in qpu_text
    assert "syndrome_source" not in manager_text

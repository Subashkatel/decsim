"""Relay-BP profile validation: the first leg must run at least once."""

import pytest

from decsim.decoders.relay_bp.window_decoder import RelayBpWindowDecoder


def test_zero_pre_iterations_is_rejected():
    """relay-bp's decode_inner loops pre_iter times and otherwise leaves the
    previous call's decoding in place (relay.rs), so a zero first leg would
    hand back stale state; the profile refuses it up front."""
    with pytest.raises(ValueError, match="pre_iterations must be positive"):
        RelayBpWindowDecoder(pre_iterations=0)


def test_single_pre_iteration_is_accepted():
    RelayBpWindowDecoder(pre_iterations=1)

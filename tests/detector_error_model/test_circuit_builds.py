"""Builds kept by what they are a function of, the oldest dropped first.

Source: the Relay-BP row keeps its compiled models first in, first out
(decoders/relay_belief_propagation/window_decoder.py).
"""

import decsim.detector_error_model.circuit_builds as circuit_builds


def test_the_oldest_build_is_dropped_once_the_count_is_held():
    kept = circuit_builds.KeptBuilds(kept_count=2)
    kept.keep(("first",), 1)
    kept.keep(("second",), 2)
    kept.keep(("third",), 3)
    assert kept.held(("first",)) is None
    assert kept.held(("third",)) == 3


def test_a_mapping_is_one_key_whatever_its_order():
    first = circuit_builds.frozen({1: 2, 3: 4})
    second = circuit_builds.frozen({3: 4, 1: 2})
    assert first == second

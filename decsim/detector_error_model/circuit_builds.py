"""What a task builds once from its circuit, kept for its later shots.

Each shot installs its own copy of the workload's circuit
(build/plan.py _installed_circuits), so a build kept by the circuit
object is found by no later shot; kept by the circuit's text, which is
what every build here is a function of, it is found by every shot of
the task. sinter likewise compiles once per task
(sinter/_decoding/_decoding_decoder_class.py, compile_decoder_for_dem).
A kept build is only read: a formation table is frozen, a strong
window is sliced on its own (window_slicer.slice_window_alone), and a
sampler draws from the seed it is handed. The oldest build goes first
once the count is held, as the Relay-BP row keeps its compiled models
(decoders/relay_belief_propagation/window_decoder.py).
"""

from typing import Any, Optional


class KeptBuilds:
    """Builds by what they are a function of, the oldest dropped first."""

    def __init__(self, kept_count: int) -> None:
        self.kept_count = kept_count
        self.builds_by_key: dict = {}

    def held(self, key: tuple) -> Optional[Any]:
        """The build kept under this key, or None."""
        return self.builds_by_key.get(key)

    def keep(self, key: tuple, build: Any) -> None:
        """Keep one build, dropping the oldest when the count is held."""
        if len(self.builds_by_key) == self.kept_count:
            keys = iter(self.builds_by_key)
            oldest = next(keys)
            del self.builds_by_key[oldest]
        self.builds_by_key[key] = build


def frozen(mapping: Optional[dict]) -> Optional[tuple]:
    """A mapping as a key: its items in order, or None for none."""
    if mapping is None:
        return None
    items = mapping.items()
    return tuple(sorted(items))

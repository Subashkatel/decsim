"""The timing-only decoder.

A row on the Decoder port (ports.py, the defaults in decoder.py):
latency(job) prices one window job's compute as a service time in
ticks, and decode(job) produces an empty DecodeResult.
"""

import dataclasses
from typing import Optional

import decsim.config as config
import decsim.decoders.decoder as decoder_module
import decsim.records.decoding as decoding_records


class PresetLatencyDecoder(decoder_module.DecoderBase):
    """Timing-only decoder with one fixed latency for every job.

    A change of conditions mid-stream (slice 06: strong service slowing)
    gives later_latency_microseconds from window from_window on, the
    window index within its operation.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The latency every job takes, and the one from a window on."""

        latency_microseconds: float = 1.0
        later_latency_microseconds: Optional[float] = None
        from_window: Optional[int] = None
        # the word the reports name this row by
        name = "preset_latency"

        def __post_init__(self) -> None:
            _check_a_change(self.later_latency_microseconds, self.from_window)

        def build(self) -> "PresetLatencyDecoder":
            """A fresh decoder of this latency."""
            return PresetLatencyDecoder(
                self.latency_microseconds,
                self.later_latency_microseconds,
                self.from_window,
            )

    def __init__(
        self,
        latency_us: float = 1.0,
        later_latency_us: Optional[float] = None,
        from_window: Optional[int] = None,
    ):
        self.latency_us = latency_us
        self.later_latency_us = later_latency_us
        self.from_window = from_window

    def latency(self, job: decoding_records.DecodeJob) -> int:
        """The preset latency in ticks, the later one from its window on."""
        latency_us = self.latency_us
        changed = self.from_window is not None
        if changed and job.window_id >= self.from_window:
            latency_us = self.later_latency_us
        return config.microseconds_to_ticks(latency_us)

    def decode(
        self, job: decoding_records.DecodeJob
    ) -> decoding_records.DecodeResult:
        """An empty timing-only result."""
        return decoding_records.DecodeResult(job.stream_id, job.window_id)


def _check_a_change(later_latency_microseconds, from_window) -> None:
    """A later latency and its window come together, the window whole."""
    if (later_latency_microseconds is None) != (from_window is None):
        raise ValueError(
            "preset_latency.later_latency_microseconds and from_window go "
            f"together, got {later_latency_microseconds!r} and "
            f"{from_window!r}"
        )
    if from_window is not None:
        config.check_whole_count(
            "preset_latency.from_window", from_window, "windows", 0
        )

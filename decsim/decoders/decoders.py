"""The timing-only decoder.

A row on the Decoder port (ports.py, the defaults in decoder.py):
latency(job) prices one window job's compute as a service time in
ticks, and decode(job) produces an empty DecodeResult.
"""

import decsim.config as config
import decsim.decoders.decoder as decoder_module
import decsim.records.decoding as decoding_records


class PresetLatencyDecoder(decoder_module.DecoderBase):
    """Timing-only decoder with one fixed latency for every job."""

    def __init__(self, latency_us: float = 1.0):
        self.latency_us = latency_us

    def latency(self, job: decoding_records.DecodeJob) -> int:
        """The preset latency in ticks, whatever the job."""
        del job
        return config.microseconds_to_ticks(self.latency_us)

    def decode(
        self, job: decoding_records.DecodeJob
    ) -> decoding_records.DecodeResult:
        """An empty timing-only result."""
        return decoding_records.DecodeResult(job.operation_id, job.window_id)

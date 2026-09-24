"""Relay-BP on a measured GPU: decsim's own answer, the device's time.

MeasuredTable is the first row of the StrongBackend port
(decsim/ports.py). It answers with decsim's own Relay-BP decode (the
relay_bp row) and prices it from a line measured on the device
(measurements.py): intercept plus slope times the iterations decsim's
decode ran, and never less than the fastest decode the cell measured.
The measured time is one decode call with the syndrome's copies to and
from the device and the launch inside it, and no link, so it is the
time beyond the echo the port asks for with the launch folded in.

The time follows decsim's own iteration count rather than a draw from
the measured samples, so a hard region is slow on the device exactly
when it is hard for decsim's decode. The two implementations agree on a
region's iteration count in distribution: on the measured regions, the
same count for 98 percent of them at d = 5 and 79 percent at d = 13, and
for every region that needed 20 iterations or fewer. On a long decode
the two differ as much as two GPU runs of the same region do, so the
law matches the distribution of times, not each long decode.

The row is "bound": one chip's regions reach one dispatcher, which runs
one decode at a time (a CUDA-Q dispatcher's decode holds it, cudaqx
docs/sphinx/examples_rst/qec/realtime_relay_bp.rst:49-50), so its
capacity is the most decodes a measured cell ran at once, which is one.
"""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import decsim.config as config
import decsim.decoders.measured_table.measurements as measurements
import decsim.decoders.strong_backend as strong_backend
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoding as decoding_records
from decsim.decoders.relay_belief_propagation import (
    decoder as relay_belief_propagation,
)

PHYSICAL = fault_models.FaultRepresentation.PHYSICAL


@dataclasses.dataclass(frozen=True)
class MeasuredTableSettings:
    """The measured_table row's keys in its tier section.

    device names the GPU measured and partition how it was shared; the
    pair must be a measured one (measurements.py).
    """

    device: str = "a100"
    partition: str = "whole"

    @classmethod
    def from_yaml(
        cls, section: Mapping, clocks: config.ClockSettings
    ) -> "MeasuredTableSettings":
        """Both keys, checked where they enter against the measured pairs."""
        del clocks
        device = section.get("device", "a100")
        partition = section.get("partition", "whole")
        rows = _measured_rows(device, partition)
        if rows:
            return cls(device=device, partition=partition)
        pairs = _measured_pairs()
        raise ValueError(
            f"measured_table has no measurement of device {device!r} with "
            f"partition {partition!r}; the measured pairs are {pairs}"
        )


class MeasuredTable:
    """A GPU's Relay-BP time from its measured line; decsim's answer.

    A region is priced by the measured region nearest to it in detector
    count, the smaller on a tie. decsim's regions are not exactly the
    measured ones (a strong window with context holds a few rounds more
    or less than 3d), and the nearest measured region is the simplest
    rule the table can state; a region far from every measured size is
    priced by the nearest all the same.
    """

    def __init__(self, settings: MeasuredTableSettings) -> None:
        self.cells = _measured_rows(settings.device, settings.partition)
        self.decoder = relay_belief_propagation.RelayBeliefPropagationDecoder()

    def run_seed_children(self) -> tuple:
        """The Relay-BP decoder's own children, at the relay_bp row's paths."""
        return self.decoder.run_seed_children()

    def capacity(self) -> int:
        """The most decodes a measured cell of this device ran at once."""
        counts = []
        for cell in self.cells:
            counts.append(cell.decodes_running)
        return max(counts)

    def submit(
        self, request: decoding_records.DecodeJob, running: int
    ) -> "_Ticket":
        """Decode with decsim's Relay-BP now; price it by the device's line."""
        result = self.decoder.decode(request)
        detectors = _region_detectors(request)
        decodes_running = running + 1
        cell = _nearest_cell(self.cells, decodes_running, detectors)
        iterations = _iterations_of(result)
        microseconds = cell.decode_microseconds(iterations)
        service_ticks = config.microseconds_to_ticks(microseconds)
        return _Ticket(result, service_ticks)

    def service_ticks(self, ticket: "_Ticket") -> int:
        """The priced time, known at submit."""
        return ticket.service_ticks

    def result(self, ticket: "_Ticket") -> decoding_records.DecodeResult:
        """The region's answer from decsim's own Relay-BP decode."""
        return ticket.result


class MeasuredTableDecoder(strong_backend.StrongBackendDecoder):
    """The measured_table row: the strong decoder over a MeasuredTable."""

    Settings = MeasuredTableSettings
    fault_model_requirement = fault_models.PHYSICAL_FAULT_MODEL_REQUIRED

    def __init__(self, settings: Optional[MeasuredTableSettings] = None):
        if settings is None:
            settings = MeasuredTableSettings()
        backend = MeasuredTable(settings)
        strong_backend.StrongBackendDecoder.__init__(self, backend)


@dataclasses.dataclass(frozen=True)
class _Ticket:
    result: decoding_records.DecodeResult
    service_ticks: int


def _measured_rows(device: str, partition: str) -> tuple:
    """The measured cells of one device and partition, in region order."""
    rows = []
    for cell in measurements.RELAY_BP_TIMES:
        if cell.device == device and cell.partition == partition:
            rows.append(cell)
    return tuple(rows)


def _measured_pairs() -> list:
    pairs = []
    for cell in measurements.RELAY_BP_TIMES:
        pair = (cell.device, cell.partition)
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _region_detectors(request: decoding_records.DecodeJob) -> int:
    model = request.detector_error_model
    assert model is not None, (
        f"{request.label}: a Relay-BP decode needs its region's model"
    )
    faults = model.require_faults(PHYSICAL)
    return faults.check.shape[0]


def _iterations_of(result: decoding_records.DecodeResult) -> int:
    """The iterations the decode ran; a model with no faults runs none."""
    if result.iterations is None:
        return 0
    return result.iterations


def _nearest_cell(
    cells: tuple, decodes_running: int, detectors: int
) -> measurements.MeasuredTime:
    """The cell with this many decodes whose region is nearest in size."""
    nearest = None
    nearest_gap = None
    for cell in cells:
        if cell.decodes_running != decodes_running:
            continue
        size_difference = cell.detectors - detectors
        size_gap = abs(size_difference)
        if nearest is None or size_gap < nearest_gap:
            nearest = cell
            nearest_gap = size_gap
    assert nearest is not None, (
        f"no measured cell runs {decodes_running} decodes at once"
    )
    return nearest

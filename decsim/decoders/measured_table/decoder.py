"""Relay-BP on a measured GPU: decsim's own answer, the device's time.

MeasuredTable is the first row of the StrongBackend port
(decsim/ports.py). It answers with decsim's own Relay-BP decode (the
relay_bp row, at the relay_bp keys its tier section sets) and prices it
from a line measured on the device at those keys (measurements.py):
intercept plus slope times the iterations decsim's decode ran, and
never less than the fastest decode the cell measured.
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
docs/sphinx/examples_rst/qec/realtime_relay_bp.rst:49-50), so the
decode is one step on the dispatcher, and the dispatcher's count is the
most decodes a measured cell ran at once, which is one.
"""

import dataclasses
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
class MeasuredTableSettings(measurements.RelaySettings):
    """The measured_table row's settings.

    The relay_bp row's nine fields set the row's own Relay-BP decode, so
    the answer, its iterations and the line that prices them come from
    one setting; bases among them says whether a region is decoded whole
    or as its X and Z parts. device names the GPU measured and partition
    how it was shared. Together they must name measured cells
    (measurements.py): a decode is priced only by a line measured at its
    own keys, and a part only by parts measured on the device.
    """

    device: str = "a100"
    partition: str = "whole"
    # the word the reports name this row by
    name = "measured_table"

    def __post_init__(self) -> None:
        measurements.RelaySettings.__post_init__(self)
        _check_measured(self)

    def build(self) -> "MeasuredTableDecoder":
        """A fresh decoder of these settings."""
        return MeasuredTableDecoder(settings=self)


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
        self.cells = _measured_rows(settings)
        decode_settings = _decode_settings(settings)
        self.decoder = relay_belief_propagation.RelayBeliefPropagationDecoder(
            settings=decode_settings
        )

    def run_seed_children(self) -> tuple:
        """The Relay-BP decoder's own children, at the relay_bp row's paths."""
        return self.decoder.run_seed_children()

    def capacities(self) -> dict:
        """One dispatcher, running the most decodes a measured cell ran."""
        counts = []
        for cell in self.cells:
            counts.append(cell.decodes_running)
        most = max(counts)
        return {strong_backend.DISPATCHER: most}

    def submit(
        self, request: decoding_records.DecodeJob, running: int
    ) -> decoding_records.Ticket:
        """Decode with decsim's Relay-BP now; price it by the device's line."""
        result = self.decoder.decode(request)
        detectors = _region_detectors(request)
        decodes_running = running + 1
        cells = _cells_running(self.cells, decodes_running)
        cell = nearest_in_size(cells, detectors)
        iterations = _iterations_of(result)
        microseconds = cell.decode_microseconds(iterations)
        decode_ticks = config.microseconds_to_ticks(microseconds)
        return decoding_records.Ticket(result, decode_ticks)

    def steps(self, ticket: decoding_records.Ticket) -> tuple:
        """One step, the measured decode, holding the dispatcher throughout.

        The measured line is one decode() call with its copies and its
        launch inside it, so it is not split here.
        """
        decode = decoding_records.Step(
            "decode", ticket.decode_ticks, strong_backend.DISPATCHER
        )
        return (decode,)

    def result(
        self, ticket: decoding_records.Ticket
    ) -> decoding_records.DecodeResult:
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
        strong_backend.StrongBackendDecoder.__init__(
            self, backend, settings.bases
        )


def nearest_in_size(rows: tuple, detectors: int):
    """The row whose region is nearest in detectors; the first on a tie.

    Rows are in region order, so the first is the smaller region.
    """
    nearest = rows[0]
    for row in rows:
        size_difference = row.detectors - detectors
        nearest_difference = nearest.detectors - detectors
        if abs(size_difference) < abs(nearest_difference):
            nearest = row
    return nearest


def _check_measured(settings: MeasuredTableSettings) -> None:
    """The settings name measured cells, or no line prices a decode."""
    rows = _measured_rows(settings)
    if rows:
        return
    decode_settings = _decode_settings(settings)
    keys = _keys_off_default(decode_settings)
    measured = _measured_cells()
    raise ValueError(
        f"device {settings.device!r} with partition {settings.partition!r}, "
        f"bases {settings.bases!r} and the Relay-BP keys {keys} has no "
        f"measurement in measured_table; the measured ones are {measured}, "
        "each with the keys it sets off the relay_bp row's defaults"
    )


def _measured_rows(settings: MeasuredTableSettings) -> tuple:
    """The cells measured at the settings, in region order."""
    decode_settings = _decode_settings(settings)
    wanted = (
        settings.device,
        settings.partition,
        settings.bases,
        decode_settings,
    )
    rows = []
    for cell in measurements.RELAY_BP_TIMES:
        measured = (
            cell.device,
            cell.partition,
            cell.bases,
            cell.relay_settings,
        )
        if measured == wanted:
            rows.append(cell)
    return tuple(rows)


def _decode_settings(
    settings: MeasuredTableSettings,
) -> measurements.RelaySettings:
    """The relay_bp row's keys each decode runs at, X and Z together.

    The strong backend cuts a region into its parts itself
    (strong_backend.part_jobs), so the decode takes each request whole.
    """
    relay_keys = {}
    for field in dataclasses.fields(measurements.RelaySettings):
        relay_keys[field.name] = getattr(settings, field.name)
    relay_keys["bases"] = "together"
    return measurements.RelaySettings(**relay_keys)


def _keys_off_default(relay_settings: measurements.RelaySettings) -> dict:
    """Only what a tier changed, so a refusal names the keys to look at."""
    defaults = measurements.RelaySettings()
    keys = {}
    for field in dataclasses.fields(relay_settings):
        value = getattr(relay_settings, field.name)
        if value != getattr(defaults, field.name):
            keys[field.name] = value
    return keys


def _measured_cells() -> list:
    """Every measured device, partition, bases and keys, once each."""
    measured = []
    for cell in measurements.RELAY_BP_TIMES:
        keys = _keys_off_default(cell.relay_settings)
        cell_settings = (cell.device, cell.partition, cell.bases, keys)
        if cell_settings not in measured:
            measured.append(cell_settings)
    return measured


def _region_detectors(request: decoding_records.DecodeJob) -> int:
    model = request.detector_error_model
    faults = model.require_faults(PHYSICAL)
    return faults.check.shape[0]


def _iterations_of(result: decoding_records.DecodeResult) -> int:
    """The iterations the decode ran; a model with no faults runs none."""
    if result.iterations is None:
        return 0
    return result.iterations


def _cells_running(cells: tuple, decodes_running: int) -> tuple:
    """The cells measured with this many decodes at once."""
    running = []
    for cell in cells:
        if cell.decodes_running == decodes_running:
            running.append(cell)
    return tuple(running)

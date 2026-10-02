"""Relay-BP on a GPU behind a CUDA-Q dispatcher, step by step.

DispatchSteps is the second row of the StrongBackend port
(decsim/ports.py). It answers with decsim's own Relay-BP decode, as
measured_table does, and states the decode as the steps a CUDA-Q
dispatcher takes, each timed on its own (measurements.py).

On the device path (cuda-quantum releases/v0.15.2
realtime/lib/daemon/dispatcher/dispatch_kernel.cu) one persistent
kernel notices a filled ring slot, checks its header, runs the handler,
fires the decode graph and holds until the fired decode ends, so every
step holds the dispatcher. The device fire is compiled only for compute
capability 9.0 and up (`#if __CUDA_ARCH__ >= 900`, line 491), so the
A100 has no device path. On the host path (host_api.md lines
1065-1113) a CPU monitor notices the slot, "acquires an idle worker"
(line 1090), launches that worker's graph on its stream and moves on;
the worker copies the syndrome in, decodes and copies the answer out,
and is idle again when its stream is done (line 1112).

Notice, check, handle and respond are inside the echo round trip the
link card prices (the StrongBackend rule), so they are zero-tick steps
naming it.

The row decodes a region's X and Z detectors together only: its kernel
lines were traced on whole regions, and a part is priced on a line
measured on parts (measured_table's bases rows) or not at all.
"""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import decsim.config as config
import decsim.decoders.dispatch_steps.measurements as measurements
import decsim.decoders.measured_table.decoder as measured_table
import decsim.decoders.strong_backend as strong_backend
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoding as decoding_records
import decsim.tables as tables
from decsim.decoders.relay_belief_propagation import (
    decoder as relay_belief_propagation,
)

PHYSICAL = fault_models.FaultRepresentation.PHYSICAL
ECHO = "the link card's echo round trip"
DISPATCHER = strong_backend.DISPATCHER
WORKER = strong_backend.WORKER


@dataclasses.dataclass(frozen=True)
class DispatchStepsSettings:
    """The dispatch_steps row's keys in its tier section.

    device names the GPU measured, path the dispatcher's (device or
    host), workers the host path's graph workers, each with its own
    stream. The device and path must have a measured card
    (measurements.py).
    """

    device: str = "gh200"
    path: str = "device"
    workers: int = 1
    # the word the yaml and the reports name this row by
    name = "dispatch_steps"

    def __post_init__(self) -> None:
        _check_path(self.device, self.path)
        config.check_whole_count("workers", self.workers, "graph workers")
        _check_workers(self.path, self.workers)

    def build(self) -> "DispatchStepsDecoder":
        """A fresh decoder of these settings."""
        return DispatchStepsDecoder(settings=self)

    @classmethod
    def from_yaml(
        cls,
        section: Mapping,
        clocks: config.ClockSettings,
        section_name: str,
    ) -> "DispatchStepsSettings":
        """The three keys the section writes; absent is the default.

        section_name is the tier section the row sits in, which a refusal
        names.
        """
        del clocks
        return tables.section_record(section_name, cls, section)


class DispatchSteps:
    """A GPU's dispatcher and workers, each step timed on its own."""

    def __init__(self, settings: DispatchStepsSettings) -> None:
        self.settings = settings
        self.card = measurements.CARDS[(settings.device, settings.path)]
        self.kernels = _kernel_rows(settings.device)
        self.decoder = relay_belief_propagation.RelayBeliefPropagationDecoder()

    def run_seed_children(self) -> tuple:
        """The Relay-BP decoder's own children, at the relay_bp row's paths."""
        return self.decoder.run_seed_children()

    def capacities(self) -> dict:
        """One dispatcher; on the host path, the workers as well."""
        if self.settings.path == "device":
            return {DISPATCHER: 1}
        return {DISPATCHER: 1, WORKER: self.settings.workers}

    def submit(
        self, request: decoding_records.DecodeJob, running: int
    ) -> decoding_records.Ticket:
        """Decode with decsim's Relay-BP now; its steps follow from it."""
        del running
        result = self.decoder.decode(request)
        faults = request.detector_error_model.require_faults(PHYSICAL)
        detectors = faults.check.shape[0]
        kernel = measured_table.nearest_in_size(self.kernels, detectors)
        iterations = result.iterations or 0
        microseconds = kernel.decode_microseconds(iterations)
        decode_ticks = config.microseconds_to_ticks(microseconds)
        return decoding_records.Ticket(result, decode_ticks)

    def steps(self, ticket: decoding_records.Ticket) -> tuple:
        """The path's steps for this decode, in the order they run."""
        if self.settings.path == "device":
            return self._device_steps(ticket.decode_ticks)
        return self._host_steps(ticket.decode_ticks)

    def result(
        self, ticket: decoding_records.Ticket
    ) -> decoding_records.DecodeResult:
        """The region's answer from decsim's own Relay-BP decode."""
        return ticket.result

    def _device_steps(self, decode_ticks: int) -> tuple:
        """Every step on the dispatcher, which a fired decode holds."""
        fire = _timed_step("fire", self.card.launch_microseconds, DISPATCHER)
        decode = decoding_records.Step("decode", decode_ticks, DISPATCHER)
        return (
            _echo_step("notice", DISPATCHER),
            _echo_step("check", DISPATCHER),
            _echo_step("handle", DISPATCHER),
            fire,
            decode,
            _echo_step("respond", DISPATCHER),
        )

    def _host_steps(self, decode_ticks: int) -> tuple:
        """The monitor's notice, then the worker's launch, copies and decode."""
        card = self.card
        launch = _timed_step("launch", card.launch_microseconds, WORKER)
        copy_in = _timed_step("copy_in", card.copy_in_microseconds, WORKER)
        decode = decoding_records.Step("decode", decode_ticks, WORKER)
        copy_out = _timed_step("copy_out", card.copy_out_microseconds, WORKER)
        return (
            _echo_step("notice", DISPATCHER),
            _echo_step("check", DISPATCHER),
            launch,
            copy_in,
            decode,
            copy_out,
            _echo_step("respond", None),
        )


class DispatchStepsDecoder(strong_backend.StrongBackendDecoder):
    """The dispatch_steps row: the strong decoder over DispatchSteps."""

    Settings = DispatchStepsSettings
    fault_model_requirement = fault_models.PHYSICAL_FAULT_MODEL_REQUIRED

    def __init__(self, settings: Optional[DispatchStepsSettings] = None):
        if settings is None:
            settings = DispatchStepsSettings()
        backend = DispatchSteps(settings)
        strong_backend.StrongBackendDecoder.__init__(self, backend)


def _echo_step(name: str, resource: Optional[str]) -> decoding_records.Step:
    """A step the link card's echo already prices, shown with no ticks."""
    return decoding_records.Step(name, 0, resource, priced_on=ECHO)


def _timed_step(
    name: str, microseconds: float, resource: str
) -> decoding_records.Step:
    ticks = config.microseconds_to_ticks(microseconds)
    return decoding_records.Step(name, ticks, resource)


def _kernel_rows(device: str) -> tuple:
    """The traced kernel lines of one device, in region order."""
    rows = []
    for kernel in measurements.KERNEL_TIMES:
        if kernel.device == device:
            rows.append(kernel)
    return tuple(rows)


def _check_path(device: str, path: str) -> None:
    """A device and path with a measured card; the A100 has no device path."""
    if (device, path) in measurements.CARDS:
        return
    measured = sorted(measurements.CARDS)
    raise ValueError(
        f"device {device!r} with path {path!r} has no card in "
        f"dispatch_steps; the measured ones are {measured} (the device "
        "path's graph fire is compiled for compute capability 9.0 and up, "
        "dispatch_kernel.cu v0.15.2 line 491)"
    )


def _check_workers(path: str, workers: int) -> None:
    """Workers other than the one are named only on the host path."""
    if path == "device" and workers != 1:
        raise ValueError(
            "workers is the host path's; the device path decodes on its one "
            "dispatcher"
        )

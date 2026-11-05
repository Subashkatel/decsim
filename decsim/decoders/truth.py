"""The truth decoder: each window commits the faults the shot really fired.

The "always right" decoder of a speed-of-light study, which asks how many
failures a decoder's own mistakes cause. It reads the shot's fired errors
from a source that keeps them (qpu/stim_device.py, ErrorModelStimDevice)
and, in each window, selects exactly the fired mechanisms among the
columns the window owns. Its commit is then the true one: the observables
those faults flip, and the detectors they flip in later rounds, which is
the boundary it hands to the next window. A window that owns no fired
fault commits nothing.

It works on the physical columns (decompose_errors=False), the same
mechanisms the source draws from: a column is a fault's detectors and
observables, reduced modulo two, and mechanisms of one identity are one
column. Two fired errors of one identity cancel, as they do on the chip.
The source fires each shot's errors before any window is decoded, and a
machine is built for each shot, so the errors held are this shot's.
"""

import dataclasses
from typing import Optional

import numpy

import decsim.config as config
import decsim.decoders.decoder as decoder_module
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models
import decsim.records.program as program_records


class TruthDecoder(decoder_module.WindowDecoderBase):
    """Selects the fired faults among a window's owned columns.

    The unit is held for a stated time; the study replays or sets it.
    """

    fault_model_requirement = fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    fault_representation = fault_models.FaultRepresentation.PHYSICAL
    reads_fired_errors = True

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The time a decode holds the unit, in microseconds."""

        latency_microseconds: float = 0.0
        # the word the reports name this row by
        name = "truth"

        def __post_init__(self) -> None:
            config.check_microseconds(
                "truth.latency_microseconds", self.latency_microseconds
            )

        def build(self) -> "TruthDecoder":
            """A fresh decoder, which holds no errors until a shot fires."""
            return TruthDecoder(self)

    def __init__(self, settings: "TruthDecoder.Settings") -> None:
        decoder_module.WindowDecoderBase.__init__(self)
        self.compile_key = (type(self), settings)
        self.latency_ticks = config.microseconds_to_ticks(
            settings.latency_microseconds
        )
        # the identities (detectors, observable bits) that fired an odd
        # number of times; None until the source fires
        self.odd_identities: Optional[set] = None
        self.operation_id = None

    def errors_sampled(
        self, operation: program_records.Operation, fired_errors: tuple
    ) -> None:
        """The source drew the shot; keep the identities that fired oddly.

        A shot of several operations would mix their detector numbers,
        so a second operation is refused.
        """
        if self.operation_id is not None and self.operation_id != operation.id:
            raise RuntimeError(
                "the truth decoder keeps one operation's errors a shot; "
                f"{operation.id!r} fired after {self.operation_id!r}"
            )
        self.operation_id = operation.id
        odd = set()
        for error in fired_errors:
            identity = (error.detectors, tuple(error.logical_observables))
            odd ^= {identity}
        self.odd_identities = odd

    def ticks_after_decode(
        self,
        result: Optional[decoding_records.DecodeResult],
        elapsed_nanoseconds: int,
        now: int,
    ) -> int:
        """The stated time, whatever the host took."""
        del result, elapsed_nanoseconds, now
        return self.latency_ticks

    def compile(
        self,
        faults: fault_models.PlacedFaultModel,
        model: fault_models.WindowErrorModel,
    ) -> dict:
        """The first owned column of each identity, keyed by the identity.

        A window can hold two columns of one identity: the linked
        catalogs (a run whose other pool decodes graphlike columns) keep
        a physical column per Stim instruction. Selecting both would
        cancel the fault, so a fired identity selects one column.
        """
        del model
        observables = faults.observables.toarray()
        column_by_identity = {}
        for column in numpy.flatnonzero(faults.owned):
            column = int(column)
            detectors = tuple(sorted(faults.boundary_flips[column]))
            bits = tuple(int(bit) for bit in observables[:, column])
            column_by_identity.setdefault((detectors, bits), column)
        return column_by_identity

    def decode_window(
        self,
        backend: dict,
        model: fault_models.WindowErrorModel,
        faults: fault_models.PlacedFaultModel,
        syndrome: numpy.ndarray,
    ) -> decoding_records.WindowDecode:
        """One owned column per identity that fired an odd number of times."""
        del model, syndrome
        if self.odd_identities is None:
            raise RuntimeError(
                "the truth decoder holds no fired errors: its source keeps "
                "none (name qpu.source.type=error_model_stim)"
            )
        selected = numpy.zeros(faults.check.shape[1], dtype=numpy.uint8)
        for identity in self.odd_identities:
            column = backend.get(identity)
            if column is not None:
                selected[column] = 1
        return decoding_records.WindowDecode(selected, None)

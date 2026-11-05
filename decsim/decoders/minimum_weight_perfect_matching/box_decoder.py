"""The box redo's strong side: correlated matching on the chip's box alone.

The chip packs an escalated window's doubtful box (escalation/
doubtful_box.py): the box's detection events, and union-find's
correction outside the box. This row reads that request and no round.
Correlated matching decodes the window's own graph given only the box's
events, and the host joins its correction to union-find's outside the
box: each cluster's correction explains exactly that cluster's events,
so the joined correction explains the window's syndrome. A request with
no box event keeps union-find's correction.

The unit is held for the fitted law at the box's detection events, then
for the join (join_microseconds). The law is the whole window's fit; a
run that measures matching on boxes alone gives the box its own scale.
With no law, the host's own time around the call, the join included,
holds the unit.
"""

import dataclasses
import math
from typing import Optional

import numpy
import pymatching

import decsim.config as config
import decsim.decoders.minimum_weight_perfect_matching.decoder as matching
import decsim.decoders.minimum_weight_perfect_matching.detection_event_time as event_time  # noqa: E501
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models

CORRELATED = matching.CorrelatedPyMatchingDecoder


class BoxCorrelatedPyMatchingDecoder(CORRELATED):
    """Correlated matching on the chip's doubtful box; union-find else."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """timing is the fitted law at the box's detection events.

        join_microseconds is the host's join of the strong correction to
        union-find's outside the box. With no timing, the host's own
        time around the call holds the unit.
        """

        timing: Optional[event_time.DetectionEventTime] = None
        join_microseconds: float = 0.0
        # the word the reports name this row by
        name = "box_correlated_pymatching"
        # it decodes the chip's doubtful box, which a redo window packs
        # under box_margin (escalation/strong_window_shapes.py)
        reads_doubtful_box = True

        def __post_init__(self) -> None:
            _check_join(self.join_microseconds)

        def build(self) -> "BoxCorrelatedPyMatchingDecoder":
            """A fresh decoder of these settings."""
            return BoxCorrelatedPyMatchingDecoder(self)

    def __init__(self, settings: "BoxCorrelatedPyMatchingDecoder.Settings"):
        CORRELATED.__init__(self, timing=settings.timing)
        self.join_ticks = config.microseconds_to_ticks(
            settings.join_microseconds
        )
        row = type(self)
        self.compile_key = (row, None)

    def job_syndrome(
        self,
        job: decoding_records.DecodeJob,
        faults: fault_models.PlacedFaultModel,
    ) -> numpy.ndarray:
        """The window's syndrome with only the box's events set."""
        box = _box_of(job)
        _check_rows(job, box, faults)
        return box.syndrome()

    def window_answer(
        self,
        job: decoding_records.DecodeJob,
        backend: pymatching.Matching,
        model: fault_models.WindowErrorModel,
        faults: fault_models.PlacedFaultModel,
        syndrome: numpy.ndarray,
    ) -> decoding_records.WindowDecode:
        """Matching's correction of the box, joined to union-find's outside."""
        box = _box_of(job)
        _refuse_a_forced_class(job)
        numbers = box.numbers()
        if len(box.event_rows) == 0:
            return decoding_records.WindowDecode(box.outside, evidence=numbers)
        strong = CORRELATED.decode_window(
            self, backend, model, faults, syndrome
        )
        if strong.decode_status is not None:
            return dataclasses.replace(strong, evidence=numbers)
        strong_selected = numpy.asarray(strong.selected_faults, numpy.uint8)
        joined = box.outside ^ strong_selected
        return decoding_records.WindowDecode(joined, evidence=numbers)

    def ticks_after_decode(
        self,
        result: Optional[decoding_records.DecodeResult],
        elapsed_nanoseconds: int,
        now: int,
    ) -> int:
        """The fitted law at the box's detection events and the join."""
        if self.timing is None:
            return CORRELATED.ticks_after_decode(
                self, result, elapsed_nanoseconds, now
            )
        box_events = int(result.evidence["box_detection_events"])
        decode_ticks = self.timing.ticks_for_events(box_events)
        return decode_ticks + self.join_ticks


def _box_of(job: decoding_records.DecodeJob) -> object:
    """The chip's doubtful box (escalation/doubtful_box.py DoubtfulBox).

    A job's request is the box a redo window packed, or none (records/
    decoding.py DecodeJob.request).
    """
    box = job.request
    if box is not None:
        return box
    raise ValueError(
        f"{job.label}: box_correlated_pymatching decodes the chip's doubtful "
        "box, and this job carries none; set "
        "switching.strong_window.box_margin on a redo window"
    )


def _check_rows(
    job: decoding_records.DecodeJob,
    box: object,  # the chip's doubtful_box.DoubtfulBox
    faults: fault_models.PlacedFaultModel,
) -> None:
    """The chip's rows and columns are the strong window's own."""
    detector_count, column_count = faults.check.shape
    same_rows = box.detector_count == detector_count
    same_columns = len(box.outside) == column_count
    if same_rows and same_columns:
        return
    raise ValueError(
        f"{job.label}: the chip packed a box of {box.detector_count} "
        f"detectors and {len(box.outside)} fault columns; the strong window "
        f"has {detector_count} and {column_count}"
    )


def _refuse_a_forced_class(job: decoding_records.DecodeJob) -> None:
    if job.forced_logical_class is None:
        return
    raise ValueError(
        f"{job.label}: box_correlated_pymatching answers no forced-class solve"
    )


def _check_join(join_microseconds: object) -> None:
    is_number = config.is_number(join_microseconds)
    is_finite = is_number and math.isfinite(join_microseconds)
    if is_finite and join_microseconds >= 0:
        return
    raise ValueError(
        "join_microseconds must be a finite number, 0 or more "
        f"(got {join_microseconds!r})"
    )

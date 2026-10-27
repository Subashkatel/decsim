"""Batches that decode: each member by the pool's algorithm, one price.

A pool an edge escalates to may serve its waiting re-decodes as one
batch (decode_queue.py). Under decoder_manager.bulk_strong the batch is
timing-only. A pool's batching row decode_each instead decodes every
member with the pool's own algorithm, hands each member its own result,
and prices the batch by a law:

    sum      the members' times added. Toshio et al. assume "a linear
             decoding time and batch decoding" for the strong decoder
             (2510.25222 lines 1096-1101), and one GPU decoding several
             blocks of a window scheme at once runs "in a near linear
             fashion with increasing batch size" (2604.12841 lines
             9968-9972 and 10011).
    slowest  the slowest member's time: a batch that steps its members
             together ends with its last, as Syndrilla's batch "must
             synchronize to its slowest sample" (Lottery BP 2605.00038
             line 3561) and cudaq-qec's batched BP+OSD is tail-free from
             its synchronous batch (2605.30765 line 3788).

The sources time batches of shots; a batch here is of windows, the
blocks 2604.12841 batches on one GPU. A strong backend row prices one
region per submit and states no batch time, so batching it is refused,
not implemented yet. A measured batch table is not built:
the measured curves (2604.12841 Fig. 18) are a network's, which no row
here runs.
"""

import dataclasses
from typing import Optional, Protocol

import decsim.decoders.decoder as decoder_module
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.refusals as refusals


class BatchLaw(Protocol):
    """How a batch's time follows from its members' times."""

    def priced(self, member_ticks: list) -> int:
        """The batch's ticks."""


class LawSettings(Protocol):
    """A law row's settings record (this module), which builds it."""

    name: str

    def build(self) -> BatchLaw:
        """The law."""


class Sum:
    """The members' times added: a decoding time linear in the batch."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The sum law, which takes no setting."""

        name = "sum"

        def build(self) -> "Sum":
            """The sum law."""
            return Sum()

    def priced(self, member_ticks: list) -> int:
        """The members' ticks added."""
        return sum(member_ticks)


class Slowest:
    """The slowest member's time: a batch whose members step together."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The slowest law, which takes no setting."""

        name = "slowest"

        def build(self) -> "Slowest":
            """The slowest law."""
            return Slowest()

    def priced(self, member_ticks: list) -> int:
        """The largest of the members' ticks."""
        return max(member_ticks)


class Batching(Protocol):
    """A pool's batching, which decodes a merged batch over its row."""

    def over(self, decoder: object) -> "BatchDecoder":
        """The batch decode over the pool's algorithm."""


class DecodeEach:
    """Each member decoded by the pool's algorithm, priced by a law."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """law is a row of this module: sum (the default) or slowest."""

        law: LawSettings = Sum.Settings()
        name = "decode_each"

        def build(self) -> "DecodeEach":
            """The batching under its law."""
            law = self.law.build()
            return DecodeEach(law)

    def __init__(self, law: BatchLaw) -> None:
        self.law = law

    def over(self, decoder: object) -> "BatchDecoder":
        """The batch decode over the pool's algorithm."""
        _refuse_a_backend_row(decoder)
        return BatchDecoder(decoder, self.law)


class BatchDecoder:
    """The pool's algorithm over one batch's members, priced by a law.

    Each member's input is masked by its own gate and read out of the
    unit's memory, as a single decode's is when it starts
    (decode_service.py begin, staged_decoder.py); each member decodes
    as an ensemble's members do, measured on the host or declared. A
    member cancelled before the batch starts is left out; one cancelled
    while it runs was decoded, and its result is dropped at delivery.
    """

    def __init__(self, decoder: object, law: BatchLaw) -> None:
        self.decoder = decoder
        self.law = law

    def latency(self, job: decoding_records.DecodeJob) -> int:
        """The law over the members' latencies."""
        latencies = []
        for member in _uncancelled(job):
            member_latency = self.decoder.latency(member)
            latencies.append(member_latency)
        return self.law.priced(latencies)

    def occupancy(self, job: decoding_records.DecodeJob) -> Optional[int]:
        """The law over the members' occupancies; None if one is measured."""
        occupancies = []
        for member in _uncancelled(job):
            occupancy = self.decoder.occupancy(member)
            if occupancy is None:
                return None
            occupancies.append(occupancy)
        return self.law.priced(occupancies)

    def start(
        self,
        job: decoding_records.DecodeJob,
        engine: engine_module.Engine,
        on_result: decoder_module.OnResult,
    ) -> None:
        """Decode every member now; answer when the law's time ends."""
        results = {}
        member_ticks = []
        for member in _uncancelled(job):
            _read_input(member)
            result, ticks = decoder_module.timed_decode(
                self.decoder, member, engine.now
            )
            results[member.request_key] = result
            member_ticks.append(ticks)
        ticks = self.law.priced(member_ticks)
        answer = decoding_records.DecodeResult(
            job.stream_id, job.window_id, member_results=results
        )
        engine.schedule(
            ticks,
            lambda: _deliver(job, answer, on_result),
            label=f"decode_done({job.label})",
        )


def _uncancelled(job: decoding_records.DecodeJob) -> list:
    """The batch's members no cancel has reached."""
    return [member for member in job.batch_members if not member.cancelled]


def _read_input(member: decoding_records.DecodeJob) -> None:
    """The member's landed input, masked by its gate, as the decode reads it."""
    if member.gate is not None:
        member.gate.mask_input(member)
    if member.decoder_input is not None:
        member.payloads = member.decoder_input.fragments()


def _deliver(
    job: decoding_records.DecodeJob,
    answer: decoding_records.DecodeResult,
    on_result: decoder_module.OnResult,
) -> None:
    if job.cancelled:
        on_result(None)
        return
    on_result(answer)


def _refuse_a_backend_row(decoder: object) -> None:
    if not getattr(decoder, "priced_by_its_backend", False):
        return
    row = type(decoder)
    raise NotImplementedError(
        f"decode_each over {row.__name__}: a strong backend prices one "
        "region per submit and states no batch time; batching it is not "
        f"implemented yet; {refusals.REFUSALS}"
    )

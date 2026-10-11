"""Batches that decode: the two laws, and a batch of two on declared ticks.

The sum law is Toshio et al.'s linear decoding time (2510.25222 lines
1096-1101) and 2604.12841's near-linear batch runtime (lines
9968-9972); the slowest law is a batch that synchronizes to its slowest
sample (Lottery BP 2605.00038 line 3561).

The timeline runs four memories, mem1 to mem4, on the declared
switching fabric (tests/declared_run.py) with a weak unit each and one
strong unit of two input slots; every window escalates, and the strong
row decodes operation k in 10k us. Each W0's weak decode ends at 30 us
and its rounds reach the strong side at 33 us: the unit takes mem1 W0
and stages mem2 W0, and mem3 W0 and mem4 W0 wait. Their inputs land at
39 us; mem1 W0 decodes 39-49 us and mem2 W0 49-69 us. At 49 us the
freed slot takes the two waiting jobs as one batch, whose inputs land
at 55 us, and the batch starts at 69 us. An answer rides
strong_decoder_to_frame (4 us) and commits 1 us later.
"""

import dataclasses

import pytest

import decsim.config as config
import decsim.decoders.batching as batching
import decsim.decoders.decoder as decoder_module
import decsim.decoders.settings as decoder_settings
import decsim.machine as machine_module
import decsim.parameters.kinds as kinds
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records
import tests.declared_run as declared_run


def test_the_sum_law_adds_the_members_ticks():
    law = batching.Sum()

    ticks = law.priced([30, 40])

    assert ticks == 70


def test_the_slowest_law_takes_the_largest_of_the_members_ticks():
    law = batching.Slowest()

    ticks = law.priced([30, 40])

    assert ticks == 40


def test_decode_each_and_its_laws_are_written_by_their_names():
    batching_rows = kinds.records_for(decoder_settings.BatchingSettings)
    law_rows = kinds.records_for(batching.LawSettings)

    each = kinds.records_named("decode_each", batching_rows)
    laws = []
    for name in ("sum", "slowest"):
        matches = kinds.records_named(name, law_rows)
        laws.append(len(matches))

    assert each == [batching.DecodeEach.Settings]
    assert laws == [1, 1]
    assert len(law_rows) == 2


def test_decode_each_refuses_a_strong_backend_row_for_now():
    row = _PatchLatency()
    row.priced_by_its_backend = True
    each = batching.DecodeEach.Settings()
    built = each.build()

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        built.over(row)


def test_a_batch_of_two_priced_by_sum_answers_when_both_decodes_are_done():
    """mem3 W0 and mem4 W0 decode 30 + 40 us, from 69 to 139 us.

    Both answers land at 143 us and commit at 144 us. Unbatched, mem3 W0
    would commit at 104 us and mem4 W0 at 144 us.
    """
    law = batching.Sum.Settings()
    machine = _four_memories(law)

    records = _records_by_window(machine)
    batch_start = _log_tick(machine, "START DECODE strong-batch x2")

    assert batch_start == _ticks(69.0)
    assert records[(3, 0)] == (_ticks(143.0), _ticks(144.0))
    assert records[(4, 0)] == (_ticks(143.0), _ticks(144.0))


def test_a_batch_of_two_priced_by_slowest_answers_with_its_slower_member():
    """The batch ends with mem4 W0's 40 us, at 109 us.

    Both answers land at 113 us and commit at 114 us.
    """
    law = batching.Slowest.Settings()
    machine = _four_memories(law)

    records = _records_by_window(machine)

    assert records[(3, 0)] == (_ticks(113.0), _ticks(114.0))
    assert records[(4, 0)] == (_ticks(113.0), _ticks(114.0))


def test_each_member_of_a_batch_is_answered_with_its_own_result():
    """The strong row predicts each operation's parity, window by window."""
    law = batching.Sum.Settings()
    ended = declared_run.EndedRequests()
    _four_memories(law, probes=(ended,))

    answers = []
    for request in ended.of_tier(window_records.STRONG_POOL_ID):
        result = request.result
        answer = (
            result.operation_id,
            result.window_id,
            result.logical_observables,
        )
        answers.append(answer)

    assert sorted(answers) == [
        (1, 0, (1,)),
        (1, 1, (1,)),
        (2, 0, (0,)),
        (2, 1, (0,)),
        (3, 0, (1,)),
        (3, 1, (1,)),
        (4, 0, (0,)),
        (4, 1, (0,)),
    ]


def _four_memories(law, probes=()) -> machine_module.Machine:
    """The declared four-memory run, its strong pool decoding each member."""
    operations = []
    for operation_id in range(1, 5):
        memory = declared_run.memory_operation(operation_id)
        operations.append(memory)
    decode_each = batching.DecodeEach.Settings(law=law)
    strong = _PatchLatency.Settings()
    return declared_run.switching_run(
        rounds=6,
        operations=operations,
        escalates=True,
        weak_units=4,
        strong_algorithm=strong,
        strong_batching=decode_each,
        probes=probes,
    )


def _records_by_window(machine: machine_module.Machine) -> dict:
    """Each window's landing and commit ticks."""
    snapshot = machine.control.pauli_frame.snapshot()
    records = {}
    for record in snapshot.records:
        records[record.window_key] = (
            record.accepted_ticks,
            record.committed_ticks,
        )
    return records


def _log_tick(machine: machine_module.Machine, text: str) -> int:
    log_lines = machine.observation.log.lines
    return declared_run.log_tick(log_lines, text)


def _ticks(microseconds: float) -> int:
    return config.microseconds_to_ticks(microseconds)


class _PatchLatency(decoder_module.DecoderBase):
    """Ten microseconds per operation id; it predicts the id's parity."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        name = "patch_latency"

        def build(self):
            return _PatchLatency()

    def latency(self, job) -> int:
        microseconds = 10.0 * job.operation_id
        return _ticks(microseconds)

    def decode(self, job) -> decoding_records.DecodeResult:
        parity = job.operation_id % 2
        return decoding_records.DecodeResult(
            job.operation_id, job.window_id, logical_observables=(parity,)
        )

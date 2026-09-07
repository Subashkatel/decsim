"""The dispatcher's law: a blocked job never displaces a startable one.

gem5 O3 issues from its ready set oldest-first and non-ready work never
displaces ready work (src/cpu/o3/inst_queue.hh, scheduleReadyInsts).
"""

import decsim.decoders.decoders as decoders
import decsim.decoders.schedulers as schedulers
import decsim.engine as engine_module
import decsim.escalation.policies as escalation_policies
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
from decsim.decoders.decoder_manager import DecoderManager


class _OpenGate:
    """A gate that lets a blocked job take a slot and start when it can."""

    def may_stage(self, job):
        del job
        return True

    def may_start(self, job):
        del job
        return True

    def mask_input(self, job):
        del job


def _window(index, deps_remaining):
    return window_records.Window(
        operation_id=1,
        k=index,
        commit_lo=1,
        commit_hi=1,
        buffer_hi=1,
        n_rounds=1,
        deps_remaining=deps_remaining,
    )


def _job(index, label, deps_remaining, gate=None):
    payload = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_id="p",
        round_index=index,
        bits=(0, 1),
        size_bits=2,
        fragment_index=0,
    )
    request_key = window_records.DecoderRequestKey(
        1, index, window_records.DecoderTier.WEAK, index
    )
    window = _window(index, deps_remaining)
    return decoding_records.DecodeJob(
        operation_id=1,
        window_id=index,
        n_rounds=1,
        payloads=[payload],
        label=label,
        request_key=request_key,
        window=window,
        gate=gate,
    )


def _ignore(_job, _result):
    pass


def test_a_blocked_job_never_displaces_a_startable_one():
    engine = engine_module.Engine()
    decoder = decoders.PresetLatencyDecoder(4.0)
    router = decoders.CodeRouter(decoder)
    scheduler = schedulers.FifoScheduler()
    policy = escalation_policies.Baseline()
    manager = DecoderManager(
        engine,
        router=router,
        scheduler=scheduler,
        num_units=1,
        escalation_policy=policy,
    )
    dispatched = []
    original_dispatch_to = manager.service.dispatch_to

    def recording_dispatch_to(pool, job, unit, claim_compute):
        dispatched.append(job.label)
        original_dispatch_to(pool, job, unit, claim_compute)

    manager.service.dispatch_to = recording_dispatch_to
    # a computes and b lands in the second slot: the unit is full, so
    # the blocked c and the startable d wait, c admitted first
    job_a = _job(0, "a", deps_remaining=0)
    manager.enqueue(job_a, None, _ignore)
    job_b = _job(1, "b", deps_remaining=0)
    manager.enqueue(job_b, None, _ignore)
    gate = _OpenGate()
    job_c = _job(2, "c", deps_remaining=1, gate=gate)
    manager.enqueue(job_c, None, _ignore)
    job_d = _job(3, "d", deps_remaining=0)
    manager.enqueue(job_d, None, _ignore)
    engine.run()
    assert dispatched == ["a", "b", "d", "c"]

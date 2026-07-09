"""Gate-6 leftover: bulk_strong merge invariants (V9 caveat closure).

Covers the previously-uncovered bulk_strong paths of DecoderManager:
merge-and-deliver, refusal of accuracy-coupled merges, running-rounds
accounting, and the cancel-one-merged-key edge. The last one exposed
TWO real bugs (2026-07-04, probe-verified before the fix): cancelling
one key of a RUNNING merged batch cancelled the whole batch — the
sibling keys' results were silently lost (their windows hung in
_windows_waiting_for_strong_result forever) — and strong_running_rounds
leaked (never decremented on a cancelled batch). Fixed in
decoder_manager.cancel_strong: a running merged batch with live
siblings survives the cancel (only the cancelled key is dropped from
delivery); a batch with no survivors cancels AND settles the rounds
accounting.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from decsim.config import us
from decsim.engine import Engine
from decsim.decoders import CodeRouter, PerRoundDecoder
from decsim.decoder_manager import DecoderManager
from decsim.message import DecodeJob
from decsim.schedulers import FifoScheduler


class _NullStrategy:
    def on_decode_outcome(self, outcome, services):
        return None


def build(bulk_strong=True):
    eng = Engine(verbose=False)
    manager = DecoderManager(
        eng, router=CodeRouter(default=PerRoundDecoder(tau_us=1.0)),
        scheduler=FifoScheduler(),
        unit_pools={"default": 1, "strong": 1}, bulk_strong=bulk_strong)
    manager.strategy = _NullStrategy()
    results = []
    manager.on_strong_window_decoded = \
        lambda key, res: results.append((eng.now, key))
    return eng, manager, results


def strong_job(op, rounds, label=None):
    return DecodeJob(op_id=op, window_id=0, n_rounds=rounds,
                     strong_decode_for=(op, 0), hint="strong",
                     label=label or f"s{op}")


def occupy_then_merge(eng, manager):
    """Blocker holds the strong unit; two 5-round strongs queue+merge."""
    manager.enqueue(strong_job(1, 10, "s-block"))
    for op in (2, 3):
        manager.enqueue(strong_job(op, 5))
    for key in [(1, 0), (2, 0), (3, 0)]:
        manager._windows_waiting_for_strong_result.add(key)


def test_bulk_merge_delivers_every_key_and_frees_units():
    eng, manager, results = build()
    occupy_then_merge(eng, manager)
    eng.run()
    assert [(t / us(1), k) for t, k in results] == \
        [(10.0, (1, 0)), (20.0, (2, 0)), (20.0, (3, 0))]
    assert manager.pool_free == {"default": 1, "strong": 1}
    assert manager.strong_running_rounds == 0
    assert not manager._windows_waiting_for_strong_result


def test_cancel_one_merged_key_keeps_sibling_result():
    """THE bug: before the fix, cancelling (2,0) killed the whole
    running batch and (3,0) hung forever with rounds leaked."""
    eng, manager, results = build()
    occupy_then_merge(eng, manager)
    eng.schedule(us(12), lambda: manager.cancel_strong((2, 0)))
    eng.run()
    keys = [k for _, k in results]
    assert (3, 0) in keys, "sibling result lost on merged-key cancel"
    assert (2, 0) not in keys, "cancelled key must not deliver"
    assert manager.strong_cancelled == 1
    assert manager.strong_running_rounds == 0, "rounds accounting leaked"
    assert manager.pool_free == {"default": 1, "strong": 1}
    assert manager._windows_waiting_for_strong_result == {(2, 0)}


def test_cancel_all_merged_keys_cancels_the_batch_once():
    eng, manager, results = build()
    occupy_then_merge(eng, manager)
    eng.schedule(us(12), lambda: (manager.cancel_strong((2, 0)),
                                  manager.cancel_strong((3, 0))))
    eng.run()
    assert [k for _, k in results] == [(1, 0)]
    assert manager.strong_cancelled == 2
    assert manager.strong_running_rounds == 0
    assert manager.pool_free == {"default": 1, "strong": 1}


def test_bulk_strong_refuses_accuracy_coupled_merges():
    eng, manager, _ = build()
    manager.enqueue(strong_job(1, 10, "s-block"))
    j2 = strong_job(2, 5)
    j2.dem = object()                       # accuracy-coupled marker
    manager.enqueue(j2)
    manager.enqueue(strong_job(3, 5))
    with pytest.raises(RuntimeError, match="bulk_strong only merges"):
        eng.run()


def test_running_rounds_tracks_merged_batch_lifecycle():
    eng, manager, _ = build()
    occupy_then_merge(eng, manager)
    seen = []

    def watch():
        seen.append((eng.now / us(1), manager.strong_running_rounds))
        if eng.now < us(25):
            eng.schedule(us(1), watch)
    eng.schedule(0, watch)
    eng.run()
    by_time = dict(seen)
    assert by_time[11.0] == 10              # merged batch (5+5) running
    assert by_time[21.0] == 0               # settled after completion

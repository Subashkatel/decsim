from dataclasses import replace
from types import MappingProxyType

import pytest

from decsim.config import us
from decsim.detector_error_model import NO_FAULT_MODEL_REQUIRED
from decsim.message import (BoundaryUpdate, DecodeResult, Operation,
                            WindowInfo)
from decsim.planner import FixedRounds, PerOpRounds
from decsim.run_spec import RunSpec, simulate
from decsim.window_interactions import DefaultWindowInteraction


class RichBoundaryDecoder:
    fault_model_requirement = NO_FAULT_MODEL_REQUIRED

    def __init__(self):
        self.first_round_bits = {}

    def latency(self, job):
        return us(0.1)

    def decode(self, job):
        first = min(job.payloads, key=lambda payload: payload.round_index)
        self.first_round_bits[job.window_id] = first.bits
        return DecodeResult(
            job.op_id,
            job.window_id,
            boundary_data={"confidence": 7 + job.window_id},
        )


class RichBoundaryInteraction:
    """A non-mask protocol implemented only against the public port."""

    def initial_boundary_state(self, window):
        return ()

    def boundary_from_result(self, result, fallback):
        if result is None or result.boundary_data is None:
            return fallback
        return result.boundary_data

    def boundaries_equal(self, left, right):
        return left == right

    def boundary_targets(self, source, windows):
        assert isinstance(source, WindowInfo)
        assert isinstance(windows, MappingProxyType)
        return list(source.dependents)

    def merge_boundary(self, delivery, destination, current_state):
        confidences = tuple(current_state or ())
        return BoundaryUpdate(
            state=confidences + (delivery.payload["confidence"],),
            accepted=True,
            release_dependency=True,
        )

    def apply_boundary(self, state, window, payload, round_key):
        if not state or round_key != window.commit_lo:
            return payload
        encoded_confidence = tuple(
            int(bit) for bit in format(sum(state), "b")
        )
        return replace(payload, bits=encoded_confidence)

    def invalidated_windows(self, source_key, windows):
        return []

    def plan_strong_region(
        self, weak_window, later_windows, operation_round_count,
    ):
        return None


def test_custom_interaction_controls_rich_boundary_handoff_end_to_end():
    decoder = RichBoundaryDecoder()
    interaction = RichBoundaryInteraction()
    operation = Operation(0, "memory", (0,))

    result = simulate(RunSpec(
        ops=[operation],
        d=3,
        rounds_policy=FixedRounds(7),
        decoder=decoder,
        window_interaction=interaction,
    ))

    assert result.window_manager.window_interaction is interaction
    assert decoder.first_round_bits[0] is None
    assert decoder.first_round_bits[1] == (1, 1, 1)
    assert result.window_manager.payloads_held == 0


def test_decoder_result_identity_is_rejected_before_interaction_or_commit():
    class WrongIdentityDecoder(RichBoundaryDecoder):
        def decode(self, job):
            return DecodeResult(
                job.op_id + 1,
                job.window_id + 1,
                logical_observables=(1,),
                boundary_data={"source": (job.op_id + 1, job.window_id + 1)},
            )

    with pytest.raises(
        (RuntimeError, ValueError),
        match="identity|op_id|window_id",
    ):
        simulate(RunSpec(
            ops=[Operation(0, "memory", (0,))],
            d=3,
            rounds_policy=FixedRounds(7),
            decoder=WrongIdentityDecoder(),
            window_interaction=RichBoundaryInteraction(),
        ))


def test_duplicate_decoder_completion_is_rejected_before_callback():
    class RecordingDecoder(_OrderedBoundaryDecoder):
        def __init__(self):
            super().__init__()
            self.jobs = []

        def decode(self, job):
            self.jobs.append(job)
            return super().decode(job)

    decoder = RecordingDecoder()
    completed_run = RunSpec(
        ops=[Operation(0, "memory", (0,))],
        d=3,
        rounds_policy=FixedRounds(3),
        decoder=decoder,
    ).build()
    completed_job = decoder.jobs[0]
    free_before = dict(completed_run.decoder_manager.pool_free)

    with pytest.raises(RuntimeError, match="duplicate decoder completion"):
        completed_run.decoder_manager._on_decode_done(completed_job)

    assert completed_run.decoder_manager.pool_free == free_before


def test_duplicate_boundary_targets_are_rejected_before_delivery():
    class DuplicateTargets(DefaultWindowInteraction):
        def boundary_targets(self, source, windows):
            return list(source.dependents) * 2

    with pytest.raises(RuntimeError, match="duplicate boundary targets"):
        simulate(RunSpec(
            ops=[Operation(0, "memory", (0,))],
            d=3,
            rounds_policy=FixedRounds(7),
            decoder=RichBoundaryDecoder(),
            window_interaction=DuplicateTargets(),
        ))


class _CrossOperationTarget(DefaultWindowInteraction):
    def boundary_targets(self, source, windows):
        return [(1, 0)] if source.key == (0, 0) else []


class _CrossOperationTargetWithoutRelease(_CrossOperationTarget):
    def merge_boundary(self, delivery, destination, current_state):
        update = super().merge_boundary(
            delivery, destination, current_state)
        return replace(update, release_dependency=False)


class _OrderedBoundaryDecoder:
    fault_model_requirement = NO_FAULT_MODEL_REQUIRED

    def __init__(self, *, slow_source=False):
        self.slow_source = slow_source

    def latency(self, job):
        return us(20 if self.slow_source and job.op_id == 0 else 1)

    def decode(self, job):
        return DecodeResult(
            job.op_id,
            job.window_id,
            logical_observables=(0,),
            boundary_defects={4: [1]} if job.op_id == 0 else None,
        )


def _independent_operations(interaction, *, slow_source=False, capture=None):
    def capture_runtime(_engine, window_manager, decoder_manager, _chip, _factory):
        capture["runtime"] = window_manager
        return []

    return RunSpec(
        ops=[
            Operation(0, "source", (0,)),
            Operation(1, "target", (1,)),
        ],
        d=3,
        rounds_policy=PerOpRounds({0: 3, 1: 30}),
        decoder=_OrderedBoundaryDecoder(slow_source=slow_source),
        num_units=2,
        window_interaction=interaction,
        make_metrics=(
            None
            if capture is None
            else capture_runtime
        ),
    )


def test_invalid_boundary_destination_is_rejected_without_partial_state():
    captured = {}
    with pytest.raises(RuntimeError, match="boundary target|live dependency"):
        _independent_operations(
            _CrossOperationTarget(),
            capture=captured,
        ).build()

    runtime = captured["runtime"]
    assert runtime.windows[(1, 0)].boundary_in == {}
    assert runtime._boundary_versions == {}
    assert runtime._boundary_delivery_versions == {}


def test_late_boundary_destination_cannot_mutate_finished_work():
    captured = {}
    with pytest.raises(RuntimeError, match="boundary target|live dependency"):
        _independent_operations(
            _CrossOperationTargetWithoutRelease(),
            slow_source=True,
            capture=captured,
        ).build()

    runtime = captured["runtime"]
    target = runtime.windows[(1, 0)]
    assert target.t_done < runtime.windows[(0, 0)].t_done
    assert target.boundary_in == {}


def test_rejected_boundary_update_cannot_mutate_aliased_state():
    class MutatingRejectedInteraction(DefaultWindowInteraction):
        def initial_boundary_state(self, window):
            return {"arrivals": []}

        def merge_boundary(self, delivery, destination, current_state):
            current_state["arrivals"].append(delivery.source_key)
            return BoundaryUpdate(
                state=current_state,
                accepted=False,
                release_dependency=True,
            )

    captured = {}

    def capture_runtime(_engine, window_manager, decoder_manager, _chip, _factory):
        captured["runtime"] = window_manager
        return []

    with pytest.raises(RuntimeError, match="rejected boundary"):
        RunSpec(
            ops=[Operation(0, "memory", (0,))],
            d=3,
            rounds_policy=FixedRounds(7),
            decoder=_OrderedBoundaryDecoder(),
            window_interaction=MutatingRejectedInteraction(),
            make_metrics=capture_runtime,
        ).build()

    destination = captured["runtime"].windows[(0, 1)]
    assert destination.boundary_in == {"arrivals": []}
    assert destination.deps_remaining == 1


def test_default_strong_region_names_the_restart_seam_owner():
    from decsim.message import SeamFaultOwner

    windows = {
        (0, 2): WindowInfo(
            op_id=0, k=2, commit_lo=7, commit_hi=9,
            buffer_lo=4, buffer_hi=12, n_rounds=9,
            dependents=((0, 3),), deps=((0, 1),),
        ),
        (0, 3): WindowInfo(
            op_id=0, k=3, commit_lo=10, commit_hi=12,
            buffer_lo=7, buffer_hi=15, n_rounds=9,
            dependents=((0, 4),), deps=((0, 2),),
        ),
        (0, 4): WindowInfo(
            op_id=0, k=4, commit_lo=13, commit_hi=15,
            buffer_lo=10, buffer_hi=18, n_rounds=9,
            dependents=((0, 5),), deps=((0, 3),),
        ),
        (0, 5): WindowInfo(
            op_id=0, k=5, commit_lo=16, commit_hi=18,
            buffer_lo=13, buffer_hi=18, n_rounds=6,
            dependents=(), deps=((0, 4),),
        ),
    }
    plan = DefaultWindowInteraction().plan_strong_region(
        windows[(0, 2)],
        [windows[(0, 3)], windows[(0, 4)], windows[(0, 5)]],
        operation_round_count=18,
    )

    assert plan.commit_hi == 15
    assert plan.restart_seam_fault_owner is SeamFaultOwner.STRONG_REGION

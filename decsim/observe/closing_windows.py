"""The windows of a stream read out early, each decoded at zero time.

A source that holds the state closes a copy of the stream after a round
with a noiseless readout (records/checkpoints.py). A stream that ended
there reads that readout as its last detectors, and its windows after
the ones it keeps are its last windows, each clipped to end on the
round, as a growing stream's tail closes
(windows/region_streams.py RegionStream.close_tail). This module builds
those windows' jobs from the raw rounds and the copy's readout, each
with the boundaries it takes folded in, as the window side folds them
(windows/decode_requests.py), and decides them on the machine's own
path (observe/closing_decisions.py). A window takes the boundary of the
window before it, as the window side links it (window_manager.py),
unless it reads rounds before its commit, a parallel A block, which
decodes them itself; a closed window, a parallel B block, also takes
the boundary of the window after it (Skoric et al. 2209.08552 lines
421-426). The checkpoints (observe/checkpoints.py) and the answer
checks (observe/answer_checks.py) close their streams with it.
"""

from typing import Any, Optional

import decsim.observe.closing_decisions as closing_decisions
import decsim.qpu.moved_stream_models as moved_stream_models
import decsim.records.checkpoints as checkpoint_records
import decsim.records.circuits as circuit_records
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records


class ClosingWindows:
    """One stream's windows, read out early and decoded at zero time."""

    def __init__(
        self,
        program: circuit_records.RepeatedStimCircuit,
        closer: closing_decisions.ClosingDecisions,
    ) -> None:
        requirement = closer.requirement
        self.models = moved_stream_models.MovedStimModels(program, requirement)
        self.closer = closer
        # how many rounds back a round's detectors read, and one more
        self.reach = self.models.template.prefix_rounds

    def tail_observables(
        self,
        snapshot: checkpoint_records.CheckpointSnapshot,
        events: dict,
        boundary_ids: tuple,
        tail: list,
    ) -> Optional[tuple]:
        """The tail's observables, each window folding the boundaries it takes.

        boundary_ids is what the last kept window hands on. Each tail
        window is clipped to end on the snapshot's round and decided on
        the machine's path, a closed window after the window after it;
        None when one answers nothing.
        """
        last_round = snapshot.round_index
        endings = [ending_on(window, last_round) for window in tail]
        handed: dict = {}
        observables = ()
        for position in _decision_order(endings):
            ending = endings[position]
            taken = _boundary_taken(endings, position, handed, boundary_ids)
            job = self.job(snapshot, ending, events, taken)
            result = self.closer.decide(job)
            if result is None:
                return None
            answer = result.logical_observables
            observables = xor_or_none(observables, answer)
            handed[position] = boundary_ids_of(result)
        return observables

    def job(
        self,
        snapshot: checkpoint_records.CheckpointSnapshot,
        window: window_records.Window,
        events: dict,
        boundary_ids: tuple,
    ) -> decoding_records.DecodeJob:
        """The window's job, its boundary folded in.

        A window that reads the snapshot's round ends the stream there
        and takes that stream's model; an earlier window takes the model
        of a stream whose end is not known, as the live window does.
        """
        last_round = snapshot.round_index
        folded = dict(events)
        for detector_id in boundary_ids:
            if detector_id in folded:
                folded[detector_id] ^= 1
        final_round: Optional[int] = None
        if window.buffer_hi >= last_round:
            final_round = last_round
        model = self.models.for_window_ending_at(window, final_round)
        syndrome = [folded[detector_id] for detector_id in model.detector_ids]
        payload = round_records.QPUReadout(
            snapshot.stream_id, (), window.start_round, bits=syndrome
        )
        return decoding_records.DecodeJob(
            snapshot.stream_id,
            window.window_index,
            window.round_count,
            detector_error_model=model,
            payloads=[payload],
            window=window,
            label=f"closing {last_round} W{window.window_index}",
        )

    def events(
        self,
        snapshot: checkpoint_records.CheckpointSnapshot,
        first_round: int,
        packets: dict,
    ) -> dict:
        """Detector id -> event, of the rounds and the copy's readout.

        The readout's detectors are the last of its round, in detector
        order, as the final fragment appends them to the repeated round.
        """
        last_round = snapshot.round_index
        table = self.models.template.table(
            last_round, is_final=True, live_reach=None
        )
        recipes = []
        for round_index in range(first_round, last_round):
            round_recipes = table.detectors_of_round(round_index)
            recipes.extend(round_recipes)
        last_recipes = table.detectors_of_round(last_round)
        readout_count = len(snapshot.readout_events)
        round_count = len(last_recipes) - readout_count
        recipes.extend(last_recipes[:round_count])
        events = {}
        for recipe in recipes:
            events[recipe.detector_index] = _formed(recipe, packets)
        readout_recipes = last_recipes[round_count:]
        readout = zip(readout_recipes, snapshot.readout_events, strict=True)
        for recipe, event in readout:
            events[recipe.detector_index] = event
        return events


def ending_on(
    window: window_records.Window, last_round: int
) -> window_records.Window:
    """The window as a stream ending on the round lays it: clipped there.

    A window holding the round commits through it; every window reads
    no round after it.
    """
    commit_hi = min(window.commit_hi, last_round)
    buffer_hi = min(window.buffer_hi, last_round)
    rounds_after_start = buffer_hi - window.start_round
    round_count = rounds_after_start + 1
    return window_records.Window(
        window.stream_id,
        window.window_index,
        window.commit_lo,
        commit_hi,
        buffer_hi,
        round_count,
        buffer_lo=window.buffer_lo,
        closed_temporal_boundaries=window.closed_temporal_boundaries,
    )


def failed(
    prediction: Optional[tuple],
    snapshot: checkpoint_records.CheckpointSnapshot,
) -> Optional[bool]:
    """Whether the prediction misses the copy's truth; None for no answer."""
    if prediction is None:
        return None
    return prediction != snapshot.observable_flips


def xor_or_none(
    first: Optional[tuple], second: Optional[tuple]
) -> Optional[tuple]:
    """Two observable tuples XORed; an empty one is all zeros, None stays."""
    if first is None or second is None:
        return None
    if not first:
        return tuple(second)
    pairs = zip(first, second, strict=True)
    return tuple(left ^ right for left, right in pairs)


def boundary_ids_of(result: decoding_records.DecodeResult) -> tuple:
    """The detectors the result's commit hands to the next window."""
    boundary = result.boundary_data
    if boundary is None:
        return ()
    return tuple(boundary.detector_ids)


def _decision_order(endings: list) -> list:
    """The tail's positions, a closed window after every open one.

    A closed window takes the boundary of the window after it, and no
    open window takes a closed one's.
    """
    open_positions = []
    closed_positions = []
    for position, window in enumerate(endings):
        if window.closed_temporal_boundaries:
            closed_positions.append(position)
            continue
        open_positions.append(position)
    return open_positions + closed_positions


def _boundary_taken(
    endings: list, position: int, handed: dict, kept_boundary_ids: tuple
) -> tuple:
    """The detectors flipped by the boundaries a tail window takes.

    The first window takes the last kept window's boundary in place of
    a window before it.
    """
    window = endings[position]
    taken = []
    if window.start_round == window.commit_lo:
        before = kept_boundary_ids
        if position > 0:
            before = handed[position - 1]
        taken.extend(before)
    after = position + 1
    if window.closed_temporal_boundaries and after < len(endings):
        taken.extend(handed[after])
    return tuple(taken)


def _formed(recipe: Any, packets: dict) -> int:
    """A recipe's event: its reference parity XOR the raw bits it reads."""
    parity = recipe.reference_parity
    for round_index, slot in recipe.records:
        bits = packets[round_index]
        parity ^= int(bits[slot])
    return parity

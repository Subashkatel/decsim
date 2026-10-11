"""Window fault models of a live stream, each sliced on a short circuit.

GrowingStimModels slices a window on the circuit from round one to the
window's horizon, and every earlier window again on it, so a stream of
R rounds costs R squared. Two facts make a window's model a short
circuit's, moved in time.

- Ownership follows the geometry. A window owns the faults touching its
  commit rounds that no earlier window committed (qLDPC's
  SlidingWindowDecoder; window_placement.py), and a stream's commit
  regions tile its rounds in window order (window_planner.py,
  arithmetic_geometries_begun and _clip), so a fault belongs to the
  window whose commit region holds its earliest round. A window's prior
  faults are then every fault touching a round before its commit
  region, read off its own circuit, with no earlier window sliced.
- A repeated round's faults are the round before's, moved one round
  (periodic_formation.py; IBM's sliding-window decoder decodes every
  window with one fixed check matrix, 2510.21600, Algorithm 2). A window
  whose faults all lie in repeated rounds is sliced on a short circuit
  that holds it just past the first round's reach, or as far before the
  final round as it is, and is moved by the rounds between.

A fault's name is its earliest round and its shape there: its detectors
less that round's first detector, its observables and, for a physical
fault, its graphlike components read the same way. The name is the same
in every short circuit, so a weak window's commits and a strong window's
priors share it. A test holds every window's model to the whole
circuit's, sliced in window order.
"""

import collections
import dataclasses
import itertools
from collections.abc import Callable, Hashable, Mapping
from typing import Optional

import numpy

import decsim.detector_error_model.detector_formation as detector_formation
import decsim.detector_error_model.periodic_formation as periodic_formation
import decsim.detector_error_model.window_slicer as window_slicer
import decsim.records.circuits as circuit_records
import decsim.records.fault_model_contracts as fault_models
import decsim.records.windows as window_records

# a fault's name is its earliest round times the stride, plus its shape's
# number; a stream has far fewer shapes than the stride
NAME_STRIDE = 1 << 32
# the short circuits and the window models kept, the least recent let go
FRAME_LIMIT = 8
MODEL_LIMIT = 64


class MovedStimModels:
    """A stream's window models, each a short circuit's, moved in time.

    It answers what GrowingStimModels answers, in flat memory and a
    constant cost a window: a stream's windows share a few short
    circuits, and its interior windows one model, moved.
    """

    def __init__(
        self,
        program: circuit_records.RepeatedStimCircuit,
        requirement: fault_models.DecoderFaultModelRequirement,
    ) -> None:
        self.template = periodic_formation.StreamTemplate(program)
        self.frames = _Frames(program, requirement)
        self.models = _Recent(MODEL_LIMIT)
        self.final_round_count: Optional[int] = None
        template_frame = self.frames.frame(self.template.round_count)
        self.span = _checked_span(template_frame.slicer, self.template)
        first_repeated = self.template.prefix_rounds + 1
        repeated_recipes = self.template.live.detectors_of_round(first_repeated)
        self.detectors_per_round = len(repeated_recipes)

    def finish(self, round_count: int) -> None:
        """Bind the models to the stream's final readout at this round."""
        if self.final_round_count is not None:
            raise RuntimeError("a physical stream can only finish once")
        self.final_round_count = round_count

    def for_window(
        self, window: window_records.Window
    ) -> fault_models.WindowErrorModel:
        """One window's model: its short circuit's, moved to its rounds."""
        geometry = self._geometry_of(window)
        placement = self._placement_of(geometry)
        local = geometry.moved(-placement.shift)
        key = (placement.round_count, local)
        model = self.models.get(
            key, lambda: self._sliced_alone(placement.round_count, local)
        )
        return self._moved(model, placement.shift)

    def for_strong_window(
        self,
        window: window_records.Window,
        fault_exclusion_ranges: tuple,
        prior_faults: Optional[dict],
    ) -> fault_models.WindowErrorModel:
        """A strong window's model, its priors named as the weak commits."""
        geometry = self._geometry_of(window)
        placement = self._placement_of(geometry)
        local = geometry.moved(-placement.shift)
        frame = self.frames.frame(placement.round_count)
        exclusions = _moved_ranges(fault_exclusion_ranges, -placement.shift)
        lowest_anchor = local.start_round - self.span
        anchors = (lowest_anchor, local.last_buffer_round)
        local_priors = frame.local_faults(
            prior_faults, placement.shift, anchors
        )
        model = frame.slicer.slice_window_alone(
            local.start_round,
            local.first_commit_round,
            local.last_commit_round,
            local.last_buffer_round,
            fault_exclusion_ranges=exclusions,
            explicitly_prior_faults=local_priors,
            is_last=local.is_last,
        )
        named = frame.named(model)
        return self._moved(named, placement.shift)

    def _geometry_of(self, window: window_records.Window) -> "_Geometry":
        if window.closed_temporal_boundaries:
            raise ValueError(
                "live Stim streams require trailing-buffer feedback"
            )
        final = self.final_round_count
        last_buffer_round = window.buffer_hi
        if final is not None:
            last_buffer_round = min(last_buffer_round, final)
        is_last = window.commit_hi == final
        return _Geometry(
            start_round=window.start_round,
            first_commit_round=window.commit_lo,
            last_commit_round=window.commit_hi,
            last_buffer_round=last_buffer_round,
            is_last=is_last,
        )

    def _placement_of(self, geometry: "_Geometry") -> "_Placement":
        """The short circuit a window is sliced on, and the move to it.

        Its faults lie in the rounds from lowest to highest. The first
        repeated-only round is the first whose faults, and the round
        before's, are a repeated round's (the template's prefix holds
        the rounds a later round reads the first round from).
        """
        lowest = geometry.start_round - self.span
        highest = geometry.last_buffer_round + self.span
        first_repeated_only = self.template.prefix_rounds + 2
        final = self.final_round_count
        near_end = final is not None and highest >= final
        if lowest <= first_repeated_only:
            return _unmoved_placement(highest, final, near_end)
        local_start = first_repeated_only + self.span + 1
        shift = geometry.start_round - local_start
        if near_end:
            local_final = final - shift
            return _Placement(local_final, shift)
        local_highest = highest - shift
        past_highest = local_highest + 2
        return _Placement(past_highest, shift)

    def _sliced_alone(
        self, round_count: int, local: "_Geometry"
    ) -> fault_models.WindowErrorModel:
        """A weak window on its short circuit, named; its priors before it."""
        frame = self.frames.frame(round_count)
        priors = frame.faults_before(local.first_commit_round)
        model = frame.slicer.slice_window_alone(
            local.start_round,
            local.first_commit_round,
            local.last_commit_round,
            local.last_buffer_round,
            fault_exclusion_ranges=(),
            explicitly_prior_faults=priors,
            is_last=local.is_last,
        )
        return frame.named(model)

    def _moved(
        self, model: fault_models.WindowErrorModel, shift: int
    ) -> fault_models.WindowErrorModel:
        """A named short-circuit model moved later by shift rounds."""
        if shift == 0:
            return model
        detectors = shift * self.detectors_per_round
        move = _Move(rounds=shift, detectors=detectors, step=self.template.step)
        return _moved_model(model, move)


@dataclasses.dataclass(frozen=True)
class _Geometry:
    """A window's rounds as the slicer reads them."""

    start_round: int
    first_commit_round: int
    last_commit_round: int
    last_buffer_round: int
    is_last: bool

    def moved(self, rounds: int) -> "_Geometry":
        """The same window, rounds later."""
        start_round = self.start_round + rounds
        first_commit_round = self.first_commit_round + rounds
        last_commit_round = self.last_commit_round + rounds
        last_buffer_round = self.last_buffer_round + rounds
        return dataclasses.replace(
            self,
            start_round=start_round,
            first_commit_round=first_commit_round,
            last_commit_round=last_commit_round,
            last_buffer_round=last_buffer_round,
        )


@dataclasses.dataclass(frozen=True)
class _Placement:
    """The rounds of a window's short circuit, and the rounds it moves."""

    round_count: int
    shift: int


@dataclasses.dataclass(frozen=True)
class _Move:
    """How far a model moves: rounds, detector ids, coordinates a round."""

    rounds: int
    detectors: int
    step: tuple


class _Frame:
    """One short circuit's slicer, and its faults' names."""

    def __init__(
        self,
        program: circuit_records.RepeatedStimCircuit,
        round_count: int,
        requirement: fault_models.DecoderFaultModelRequirement,
        shapes: "_Shapes",
    ) -> None:
        circuit, measurement_rounds = program.assemble(round_count)
        table = detector_formation.build_formation_table(
            circuit, round_count, measurement_rounds=measurement_rounds
        )
        detector_rounds = table.detector_rounds()
        self.slicer = window_slicer.WindowSlicer(
            circuit,
            round_count=round_count,
            detector_rounds=detector_rounds,
            fault_model_requirement=requirement,
        )
        self.names_by_representation = shapes.names_of(self.slicer)
        self.index_by_name = {
            representation: _positions(names)
            for representation, names in self.names_by_representation.items()
        }

    def faults_before(self, round_index: int) -> dict:
        """Every fault touching a round before this one, by representation."""
        fault_index = self.slicer.fault_index
        return {
            representation: _touching_before(by_round, round_index)
            for representation, by_round in fault_index.faults_by_round.items()
        }

    def named(
        self, model: fault_models.WindowErrorModel
    ) -> fault_models.WindowErrorModel:
        """The model with each column's catalog position as its name."""
        graphlike = self._named_faults(model.graphlike_faults)
        physical = self._named_faults(model.physical_faults)
        return dataclasses.replace(
            model, graphlike_faults=graphlike, physical_faults=physical
        )

    def local_faults(
        self, prior_faults: Optional[dict], shift: int, anchors: tuple
    ) -> Optional[dict]:
        """Named priors as this circuit's positions, the ones it can hold.

        A named fault whose earliest round lies outside anchors, moved
        here, touches no round of the window and is let go; one inside
        must be a fault of this circuit.
        """
        if prior_faults is None:
            return None
        name_shift = shift * NAME_STRIDE
        local = {}
        for representation, names in prior_faults.items():
            index_by_name = self.index_by_name[representation]
            local[representation] = _local_positions(
                index_by_name, names, name_shift, anchors
            )
        return local

    def _named_faults(
        self, faults: Optional[fault_models.PlacedFaultModel]
    ) -> Optional[fault_models.PlacedFaultModel]:
        if faults is None:
            return None
        names = self.names_by_representation[faults.representation]
        named_ids = tuple(names[index] for index in faults.source_fault_ids)
        return dataclasses.replace(faults, source_fault_ids=named_ids)


class _Frames:
    """The short circuits of one program, by their rounds."""

    def __init__(
        self,
        program: circuit_records.RepeatedStimCircuit,
        requirement: fault_models.DecoderFaultModelRequirement,
    ) -> None:
        self.program = program
        self.requirement = requirement
        self.shapes = _Shapes()
        self.recent = _Recent(FRAME_LIMIT)

    def frame(self, round_count: int) -> _Frame:
        """The short circuit of round_count rounds, its last one final."""
        return self.recent.get(
            round_count,
            lambda: _Frame(
                self.program, round_count, self.requirement, self.shapes
            ),
        )


class _Shapes:
    """A number for every fault shape a stream's circuits hold."""

    def __init__(self) -> None:
        self.number_by_shape: dict = {}

    def names_of(self, slicer: window_slicer.WindowSlicer) -> dict:
        """Each catalog fault's name, by representation."""
        chronology = slicer.chronology
        first_detector_by_round = {
            round_index: min(detectors)
            for round_index, detectors in chronology.detectors_by_round.items()
        }
        catalogs = slicer.catalogs
        reading = _ShapeReading(catalogs, first_detector_by_round)
        names = {}
        for representation, catalog in catalogs.by_representation.items():
            fault_rounds = slicer.fault_index.fault_rounds[representation]
            names[representation] = tuple(
                self._name(reading, catalog, fault_rounds, index)
                for index in range(len(fault_rounds))
            )
        return names

    def _name(
        self,
        reading: "_ShapeReading",
        catalog: fault_models.FaultCatalog,
        fault_rounds: tuple,
        index: int,
    ) -> int:
        """The fault's earliest round and its shape's number, as one name."""
        rounds = fault_rounds[index]
        anchor = min(rounds, default=0)
        shape = reading.shape(catalog, index, anchor)
        shape_count = len(self.number_by_shape)
        number = self.number_by_shape.setdefault(shape, shape_count)
        if number >= NAME_STRIDE:
            raise RuntimeError("a stream's fault shapes outgrew their names")
        round_part = anchor * NAME_STRIDE
        return round_part + number


@dataclasses.dataclass(frozen=True)
class _ShapeReading:
    """What reading a fault's shape needs: the catalogs, each round's base."""

    catalogs: fault_models.FaultCatalogs
    first_detector_by_round: dict

    def shape(
        self, catalog: fault_models.FaultCatalog, index: int, anchor: int
    ) -> tuple:
        """The fault's detectors from its round's first, and the rest."""
        representation = catalog.representation
        base = self.first_detector_by_round.get(anchor, 0)
        detectors = _less(catalog.detector_sets[index], base)
        observables = catalog.observable_sets[index]
        components = self._components(representation, index, base)
        return (representation, detectors, observables, components)

    def _components(
        self,
        representation: fault_models.FaultRepresentation,
        index: int,
        base: int,
    ) -> tuple:
        """A physical fault's graphlike parts, read from the same base."""
        link = self.catalogs.link
        if link is None:
            return ()
        if representation is not fault_models.FaultRepresentation.PHYSICAL:
            return ()
        graphlike = self.catalogs.by_representation[
            fault_models.FaultRepresentation.GRAPHLIKE
        ]
        column = link.getcol(index)
        column = column.tocoo()
        components = []
        for component in column.row:
            detectors = _less(graphlike.detector_sets[component], base)
            observables = graphlike.observable_sets[component]
            components.append((detectors, observables))
        return tuple(sorted(components))


class _Recent:
    """The last few things made, by key; the least recently used let go."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.by_key: collections.OrderedDict = collections.OrderedDict()

    def get(self, key: Hashable, make: Callable) -> object:
        """The thing made for the key, made now when not kept."""
        if key in self.by_key:
            self.by_key.move_to_end(key)
            return self.by_key[key]
        made = make()
        self.by_key[key] = made
        if len(self.by_key) > self.limit:
            self.by_key.popitem(last=False)
        return made


def _unmoved_placement(
    highest: int, final: Optional[int], near_end: bool
) -> _Placement:
    """A window near the first round, on a circuit of the stream's rounds.

    Near the final round it is the stream's own circuit; otherwise the
    circuit ends two rounds past the window's last fault, so its final
    round reaches none of them.
    """
    if near_end:
        return _Placement(final, 0)
    past_highest = highest + 2
    return _Placement(past_highest, 0)


def _checked_span(
    slicer: window_slicer.WindowSlicer,
    template: periodic_formation.StreamTemplate,
) -> int:
    """The most rounds one fault reaches past its earliest, on the template."""
    span = 0
    for fault_rounds in slicer.fault_index.fault_rounds.values():
        for rounds in fault_rounds:
            reach = max(rounds, default=0) - min(rounds, default=0)
            span = max(span, reach)
    if span >= template.round_count - 1:
        raise ValueError(
            f"not implemented yet: a fault of this program reaches {span} "
            "rounds past its earliest, more than its template shows"
        )
    return span


def _moved_ranges(ranges: tuple, rounds: int) -> tuple:
    moved = []
    for first, last in ranges:
        moved_first = first + rounds
        moved_last = last + rounds
        moved.append((moved_first, moved_last))
    return tuple(moved)


def _touching_before(by_round: dict, round_index: int) -> set:
    """The faults touching any round before this one."""
    faults = set()
    for touched_round, touching in by_round.items():
        if touched_round < round_index:
            faults.update(touching)
    return faults


def _local_positions(
    index_by_name: dict, names: frozenset, name_shift: int, anchors: tuple
) -> set:
    """The names whose earliest round lies in anchors, as positions."""
    lowest, highest = anchors
    positions = set()
    for name in names:
        local_name = name - name_shift
        anchor = local_name // NAME_STRIDE
        if not lowest <= anchor <= highest:
            continue
        position = _position_of(index_by_name, local_name)
        positions.add(position)
    return positions


def _positions(names: tuple) -> dict:
    return {name: position for position, name in enumerate(names)}


def _position_of(index_by_name: dict, name: int) -> int:
    position = index_by_name.get(name)
    if position is None:
        raise RuntimeError("committed faults changed during stream growth")
    return position


def _less(detectors: tuple, base: int) -> tuple:
    return tuple(detector - base for detector in detectors)


def _moved_model(
    model: fault_models.WindowErrorModel, move: _Move
) -> fault_models.WindowErrorModel:
    """Every detector id, round, coordinate and name moved; the rest kept."""
    detector_ids = _plus(model.detector_ids, move.detectors)
    coordinates = _moved_coordinate_rows(model.detector_coordinates, move)
    defect_positions = _moved_positions(model.defect_positions, move)
    first_commit_round = model.first_commit_round + move.rounds
    graphlike = _moved_faults(model.graphlike_faults, move)
    physical = _moved_faults(model.physical_faults, move)
    detector_bases = _moved_keys(model.detector_bases, move.detectors)
    return dataclasses.replace(
        model,
        detector_ids=detector_ids,
        detector_coordinates=coordinates,
        defect_positions=defect_positions,
        first_commit_round=first_commit_round,
        graphlike_faults=graphlike,
        physical_faults=physical,
        detector_bases=detector_bases,
    )


def _moved_faults(
    faults: Optional[fault_models.PlacedFaultModel], move: _Move
) -> Optional[fault_models.PlacedFaultModel]:
    if faults is None:
        return None
    name_shift = move.rounds * NAME_STRIDE
    names = _plus(faults.source_fault_ids, name_shift)
    boundary_flips = _moved_flips(faults.boundary_flips, move.detectors)
    return dataclasses.replace(
        faults, source_fault_ids=names, boundary_flips=boundary_flips
    )


def _moved_flips(boundary_flips: Mapping, detectors: int) -> dict:
    """Each owned column's detectors, moved in one array operation.

    A window hands off thousands of short tuples, so they are moved as
    one flat array and cut back into tuples.
    """
    every_flips = boundary_flips.values()
    flat = itertools.chain.from_iterable(every_flips)
    flat_flips = tuple(flat)
    moved = _plus(flat_flips, detectors)
    moved_flips = {}
    start = 0
    for column, flips in boundary_flips.items():
        end = start + len(flips)
        moved_flips[column] = moved[start:end]
        start = end
    return moved_flips


def _moved_coordinate_rows(
    rows: Optional[tuple], move: _Move
) -> Optional[tuple]:
    if rows is None:
        return None
    return tuple(
        periodic_formation.moved_coordinates(row, move.rounds, move.step)
        for row in rows
    )


def _moved_positions(positions: Mapping, move: _Move) -> dict:
    """Each detector's (round, position in round), moved."""
    moved = {}
    for detector_id, (round_index, position) in positions.items():
        moved_detector = detector_id + move.detectors
        moved_round = round_index + move.rounds
        moved[moved_detector] = (moved_round, position)
    return moved


def _moved_keys(
    by_detector: Optional[Mapping], detectors: int
) -> Optional[dict]:
    if by_detector is None:
        return None
    return {
        detector_id + detectors: value
        for detector_id, value in by_detector.items()
    }


def _plus(values: tuple, amount: int) -> tuple:
    array = numpy.asarray(values, dtype=numpy.int64)
    moved = array + amount
    moved_list = moved.tolist()
    return tuple(moved_list)

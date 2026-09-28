"""decsim's Relay-BP as a sinter decoder, built from the point's circuit.

sinter hands a decoder only the detector error model it made with
decompose_errors=True (sinter/_collection/_collection_worker_state.py:
28-33). Joined at its `^` separators, that model holds the row's
physical faults, but in another column order, with priors a rounding
apart, and with no detector's X or Z type. The row draws its gamma table
per column and splits X from Z by each detector's type on the circuit
(detector_error_model/basis_split.py), so the adapter is built from the
circuit. It builds the machine's one window over the whole circuit
(window_model_builders), splits it as the row splits a window with
bases apart (strong_backend.part_jobs), decodes each part with the row's
own call, and joins the parts' observables by XOR
(strong_backend.joined_result), so an offline run answers as the row.

A shot whose backend produced no correction, which the machine leaves
unscored, is answered with the empty correction the row commits, since
sinter scores every shot.
"""

import dataclasses
from collections.abc import Mapping

import numpy
import sinter
import stim

import decsim.decoders.decoder as decoder_module
import decsim.decoders.relay_belief_propagation.decoder as relay_decoder
import decsim.detector_error_model.basis_split as basis_split
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.detector_error_model.window_model_builders as window_builders
import decsim.tables as tables

# the name a refused settings key is reported under
SETTINGS_NAME = "relay_bp_adapter"


class RelayBeliefPropagationDecoder(sinter.Decoder):
    """The factory sinter pickles to each worker, for one point's circuit.

    settings holds the relay_bp row's own yaml keys, read and checked by
    the row's Settings (decoders/relay_belief_propagation/decoder.py).
    seed is the seed the row's gamma table is drawn from, so every
    worker of a point decodes with one table and a rerun repeats it.
    """

    def __init__(
        self, circuit: stim.Circuit, settings: Mapping, seed: int
    ) -> None:
        self.circuit = circuit
        # the row reads no clock (its Settings.from_yaml deletes it)
        self.settings = tables.row_settings(
            relay_decoder.RelayBeliefPropagationDecoder,
            SETTINGS_NAME,
            settings,
            (),
            None,
            SETTINGS_NAME,
        )
        self.seed = seed

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> "CompiledRelayBeliefPropagationDecoder":
        """The row and its window's parts; sinter's dem is not read."""
        del dem
        row = relay_decoder.RelayBeliefPropagationDecoder(
            settings=self.settings
        )
        window_decoder = row.window_decoder
        reservation = window_decoder.reserve_run_seed(self.seed)
        window_decoder.commit_run_seed(reservation)
        model = _whole_circuit_window(self.circuit, row.fault_model_requirement)
        parts = _parts(model, row.splits_by_basis)
        return CompiledRelayBeliefPropagationDecoder(row, model, parts)


class CompiledRelayBeliefPropagationDecoder(sinter.CompiledDecoder):
    """One point's row, decoding bit-packed shots one at a time.

    relay-bp's decode_batch runs decode on each shot in turn
    (crates/relay_bp/src/decoder.rs:52-58), so the row's own call per
    shot costs the same and keeps the row's handling of each answer.
    """

    def __init__(
        self,
        row: relay_decoder.RelayBeliefPropagationDecoder,
        model: fault_models.WindowErrorModel,
        parts: list,
    ) -> None:
        self.row = row
        self.detector_count = len(model.detector_ids)
        self.observable_count = model.physical_faults.observables.shape[0]
        self.parts = parts

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: numpy.ndarray
    ) -> numpy.ndarray:
        """Each shot's predicted observable flips, bit packed as sinter's."""
        events = numpy.unpackbits(
            bit_packed_detection_event_data,
            axis=1,
            count=self.detector_count,
            bitorder="little",
        )
        shot_count = events.shape[0]
        shape = (shot_count, self.observable_count)
        predictions = numpy.zeros(shape, dtype=numpy.uint8)
        for part in self.parts:
            part_events = events[:, part.rows]
            for shot, syndrome in enumerate(part_events):
                flips = self._observable_flips(part, syndrome)
                predictions[shot] ^= flips
        return numpy.packbits(predictions, axis=1, bitorder="little")

    def _observable_flips(self, part: "_Part", syndrome) -> numpy.ndarray:
        """The observables one part's correction flips, as the row reads them.

        The whole-circuit window owns every column, so every selected
        fault is committed (decoders/decoder.py result_from_selected_faults).
        """
        faults = part.model.physical_faults
        window_decode = self.row.decode_window(
            self.row.window_decoder, part.model, faults, syndrome
        )
        selected = numpy.asarray(window_decode.selected_faults)
        flips = decoder_module.parity_product(faults.observables, selected)
        return flips.astype(numpy.uint8)


@dataclasses.dataclass(frozen=True)
class _Part:
    """One part's window model and the whole window's rows it reads."""

    model: fault_models.WindowErrorModel
    rows: list


def _whole_circuit_window(
    circuit: stim.Circuit,
    requirement: fault_models.DecoderFaultModelRequirement,
) -> fault_models.WindowErrorModel:
    """The machine's one window over the whole circuit.

    The row reads no round, so the circuit is declared one round and the
    window that round: its rows are the detectors in id order and its
    columns the physical catalog in order, every one owned, as in the
    naive_online window the machine lays (windows/schemes/naive_online.py).
    """
    one_round = dict.fromkeys(range(circuit.num_detectors), 1)
    whole_window = (1, 1, 1, 1)
    (model,) = window_builders.build_window_error_models(
        circuit,
        [whole_window],
        round_count=1,
        detector_rounds=one_round,
        fault_model_requirement=requirement,
        fault_exclusion_ranges=(),
    )
    return model


def _parts(model: fault_models.WindowErrorModel, splits_by_basis: bool) -> list:
    """The window whole, or its X part and Z part with the rows of each."""
    if not splits_by_basis:
        rows = list(range(len(model.detector_ids)))
        return [_Part(model, rows)]
    parts = []
    part_models = basis_split.split_by_basis(model)
    for basis, part_model in part_models.items():
        rows = basis_split.rows_of_basis(model, basis)
        part = _Part(part_model, rows)
        parts.append(part)
    return parts

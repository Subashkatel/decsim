"""The Tesseract adapter: the referee's decoder as a tier of its own.

Physical fault mechanisms decoded by the official backend; the window
decoder module is the backend this row compiles.
"""

import dataclasses
from typing import Optional

import numpy

import decsim.config as config
import decsim.decoders.backend_outcome as backend_outcome
import decsim.decoders.decoder as decoder_module
import decsim.decoders.tesseract.window_decoder as window_decoder
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models
import decsim.records.seeds as seed_records


class TesseractDecoder(decoder_module.WindowDecoderBase):
    """Decode physical fault mechanisms with the official Tesseract backend."""

    fault_model_requirement = fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    fault_representation = fault_models.FaultRepresentation.PHYSICAL
    backend_is_seeded = True

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The Tesseract row's settings: its search.

        detector_beam, beam_climbing, no_revisit_detectors,
        priority_queue_limit and merge_errors are tesseract_decoder's
        TesseractConfig det_beam, beam_climbing, no_revisit_dets, pqlimit
        and merge_errors;
        detector_order_count and detector_order_method are the
        num_det_orders and DetOrder its utils.build_det_orders takes.
        The defaults are the package's tesseract-short-beam profile
        (tesseract-decoder src/tesseract_sinter_compat.pybind.h, the
        profile the Tesseract paper 2503.10988 runs); its
        tesseract-long-beam is beam 20, queue 1,000,000 and 21 orders,
        and both profiles merge errors and fix the order seed at 2384753.
        merge_errors off by default keeps every physical column a search
        choice of its own; on, the backend searches one error per set of
        columns with the same detectors and observables and answers with
        the set's first column (src/tesseract.cc:153-168, 461-465), so
        the correction still names physical columns and flips the same
        detectors and observables. detector_order_seed is the seed
        build_det_orders draws the orders from; None draws them from the
        run seed (decsim/seeding.py).
        """

        detector_beam: int = 15
        beam_climbing: bool = True
        no_revisit_detectors: bool = True
        priority_queue_limit: int = 200_000
        detector_order_method: str = "index"
        detector_order_count: int = 16
        detector_order_seed: Optional[int] = None
        merge_errors: bool = False
        # the word the reports name this row by
        name = "tesseract"

        def __post_init__(self) -> None:
            for key, (unit, minimum) in _COUNT_KEYS.items():
                value = getattr(self, key)
                config.check_whole_count(key, value, unit, minimum)
            _check_detector_order_seed(self.detector_order_seed)

        def build(self) -> "TesseractDecoder":
            """A fresh decoder of these settings."""
            return TesseractDecoder(settings=self)

    def __init__(
        self,
        latency_model: Optional[decoder_module.DecoderBase] = None,
        settings: Optional["TesseractDecoder.Settings"] = None,
    ) -> None:
        decoder_module.WindowDecoderBase.__init__(self, latency_model)
        if settings is None:
            settings = TesseractDecoder.Settings()
        self.compile_key = (TesseractDecoder, settings)
        self.window_decoder = window_decoder.TesseractWindowDecoder(settings)

    def run_seed_children(self) -> tuple:
        """The timing and detector-order seed owners by semantic role.

        A fixed detector_order_seed draws the orders from itself and
        never from the run seed, so the window decoder owns no run seed
        then.
        """
        latency_path = (
            seed_records.RunSeedPathSegment("field", "latency_model"),
        )
        timing = seed_records.RunSeedChild(latency_path, self.latency_model)
        if self.window_decoder.settings.detector_order_seed is not None:
            return (timing,)
        decoder_path = (
            seed_records.RunSeedPathSegment("field", "window_decoder"),
        )
        orders = seed_records.RunSeedChild(decoder_path, self.window_decoder)
        return (timing, orders)

    def compile(
        self,
        faults: fault_models.PlacedFaultModel,
        model: fault_models.WindowErrorModel,
    ) -> window_decoder.TesseractWindowDecoder:
        """The window decoder, with this model's backend already built.

        The build is setup, outside the timed decode, as Tesseract's own
        benchmark builds its decoder outside the timer (tesseract-decoder
        src/tesseract_main.cc:568-579).
        """
        del faults
        self.window_decoder.prepare(model)
        return self.window_decoder

    def decode_window(
        self,
        backend: window_decoder.TesseractWindowDecoder,
        model: fault_models.WindowErrorModel,
        faults: fault_models.PlacedFaultModel,
        syndrome: numpy.ndarray,
    ) -> decoding_records.WindowDecode:
        """One backend call; a produced correction is committed as it stands."""
        outcome = backend.decode(model, syndrome)
        fault_count = faults.check.shape[1]
        return backend_outcome.window_decode_of(outcome, fault_count)


# each whole-number key, the unit its refusal names and its least value
_COUNT_KEYS = {
    "detector_beam": ("detection events", 0),
    "priority_queue_limit": ("search states", 1),
    "detector_order_count": ("orders", 1),
}

# build_det_orders takes its seed as a uint64 (tesseract-decoder
# src/utils.h:42-45)
_LARGEST_SEED = 2**64 - 1


def _check_detector_order_seed(seed) -> None:
    """None, or a whole number build_det_orders can take as its seed."""
    if seed is None:
        return
    if config.is_whole_count(seed, 0) and seed <= _LARGEST_SEED:
        return
    raise ValueError(
        "detector_order_seed must be None or a whole number from 0 to "
        f"{_LARGEST_SEED} (got {seed!r})"
    )

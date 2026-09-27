"""The Tesseract adapter: the referee's decoder as a tier of its own.

Physical fault mechanisms decoded by the official backend; the window
decoder module is the backend this row compiles.
"""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import decsim.config as config
import decsim.decoders.backend_outcome as backend_outcome
import decsim.decoders.decoder as decoder_module
import decsim.decoders.tesseract.window_decoder as window_decoder
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.seeds as seed_records
import decsim.tables as tables


class TesseractDecoder(decoder_module.WindowDecoderBase):
    """Decode physical fault mechanisms with the official Tesseract backend."""

    fault_model_requirement = fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    fault_representation = fault_models.FaultRepresentation.PHYSICAL

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The row's own keys in its tier section: Tesseract's search.

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

        @classmethod
        def from_yaml(
            cls,
            section: Mapping,
            clocks: config.ClockSettings,
            section_name: str,
        ) -> "TesseractDecoder.Settings":
            """Every key, checked where it enters; absent is the default.

            section_name is the tier section the row sits in, which a
            refusal names.
            """
            del clocks
            counts = config.whole_counts(
                section, section_name, _COUNT_KEYS, _DEFAULTS
            )
            switches = _switches(section, section_name)
            order_method = _detector_order_method(section, section_name)
            order_seed = _detector_order_seed(section, section_name)
            return cls(
                detector_order_method=order_method,
                detector_order_seed=order_seed,
                **counts,
                **switches,
            )

    def __init__(
        self,
        latency_model: Optional[decoder_module.DecoderBase] = None,
        settings: Optional["TesseractDecoder.Settings"] = None,
    ) -> None:
        decoder_module.WindowDecoderBase.__init__(self, latency_model)
        if settings is None:
            settings = TesseractDecoder.Settings()
        self.window_decoder = window_decoder.TesseractWindowDecoder(settings)

    def run_seed_children(self) -> tuple:
        """The timing and detector-order seed owners by semantic role."""
        latency_path = (
            seed_records.RunSeedPathSegment("field", "latency_model"),
        )
        decoder_path = (
            seed_records.RunSeedPathSegment("field", "window_decoder"),
        )
        return (
            seed_records.RunSeedChild(latency_path, self.latency_model),
            seed_records.RunSeedChild(decoder_path, self.window_decoder),
        )

    def compile(self, faults, model):
        """The window decoder, which compiles the backend per model itself."""
        del faults
        del model
        return self.window_decoder

    def decode_window(self, backend, model, faults, syndrome):
        """One backend call; a produced correction is committed as it stands."""
        outcome = backend.decode(model, syndrome)
        fault_count = faults.check.shape[1]
        return backend_outcome.window_decode_of(outcome, fault_count)


# the value a key the section leaves out takes
_DEFAULTS = TesseractDecoder.Settings()

# each whole-number key, the unit its refusal names and its least value
_COUNT_KEYS = {
    "detector_beam": ("detection events", 0),
    "priority_queue_limit": ("search states", 1),
    "detector_order_count": ("orders", 1),
}

# build_det_orders takes its seed as a uint64 (tesseract-decoder
# src/utils.h:42-45)
_LARGEST_SEED = 2**64 - 1

# the on-or-off keys
_SWITCH_KEYS = ("beam_climbing", "no_revisit_detectors", "merge_errors")


def _switches(section: Mapping, section_name: str) -> dict:
    """Beam climbing, no-revisit and merging, each true or false."""
    switches = {}
    for key in _SWITCH_KEYS:
        default = getattr(_DEFAULTS, key)
        switches[key] = config.boolean(section, section_name, key, default)
    return switches


def _detector_order_method(section: Mapping, section_name: str) -> str:
    """A row of window_decoder.DETECTOR_ORDER_METHODS, index by default."""
    method = section.get("detector_order_method", "index")
    key = f"{section_name}.detector_order_method"
    tables.row(window_decoder.DETECTOR_ORDER_METHODS, key, method)
    return method


def _detector_order_seed(section: Mapping, section_name: str):
    """None, or a whole number build_det_orders can take as its seed."""
    seed = section.get("detector_order_seed")
    if seed is None:
        return None
    is_whole = isinstance(seed, int) and not isinstance(seed, bool)
    if is_whole and 0 <= seed <= _LARGEST_SEED:
        return seed
    raise ValueError(
        f"{section_name}.detector_order_seed must be null or a whole number "
        f"from 0 to {_LARGEST_SEED} (got {seed!r})"
    )

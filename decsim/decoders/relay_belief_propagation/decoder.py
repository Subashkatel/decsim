"""The Relay-BP adapter: corrections from relay-bp, time from a model."""

import dataclasses
import math
from typing import Optional

import decsim.config as config
import decsim.decoders.backend_outcome as backend_outcome
import decsim.decoders.decoder as decoder_module
import decsim.decoders.relay_belief_propagation.window_decoder as window_decoder
import decsim.decoders.strong_backend as strong_backend
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoding as decoding_records
import decsim.records.seeds as seed_records


class RelayBeliefPropagationDecoder(decoder_module.WindowDecoderBase):
    """Use Relay-BP for corrections and an injected model for service time."""

    fault_model_requirement = fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    fault_representation = fault_models.FaultRepresentation.PHYSICAL
    backend_is_seeded = True

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The row's own keys in its tier section.

        Eight keys are relay-bp 0.2.2's RelayDecoderF32 arguments: alpha
        (null leaves relay-bp its own choice) and
        alpha_iteration_scaling_factor; gamma0, the first leg's memory
        strength; pre_iterations, pre_iter, the first leg's iteration
        limit T0; relay_set_count, num_sets, the legs after the first,
        so the paper's leg count R is relay_set_count + 1;
        iterations_per_set, set_max_iter, each later leg's limit Tr;
        gamma_interval, gamma_dist_interval, the range each later leg's
        memory strengths are drawn from; converged_solution_count,
        stop_nconv, the solutions S sought before stopping (Mueller et
        al. 2506.01779 lines 255-260). Their defaults are RelayDecoderF32's
        own, and the interval SinterDecoder_RelayBP's
        (relay_bp/stim/sinter/decoders.py), the paper's gross-code
        interval (line 332); T0 80 and Tr 60 are the paper's (lines
        304-305). The paper's surface code values are gamma0 0.35 (line
        307), the interval [-0.254, 0.985] (line 332), and Relay-BP-1,
        R 301 and S 1, or Relay-BP-5, R 601 and S 5 (line 343). The
        gamma table is drawn from the run seed (decsim/seeding.py), so
        no key sets it.

        bases names a row of strong_backend.BASIS_DECODES, read as the
        measured_table row reads it: together decodes the window's X and
        Z detectors as one problem, the paper's XYZ-decoding; apart
        decodes them as two, the paper's default XZ-decoding (lines
        289-298), cut and joined by the strong backend's own split
        (strong_backend.part_jobs and joined_result).
        """

        alpha: Optional[float] = None
        alpha_iteration_scaling_factor: float = 1.0
        gamma0: float = 0.1
        pre_iterations: int = 80
        relay_set_count: int = 300
        iterations_per_set: int = 60
        gamma_interval: tuple[float, float] = (-0.24, 0.66)
        converged_solution_count: int = 1
        bases: str = "together"
        # the word the reports name this row by
        name = "relay_bp"

        def __post_init__(self) -> None:
            """Every value checked, each real number held as a float."""
            for key, (unit, minimum) in _COUNT_KEYS.items():
                value = getattr(self, key)
                config.check_whole_count(key, value, unit, minimum)
            for key in _REAL_KEYS:
                _hold_as_float(self, key)
            if self.alpha is not None:
                _hold_as_float(self, "alpha")
            interval = _ordered_interval(self.gamma_interval)
            object.__setattr__(self, "gamma_interval", interval)

        def build(self) -> "RelayBeliefPropagationDecoder":
            """A fresh decoder of these settings."""
            return RelayBeliefPropagationDecoder(settings=self)

    def __init__(
        self,
        latency_model: Optional[decoder_module.DecoderBase] = None,
        settings: Optional["RelayBeliefPropagationDecoder.Settings"] = None,
    ) -> None:
        decoder_module.WindowDecoderBase.__init__(self, latency_model)
        if settings is None:
            settings = RelayBeliefPropagationDecoder.Settings()
        self.compile_key = (RelayBeliefPropagationDecoder, settings)
        self.window_decoder = (
            window_decoder.RelayBeliefPropagationWindowDecoder(settings)
        )
        self.splits_by_basis = strong_backend.BASIS_DECODES[settings.bases]
        if self.splits_by_basis:
            row_requirement = (
                RelayBeliefPropagationDecoder.fault_model_requirement
            )
            self.fault_model_requirement = row_requirement.joined(
                fault_models.DETECTOR_BASES_REQUIRED
            )

    def run_seed_children(self) -> tuple:
        """The timing and fixed-gamma owners at stable semantic paths."""
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

    def decode_timed(self, job: decoding_records.DecodeJob) -> tuple:
        """(result, nanoseconds of the backend calls), apart or together.

        Apart, the window's X part and Z part are each decoded as a
        window of their own and the answers joined; one weak unit runs
        the two calls one after the other, so its time is their sum.
        """
        if not self.splits_by_basis:
            return decoder_module.WindowDecoderBase.decode_timed(self, job)
        parts = strong_backend.part_jobs(job)
        results = {}
        elapsed_nanoseconds = 0
        for basis, part in parts.items():
            result, part_nanoseconds = (
                decoder_module.WindowDecoderBase.decode_timed(self, part)
            )
            results[basis] = result
            elapsed_nanoseconds += part_nanoseconds
        joined = strong_backend.joined_result(job, results)
        return joined, elapsed_nanoseconds

    def compile(self, faults, model):
        """The window decoder, its backend for this model built untimed."""
        del model
        self.window_decoder.compiled_model(faults)
        return self.window_decoder

    def decode_window(self, backend, model, faults, syndrome):
        """One backend call; a produced correction is committed as it stands."""
        outcome = backend.decode(model, syndrome)
        fault_count = faults.check.shape[1]
        return backend_outcome.window_decode_of(outcome, fault_count)


# each whole-number key, the unit its refusal names and its least value.
# A first leg of zero iterations is refused: relay-bp runs its first leg
# for pre_iter iterations and keeps the previous call's decoding when
# that loop never runs (relay.rs decode_inner), so every window would
# get a stale correction. The later legs may be none at all.
_COUNT_KEYS = {
    "pre_iterations": ("iterations", 1),
    "relay_set_count": ("legs", 0),
    "iterations_per_set": ("iterations", 0),
    "converged_solution_count": ("solutions", 1),
}

# the keys that are one finite real number each; alpha may also be None
_REAL_KEYS = ("alpha_iteration_scaling_factor", "gamma0")


def _hold_as_float(settings, key: str) -> None:
    """A finite real number, held as a float; a flag or a text is refused."""
    value = getattr(settings, key)
    if not _is_finite_number(value):
        raise ValueError(f"{key} must be a finite real number (got {value!r})")
    object.__setattr__(settings, key, float(value))


def _ordered_interval(interval) -> tuple:
    """(low, high), two finite real numbers, low below high, as floats.

    relay-bp draws each later leg's memory strengths uniformly from the
    interval and panics on an empty one (rand's Uniform::new, low >= high).
    """
    if _is_ordered_pair(interval):
        low, high = interval
        return (float(low), float(high))
    raise ValueError(
        "gamma_interval must be [low, high], two finite real numbers with "
        f"low below high (got {interval!r})"
    )


def _is_ordered_pair(interval) -> bool:
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return False
    low, high = interval
    if not _is_finite_number(low) or not _is_finite_number(high):
        return False
    return low < high


def _is_finite_number(value) -> bool:
    if not config.is_number(value):
        return False
    return math.isfinite(value)

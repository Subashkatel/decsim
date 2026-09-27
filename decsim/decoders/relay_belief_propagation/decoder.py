"""The Relay-BP adapter: corrections from relay-bp, time from a model."""

import dataclasses
import math
from collections.abc import Mapping
from typing import Optional

import decsim.config as config
import decsim.decoders.backend_outcome as backend_outcome
import decsim.decoders.decoder as decoder_module
import decsim.decoders.relay_belief_propagation.window_decoder as window_decoder
import decsim.decoders.strong_backend as strong_backend
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoding as decoding_records
import decsim.records.seeds as seed_records
import decsim.tables as tables


class RelayBeliefPropagationDecoder(decoder_module.WindowDecoderBase):
    """Use Relay-BP for corrections and an injected model for service time."""

    fault_model_requirement = fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    fault_representation = fault_models.FaultRepresentation.PHYSICAL

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

        @classmethod
        def from_yaml(
            cls,
            section: Mapping,
            clocks: config.ClockSettings,
            section_name: str,
        ) -> "RelayBeliefPropagationDecoder.Settings":
            """Every key, checked where it enters; absent is the default.

            section_name is the tier section the row sits in, which a
            refusal names.
            """
            del clocks
            alpha = _alpha(section, section_name)
            scaling_key = "alpha_iteration_scaling_factor"
            scaling = _real(section, section_name, scaling_key)
            gamma0 = _real(section, section_name, "gamma0")
            gamma_interval = _gamma_interval(section, section_name)
            counts = _counts(section, section_name)
            bases = section.get("bases", cls.bases)
            bases_key = f"{section_name}.bases"
            tables.row(strong_backend.BASIS_DECODES, bases_key, bases)
            return cls(
                alpha=alpha,
                alpha_iteration_scaling_factor=scaling,
                gamma0=gamma0,
                gamma_interval=gamma_interval,
                bases=bases,
                **counts,
            )

    def __init__(
        self,
        latency_model: Optional[decoder_module.DecoderBase] = None,
        settings: Optional["RelayBeliefPropagationDecoder.Settings"] = None,
    ) -> None:
        decoder_module.WindowDecoderBase.__init__(self, latency_model)
        if settings is None:
            settings = RelayBeliefPropagationDecoder.Settings()
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
        """The window decoder, which compiles the backend per model itself."""
        del faults
        del model
        return self.window_decoder

    def decode_window(self, backend, model, faults, syndrome):
        """One backend call; a produced correction is committed as it stands."""
        del faults
        outcome = backend.decode(model, syndrome)
        return backend_outcome.window_decode_of(outcome)


# the value a key the section leaves out takes
_DEFAULTS = RelayBeliefPropagationDecoder.Settings()

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


def _alpha(section: Mapping, section_name: str) -> Optional[float]:
    """alpha, or None when the section leaves it null or out."""
    alpha = section.get("alpha")
    if alpha is None:
        return None
    return _real(section, section_name, "alpha")


def _real(section: Mapping, section_name: str, key: str) -> float:
    """A finite real number; a boolean or a string is refused."""
    default = getattr(_DEFAULTS, key)
    value = section.get(key, default)
    if _is_finite_number(value):
        return float(value)
    raise ValueError(
        f"{section_name}.{key} must be a finite real number (got {value!r})"
    )


def _counts(section: Mapping, section_name: str) -> dict:
    """The iteration limits, the later legs and the solutions sought."""
    counts = {}
    for key, (unit, minimum) in _COUNT_KEYS.items():
        default = getattr(_DEFAULTS, key)
        counts[key] = config.whole_count(
            section, section_name, key, default, unit, minimum
        )
    return counts


def _gamma_interval(section: Mapping, section_name: str) -> tuple:
    """[low, high], two finite real numbers, low below high.

    relay-bp draws each later leg's memory strengths uniformly from the
    interval and panics on an empty one (rand's Uniform::new, low >= high).
    """
    interval = section.get("gamma_interval", _DEFAULTS.gamma_interval)
    if _is_ordered_pair(interval):
        low, high = interval
        return (float(low), float(high))
    raise ValueError(
        f"{section_name}.gamma_interval must be [low, high], two finite "
        f"real numbers with low below high (got {interval!r})"
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

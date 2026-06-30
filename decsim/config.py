"""Central simulator configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
 
if TYPE_CHECKING:
    from .engine import Engine
    from .links import LinkModel
    from .protocols import CodeModel, Controller, Decoder, DecodingScheme

TICKS_PER_US = 1_000_000


def us(microseconds: float) -> int:
    """Convert microseconds to ticks."""
    return int(round(microseconds * TICKS_PER_US))


def fmt(ticks: int) -> str:
    """Format ticks as microseconds for readability in logs."""
    return f"{ticks / TICKS_PER_US:7.3f} us"


# Simple (alpha, beta) latency fits for the default LatencyModelDecoder
# (per-round time = alpha * nodes**beta). NOTE: a simple model for now; TODO swap
# in measured/cited latencies when the latency model is built out.
DECODER_FITS = {
    "cc_fpga": (2.85e-10, 1.2),
    "cc_asic": (5.53e-11, 1.34),
    "alphaqubit": (4.8e-6, 0.503),
    "pymatching": (5.91e-9, 1.17),
}

SCHEME_NAMES = ("sliding", "naive", "parallel")
SWITCH_MODES = ("none", "serial", "parallel")
IDLE_ROUND_MODES = ("ignore", "separate_decode_jobs", "extend_stream")
FEEDBACK_BOUNDARY_MODES = ("trailing_buffer", "measurement_closed")


def _scheme_registry() -> dict:
    """Return scheme classes by config name."""
    from .schemes import SlidingWindowScheme, NaiveOnlineScheme, ParallelWindowScheme
    return {"sliding": SlidingWindowScheme, "naive": NaiveOnlineScheme,
            "parallel": ParallelWindowScheme}


@dataclass(frozen=True)
class SimConfig:
    """All main simulator knobs in one place."""

    round_us: float = 1.1                 # Syndrome round duration.
    rounds_per_op: int = 11               # Default rounds for timing-only operations.
    num_units: int = 1                    # Default number of decoder workers.

    t_qc_us: float = 0.15                 # QPU to controller syndrome latency.
    t_cd_us: float = 2.0                  # Controller to decoder syndrome latency.
    t_dd_us: float = 0.5                  # Decoder to decoder boundary latency.
    t_do_us: float = 1.0                  # Decoder to orchestrator result latency.
    t_oc_us: float = 4.0                  # Orchestrator to controller feedback latency.
    t_cq_us: float = 0.15                 # Controller to QPU feedback latency.
    t_ws_us: Optional[float] = None       # Weak to strong decoder handoff latency.
    t_pack_us: float = 0.0                # Controller packet packing delay.

    decoder_model: Optional[str] = None   # Named latency fit, or None for alpha and beta.
    decoder_alpha: float = 2.85e-10       # Decoder latency scale when no named fit is used.
    decoder_beta: float = 1.2             # Decoder latency exponent when no named fit is used.

    switch_gamma: float = 0.0             # Probability scale for sampled decoder switching.
    switch_handoff_us: float = 0.5        # Extra handoff latency around a strong redo.
    switch_comm_weak_us: float = 0.0      # Communication cost paid on each weak decode.
    switch_seed: int = 0                  # Random seed for sampled switching decisions.
    switch_confidence_threshold: float = 0.0  # Minimum soft output to keep weak result.
    switch_mode: str = "none"             # none, serial, or parallel strong-decoder mode.
    switch_weak_keepup_ratio: Optional[float] = None  # Weak decoder time per syndrome round.
    switch_bulk_strong: bool = False      # Batch queued strong redos in serial mode.

    relaybp_iterations: int = 40          # Relay-BP iteration count.
    relaybp_t_iter_ns: float = 24.0       # Relay-BP time per iteration.

    scheme_name: str = "sliding"          # Window scheme selected by config.
    idle_round_mode: str = "ignore"       # ignore, separate_decode_jobs, or extend_stream.
    feedback_boundary_mode: str = "trailing_buffer"  # How final feedback windows close.

    def __post_init__(self) -> None:
        """Validate config values after dataclass initialization."""
        if self.round_us <= 0:
            raise ValueError(f"round_us must be > 0 (got {self.round_us})")
        if self.rounds_per_op < 1:
            raise ValueError(f"rounds_per_op must be >= 1 (got {self.rounds_per_op})")
        if self.num_units < 1:
            raise ValueError(f"num_units must be >= 1 (got {self.num_units})")
        for name in ("t_qc_us", "t_cd_us", "t_dd_us", "t_do_us", "t_oc_us", "t_cq_us",
                     "t_pack_us"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0 (got {getattr(self, name)})")
        if self.t_ws_us is not None and self.t_ws_us < 0:
            raise ValueError(f"t_ws_us must be >= 0 when set (got {self.t_ws_us})")
        if self.decoder_alpha < 0 or self.decoder_beta < 0:
            raise ValueError("decoder_alpha and decoder_beta must be >= 0")
        if self.decoder_model is not None and self.decoder_model not in DECODER_FITS:
            raise ValueError(f"decoder_model must be None or one of {sorted(DECODER_FITS)} "
                             f"(got {self.decoder_model!r})")
        if not 0.0 <= self.switch_gamma <= 1.0:
            raise ValueError(f"switch_gamma must be in [0, 1] (got {self.switch_gamma})")
        if self.switch_handoff_us < 0 or self.switch_comm_weak_us < 0:
            raise ValueError("switch_handoff_us and switch_comm_weak_us must be >= 0")
        if self.relaybp_iterations < 1:
            raise ValueError(f"relaybp_iterations must be >= 1 (got {self.relaybp_iterations})")
        if self.relaybp_t_iter_ns < 0:
            raise ValueError("relaybp_t_iter_ns must be >= 0")
        if self.scheme_name not in SCHEME_NAMES:
            raise ValueError(f"scheme_name must be one of {SCHEME_NAMES} (got {self.scheme_name!r})")
        if self.switch_mode not in SWITCH_MODES:
            raise ValueError(f"switch_mode must be one of {SWITCH_MODES} (got {self.switch_mode!r})")
        if self.switch_bulk_strong and self.switch_mode != "serial":
            raise ValueError("switch_bulk_strong requires switch_mode='serial'")
        if self.idle_round_mode not in IDLE_ROUND_MODES:
            raise ValueError(
                f"idle_round_mode must be one of {IDLE_ROUND_MODES} "
                f"(got {self.idle_round_mode!r})")
        if self.feedback_boundary_mode not in FEEDBACK_BOUNDARY_MODES:
            raise ValueError(
                f"feedback_boundary_mode must be one of {FEEDBACK_BOUNDARY_MODES} "
                f"(got {self.feedback_boundary_mode!r})")

    def make_links(self) -> "LinkModel":
        """Build the communication fabric from the link latency knobs."""
        from .links import LinkModel
        return LinkModel(qc=us(self.t_qc_us), cd=us(self.t_cd_us),
                         dd=us(self.t_dd_us), do=us(self.t_do_us),
                         oc=us(self.t_oc_us), cq=us(self.t_cq_us),
                         ws=us(self.t_dd_us if self.t_ws_us is None else self.t_ws_us))

    def make_controller(self, engine: "Engine") -> "Controller":
        """Build the default controller."""
        from .controllers import ModularController
        return ModularController(engine, links=self.make_links(),
                                 t_pack=us(self.t_pack_us))

    def decoder_fit(self) -> tuple:
        """Return the selected decoder latency fit."""
        if self.decoder_model is not None:
            return DECODER_FITS[self.decoder_model]
        return (self.decoder_alpha, self.decoder_beta)

    def make_decoder(self, code: "CodeModel") -> "Decoder":
        """Build the default latency-model decoder for this code."""
        from .decoders import LatencyModelDecoder
        alpha, beta = self.decoder_fit()
        return LatencyModelDecoder(d=code.distance, alpha=alpha, beta=beta)

    def make_switching_decoder(self, weak: "Decoder", strong: "Decoder") -> "Decoder":
        """Wrap weak and strong decoders in a sampled switching decoder."""
        from .decoders import SwitchingDecoder
        return SwitchingDecoder(weak, strong, gamma_switch=self.switch_gamma,
                                handoff_us=self.switch_handoff_us, seed=self.switch_seed,
                                t_comm_weak_us=self.switch_comm_weak_us)

    def make_relaybp_decoder(self) -> "Decoder":
        """Build a Relay-BP latency decoder."""
        from .decoders import RelayBPDecoder
        return RelayBPDecoder(iterations=self.relaybp_iterations, t_iter_ns=self.relaybp_t_iter_ns)

    def make_scheme(self) -> "DecodingScheme":
        """Build the configured decoding scheme."""
        return _scheme_registry()[self.scheme_name]()

    def make_switching(self):
        """Build the configured decoder-switching policy, or None."""
        if self.switch_mode == "none":
            return None
        from .switching import Switching
        return Switching(confidence_threshold=self.switch_confidence_threshold,
                         run_both_at_once=(self.switch_mode == "parallel"),
                         weak_keepup_ratio=self.switch_weak_keepup_ratio,
                         bulk_strong=self.switch_bulk_strong)

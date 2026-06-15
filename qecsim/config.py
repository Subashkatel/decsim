from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
 
if TYPE_CHECKING:
    from .engine import Engine
    from .links import LinkModel
    from .protocols import CodeModel, Controller, Decoder, DecodingScheme

#===============================================================================
# CONFIGURATION 
#===============================================================================

# Time Units : The simulator uses integer ticks (similar to the classical gem5 simulator)
# 1 tick = 1 picosecond (global frequency)
TICKS_PER_US = 1_000_000

def us(microseconds : float) -> int:
    """Convert microseconds to ticks."""
    return int(round(microseconds * TICKS_PER_US))

def fmt(ticks : int ) -> str:
    """Format ticks as microseconds for readability in logs."""
    return f"{ticks / TICKS_PER_US:7.3f} us"


# Named latency-model fits tau_d(N)=alpha*N^beta (arXiv:2511.10633 Table 3; PyMatching @ p=0.1%).
# Select one with SimConfig(decoder_model="cc_asic"); leave decoder_model=None to use the raw
# decoder_alpha/decoder_beta fields instead (their defaults ARE the cc_fpga fit, so behaviour
# is unchanged by default).
DECODER_FITS = {
    "cc_fpga":    (2.85e-10, 1.2),    # Collision Cluster on FPGA (the paper's headline fit)
    "cc_asic":    (5.53e-11, 1.34),   # Collision Cluster on ASIC
    "alphaqubit": (4.8e-6,   0.503),  # AlphaQubit (ML decoder)
    "pymatching": (5.91e-9,  1.17),   # PyMatching (MWPM) at p = 0.1%
}

# Decoding schemes selectable by name from config (schemes.py). "double" is a documented stub.
SCHEME_NAMES = ("sliding", "naive", "parallel", "double")


def _scheme_registry() -> dict:
    """name -> scheme class (lazy import: schemes.py imports nothing from here at module load)."""
    from .schemes import (SlidingWindowScheme, NaiveOnlineScheme,
                          ParallelWindowScheme, DoubleWindowScheme)
    return {"sliding": SlidingWindowScheme, "naive": NaiveOnlineScheme,
            "parallel": ParallelWindowScheme, "double": DoubleWindowScheme}


@dataclass(frozen=True)
class SimConfig:
    """All the changable parameters in one place so its easy to modify.

    Defaults are grounded in Khalid et al., arXiv:2511.10633: the six link latencies are
    Table 2; (decoder_alpha, decoder_beta) are the Table 3 monomial fit tau_d(N)=alpha*N^beta
    for the Collision Cluster decoder on FPGA. Syndrome-round time is platform-dependent
    (~1 us superconducting: 1 us/cycle in arXiv:2510.21600; ~0.5 us stabilization rounds in
    arXiv:2411.10406 Sec I.2.1)."""
    round_us: float = 1.1 # one syndrome round = one parity check cycle
    rounds_per_op: int = 11 # number of rounds per logical operation (two_qubit op + bus)
    num_units: int = 1             # decoder units in the cluster

    # link latencies: Table 2 of arXiv:2511.10633 (sum ~= t_com ~ 10 us)
    t_qc_us: float = 0.15  # chip -> controller latency (microseconds)
    t_cd_us: float = 2.0   # controller -> decoder cluster latency (microseconds)
    t_dd_us: float = 0.5   # decoder -> decoder message passing latency (microseconds)
    t_do_us: float = 1.0   # decoder -> orchestrator latency (microseconds)
    t_oc_us: float = 4.0   # orchestrator -> controller latency (microseconds)
    t_cq_us: float = 0.15  # controller -> chip latency (microseconds)
    t_pack_us: float = 0.0 # controller packaging cost per round packet (microseconds): the controller aggregates a round's fragments into one t_cd packet (arXiv:2511.10633 Sec III.1); this prices the serialization/compression step (0 = free, the paper folds it into t_cd)

    # Decoder speed model tau_d(N) = alpha * N^beta (arXiv:2511.10633 Eq. 12): time to decode
    # one round of a decoding graph with N nodes (N ~ d^2 for a distance-d patch).
    #   alpha = the hardware's raw speed, in seconds (smaller = faster decoder)
    #   beta  = how decode time grows with patch size (>1 = superlinear: doubling the
    #           graph more than doubles the decode time)
    # Defaults: the paper's Table 3 fit for the Collision Cluster decoder on FPGA. Other
    # Table 3 fits to swap in: ASIC (5.53e-11, 1.34), AlphaQubit (4.8e-6, 0.503),
    # PyMatching at p=0.1% (5.91e-9, 1.17).
    # Decoder selection: a named fit from DECODER_FITS, or None to use the raw alpha/beta below.
    decoder_model: Optional[str] = None
    decoder_alpha: float = 2.85e-10    # raw monomial fit, used when decoder_model is None
    decoder_beta: float = 1.2

    # Switching decoder (arXiv:2510.25222): a fast weak decoder backed by a slow strong one.
    switch_gamma: float = 0.0          # P(escalate to strong) per window; 0 = never switch
    switch_handoff_us: float = 0.5     # decoder<->decoder handoff cost (default = the t_dd link)
    switch_comm_weak_us: float = 0.0   # T_comm^weak paid on every decode (0 in a full-stack run)
    switch_seed: int = 0               # RNG seed for the switch draw

    # Relay-BP decoder (arXiv:2510.21600) for qLDPC / bivariate-bicycle codes
    relaybp_iterations: int = 40       # BP iteration budget (a worst-case cap, not the average)
    relaybp_t_iter_ns: float = 24.0    # per-iteration FPGA time (ns)

    # Decoding scheme / windowing (schemes.py); default "sliding" == the cluster's own default
    scheme_name: str = "sliding"

    def __post_init__(self):
        """ This method is called after the dataclass is initialized so it can perform validation."""
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

    def make_links(self) -> "LinkModel":
        """Build the communication fabric (links.py) from these link-latency knobs --
        flat constants, arXiv:2511.10633 Table 2. Imported lazily so this module has no
        import-time dependency on links.py (which imports `us` from here)."""
        from .links import LinkModel
        return LinkModel(qc=us(self.t_qc_us), cd=us(self.t_cd_us),
                         dd=us(self.t_dd_us), do=us(self.t_do_us),
                         oc=us(self.t_oc_us), cq=us(self.t_cq_us))

    def make_controller(self, engine: "Engine") -> "Controller":
        """Build the modular controller on a fabric from these knobs (lazy import, as above)."""
        from .controllers import ModularController
        return ModularController(engine, links=self.make_links(),
                                 t_pack=us(self.t_pack_us))

    def decoder_fit(self) -> tuple:
        """Resolve (alpha, beta): the named DECODER_FITS entry if decoder_model is set, else the
        raw decoder_alpha/decoder_beta. Default (decoder_model=None) -> (2.85e-10, 1.2) = cc_fpga."""
        if self.decoder_model is not None:
            return DECODER_FITS[self.decoder_model]
        return (self.decoder_alpha, self.decoder_beta)

    def make_decoder(self, code: "CodeModel") -> "Decoder":
        """Build the default latency-model decoder for the given code (lazy import, as above)."""
        from .decoders import LatencyModelDecoder
        alpha, beta = self.decoder_fit()
        return LatencyModelDecoder(d=code.distance, alpha=alpha, beta=beta)

    def make_switching_decoder(self, weak: "Decoder", strong: "Decoder") -> "Decoder":
        """Build a SwitchingDecoder (arXiv:2510.25222) from the switch_* knobs, wrapping the two
        given sub-decoders (each carries its own latency model)."""
        from .decoders import SwitchingDecoder
        return SwitchingDecoder(weak, strong, gamma_switch=self.switch_gamma,
                                handoff_us=self.switch_handoff_us, seed=self.switch_seed,
                                t_comm_weak_us=self.switch_comm_weak_us)

    def make_relaybp_decoder(self) -> "Decoder":
        """Build a Relay-BP latency decoder (arXiv:2510.21600) from the relaybp_* knobs."""
        from .decoders import RelayBPDecoder
        return RelayBPDecoder(iterations=self.relaybp_iterations, t_iter_ns=self.relaybp_t_iter_ns)

    def make_scheme(self) -> "DecodingScheme":
        """Build the named decoding scheme (schemes.py); default 'sliding' == the cluster default."""
        return _scheme_registry()[self.scheme_name]()

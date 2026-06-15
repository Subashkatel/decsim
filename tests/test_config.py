#==================================================================
# TESTS FOR CONFIG
#==================================================================

from qecsim.config import TICKS_PER_US, us , fmt

def test_ticks_per_us():
    assert TICKS_PER_US == 1_000_000

def test_us():
    assert us(1.1) == 1_100_000
    assert us(2.0) == 2_000_000

def test_fmt():
    assert fmt(1_100_000) == "  1.100 us"
    assert fmt(2_000_000) == "  2.000 us"

def test_fmt_output():
    print(repr(fmt(1_100_000)))


#==================================================================
# TESTS FOR THE CONSOLIDATED SimConfig KNOBS
# decoder fits / switching / relay-BP / scheme -- all in one place.
# Covers backward compat (default unchanged), resolution, validation, flow-through.
#==================================================================
import pytest
from qecsim.config import SimConfig, DECODER_FITS, SCHEME_NAMES
from qecsim.codes import SurfaceCodeModel
from qecsim.decoders import (LatencyModelDecoder, SwitchingDecoder, RelayBPDecoder,
                             PresetLatencyDecoder)
from qecsim.schemes import (SlidingWindowScheme, NaiveOnlineScheme,
                            ParallelWindowScheme, DoubleWindowScheme)

_CODE = SurfaceCodeModel(d=3)


def test_default_decoder_unchanged():
    """The DEFAULT config must build the exact pre-refactor cc_fpga decoder (back compat)."""
    cfg = SimConfig()
    assert cfg.decoder_model is None
    assert cfg.decoder_fit() == (2.85e-10, 1.2)
    dec = cfg.make_decoder(_CODE)
    assert isinstance(dec, LatencyModelDecoder)
    assert (dec.alpha, dec.beta) == (2.85e-10, 1.2)


def test_default_scheme_is_sliding():
    assert SimConfig().scheme_name == "sliding"
    assert isinstance(SimConfig().make_scheme(), SlidingWindowScheme)


@pytest.mark.parametrize("name,fit", list(DECODER_FITS.items()))
def test_named_decoder_fits(name, fit):
    cfg = SimConfig(decoder_model=name)
    assert cfg.decoder_fit() == fit
    dec = cfg.make_decoder(_CODE)
    assert (dec.alpha, dec.beta) == fit


def test_raw_alpha_beta_override():
    """decoder_model=None uses the raw alpha/beta fields (explicit override path)."""
    cfg = SimConfig(decoder_alpha=1e-9, decoder_beta=1.5)
    assert cfg.decoder_fit() == (1e-9, 1.5)
    assert (cfg.make_decoder(_CODE).alpha, cfg.make_decoder(_CODE).beta) == (1e-9, 1.5)


def test_make_switching_decoder():
    cfg = SimConfig(switch_gamma=0.3, switch_handoff_us=0.5, switch_comm_weak_us=0.1, switch_seed=7)
    weak, strong = PresetLatencyDecoder(1.0), PresetLatencyDecoder(10.0)
    sw = cfg.make_switching_decoder(weak, strong)
    assert isinstance(sw, SwitchingDecoder)
    assert sw.gamma_switch == 0.3
    assert sw.handoff == us(0.5) and sw.t_comm_weak == us(0.1)
    assert sw.weak is weak and sw.strong is strong


def test_switching_never_switches_at_gamma_zero():
    sw = SimConfig(switch_gamma=0.0).make_switching_decoder(
        PresetLatencyDecoder(1.0), PresetLatencyDecoder(10.0))

    class _Job:
        spatial_nodes = 9; n_rounds = 1; hint = None
    assert sw.latency(_Job()) == us(1.0)        # weak path only
    assert sw.switches == 0


def test_make_relaybp_decoder():
    rb = SimConfig(relaybp_iterations=10, relaybp_t_iter_ns=24.0).make_relaybp_decoder()
    assert isinstance(rb, RelayBPDecoder)
    assert rb.iterations == 10 and rb.t_iter_ns == 24.0


@pytest.mark.parametrize("name,cls", [("sliding", SlidingWindowScheme),
                                       ("naive", NaiveOnlineScheme),
                                       ("parallel", ParallelWindowScheme),
                                       ("double", DoubleWindowScheme)])
def test_make_scheme(name, cls):
    assert isinstance(SimConfig(scheme_name=name).make_scheme(), cls)


@pytest.mark.parametrize("kw", [
    {"decoder_model": "nope"}, {"switch_gamma": 1.5}, {"switch_gamma": -0.1},
    {"switch_handoff_us": -1}, {"relaybp_iterations": 0}, {"relaybp_t_iter_ns": -1},
    {"scheme_name": "bogus"},
])
def test_validation_rejects_bad_values(kw):
    with pytest.raises(ValueError):
        SimConfig(**kw)


def test_config_flows_through_build_and_run():
    """decoder_model + scheme_name + num_units all take effect via build_and_run."""
    from qecsim.message import Operation
    from qecsim.wiring import build_and_run
    from qecsim.planner import FixedRounds
    op = Operation(id=1, name="memory", qubits=(0,), clifford=True)
    cfg = SimConfig(decoder_model="cc_asic", scheme_name="naive", num_units=2)
    res = build_and_run(ops=[op], d=3, rounds_policy=FixedRounds(6),
                        code=SurfaceCodeModel(d=3), config=cfg, verbose=False)
    assert isinstance(res["cluster"].scheme, NaiveOnlineScheme)
    assert res["cluster"].decoder.alpha == DECODER_FITS["cc_asic"][0]

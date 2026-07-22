#==================================================================
# TESTS FOR CONTROLLER-SIDE ROUND PACKAGING
# arXiv:2511.10633 Sec III.1: the controller aggregates per-qubit readout and
# forwards batched packets to the decoders -- so a round arriving in fragments
# (possibly at different times) must ship as ONE packet after its last fragment.
#==================================================================
from decsim.config import us
from decsim.controllers import ModularController
from decsim.engine import Engine
from decsim.links import Link, LinkModel
from decsim.message import SyndromePayload


def _frag(patch, n=2):
    return SyndromePayload(0, patch, 1, n_fragments=n)


def test_staggered_fragments_ship_as_one_packet_after_the_last():
    eng = Engine(verbose=False)
    ctrl = ModularController(eng, log_syndromes=False)
    arrivals = []
    deliver = lambda p: arrivals.append((eng.now, p.patch_id))
    # the device emits the round in two chunks, 0.4 us apart (staggered readout)
    eng.schedule(0, lambda: ctrl.relay_syndrome(_frag(0), deliver))
    eng.schedule(us(0.4), lambda: ctrl.relay_syndrome(_frag(1), deliver))
    eng.run()
    # both fragments arrive TOGETHER, one t_qc + t_cd after the LAST chunk left the chip
    expected = us(0.4) + us(0.15) + us(2.0)
    assert arrivals == [(expected, 0), (expected, 1)]


def test_packaging_cost_is_priced_per_packet():
    eng = Engine(verbose=False)
    ctrl = ModularController(eng, log_syndromes=False, t_pack=us(0.3))
    arrivals = []
    eng.schedule(0, lambda: ctrl.relay_syndrome(_frag(0), lambda p: arrivals.append(eng.now)))
    eng.schedule(0, lambda: ctrl.relay_syndrome(_frag(1), lambda p: arrivals.append(eng.now)))
    eng.run()
    assert arrivals == [us(0.15) + us(0.3) + us(2.0)] * 2


def test_whole_round_payloads_take_the_original_path():
    # n_fragments=1 (every default device): no buffering, no t_pack -- two plain hops
    eng = Engine(verbose=False)
    ctrl = ModularController(eng, log_syndromes=False, t_pack=us(9.9))
    arrivals = []
    eng.schedule(0, lambda: ctrl.relay_syndrome(_frag(0, n=1),
                                                lambda p: arrivals.append(eng.now)))
    eng.run()
    assert arrivals == [us(0.15) + us(2.0)]


#------------------------------------------------------------------
# BANDWIDTH AND SERIALIZATION: the controller must hand each hop the
# payload's size_bits (and, for shared buses, the current time) or the
# link prices every transfer at zero.
#------------------------------------------------------------------

def test_bandwidth_links_price_the_syndrome_bits():
    # 1000 bits over 1000 bits/us (qc) then 2000 bits/us (cd): 1.5 us total
    eng = Engine(verbose=False)
    links = LinkModel(qc=Link(0, bandwidth_bits_per_us=1000),
                      cd=Link(0, bandwidth_bits_per_us=2000))
    ctrl = ModularController(eng, links=links, log_syndromes=False)
    arrivals = []
    payload = SyndromePayload(0, 0, 1, size_bits=1000)
    eng.schedule(0, lambda: ctrl.relay_syndrome(
        payload, lambda p: arrivals.append(eng.now)))
    eng.run()
    assert arrivals == [us(1.5)]


def test_serialized_chip_link_queues_concurrent_rounds():
    # two 1000-bit rounds on a shared 1000 bits/us qc bus: 1 us then 2 us
    eng = Engine(verbose=False)
    links = LinkModel(qc=Link(0, bandwidth_bits_per_us=1000, serialize=True),
                      cd=Link(0))
    ctrl = ModularController(eng, links=links, log_syndromes=False)
    arrivals = []
    for round_index in (1, 2):
        payload = SyndromePayload(0, 0, round_index, size_bits=1000)
        eng.schedule(0, lambda p=payload: ctrl.relay_syndrome(
            p, lambda q: arrivals.append(eng.now)))
    eng.run()
    assert arrivals == [us(1.0), us(2.0)]


def test_generic_send_prices_bits_and_queues_on_serialized_links():
    # the same two-message queueing through Transport.send on a dd bus
    eng = Engine(verbose=False)
    links = LinkModel(dd=Link(0, bandwidth_bits_per_us=1000, serialize=True))
    ctrl = ModularController(eng, links=links, log_syndromes=False)
    arrivals = []

    def send_both():
        for round_index in (1, 2):
            ctrl.send("dd", SyndromePayload(0, 0, round_index, size_bits=1000),
                      lambda p: arrivals.append(eng.now), now=eng.now)

    eng.schedule(0, send_both)
    eng.run()
    assert arrivals == [us(1.0), us(2.0)]


def test_packed_fragments_are_priced_by_their_total_bits():
    # 400 + 600 bit fragments = one 1000-bit packet over 1000 bits/us (cd)
    eng = Engine(verbose=False)
    links = LinkModel(qc=Link(0), cd=Link(0, bandwidth_bits_per_us=1000))
    ctrl = ModularController(eng, links=links, log_syndromes=False)
    arrivals = []
    for patch, bits in ((0, 400), (1, 600)):
        payload = SyndromePayload(0, patch, 1, n_fragments=2, size_bits=bits)
        eng.schedule(0, lambda p=payload: ctrl.relay_syndrome(
            p, lambda q: arrivals.append(eng.now)))
    eng.run()
    assert arrivals == [us(1.0)] * 2

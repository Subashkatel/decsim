"""The ported syndrome buffer against the memory law it states.

The law, written here from its sources and not from the row: a round is
stored in whole words, so a request needs k = sum of ceil(bits /
word_bits) accesses (gem5 src/mem/xbar.cc:135 divCeil per packet;
Helios_scalable_QEC control_node_single_FPGA.v lines 35-36); requests are
served in arrival order on the port that frees first among those their
direction may use (gem5 src/mem/xbar.cc:206 and 288-289); a request holds
its port k x cycles_per_access cycles from the later of its clock edge and
the port's free tick, and completes access_latency_cycles after that
(gem5 src/mem/simple_mem.cc:154-174). The d = 11 window sizes are the
machine's: rounds of 60 then 120 bits, or 120 bits closed by a readout
round of 180.
"""

import math
import random

import pytest

import decsim.config as config
import decsim.engine as engine_module
import decsim.records.rounds as round_records
import decsim.syndrome_buffer.ported_syndrome_buffer as ported_syndrome_buffer
import decsim.syndrome_buffer.settings as syndrome_buffer_settings
import tests.declared_run as declared_run

PERIOD_TICKS = 10
# the last window of the d = 3 machine: six rounds of 8 bits, one of 12
D3_LAST_WINDOW_BITS = (8, 8, 8, 8, 8, 8, 12)
# the d = 11 windows: 60 then 21 x 120 bits, or 22 x 120 then 180
D11_FIRST_WINDOW_BITS = (60,) + (120,) * 21
D11_READOUT_WINDOW_BITS = (120,) * 22 + (180,)
PORTED_ROWS = ported_syndrome_buffer.SYNDROME_BUFFERS
# read, write and read/write port counts; each reads and writes somewhere
PORT_SHAPES = (
    (1, 1, 0),
    (2, 1, 0),
    (4, 1, 0),
    (8, 2, 0),
    (0, 0, 1),
    (0, 0, 2),
    (1, 0, 1),
    (0, 1, 1),
    (2, 2, 2),
)


def law_completions(requests, port_kinds, cycles_per_access, latency_cycles):
    """The law's completion cycle of every request, in arrival order.

    requests are (arrival cycle, direction, words); a port kind is "read",
    "write" or "read_write". This is the recursion of first-come service
    over several servers (Kiefer and Wolfowitz 1955), each request on the
    eligible server that frees first.
    """
    free_cycles = [0] * len(port_kinds)
    completions = []
    for arrival, direction, words in requests:
        eligible = [
            (free_cycles[index], index)
            for index, kind in enumerate(port_kinds)
            if kind in (direction, "read_write")
        ]
        _free, port = min(eligible)
        start = max(arrival, free_cycles[port])
        free_cycles[port] = start + words * cycles_per_access
        completion = free_cycles[port] + latency_cycles
        completions.append(completion)
    return completions


def _store(engine, **row_keys) -> ported_syndrome_buffer.PortedSyndromeBuffer:
    clock = config.Clock(PERIOD_TICKS)
    row_settings = ported_syndrome_buffer.PortedSyndromeBuffer.Settings(
        **row_keys
    )
    settings = syndrome_buffer_settings.SyndromeBufferSettings(
        kind="ported_syndrome_buffer", clock=clock, row_settings=row_settings
    )
    return ported_syndrome_buffer.PortedSyndromeBuffer(settings, engine)


def _packet(round_index: int, bits: int) -> round_records.SyndromeRoundPacket:
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=(0,),
        round_index=round_index,
        bits=None,
        size_bits=bits,
        fragment_index=0,
    )
    return round_records.SyndromeRoundPacket(1, round_index, (fragment,))


def _stored_window(store, round_bits) -> tuple:
    """Store one round per width, then name them as one window."""
    round_keys = []
    for round_index, bits in enumerate(round_bits, start=1):
        packet = _packet(round_index, bits)
        store.accept_packed_round(packet, publication_tick=None)
        round_keys.append((1, round_index))
    return tuple(round_keys)


def _port_kinds(read_ports, write_ports, read_write_ports) -> list:
    kinds = ["read"] * read_ports
    kinds += ["write"] * write_ports
    kinds += ["read_write"] * read_write_ports
    return kinds


def _book(store, direction, round_key, bits) -> int:
    if direction == "write":
        return store.book_write(round_key, bits)
    return store.book_read((round_key,))


def _random_program(generator, word_bits):
    """Up to 40 requests at random edges, several often on one edge."""
    program = []
    arrival = 0
    request_count = generator.randint(1, 40)
    widest_bits = 3 * word_bits
    for _ in range(request_count):
        arrival += generator.choice([0, 0, 1, 2, 5])
        direction = generator.choice(["read", "write"])
        bits = generator.randint(0, widest_bits)
        program.append((arrival, direction, bits))
    return program


def _law_requests(program, word_bits) -> list:
    """The program as the law reads it: whole words per request."""
    requests = []
    for arrival, direction, bits in program:
        exact_words = bits / word_bits
        words = math.ceil(exact_words)
        requests.append((arrival, direction, words))
    return requests


def _run_program(program, shape) -> list:
    """Every request booked at its arrival on the row; the completions.

    Each request's round is stored first, so a read has its width.
    """
    engine = engine_module.Engine()
    store = _store(engine, **shape)
    completions = []
    for round_index, (arrival, direction, bits) in enumerate(program, 1):
        engine.now = arrival * PERIOD_TICKS
        packet = _packet(round_index, bits)
        store.accept_packed_round(packet, publication_tick=None)
        round_key = (1, round_index)
        completion_ticks = _book(store, direction, round_key, bits)
        completion_cycles = completion_ticks // PERIOD_TICKS
        completions.append(completion_cycles)
    return completions


def test_every_completion_is_the_fifo_multi_port_law_property():
    for seed in range(300):
        generator = random.Random(seed)
        port_shape = generator.choice(PORT_SHAPES)
        read_ports, write_ports, read_write_ports = port_shape
        shape = {
            "read_ports": read_ports,
            "write_ports": write_ports,
            "read_write_ports": read_write_ports,
            "word_bits": generator.choice([8, 16, 32]),
            "cycles_per_access": generator.choice([1, 4]),
            "access_latency_cycles": generator.choice([0, 4]),
        }
        program = _random_program(generator, shape["word_bits"])
        kinds = _port_kinds(read_ports, write_ports, read_write_ports)
        requests = _law_requests(program, shape["word_bits"])
        expected = law_completions(
            requests,
            kinds,
            shape["cycles_per_access"],
            shape["access_latency_cycles"],
        )
        assert _run_program(program, shape) == expected, seed


@pytest.mark.parametrize(
    ("round_bits", "cycles"),
    [(D11_FIRST_WINDOW_BITS, 323), (D11_READOUT_WINDOW_BITS, 353)],
)
def test_a_d11_window_over_a_byte_word_takes_one_access_a_word(
    round_bits, cycles
):
    """8 + 21 x 15 = 323 and 22 x 15 + 23 = 353 cycles of one read port."""
    engine = engine_module.Engine()
    store = _store(engine)
    window = _stored_window(store, round_bits)

    assert store.book_read(window) == cycles * PERIOD_TICKS


def test_two_readers_on_one_read_port_finish_one_read_apart():
    """The d = 3 last window is 6 x 8 + 12 bits: 8 words, then 8 more."""
    engine = engine_module.Engine()
    store = _store(engine)
    window = _stored_window(store, D3_LAST_WINDOW_BITS)

    first = store.book_read(window)
    second = store.book_read(window)

    assert (first, second) == (8 * PERIOD_TICKS, 16 * PERIOD_TICKS)


def test_a_write_waits_for_no_read_on_its_own_port():
    engine = engine_module.Engine()
    store = _store(engine)
    window = _stored_window(store, D3_LAST_WINDOW_BITS)
    store.book_read(window)

    assert store.book_write((1, 8), 8) == 1 * PERIOD_TICKS


def test_a_single_read_write_port_makes_a_write_wait_for_the_read():
    engine = engine_module.Engine()
    store = _store(engine, read_ports=0, write_ports=0, read_write_ports=1)
    window = _stored_window(store, D3_LAST_WINDOW_BITS)
    store.book_read(window)

    assert store.book_write((1, 8), 8) == 9 * PERIOD_TICKS


def test_an_arrival_between_edges_starts_at_the_next_edge():
    engine = engine_module.Engine()
    engine.now = 1
    store = _store(engine, access_latency_cycles=4)

    assert store.book_write((1, 1), 16) == (1 + 2 + 4) * PERIOD_TICKS


def test_each_access_is_reported_with_its_port_and_its_three_ticks():
    engine = engine_module.Engine()
    store = _store(engine)
    served = []
    store.trace.access_served.connect(lambda *access: served.append(access))
    window = _stored_window(store, (8, 8))

    store.book_read(window)
    store.book_read(window)

    assert served == [
        ("read", 0, window, 0, 0, 20),
        ("read", 0, window, 0, 20, 40),
    ]


def test_the_default_shape_is_one_byte_fifo_port_each_way():
    """sky130_sram_1kbyte_1r1w_8x1024_8.py lines 6 and 15-16."""
    defaults = ported_syndrome_buffer.PortedSyndromeBuffer.Settings()

    assert (defaults.read_ports, defaults.write_ports) == (1, 1)
    assert defaults.read_write_ports == 0
    assert defaults.word_bits == 8
    assert defaults.cycles_per_access == 1
    assert defaults.access_latency_cycles == 0


def _section_settings(section):
    clocks = config.ClockSettings.from_yaml({"fridge": 250.0})
    return syndrome_buffer_settings.SyndromeBufferSettings.from_yaml(
        section, "weak_syndrome_buffer", clocks, PORTED_ROWS
    )


@pytest.mark.parametrize("key", ["write_cycles", "read_cycles"])
def test_a_flat_access_cost_under_the_ported_row_is_refused_by_name(key):
    section = {"kind": "ported_syndrome_buffer", key: 1}

    with pytest.raises(
        ValueError, match=rf"weak_syndrome_buffer does not know \['{key}'\]"
    ):
        _section_settings(section)


def test_a_ported_store_with_no_port_that_reads_is_refused():
    section = {
        "kind": "ported_syndrome_buffer",
        "read_ports": 0,
        "read_write_ports": 0,
    }

    with pytest.raises(ValueError, match="needs a port that reads"):
        _section_settings(section)


@pytest.mark.parametrize(
    ("key", "value", "sentence"),
    [
        ("word_bits", 0, "word_bits must be at least one bit"),
        ("word_bits", None, "word_bits must be at least one bit"),
        ("cycles_per_access", 0, "cycles_per_access must be at least one"),
        ("access_latency_cycles", -1, "access_latency_cycles must not be"),
        ("read_ports", True, "read_ports must be a nonnegative integer"),
    ],
)
def test_a_ported_key_out_of_its_domain_is_refused_by_name(
    key, value, sentence
):
    section = {"kind": "ported_syndrome_buffer", key: value}

    with pytest.raises(ValueError, match=sentence):
        _section_settings(section)


def test_a_ported_strong_syndrome_buffer_is_refused():
    section = {"kind": "ported_syndrome_buffer"}

    with pytest.raises(
        ValueError, match="strong_syndrome_buffer.kind ported_syndrome_buffer"
    ):
        syndrome_buffer_settings.check_strong_section_charges_nothing(section)


def test_a_declared_weak_run_on_a_ported_store_delays_the_decode_only():
    """The byte port's reads and writes cost cycles the flat row does not."""
    clocks = config.ClockSettings.from_yaml({"storage": 1.0})
    section = {"kind": "ported_syndrome_buffer", "clock": "storage"}
    settings = syndrome_buffer_settings.SyndromeBufferSettings.from_yaml(
        section, "weak_syndrome_buffer", clocks, PORTED_ROWS
    )
    free = declared_run.weak_only_run()
    ported = declared_run.weak_only_run(weak_syndrome_buffer=settings)
    free_ticks = declared_run.reaction_ticks(free)
    ported_ticks = declared_run.reaction_ticks(ported)

    assert isinstance(
        ported.readout.weak_syndrome_buffer,
        ported_syndrome_buffer.PortedSyndromeBuffer,
    )
    assert ported_ticks[3] > free_ticks[3]

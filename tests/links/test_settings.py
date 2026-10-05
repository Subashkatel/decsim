"""The link settings refuse what no card may say.

The number cards are where link numbers enter decsim, so each check
here is a boundary check (STYLE.md rule 4). The arithmetic
is ns-3's DataRate (point-to-point-net-device.cc): the wire's rate is
read exactly as the card writes it.
"""

import fractions

import pytest

import decsim.experiments.collect as collect
import decsim.links.settings as link_settings
import decsim.records.transfers as transfer_records

FREE_CHANNEL = link_settings.ChannelSettings("free", 0, None, "test")
FREE_PATH = link_settings.PathSettings(
    FREE_CHANNEL, None, "test payload", excludes_receiver_processing=False
)


def capacity(rate):
    return link_settings.CapacitySettings(rate, "test")


def payload(bits):
    return link_settings.PayloadSettings(bits, "test")


def channel(name="test", latency_ticks=0, capacity_settings=None):
    return link_settings.ChannelSettings(
        name, latency_ticks, capacity_settings, "test"
    )


def actual_path(channel_settings, setup_ticks=0, header_bits=0):
    return link_settings.PathSettings(
        channel_settings,
        None,
        "test payload",
        setup_ticks,
        header_bits_per_transfer=header_bits,
        excludes_receiver_processing=False,
    )


def every_path_free():
    """One free path per hop, keyed by the hop's name.

    A card names every hop, so a test that cares about one path wires
    that one and leaves the rest on the shared free channel.
    """
    names = []
    for path in transfer_records.LinkPath:
        names.append(path.value)
    return dict.fromkeys(names, FREE_PATH)


def card(**paths):
    wiring = every_path_free()
    wiring.update(paths)
    return link_settings.FabricSettings(profile_name="test", **wiring)


def test_an_aggregate_rate_is_read_as_the_decimal_on_the_card():
    decimal_rate = capacity(0.20846)
    exact = decimal_rate.input_bits_per_microsecond
    assert exact == fractions.Fraction("0.20846")


def test_a_rate_written_as_a_float_or_as_its_fraction_has_one_task_id():
    """The rate is one exact type, so equal rates write equal json."""
    half = fractions.Fraction(1, 2)
    float_rate = capacity(0.5)
    fraction_rate = capacity(half)

    float_value = collect.json_value(float_rate, keep_labels=False)
    fraction_value = collect.json_value(fraction_rate, keep_labels=False)

    assert type(float_rate.input_bits_per_microsecond) is fractions.Fraction
    assert float_value == fraction_value


def test_a_zero_capacity_is_refused():
    with pytest.raises(ValueError, match="input_bits_per_microsecond"):
        capacity(0.0)


def test_a_negative_payload_is_refused():
    with pytest.raises(ValueError, match="input_bits must be nonnegative"):
        payload(-1)


def test_a_whole_float_latency_is_stored_as_an_int():
    settings = channel(latency_ticks=3.0)
    assert settings.propagation_latency_ticks == 3
    assert type(settings.propagation_latency_ticks) is int


def test_a_negative_latency_is_refused():
    with pytest.raises(ValueError, match="propagation_latency_ticks"):
        channel(latency_ticks=-1)


def test_a_negative_setup_cost_is_refused():
    with pytest.raises(ValueError, match="setup_ticks must be nonnegative"):
        actual_path(FREE_CHANNEL, setup_ticks=-5)


def test_a_negative_header_is_refused():
    with pytest.raises(ValueError, match="header_bits_per_transfer"):
        actual_path(FREE_CHANNEL, header_bits=-8)


def test_a_path_without_a_payload_rule_is_refused():
    with pytest.raises(ValueError, match="a path needs a default payload"):
        link_settings.PathSettings(
            FREE_CHANNEL, None, None, excludes_receiver_processing=False
        )


def test_one_channel_name_with_two_settings_is_refused():
    five_ticks = channel("shared", 5)
    seven_ticks = channel("shared", 7)
    on_five = actual_path(five_ticks)
    on_seven = actual_path(seven_ticks)
    with pytest.raises(ValueError, match="is declared with different settings"):
        card(qpu_to_controller=on_five, weak_buffer_to_weak_decoder=on_seven)


def test_cards_built_from_the_same_numbers_compare_equal():
    first = card()
    second = card()
    assert first == second


def test_readout_routes_refuse_unstable_identities() -> None:
    patches = (1.0,)
    with pytest.raises(ValueError, match="readout route patches"):
        link_settings.ReadoutRoute(patches, FREE_PATH)


def test_readout_routes_require_distinct_patches() -> None:
    patches = (1, 1)
    with pytest.raises(ValueError, match="a readout route needs nonempty"):
        link_settings.ReadoutRoute(patches, FREE_PATH)


def test_readout_routes_cannot_repeat_a_complete_footprint() -> None:
    first = link_settings.ReadoutRoute((1, "left"), FREE_PATH)
    second = link_settings.ReadoutRoute(("left", 1), FREE_PATH)
    with pytest.raises(ValueError, match="readout routes must have distinct"):
        card(readout_routes=(first, second))


def test_readout_routes_share_the_same_channel_settings_contract() -> None:
    delayed_channel = channel("free", 5)
    delayed_path = actual_path(delayed_channel)
    route = link_settings.ReadoutRoute((1,), delayed_path)
    with pytest.raises(ValueError, match="is declared with different settings"):
        card(readout_routes=(route,))

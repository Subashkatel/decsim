"""The link settings refuse what no card may say, with a sentence.

The yaml and the number cards are where link numbers enter decsim, so
each check here is a boundary check (REWRITE.md rule 4). The arithmetic
is ns-3's DataRate (point-to-point-net-device.cc): a per-lane rate times
the lane count is the wire's rate, exactly.
"""

import fractions
import math

import pytest

import decsim.links.settings as link_settings

AGGREGATE = link_settings.QuantityBasis.AGGREGATE
PER_LANE = link_settings.QuantityBasis.PER_LANE
FREE_CHANNEL = link_settings.ChannelSettings("free", 0, None, "test")
FREE_PATH = link_settings.PathSettings(FREE_CHANNEL, None, "test payload")


def capacity(rate, basis=AGGREGATE, lane_count=None):
    return link_settings.CapacitySettings(rate, basis, lane_count, "test")


def payload(bits, basis=AGGREGATE, lane_count=None):
    return link_settings.PayloadSettings(bits, basis, lane_count, "test")


def channel(name="test", latency_ticks=0, capacity_settings=None):
    return link_settings.ChannelSettings(
        name, latency_ticks, capacity_settings, "test"
    )


def actual_path(channel_settings, setup_ticks=0):
    return link_settings.PathSettings(
        channel_settings, None, "test payload", setup_ticks
    )


def card(**paths):
    wiring = dict.fromkeys(
        (
            "qpu_to_controller",
            "weak_buffer_to_weak_decoder",
            "weak_decoder_to_strong_decoder",
            "strong_buffer_to_strong_decoder",
            "weak_decoder_to_frame",
            "decoder_to_decoder",
            "strong_decoder_to_frame",
            "frame_to_controller",
            "controller_to_qpu",
        ),
        FREE_PATH,
    )
    wiring.update(paths)
    return link_settings.FabricSettings(profile_name="test", **wiring)


def test_a_per_lane_rate_reports_the_input_times_the_lane_count():
    per_lane = capacity(2.0, PER_LANE, 4)
    assert per_lane.aggregate_bits_per_microsecond == 8.0


def test_a_per_lane_rate_is_multiplied_exactly():
    per_lane = capacity(0.7, PER_LANE, 3)
    exact = per_lane.exact_aggregate_bits_per_microsecond()
    assert exact == fractions.Fraction(21, 10)


def test_an_aggregate_rate_is_read_as_the_decimal_on_the_card():
    decimal_rate = capacity(0.20846)
    exact = decimal_rate.exact_aggregate_bits_per_microsecond()
    assert exact == fractions.Fraction("0.20846")


def test_a_per_lane_payload_is_the_input_times_the_lane_count():
    per_lane = payload(9, PER_LANE, 4)
    assert per_lane.aggregate_bits == 36


def test_a_zero_capacity_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        capacity(0.0)


def test_a_capacity_that_is_not_finite_is_refused():
    with pytest.raises(ValueError, match="must be a finite number"):
        capacity(math.inf)


def test_an_aggregate_capacity_with_a_lane_count_is_refused():
    with pytest.raises(ValueError, match="an aggregate capacity has no lane"):
        capacity(1.0, AGGREGATE, 2)


def test_a_per_lane_capacity_without_a_lane_count_is_refused():
    with pytest.raises(ValueError, match="per-lane capacity count must be"):
        capacity(1.0, PER_LANE, None)


def test_a_per_lane_capacity_with_a_zero_lane_count_is_refused():
    with pytest.raises(ValueError, match="per-lane capacity count must be"):
        capacity(1.0, PER_LANE, 0)


def test_a_negative_payload_is_refused():
    with pytest.raises(ValueError, match="input_bits must be nonnegative"):
        payload(-1)


def test_a_fractional_payload_is_refused():
    with pytest.raises(ValueError, match="input_bits must be a finite whole"):
        payload(1.5)


def test_a_whole_float_latency_is_stored_as_an_int():
    settings = channel(latency_ticks=3.0)
    assert settings.propagation_latency_ticks == 3
    assert type(settings.propagation_latency_ticks) is int


def test_a_negative_latency_is_refused():
    with pytest.raises(ValueError, match="propagation_latency_ticks must be"):
        channel(latency_ticks=-1)


def test_an_unset_latency_is_refused_with_the_field_named():
    with pytest.raises(
        ValueError, match="propagation_latency_ticks must be a finite whole"
    ):
        channel(latency_ticks=None)


def test_a_negative_setup_cost_is_refused():
    with pytest.raises(ValueError, match="setup_ticks must be nonnegative"):
        actual_path(FREE_CHANNEL, setup_ticks=-5)


def test_a_path_without_a_payload_rule_is_refused():
    with pytest.raises(ValueError, match="a path needs a default payload"):
        link_settings.PathSettings(FREE_CHANNEL, None, None)


def test_a_default_payload_on_another_basis_than_the_capacity_is_refused():
    four_lanes = capacity(2.0, PER_LANE, 4)
    bounded = channel(capacity_settings=four_lanes)
    aggregate_payload = payload(8)
    with pytest.raises(ValueError, match="bases must match"):
        link_settings.PathSettings(bounded, aggregate_payload, None)


def test_a_default_payload_with_another_lane_count_is_refused():
    four_lanes = capacity(2.0, PER_LANE, 4)
    bounded = channel(capacity_settings=four_lanes)
    three_lane_payload = payload(8, PER_LANE, 3)
    with pytest.raises(ValueError, match="lane counts must match"):
        link_settings.PathSettings(bounded, three_lane_payload, None)


def test_a_card_without_a_required_path_is_refused():
    with pytest.raises(
        ValueError, match="frame_to_controller is a required link path"
    ):
        card(frame_to_controller=None)


def test_a_card_lists_its_wired_paths_in_vocabulary_order():
    store_channel = channel("store")
    store_path = actual_path(store_channel)
    settings = card(controller_to_strong_buffer=store_path)
    wired = settings.wired_paths()
    assert [path.value for path in wired] == [
        "qpu_to_controller",
        "weak_buffer_to_weak_decoder",
        "weak_decoder_to_strong_decoder",
        "strong_buffer_to_strong_decoder",
        "weak_decoder_to_frame",
        "decoder_to_decoder",
        "strong_decoder_to_frame",
        "frame_to_controller",
        "controller_to_qpu",
        "controller_to_strong_buffer",
    ]


def test_one_channel_name_with_two_settings_is_refused():
    five_ticks = channel("shared", 5)
    seven_ticks = channel("shared", 7)
    on_five = actual_path(five_ticks)
    on_seven = actual_path(seven_ticks)
    with pytest.raises(
        ValueError, match="channel 'shared' is declared with different"
    ):
        card(qpu_to_controller=on_five, weak_buffer_to_weak_decoder=on_seven)


def test_capacities_built_from_the_same_numbers_compare_equal():
    first = capacity(8.0)
    second = capacity(8.0)
    assert first == second


def test_payloads_built_from_the_same_numbers_compare_equal():
    first = payload(9)
    second = payload(9)
    assert first == second


def test_channels_built_from_the_same_numbers_compare_equal():
    first = channel("wire", 7)
    second = channel("wire", 7)
    assert first == second


def test_paths_built_from_the_same_numbers_compare_equal():
    first = actual_path(FREE_CHANNEL, 5)
    second = actual_path(FREE_CHANNEL, 5)
    assert first == second


def test_cards_built_from_the_same_numbers_compare_equal():
    first = card()
    second = card()
    assert first == second

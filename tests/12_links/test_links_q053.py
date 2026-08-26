"""Contract tests for link cards, vocabulary, and trust boundaries."""

from dataclasses import replace
from types import MappingProxyType
from fractions import Fraction
import math
from pathlib import Path
import re

import pytest

import decsim
import decsim.links.link_profiles as link_profiles
import decsim.links.links as links_module
from decsim.config import us
from decsim.links.link_profiles import bandwidth_limited_profile, logical_reference_profile
from decsim.links.links import (
    BoundaryTransferRelation,
    Link,
    LinkCapacityConfig,
    LinkConfig,
    LinkEdgeConfig,
    LinkPath,
    LinkQuantityBasis,
    PayloadSizeConfig,
    RequestTransferRelation,
    TrafficAttribution,
)
from decsim.message import DecoderRequestKey, DecoderTier
from decsim.run_spec import RunSpec
from decsim.links.link_traffic_report import topology_json_value, traffic_json_value


_OPERATION_ID = ("experiment", 7)
_PATCH_IDS = (1, 2)
_ENDPOINTS = {
    "qc": ("qpu", "controller"),
    "cwb": ("controller", "syndrome buffer 0"),
    "wbd": ("weak buffer", "weak decoder"),
    "wsd": ("weak decoder", "strong decoder"),
    "sbd": ("strong buffer", "strong decoder"),
    "wdo": ("weak decoder", "pauli frame"),
    "dd": ("decoder", "decoder"),
    "do": ("strong decoder", "pauli frame"),
    "oc": ("pauli frame", "controller"),
    "cq": ("controller", "qpu"),
    "csb": ("controller", "syndrome buffer 1"),
}


class _WholeInt(int):
    pass


def _request_relation(tier, *, operation_id=_OPERATION_ID, window_id=3, sequence=0):
    return RequestTransferRelation(
        DecoderRequestKey(operation_id, window_id, tier, sequence)
    )


def _valid_attribution(path):
    if path in (LinkPath.QC, LinkPath.CSB):
        return TrafficAttribution(_OPERATION_ID, _PATCH_IDS, None, 1, 2)
    if path is LinkPath.WBD:
        return TrafficAttribution(
            _OPERATION_ID,
            _PATCH_IDS,
            3,
            1,
            2,
            _request_relation(DecoderTier.WEAK),
        )
    if path in (LinkPath.WSD, LinkPath.SBD, LinkPath.DO):
        relation = _request_relation(DecoderTier.STRONG)
        return TrafficAttribution(_OPERATION_ID, _PATCH_IDS, 3, 1, 2, relation)
    if path is LinkPath.WDO:
        relation = _request_relation(DecoderTier.WEAK)
        return TrafficAttribution(_OPERATION_ID, _PATCH_IDS, 3, 1, 2, relation)
    if path is LinkPath.DD:
        key = DecoderRequestKey(_OPERATION_ID, 3, DecoderTier.STRONG, 0)
        relation = BoundaryTransferRelation(
            key, (_OPERATION_ID, 3), (_OPERATION_ID, 4), 2, 5
        )
        return TrafficAttribution(_OPERATION_ID, _PATCH_IDS, 3, 1, 2, relation)
    return TrafficAttribution(_OPERATION_ID, _PATCH_IDS, None, None, None)


def _topology(config):
    return topology_json_value(config.resolve().snapshot())


def test_link_module_documents_every_segment_endpoint_and_extension_step():
    """The module explains every fixed segment, endpoint pair, and extension step."""
    doc = links_module.__doc__ or ""
    assert len(LinkPath) == 11
    assert LinkPath.CWB in LinkPath          # optional controller-to-buffer hop
    assert LinkPath.CSB in LinkPath     # optional room-side dual write
    for path in LinkPath:
        lines = [line.lower() for line in doc.splitlines() if f"``{path.name}``" in line]
        assert lines, f"missing documentation for {path.name}"
        endpoint_line = next(
            line for line in lines
            if all(endpoint in line for endpoint in _ENDPOINTS[path.value])
        )
        assert path.value in endpoint_line
    normalized = re.sub(r"\s+", " ", doc.lower())
    for phrase in (
        "add the member to ``linkpath``",
        "``_path_rules``",
        "``optional[linkedgeconfig]`` field",
        "link_profiles.py",
    ):
        assert phrase in normalized


def test_rule_table_and_reference_cards_cover_the_closed_vocabulary():
    """One complete rule table; both cards wire the nine required paths."""
    assert tuple(links_module._PATH_RULES) == tuple(LinkPath)
    assert not links_module._PATH_RULES[LinkPath.CWB].required
    assert not links_module._PATH_RULES[LinkPath.CSB].required
    paths = tuple(p for p in LinkPath
                  if p is not LinkPath.CWB and p is not LinkPath.CSB)
    assert all(links_module._PATH_RULES[path].required for path in paths)
    for profile in (logical_reference_profile, bandwidth_limited_profile):
        config = profile()
        assert config.wired_paths() == paths
        assert config.resolve().paths == paths
        for path in paths:
            incomplete = replace(config, **{path.value: None})
            with pytest.raises(ValueError, match=path.value):
                incomplete.wired_paths()
            with pytest.raises(ValueError, match=path.value):
                incomplete.resolve()


def test_unwired_optional_path_is_rejected_before_attribution(monkeypatch):
    """An unwired declared optional path fails before attribution is inspected."""
    path = LinkPath.CQ
    rule = links_module._PATH_RULES[path]
    monkeypatch.setattr(
        links_module, "_PATH_RULES",
        MappingProxyType({**links_module._PATH_RULES, path: replace(rule, required=False)}))
    model = replace(logical_reference_profile(), cq=None).resolve()
    assert path not in model.paths

    class ExplodingAttribution:
        @property
        def operation_id(self):
            raise AssertionError("attribution was inspected")

    with pytest.raises(ValueError, match="not wired"):
        model.reserve(
            path,
            payload_bits=None,
            now_ticks=0,
            attribution=ExplodingAttribution(),
        )


def test_whole_quantity_boundaries_accept_and_normalize_whole_scalars():
    """Whole quantity boundaries accept int subclasses and whole scalar values."""
    for value in (_WholeInt(7), Fraction(14, 2)):
        channel = LinkConfig(value, None, "channel")
        payload = PayloadSizeConfig(
            value, LinkQuantityBasis.DIRECT_AGGREGATE, None, "payload"
        )
        capacity = LinkCapacityConfig(
            2.5, LinkQuantityBasis.PER_CHANNEL, value, "capacity"
        )
        reservation = Link(channel).reserve(payload_bits=value, now_ticks=value)
        for normalized in (
            channel.propagation_latency_ticks,
            payload.input_bits,
            capacity.channel_count,
            reservation.payload_bits,
            reservation.send_ticks,
        ):
            assert normalized == 7
            assert type(normalized) is int


@pytest.mark.parametrize("invalid", [1.25, math.nan, math.inf, -math.inf, "7"])
def test_whole_quantity_boundaries_reject_invalid_values_with_context(invalid):
    """Fractional, nonfinite, and string whole quantities name their field."""
    factories = (
        ("propagation_latency_ticks", lambda: LinkConfig(invalid, None, "channel")),
        (
            "input_bits",
            lambda: PayloadSizeConfig(
                invalid,
                LinkQuantityBasis.DIRECT_AGGREGATE,
                None,
                "payload",
            ),
        ),
        (
            "per-channel capacity count",
            lambda: LinkCapacityConfig(
                2.5,
                LinkQuantityBasis.PER_CHANNEL,
                invalid,
                "capacity",
            ),
        ),
        (
            "payload_bits",
            lambda: Link(LinkConfig(0, None, "channel")).reserve(
                payload_bits=invalid, now_ticks=0
            ),
        ),
    )
    for field_name, factory in factories:
        with pytest.raises((TypeError, ValueError), match=field_name):
            factory()


def test_capacity_preserves_finite_real_rates_and_huge_integers():
    """Finite real capacity rates retain their input value without normalization."""
    fractional = Fraction(5, 2)
    huge = 10**400
    for value in (fractional, 2.5, huge):
        capacity = LinkCapacityConfig(
            value, LinkQuantityBasis.DIRECT_AGGREGATE, None, "capacity"
        )
        assert capacity.input_bits_per_us == value
        assert type(capacity.input_bits_per_us) is type(value)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_capacity_rejects_nonfinite_rates_with_field_context(invalid):
    """Nonfinite rates fail with the affected capacity field named."""
    with pytest.raises(ValueError, match="input_bits_per_us"):
        LinkCapacityConfig(
            invalid, LinkQuantityBasis.DIRECT_AGGREGATE, None, "capacity"
        )


def test_scientific_guards_and_relation_rules_remain_at_trust_boundaries():
    """Scientific signs, payload policy, path shapes, tiers, and time stay guarded."""
    with pytest.raises(ValueError, match="positive"):
        LinkCapacityConfig(
            0, LinkQuantityBasis.DIRECT_AGGREGATE, None, "capacity"
        )
    with pytest.raises(ValueError, match="nonnegative"):
        PayloadSizeConfig(
            -1, LinkQuantityBasis.DIRECT_AGGREGATE, None, "payload"
        )
    with pytest.raises(ValueError, match="configured default or actual"):
        LinkEdgeConfig(LinkConfig(0, None, "channel"), None, None)
    link = Link(LinkConfig(0, None, "channel"))
    link.reserve(payload_bits=1, now_ticks=2)
    with pytest.raises(ValueError, match="prior reservation"):
        link.reserve(payload_bits=1, now_ticks=1)

    model = logical_reference_profile().resolve()
    for path in logical_reference_profile().wired_paths():
        edge = getattr(logical_reference_profile(), path.value)
        payload_bits = 1 if edge.actual_payload_source is not None else None
        model.reserve(
            path,
            payload_bits=payload_bits,
            now_ticks=0,
            attribution=_valid_attribution(path),
        )


def test_round_only_controller_to_weak_transfer_rejects_a_relation():
    """A round-only controller-to-weak transfer cannot carry a window relation."""
    attribution = TrafficAttribution(
        _OPERATION_ID,
        _PATCH_IDS,
        None,
        1,
        1,
        _request_relation(DecoderTier.WEAK),
    )
    with pytest.raises(ValueError, match="does not accept a relation"):
        logical_reference_profile().resolve().reserve(
            LinkPath.WBD,
            payload_bits=8,
            now_ticks=0,
            attribution=attribution,
        )


def test_profile_functions_live_only_in_the_configuration_data_module():
    """Reference cards have one configuration-owned module and no compatibility shim."""
    for name in ("logical_reference_profile", "bandwidth_limited_profile"):
        function = getattr(link_profiles, name)
        assert callable(function)
        assert function.__module__ == "decsim.links.link_profiles"
        assert not hasattr(links_module, name)
        assert not hasattr(decsim, name)


def test_reference_cards_preserve_numeric_values_and_unicode_sources():
    """Reference cards preserve their calibrated numbers and Unicode source labels."""
    logical = _topology(logical_reference_profile())
    bandwidth = _topology(bandwidth_limited_profile())
    expected_latency = {
        "qc": us(0.15),
        "wbd": us(2.0),
        "wsd": us(0.5),
        "sbd": us(2.0),
        "wdo": us(1.0),
        "dd": us(0.5),
        "do": us(1.0),
        "oc": us(4.0),
        "cq": us(0.15),
    }
    for topology in (logical, bandwidth):
        actual_latency = {
            path: next(
                channel["propagation_latency_ticks"]
                for channel in topology["physical_channels"]
                if path in channel["member_paths"]
            )
            for path in expected_latency
        }
        assert actual_latency == expected_latency
        for channel in topology["physical_channels"]:
            source = channel["configuration_source"]
            assert type(source) is str
            source.encode("utf-8")
    capacities = {
        channel["member_paths"][0]: channel["capacity"]["aggregate_bits_per_us"]
        for channel in bandwidth["physical_channels"]
    }
    assert capacities == {
        "qc": 24.0,
        "wbd": 48.0,
        "wsd": 24.0,
        "sbd": 72.0,
        "wdo": 1_000_000.0,
        "dd": 24.0,
        "do": 1_000_000.0,
        "oc": 4_000_000.0,
        "cq": 1_000_000.0,
    }


def test_default_run_spec_build_matches_an_explicit_logical_card():
    """RunSpec selects the logical card at build time when links is unset."""
    default_result = RunSpec(ops=[]).build().result
    explicit_result = RunSpec(
        ops=[], links=logical_reference_profile()
    ).build().result
    assert default_result == explicit_result


def test_orientation_describes_card_customization_and_declared_extension():
    """Orientation distinguishes card customization from vocabulary extension."""
    orientation = (
        Path(__file__).parents[2] / "tmp" / "validation" / "ORIENTATION.md"
    ).read_text()
    normalized = re.sub(r"\s+", " ", orientation.lower())
    assert "segment vocabulary itself is the fixed measurement decomposition" in normalized
    assert "closed in code" in normalized
    assert "new measured segment is added by a declared three-line extension" in normalized
    assert "rather than by run configuration" in normalized



def test_unknown_relation_kind_is_rejected_before_ledger_append():
    """An unknown relation kind cannot enter the immutable traffic ledger."""
    attribution = TrafficAttribution(
        _OPERATION_ID, _PATCH_IDS, 3, 1, 2, object()
    )
    model = logical_reference_profile().resolve()
    with pytest.raises(ValueError, match="requires a request relation"):
        model.reserve(
            LinkPath.WSD,
            payload_bits=1,
            now_ticks=0,
            attribution=attribution,
        )
    assert traffic_json_value(model.snapshot())["transfers"] == []


def test_negative_reservation_payload_is_rejected_before_accounting():
    """A negative payload cannot create a physical reservation or counter entry."""
    link = Link(LinkConfig(0, None, "channel"))
    with pytest.raises(ValueError, match="payload_bits must be nonnegative"):
        link.reserve(payload_bits=-1, now_ticks=0)
    assert link.counters_snapshot().transfer_count == 0


def test_negative_reservation_time_is_rejected_before_accounting():
    """A negative send time cannot create a physical reservation or counter entry."""
    link = Link(LinkConfig(0, None, "channel"))
    with pytest.raises(ValueError, match="now_ticks must be nonnegative"):
        link.reserve(payload_bits=1, now_ticks=-1)
    assert link.counters_snapshot().transfer_count == 0


def test_decoder_boundary_transfer_requires_boundary_provenance():
    """A decoder boundary transfer cannot omit its boundary provenance record."""
    attribution = TrafficAttribution(
        _OPERATION_ID, _PATCH_IDS, 3, 1, 2
    )
    model = logical_reference_profile().resolve()
    with pytest.raises(ValueError, match="dd requires a boundary relation"):
        model.reserve(
            LinkPath.DD,
            payload_bits=None,
            now_ticks=0,
            attribution=attribution,
        )
    assert traffic_json_value(model.snapshot())["transfers"] == []



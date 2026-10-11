"""The records a part's slot takes, found by name (parameters/kinds.py).

The referent is the slot's own type: a record is offered where the type
admits it (a record class and its subclasses, a union's members, or a
Protocol's members with the same method names and a result alike),
which is what OMNeT++ checks of a `like` slot's typename (simulation
manual §3.11).
"""

import dataclasses
import typing

import pytest

import decsim.decoders.settings as decoder_settings
import decsim.parameters.fields as fields
import decsim.parameters.kinds as kinds
import decsim.ports as ports
import decsim.windows.schemes.sliding as sliding_scheme
import decsim.windows.settings as window_settings
from decsim.decoders.relay_belief_propagation import (
    decoder as relay_belief_propagation,
)
from decsim.decoders.union_find import decoder as union_find


@dataclasses.dataclass(frozen=True)
class PluginDecoderSettings:
    """A decoder record decsim does not define, as a lab's would be."""

    name: str = "plugin"

    def build(self) -> ports.Decoder:
        """Never called here: the slot reads only its shape."""


class PluginDecoder:
    """A lab's decoder whose settings record is nested in it."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The nested record, the docs' own convention."""

        name: str = "nested_plugin"

        def build(self) -> ports.Decoder:
            """Never called here: the slot reads only its shape."""


def _slot_hint(record_class: type, field_name: str):
    hints = typing.get_type_hints(record_class)
    return hints[field_name]


def test_a_decoder_slot_offers_every_decoder_and_no_window_scheme():
    hint = _slot_hint(decoder_settings.DecoderPoolSettings, "algorithm")

    offered = kinds.records_for(hint)

    assert union_find.UnionFindDecoder.Settings in offered
    assert relay_belief_propagation.RelayBeliefPropagationDecoder.Settings in (
        offered
    )
    assert sliding_scheme.SlidingWindowScheme.Settings not in offered


def test_a_window_scheme_slot_offers_the_sliding_scheme():
    hint = _slot_hint(window_settings.WindowSettings, "scheme")

    offered = kinds.records_for(hint)

    assert sliding_scheme.SlidingWindowScheme.Settings in offered


def test_a_record_is_written_by_its_row_name_and_its_class_name():
    pool_names = kinds.names_of(decoder_settings.DecoderPoolSettings)
    union_find_names = kinds.names_of(union_find.UnionFindDecoder.Settings)

    assert pool_names == ("DecoderPool", "DecoderPoolSettings")
    assert union_find_names[0] == "union_find"
    assert "UnionFindDecoder" in union_find_names


def test_a_name_the_slot_offers_is_its_record_class():
    hint = _slot_hint(decoder_settings.DecoderPoolSettings, "algorithm")

    chosen = kinds.record_class_named("union_find", hint, "weak_decoder")

    assert chosen is union_find.UnionFindDecoder.Settings


def test_an_unknown_name_is_refused_with_the_near_names_and_the_choices():
    hint = _slot_hint(decoder_settings.DecoderPoolSettings, "algorithm")

    with pytest.raises(fields.ParameterError) as refused:
        kinds.record_class_named("union_fnd", hint, "weak_decoder.algorithm")

    message = str(refused.value)
    assert "Did you mean union_find?" in message
    assert "relay_bp" in message


def test_a_record_outside_decsim_fills_a_slot_by_its_import_path():
    hint = _slot_hint(decoder_settings.DecoderPoolSettings, "algorithm")
    path = f"{__name__}.PluginDecoderSettings"

    chosen = kinds.record_class_named(path, hint, "weak_decoder.algorithm")

    assert chosen is PluginDecoderSettings


def test_a_record_the_slot_does_not_take_is_refused_by_its_import_path():
    hint = _slot_hint(window_settings.WindowSettings, "scheme")
    path = f"{__name__}.PluginDecoderSettings"

    with pytest.raises(fields.ParameterError, match="not a record this slot"):
        kinds.record_class_named(path, hint, "windows.scheme")


def test_an_import_path_reaches_a_record_nested_in_a_class():
    hint = _slot_hint(decoder_settings.DecoderPoolSettings, "algorithm")
    path = f"{__name__}.PluginDecoder.Settings"

    record_class = kinds.record_class_named(path, hint, "pool.algorithm")

    assert record_class is PluginDecoder.Settings


def test_an_import_path_no_prefix_of_which_imports_is_refused():
    with pytest.raises(fields.ParameterError, match="no module of"):
        kinds.imported_class("no_such_lab.parts.Decoder", "a test")


def test_a_record_outside_decsim_is_written_by_its_import_path():
    written = kinds.unique_name(PluginDecoder.Settings)
    by_row = kinds.unique_name(union_find.UnionFindDecoder.Settings)

    assert written == f"{__name__}.PluginDecoder.Settings"
    assert by_row == "union_find"

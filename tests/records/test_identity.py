"""The stable-identity helpers of decsim/records/identity.py.

The canonical bytes are decsim's own frame, not a referent's: a type tag,
a length and the payload, checked here against the frames written out by
hand.
"""

import decsim.records.identity as identity_records


def test_a_str_identity_is_exactly_a_str_without_surrogates():
    """A subclass, a nonstring and a surrogate code point are not identities."""

    class StringSubclass(str):
        pass

    assert identity_records.is_stable_identity("")
    assert identity_records.is_stable_identity("logical-π")
    subclass_value = StringSubclass("logical")
    assert not identity_records.is_stable_identity(subclass_value)
    assert not identity_records.is_stable_identity(3.0)
    assert not identity_records.is_stable_identity("\ud800")


def test_stable_identity_accepts_recursive_exact_values_only():
    """Stable identities contain only exact integers, strings, and tuples."""
    assert identity_records.is_stable_identity((1, "patch", (2, "round")))
    assert identity_records.is_stable_identity(())
    assert not identity_records.is_stable_identity(True)
    assert not identity_records.is_stable_identity([1, "patch"])
    with_an_object = (1, object())
    assert not identity_records.is_stable_identity(with_an_object)


def test_stable_identity_comparison_rejects_cross_type_equality():
    """Comparison never aliases values of different runtime types."""
    assert identity_records.same_stable_identity((1, "1"), (1, "1"))
    assert not identity_records.same_stable_identity(1, True)
    assert not identity_records.same_stable_identity((1,), (True,))
    assert not identity_records.same_stable_identity((1,), (1, 2))


def test_canonical_identity_bytes_are_injective_across_identity_kinds():
    """The bytes distinguish integer, string, and tuple structure."""
    identities = (
        1,
        "1",
        (1,),
        ("1",),
        (1, "1"),
        ((1,),),
    )
    encodings = tuple(
        identity_records.stable_identity_bytes(value) for value in identities
    )
    assert len(set(encodings)) == len(identities)
    encoded_integer = b"I" + (1).to_bytes(8, "big") + b"1"
    encoded_string = b"S" + (1).to_bytes(8, "big") + b"1"
    framed_integer_length = len(encoded_integer)
    encoded_tuple = (
        b"T"
        + (1).to_bytes(8, "big")
        + framed_integer_length.to_bytes(8, "big")
        + encoded_integer
    )
    unicode_payload = "π".encode()
    unicode_length = len(unicode_payload)
    encoded_unicode = b"S" + unicode_length.to_bytes(8, "big") + unicode_payload
    assert encodings[0] == encoded_integer
    assert encodings[1] == encoded_string
    assert encodings[2] == encoded_tuple
    assert identity_records.stable_identity_bytes("π") == encoded_unicode


def test_stable_identity_ordering_and_json_preserve_typed_structure():
    """The order key is the canonical bytes, and the json keeps the types."""
    identity = (3, "patch", (4,))
    assert identity_records.stable_identity_order_key(
        identity
    ) == identity_records.stable_identity_bytes(identity)
    assert identity_records.stable_identity_json(identity) == {
        "kind": "tuple",
        "value": None,
        "items": [
            {"kind": "integer", "value": "3", "items": None},
            {"kind": "string", "value": "patch", "items": None},
            {
                "kind": "tuple",
                "value": None,
                "items": [{"kind": "integer", "value": "4", "items": None}],
            },
        ],
    }

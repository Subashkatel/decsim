"""What decsim accepts as an operation, patch or window key.

An identity is an int, a str with no surrogate code point, or a tuple of
identities. Every ordering and every recorded key goes through these
bytes, so a run's order and its json never depend on Python's cross-type
comparison or on a hash seed.
"""

from typing import Any


def is_stable_identity(value: Any) -> bool:
    """Whether a value may serve as a key decsim orders and records."""
    value_type = type(value)
    if value_type is int:
        return True
    if value_type is str:
        return _is_stable_string(value)
    if value_type is tuple:
        return all(is_stable_identity(item) for item in value)
    return False


def same_stable_identity(left: Any, right: Any) -> bool:
    """Compare stable identities without Python's cross-type equality."""
    if type(left) is not type(right):
        return False
    if type(left) is tuple:
        if len(left) != len(right):
            return False
        return all(
            same_stable_identity(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if type(left) is int or type(left) is str:
        return left == right
    return False


def stable_identity_bytes(identity: Any) -> bytes:
    """The identity's bytes, type-tagged and length-framed.

    An int, a str and a tuple never encode alike, and a tuple frames
    every item, so no two identities share an encoding.
    """
    if type(identity) is int:
        text = str(identity)
        encoded = text.encode("ascii")
        length = len(encoded)
        framed_length = length.to_bytes(8, "big")
        return b"I" + framed_length + encoded
    if type(identity) is str:
        encoded = identity.encode("utf-8")
        length = len(encoded)
        framed_length = length.to_bytes(8, "big")
        return b"S" + framed_length + encoded
    encoded_items = tuple(stable_identity_bytes(item) for item in identity)
    item_count = len(encoded_items)
    framed_count = item_count.to_bytes(8, "big")
    framed_items = []
    for item in encoded_items:
        item_length = len(item)
        framed_length = item_length.to_bytes(8, "big")
        framed_item = framed_length + item
        framed_items.append(framed_item)
    joined_items = b"".join(framed_items)
    return b"T" + framed_count + joined_items


def stable_identity_order_key(identity: Any) -> bytes:
    """The sort key of an identity: its canonical bytes.

    Sorting mixed identity types is therefore sorting those bytes, which
    is total and does not depend on the types' own comparisons.
    """
    return stable_identity_bytes(identity)


def stable_identity_json(identity: Any) -> dict:
    """The identity as recorded json: its kind, its value, its items.

    An integer's value is its decimal text, so a key too large for a
    json number survives the round trip.
    """
    if type(identity) is int:
        return {"kind": "integer", "value": str(identity), "items": None}
    if type(identity) is str:
        return {"kind": "string", "value": identity, "items": None}
    items = [stable_identity_json(item) for item in identity]
    return {"kind": "tuple", "value": None, "items": items}


def _is_stable_string(value: Any) -> bool:
    """Whether a str is exactly a str and free of surrogate code points.

    A surrogate has no utf-8 encoding, so it could not be framed into
    the canonical bytes at all.
    """
    if type(value) is not str:
        return False
    for character in value:
        code_point = ord(character)
        if 0xD800 <= code_point <= 0xDFFF:
            return False
    return True

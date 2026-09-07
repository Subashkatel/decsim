"""The seed-path records of decsim/records/seeds.py.

The framed bytes are decsim's own frame (a tag, a length, the payload),
checked against frames written out by hand; decsim/seeding.py hashes them
into a component's seed.
"""

import pytest

import decsim.records.seeds as seed_records


def test_seed_path_segments_have_distinct_framed_encodings():
    """Field, string, none and integer edges encode distinctly."""
    segments = (
        seed_records.RunSeedPathSegment("field", "node"),
        seed_records.RunSeedPathSegment("string_key", "node"),
        seed_records.RunSeedPathSegment("none_key", None),
        seed_records.RunSeedPathSegment("integer_key", 12),
    )
    encodings = tuple(segment.canonical_bytes() for segment in segments)
    assert len(set(encodings)) == len(encodings)
    assert encodings == (
        b"F" + (4).to_bytes(4, "big") + b"node",
        b"S" + (4).to_bytes(4, "big") + b"node",
        b"N" + (0).to_bytes(4, "big"),
        b"I" + (2).to_bytes(4, "big") + b"12",
    )


def test_seed_path_segments_reject_unknown_kinds():
    """An unknown segment kind has no canonical encoding."""
    segment = seed_records.RunSeedPathSegment("unknown", "value")
    with pytest.raises(KeyError):
        segment.canonical_bytes()


def test_two_reservations_of_the_same_seed_are_distinct_values():
    """A reservation is its own identity, so a leaf holds exactly its own."""
    prepared_state = object()
    reservation = seed_records.RunSeedReservation("entropy", 19, prepared_state)
    same_proposal = seed_records.RunSeedReservation(
        "entropy", 19, prepared_state
    )
    assert reservation != same_proposal
    assert reservation == reservation

"""The seed-path records of decsim/records/seeds.py.

The framed bytes are decsim's own frame (a tag, a length, the payload),
checked against frames written out by hand; decsim/seeding.py hashes them
into a component's seed.
"""

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

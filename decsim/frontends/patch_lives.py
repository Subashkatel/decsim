"""Each code block's life as one decode stream: the patch streams.

A lattice-surgery decoder decodes each patch's syndrome history: a
region is one patch over a run of its rounds (SWIPER 2412.05115,
swiper/window_builder.py SpacetimeRegion; Bombín et al. 2303.04846 Sec.
IV). A patch's history runs from the operation that starts it to the
discard that ends it (swiper/device_manager.py _get_initialized_patches),
every round it measures, an operation's or an idle one, in between. A
patch is an opaque id for one code block, so the lives hold for any
code whose program names its blocks.
"""

import decsim.records.program as program_records


def lives(operations: list) -> tuple:
    """Each patch's lives that emit detector data, in the order they end.

    A life whose operations emit none has no rounds to decode, so no
    stream.
    """
    open_by_patch = {}
    closed = []
    for operation in operations:
        _add_to_lives(operation, open_by_patch, closed)
    for patch, members in open_by_patch.items():
        closed.append((patch, members))
    return _decoded_lives(closed)


def owner(life: program_records.PatchLife) -> program_records.Operation:
    """The life's stream owner: the patch it measures, open-ended."""
    return program_records.Operation(
        life.stream_id,
        f"patch {life.patch} life {life.stream_id[2]}",
        (life.patch,),
        patches=(life.patch,),
        kind=program_records.OpKind.MEMORY,
    )


def _add_to_lives(operation, open_by_patch: dict, closed: list) -> None:
    """The operation joins its patches' lives; a discard ends them."""
    for patch in operation.patches:
        members = open_by_patch.setdefault(patch, [])
        members.append(operation)
    if operation.kind is not program_records.OpKind.DISCARD:
        return
    for patch in operation.patches:
        ended = open_by_patch.pop(patch)
        closed.append((patch, ended))


def _decoded_lives(closed: list) -> tuple:
    """The lives that emit, each numbered among its patch's lives."""
    count_by_patch = {}
    decoded = []
    for patch, members in closed:
        index = count_by_patch.get(patch, 0)
        count_by_patch[patch] = index + 1
        if not any(member.emits_detector_data for member in members):
            continue
        operation_ids = tuple(member.id for member in members)
        stream_id = ("patch", patch, index)
        life = program_records.PatchLife(stream_id, patch, operation_ids)
        decoded.append(life)
    return tuple(decoded)

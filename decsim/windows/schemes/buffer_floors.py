"""The buffer floor a scheme refuses to run below without a reason.

Windowed accuracy degrades when the buffer is thinner than the
literature's floor (Skoric 2209.08552, Tan PRX Quantum 4, 040344,
Bombin 2303.04846), so a run below it is deliberate and says so in the
yaml.
"""


def require_buffer_floor(geometry, floor: int, floor_label: str) -> None:
    """Refuse a buffer below the literature floor without a justification.

    A justification above the floor is a stale one and is refused too.
    """
    below = geometry.buffer_round_count < floor
    justification = geometry.window_floor_justification
    if below and not justification:
        raise ValueError(
            f"buffer_rounds={geometry.buffer_round_count} is below the "
            f"{floor_label} {floor} for {geometry.code_name}; windowed "
            f"accuracy degrades (Skoric 2209.08552, Tan PRX Quantum 4, "
            f"040344, Bombin 2303.04846). Raise buffer_rounds_override to "
            f"{floor}, or set window_floor_justification to run below the "
            "floor deliberately."
        )
    if justification and not below:
        raise ValueError(
            f"window_floor_justification is set but buffer_rounds="
            f"{geometry.buffer_round_count} is not below the {floor_label} "
            f"{floor}; remove the justification."
        )

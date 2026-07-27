"""Shared helpers for turning window decoder selections into DecodeResult objects."""

from __future__ import annotations

from ..message import DecodeJob, DecodeResult


def payload_syndrome(job: DecodeJob):
    """Concatenate payload bits into one syndrome vector."""
    import numpy as np

    if not job.payloads:
        return np.zeros(0, dtype=np.uint8)
    return np.concatenate([
        np.asarray(payload.bits, dtype=np.uint8)
        for payload in job.payloads
        if payload.bits is not None
    ])


def check_syndrome_size(job: DecodeJob, syndrome, model) -> None:
    """Fail when payload bits and detector rows do not line up."""
    if syndrome.size == model.check.shape[0]:
        return
    raise ValueError(
        f"{job.label}: payload bits ({syndrome.size}) do not match the window "
        f"error model's detectors ({model.check.shape[0]}). The device and "
        "the cluster's model build must use the same folded-round convention."
    )


def result_from_selected_faults(job: DecodeJob, model, selected) -> DecodeResult:
    """Keep owned selected faults and convert them into a DecodeResult."""
    import numpy as np

    selected = np.asarray(selected, dtype=np.uint8)
    committed = selected.astype(bool) & model.owned
    observable_flips = (model.obs @ committed.astype(np.uint8)) % 2
    return DecodeResult(
        job.op_id,
        job.window_id,
        correction=committed.astype(np.uint8),
        logical_observables=tuple(int(bit) for bit in observable_flips),
        boundary_defects=_boundary_defects(model, committed),
    )


def _boundary_defects(model, committed) -> dict | None:
    """Artificial defects that cross this window's commit boundary."""
    import numpy as np

    defects: dict = {}
    for column_index in np.nonzero(committed)[0]:
        for detector_id in model.future_flips.get(int(column_index), ()):
            round_index, position = model.defect_positions[detector_id]
            mask = defects.setdefault(round_index, [])
            if len(mask) <= position:
                mask.extend([0] * (position + 1 - len(mask)))
            mask[position] ^= 1
    return defects or None

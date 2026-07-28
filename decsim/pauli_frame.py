"""Software Pauli frame: the X/Z correction bits owed to each qubit.

Instead of applying physical corrections, the simulator XORs decoder
corrections into this frame and folds the frame into measurement outcomes
(``fold``). Pure state — it only receives final values and imports nothing
from the rest of decsim.
"""

from __future__ import annotations


_X_PART = {"I": 0, "X": 1, "Y": 1, "Z": 0}
_Z_PART = {"I": 0, "X": 0, "Y": 1, "Z": 1}


class PauliFrame:
    """Per-qubit X/Z Pauli frame; pure state, receives only final values."""

    def __init__(self) -> None:
        self.x: dict = {}
        self.z: dict = {}

    def run_manifest_config(self):
        return {"kind": "pauli_frame"}

    def accumulate(self, qubit, *, x: int = 0, z: int = 0) -> None:
        """XOR an X/Z correction into one qubit's frame (Pauli corrections commute)."""
        if x:
            self.x[qubit] = self.x.get(qubit, 0) ^ (x & 1)
        if z:
            self.z[qubit] = self.z.get(qubit, 0) ^ (z & 1)

    def apply_pauli(self, qubit, pauli: str) -> None:
        """Fold a named Pauli byproduct ('I'/'X'/'Y'/'Z') into a qubit's frame."""
        pauli = pauli.upper()
        if pauli not in _X_PART:
            raise ValueError(f"pauli must be one of I/X/Y/Z (got {pauli!r})")
        self.accumulate(qubit, x=_X_PART[pauli], z=_Z_PART[pauli])

    def apply_s(self, qubit) -> None:
        """Conjugate the frame by the Clifford S gate: X -> Y, i.e. z ^= x."""
        if self.x.get(qubit, 0):
            self.z[qubit] = self.z.get(qubit, 0) ^ 1

    def x_of(self, qubit) -> int:
        return self.x.get(qubit, 0)

    def z_of(self, qubit) -> int:
        return self.z.get(qubit, 0)

    def measurement_flip(self, qubit, basis: str) -> int:
        """Return the frame-induced flip bit for a measurement of ``qubit`` in ``basis``."""
        base = basis.upper().lstrip("M") or "Z"
        if base == "Z":
            return self.x.get(qubit, 0)
        if base == "X":
            return self.z.get(qubit, 0)
        if base == "Y":
            return self.x.get(qubit, 0) ^ self.z.get(qubit, 0)
        raise ValueError(f"basis must be X/Y/Z (or MX/MY/MZ); got {basis!r}")

    def fold(self, qubit, basis: str, raw_bit: int) -> int:
        """Fold the frame into a raw measurement outcome, returning the corrected bit."""
        return (int(raw_bit) ^ self.measurement_flip(qubit, basis)) & 1

    def clear(self, qubit) -> None:
        """Drop a qubit's frame (e.g. after a destructive measurement)."""
        self.x.pop(qubit, None)
        self.z.pop(qubit, None)

    def snapshot(self) -> dict:
        """A compact copy of the live frame (for audit / assertions)."""
        qubits = set(self.x) | set(self.z)
        return {q: (self.x.get(q, 0), self.z.get(q, 0)) for q in sorted(qubits)
                if self.x.get(q, 0) or self.z.get(q, 0)}

    def __repr__(self) -> str:
        return f"PauliFrame({self.snapshot()})"

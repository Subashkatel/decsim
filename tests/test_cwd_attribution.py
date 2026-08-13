"""Regression for canonical CWD traffic attribution.

Stable metadata ordering is a decsim repository invariant implemented by
``stable_identity_order_key`` and already used by
``SyndromeIngress._round_attribution``. It is not a paper-derived decoding
rule. Packet fragment order remains device/payload order.
"""

import pytest

from decsim.codes import SurfaceCodeModel
from decsim.decoders import PerRoundDecoder
from decsim.devices import SyndromeBitDevice, TimingOnlyDevice
from decsim.links import LinkPath
from decsim.message import Operation, QPUReadout, SyndromePayload, stable_identity_json
from decsim.planner import FixedRounds
from decsim.run_spec import RunSpec
from decsim.window_manager import WindowManager


class _NoSyndromePerPatchDevice(TimingOnlyDevice):
    """Emit one patch fragment per round without syndrome bits."""

    def round_payloads(self, op, round_index):
        """Preserve operation patch order while leaving bits/size unresolved."""
        return [
            QPUReadout(op.id, patch, round_index)
            for patch in op.patches
        ]


@pytest.mark.parametrize("with_syndrome", [False, True])
def test_unsorted_fragment_order_is_canonicalized_only_for_cwd_attribution(
    monkeypatch,
    with_syndrome,
):
    """Canonicalize metadata for packets both with and without syndrome bits."""
    observed_packets = []
    original = WindowManager.accept_window_input

    def record_packet(window_manager, packet):
        """Observe the production CWD boundary without changing its behavior."""
        observed_packets.append((
            packet.round_index,
            tuple(fragment.patch_id for fragment in packet.fragments),
            tuple(fragment.size_bits for fragment in packet.fragments),
            tuple(fragment.bits for fragment in packet.fragments),
        ))
        return original(window_manager, packet)

    monkeypatch.setattr(WindowManager, "accept_window_input", record_packet)
    code = SurfaceCodeModel(d=3)
    device = (
        SyndromeBitDevice(code, per_patch=True)
        if with_syndrome
        else _NoSyndromePerPatchDevice()
    )
    completed = RunSpec(
        ops=[Operation(
            0,
            "two-patch",
            (0, 1),
            patches=(1, 0),
        )],
        code=code,
        rounds_policy=FixedRounds(3),
        device=device,
        decoder=PerRoundDecoder(0.0),
        seed=7,
    ).build()

    assert observed_packets
    assert all(packet[1] == (1, 0) for packet in observed_packets)
    assert all(
        all(bits is not None for bits in packet[3]) == with_syndrome
        for packet in observed_packets
    )

    expected_payload_bits = {
        round_index: (
            sum(fragment_sizes)
            if all(size is not None for size in fragment_sizes)
            else None
        )
        for round_index, _patch_ids, fragment_sizes, _bits
        in observed_packets
    }
    cwd_transfers = [
        transfer
        for transfer in completed.window_manager.links.traffic_json_value()[
            "transfers"
        ]
        if transfer["path"] == LinkPath.CWD.value
    ]
    assert len(cwd_transfers) == completed.window_manager.total_windows
    expected_patch_ids = [stable_identity_json(0), stable_identity_json(1)]
    for transfer in cwd_transfers:
        attribution = transfer["attribution"]
        assert attribution["patch_ids"] == expected_patch_ids
        assert attribution["window_id"] is not None
        selected = range(attribution["round_lo"], attribution["round_hi"] + 1)
        sizes = [expected_payload_bits[index] for index in selected]
        assert transfer["payload_bits"] == (
            sum(sizes) if all(size is not None for size in sizes) else None
        )

"""The QPU: its cycle clock, syndrome sources, code cards and factories.

cycle_clock runs one QEC cycle clock and emits one syndrome round per
cycle on every live patch; syndrome_devices and stim_device are the
syndrome sources it drives; code_geometry holds the code cards and
layouts the patch-to-code map; round_policies say how many rounds an
operation occupies; magic_state_factories supply the magic states that
non-Clifford operations wait for. Circuits come from
stim.Circuit.generated at every call site.
"""

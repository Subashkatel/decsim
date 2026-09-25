[decsim docs](../README.md) › [How-to guides](README.md)

# How to run a workload whose next operation waits on a decision

A memory experiment measures one patch and nothing waits on the answer,
so the two links that carry a decision back to the machine,
`frame_to_controller` and `controller_to_qpu`, never fire on it. This
page builds the smallest workload that does close that loop: two
operations, the second held until the first operation's decision reaches
the QPU.

To use only the strong decoder, select `escalation.kind: strong_only` and
configure `strong_decoder`. In Python, pass
`EscalationSettings(kind="strong_only")` from `decsim.escalation.settings`
as the machine's escalation settings. The same selection works for finite
streams and live protection: input and idle rounds use the strong buffer,
window jobs use the strong decoder, and corrections return through the frame.
There is no weak decode or escalation hop. Detection events may be formed at
the controller or decoder; formation in the weak buffer is incompatible.

## 1. Know why this one is Python

`decsim.producers:memory_circuit`, the maker every shipped config names,
builds one operation for the whole shot, so there is nothing for a
second operation to wait on. This page builds the operation list as a
machine, in Python, to show each field; a maker function that returns
the same list runs it from a yaml
([plug in a workload maker](plug_in_a_workload_maker.md)).

## 2. Build the two operations

The waiting is one field: `blocked_by` on the operation that waits, set
to the id of the operation whose decision releases it
(`decsim/records/program.py`, the `Operation` record). Both operations
run the same Stim memory circuit, on patches of their own.

```python
"""The smallest workload with a decision fed back to the QPU."""

import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.machine as machine_module
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.records.program as program_records
import decsim.settings as machine_settings

circuit = workload_settings.memory_circuit(
    "surface_code:rotated_memory_z", 6, 3, 0.001
)
first = program_records.Operation(
    id=1, name="mem0", qubits=(0,), patches=(0,), circuit=circuit
)
second = program_records.Operation(
    id=2, name="mem1", qubits=(1,), patches=(1,), circuit=circuit, blocked_by=1
)
workload = workload_settings.WorkloadSettings(
    operations=(first, second),
    rounds_policy=round_policies.FixedRounds(6),
)
settings = machine_settings.MachineSettings(
    workload=workload,
    qpu=qpu_settings.QpuSettings(distance=3, device=stim_device.StimDevice()),
    weak_decoder=decoder_settings.DecoderSettings(
        kind=1.0, engine_clock=config.Clock(1000)
    ),
    links=link_profiles.logical_reference_profile(),
)
machine = machine_module.Machine.build(settings, 0)
result = machine.run()
```

Three of those lines are choices worth naming.

**`rounds_policy`.** A workload built in Python does not fix its own
rounds, so say how many each operation runs. `FixedRounds(6)` is six for
every operation, matching the circuit built above.

**The decoder is priced.** `kind=1.0` charges the decode one microsecond
from a card instead of the wall clock a real decode took, so the ticks
below are the same on your machine as on this page. A card needs its
engine's clock, which is what `engine_clock` is;
[how to run a timing study](run_a_timing_only_study.md) is the longer
version of this choice.

**The links are the default card.** `logical_reference_profile` prices
`frame_to_controller` at no propagation and one 32-bit word a 250 MHz
cycle, and `controller_to_qpu` at 0.088 microseconds and one 128-bit word
a cycle, which are the numbers the next step checks.

## 3. Read the loop closing

Append this to the script:

```python
for transfer in result.link_traffic["transfers"]:
    if transfer["path"] in ("frame_to_controller", "controller_to_qpu"):
        print(
            transfer["path"],
            transfer["send_ticks"] / 1000000,
            "->",
            transfer["delivery_ticks"] / 1000000,
            "us,",
            transfer["payload_bits"],
            "bits",
        )
for event in machine.observation.command_events.events:
    print(
        event.kind,
        event.tick / 1000000,
        "us, operation",
        event.command.operation.id,
    )
```

```
frame_to_controller 7.703125 -> 7.707125 us, 32 bits
controller_to_qpu 7.707125 -> 7.799125 us, 128 bits
ARRIVED 0.0 us, operation 1
STARTED 0.0 us, operation 1
ARRIVED 7.799125 us, operation 2
STARTED 8.8 us, operation 2
```

That is the whole loop. Operation 1's last window committed, the Pauli
frame decided, and the decision left for the controller at 7.703125 as
one 32-bit control bus word. The word took one 4 ns cycle to cross, the
released command took the card's 0.088 microseconds plus one 4 ns cycle
for its 128-bit instruction word, and operation 2's command arrived at
the QPU at 7.799125, which is 7.703125 plus 0.096. Run the same script with `blocked_by`
removed and both lines disappear: no operation waits, so no decision
travels.

The second STARTED line at 8.8 is later than its ARRIVED line because
the QPU's own cycle clock takes the command at the next boundary it can
start on, which is its business and not the loop's.

## 4. Check it is the whole path

Two numbers are worth reading against the configuration rather than
against each other.

- **The payload widths are stated, not measured.** A decision is one bus
  word and a command is one instruction word, both from the link card
  and not from the run, because the run has no bit-level model of either
  (`decsim/links/link_profiles.py`, the two default-payload paths).
- **The wait is the decision's, not the decode's.** Operation 2 was not
  waiting for a decoder to be free; it was waiting for an answer to
  travel. Add 4.0 microseconds to the propagation latency of the
  profile's `frame_to_controller` channel and the command arrives at
  11.799125: exactly 4.0 later, with its own hop unchanged.

`tests/links/test_data_through.py` runs this same workload as a test, on
a measured decoder rather than a card, and asserts the two transfers and
their widths.

## Read next

- [The data path](../explanation/data_path.md): every hop of the loop,
  in order, and who owns each one.
- [How to read a trace and follow one round or one window](read_a_trace.md):
  the same run seen as a trace.

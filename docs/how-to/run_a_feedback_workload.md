[decsim docs](../README.md) › [How-to guides](README.md)

# How to run a workload whose next operation waits on a decision

A memory experiment measures one patch and nothing waits on the answer,
so the two links that carry a decision back to the machine,
`frame_to_controller` and `controller_to_qpu`, never fire on it. This
page builds the smallest workload that does close that loop: two
operations, the second held until the first operation's decision reaches
the QPU.

## 1. Know why this one is Python and not yaml

`workload.kind` names one of four rows, and the row that takes an
operation list refuses a yaml:

```
workload.kind circuit_list takes a list of Operation records with their
Stim circuits, which a yaml scalar cannot carry; build it in Python
(WorkloadSettings(operations=...))
```

`memory_circuit`, the row every shipped config uses, builds one
operation for the whole shot, so there is nothing for a second operation
to wait on. Build this one as a machine, in Python.

## 2. Build the two operations

The waiting is one field: `blocked_by` on the operation that waits, set
to the id of the operation whose decision releases it
(`decsim/records/program.py`, the `Operation` record). Both operations
run the same Stim memory circuit, on patches of their own.

```python
"""The smallest workload with a decision fed back to the QPU."""

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
    qpu=qpu_settings.QpuSettings(
        distance=3, device=stim_device.StimDevice()
    ),
    weak_decoder=decoder_settings.DecoderSettings(
        kind=1.0, engine_megahertz=1000.0
    ),
    links=link_profiles.logical_reference_profile(),
)
machine = machine_module.Machine.build(settings, 0)
result = machine.run()
```

Three of those lines are choices worth naming.

**`rounds_policy`.** A `circuit_list` workload does not fix its own
rounds, so say how many each operation runs. `FixedRounds(6)` is six for
every operation, matching the circuit built above.

**The decoder is priced.** `kind=1.0` charges the decode one microsecond
from a card instead of the wall clock a real decode took, so the ticks
below are the same on your machine as on this page. A card needs its
engine's clock, which is what `engine_megahertz` is;
[how to run a timing study](run_a_timing_only_study.md) is the longer
version of this choice.

**The links are the default card.** `logical_reference_profile` prices
`frame_to_controller` at 4.0 microseconds and `controller_to_qpu` at
0.15, which is the number the next step checks.

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
    print(event.kind, event.tick / 1000000, "us, operation",
          event.command.operation.id)
```

```
frame_to_controller 10.797 -> 14.797 us, 32 bits
controller_to_qpu 14.797 -> 14.947 us, 128 bits
ARRIVED 0.0 us, operation 1
STARTED 0.0 us, operation 1
ARRIVED 14.947 us, operation 2
STARTED 15.4 us, operation 2
```

That is the whole loop. Operation 1's last window committed, the Pauli
frame decided, and the decision left for the controller at 10.797 as one
32-bit control bus word. It took the card's 4.0 microseconds to get
there, the released command took the card's 0.15 more as one 128-bit
instruction word, and operation 2's command arrived at the QPU at
14.947, which is 10.797 plus 4.15. Run the same script with `blocked_by`
removed and both lines disappear: no operation waits, so no decision
travels.

The second STARTED line at 15.4 is later than its ARRIVED line because
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
  travel. Double the propagation latency of the profile's
  `frame_to_controller` channel, from 4.0 microseconds to 8.0, and the
  command arrives at 18.947: exactly 4.0 later, with its own hop
  unchanged.

`tests/links/test_data_through.py` runs this same workload as a test, on
a measured decoder rather than a card, and asserts the two transfers and
their widths.

## Read next

- [The data path](../explanation/data_path.md): every hop of the loop,
  in order, and who owns each one.
- [How to read a trace and follow one round or one window](read_a_trace.md):
  the same run seen as a trace.

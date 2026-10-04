[decsim docs](../README.md) › [How-to guides](README.md)

# How to add a component to a part

You have a new kind of component and you want it in the machine: a new
stage on the readout path, say, or a new step between the frame and the
controller. It goes in the part that holds that stage of the loop, and
the part builds it and wires it.

If what you have is another kind of a component decsim already has,
you want [How to add a decoder backend](add_a_decoder_backend.md) or
[How to add a store, a link card, a signal or a policy](add_a_store_card_signal_or_policy.md)
instead: a class that fills the component's port, with a settings
record a machine names.

The worked example on this page is `DecisionDispatch`
(`decsim/pauli_frame/decision_dispatch.py`), which sends a released
decision from the Pauli frame to the controller. It lives in the control
part, `decsim/build/control.py`.

## 1. Write the class

Declare each neighbour the class calls as a port, and take only the
engine and settings in the constructor:

```python
class DecisionDispatch:
    """Sends one released decision over frame_to_controller."""

    link = ports.Port(ports.Link, optional=True)
    instruction_output = ports.Port(ports.InstructionReceiver)

    def __init__(self, engine) -> None:
        self.engine = engine
```

A port names a `typing.Protocol` in `decsim/ports.py`: the methods this
class calls on its neighbour. If none of the protocols there has them,
add one. An optional port reads as `None` until something binds it,
which is how a run leaves out a neighbour it does not have.

Schedule nothing in the constructor. If the class has a first event,
give it a `start` method; the part calls it once every part is wired.

## 2. Pick the part

The part is the stage of the loop the component belongs to:

| Part | File | For a component that |
| --- | --- | --- |
| `Qpu` | `decsim/build/qpu.py` | makes readouts or supplies the QPU |
| `Control` | `decsim/build/control.py` | runs the program or closes the feedback loop |
| `Readout` | `decsim/build/readout.py` | carries or stores a round before a window reads it |
| `Windows` | `decsim/build/windows.py` | plans, reads, decides on or commits a window |
| `Decoders` | `decsim/build/decoders.py` | schedules or runs a decode |
| `Switching` | `decsim/build/escalation.py` | decides on a weak result or re-decodes a window on the strong tier, in a switching machine |

## 3. Build it in the part

Add a field to the part's record and one line to its `build`:

```python
    decision_dispatch: decision_dispatch_module.DecisionDispatch
```

```python
        decision_dispatch = decision_dispatch_module.DecisionDispatch(engine)
```

Pass it to the record's constructor at the end of `build`, as every
other field is.

## 4. Wire it inside the part

Bind each port whose neighbour is in the same part, one assignment a
line, in the part's wiring method (`_wire_inside` in the control part):

```python
        self.conditional_release.dispatch = self.decision_dispatch
        self.decision_dispatch.instruction_output = self.instruction_output
        self.decision_dispatch.link = links
```

## 5. Wire it to another part

A neighbour that lives in another part is one more argument of this
part's `connect`, and one more keyword where `Machine.assemble` calls
it in `decsim/machine.py`. The windows part's operation results stop
the QPU part's factory once the last result is in, and are wired to it
this way:

```python
    def connect(self, ..., factory: ports.MagicStateFactory) -> None:
        ...
        self.results.factory = factory
```

```python
        windows.connect(
            ...
            factory=qpu.factory,
        )
```

## 6. The ends of a run

- If it draws random numbers, add it to the part's `seed_roots`, under
  the name its seed is derived from.
- If it can still hold work when the run ends, give it a
  `check_settled` and call that from the part's `check_settled`.
- If an observer should hear it, give it a trace callback and connect
  the listener in `decsim/observe/wiring.py`. The component runs the
  same with no listener.

## 7. Check it

`test_every_required_port_is_bound_once_the_parts_connect`, in
`tests/machine/test_machine.py`, reads every port that is not optional on every component of a built
machine, so a wire you forgot fails there by name. Then add one test of
what the new component does, in the test folder of its package.

## Read next

- [Build a machine step by step](../tutorials/build_a_machine.md): the
  parts, built and assembled by hand.
- [The parts](../reference/parts.md): every settings record a machine is
  built from.

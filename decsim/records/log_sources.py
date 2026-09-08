"""The name each component narrates under, in one place.

`Engine.log` and `Engine.log_io` take the component's name as their
first argument, and the line a listener writes begins with it, so the
name is a fact two components share whenever one narrates an event the
other owns: the window side and the escalation both narrate on the
decoder manager's line, because the manager owns the queue they are
talking about. A shared fact is a record (STYLE.md rule 6), and one home
also makes the whole cast readable at once.

The text is what the frozen gate's log hash pins, so a name changes only
in a commit that regenerates that hash and says so.
"""

QPU = "QPU"
CONTROLLER = "Controller"
WEAK_BUFFER = "Buffer 0"
STRONG_BUFFER = "SyndromeBuffer1"
DECODER_MANAGER = "Decoder manager"
DECODER_UNIT = "Decoder unit"
PAULI_FRAME = "PauliFrame"
EXECUTION_RUNTIME = "ExecutionRuntime"
MAGIC_STATE_FACTORY = "Factory"

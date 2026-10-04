"""The name each component narrates under, in one place.

Two components narrate on one line when one owns what the other
reports: the window side and the escalation both narrate on the decoder
manager's line, since the manager owns the queue. The text is what the
gate's log hash pins, so a name changes only with that hash.
"""

QPU = "QPU"
CONTROLLER = "Controller"
WEAK_BUFFER = "weak syndrome buffer"
STRONG_BUFFER = "strong syndrome buffer"
DECODER_MANAGER = "Decoder manager"
DECODER_UNIT = "Decoder unit"
PAULI_FRAME = "PauliFrame"
EXECUTION_RUNTIME = "ExecutionRuntime"
MAGIC_STATE_FACTORY = "Factory"

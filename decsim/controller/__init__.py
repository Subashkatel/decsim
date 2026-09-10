"""The controller: the room-side machine between the QPU and the stores.

On the way in, controller.py takes every readout the QPU emits,
round_assembly.py makes the fragments of one round into one packed
round (the DETECTION_EVENT_FORMATION row in settings.py says whether
the detection events are formed before that round leaves),
round_writes.py writes the finished round into every store that must
hold it and holds it back while a store has no room, and
round_transmission.py tells the window side what landed.

On the way out, operation_issue.py admits one operation and issues it
as one QPU command, feedback_streams.py keeps the protected cycle of a
stream that feeds back, idle_rounds.py routes the rounds of a waiting
patch under the IDLE_POLICIES row in policies.py,
conditional_release.py lets go of the operations that waited on a
result, and instruction_output.py sends the instruction to the QPU.
settings.py holds the controller's per-round costs and both tables.
"""

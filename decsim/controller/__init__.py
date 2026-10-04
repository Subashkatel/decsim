"""The controller: the room-side machine between the QPU and the stores.

In: controller.py takes each readout, round_assembly.py packs a round's
fragments, syndrome_round_sender.py writes it to its store or holds it,
round_transmission.py tells the window side what landed. Out:
operation_issue.py issues operations, feedback_streams.py keeps the
protected cycle, idle_rounds.py and policies.py route a waiting patch's
rounds, conditional_release.py frees the operations waiting on a result,
instruction_output.py sends to the QPU.
"""

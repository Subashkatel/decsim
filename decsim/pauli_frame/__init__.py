"""One final correction per window, folded into a stream's logical frame.

pauli_frame.py holds the frame itself, one correction per window at the
price of one write. decision_dispatch.py is the frame side's end of
frame_to_controller: the decision a final result releases leaves by it
and lands at the controller.
"""

"""The Pauli frame and the releases it triggers.

pauli_frame keeps one final correction per window and folds a stream's
corrections into its logical frame. conditional_release lets go of the
operations that were waiting on a finished result.
"""

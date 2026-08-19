"""Program execution: plans the decoding windows and job dependencies ahead
of time (planner), decides which operation runs when (execution_runtime),
and releases the operations conditioned on a final result (conditional_release).
Planning is build time, off the reaction path; the release is on it. The
frontends that produce the program and the Pauli frame that holds outcomes
are their own components."""

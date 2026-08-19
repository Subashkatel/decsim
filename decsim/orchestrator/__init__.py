"""The orchestrator of Khalid et al. Fig. 2, the execution provider: parses
the program (qlx_frontend, circuit_frontend), plans the decoding windows and
job dependencies ahead of time (planner), decides which operation runs when
(execution_runtime), holds the Pauli frame (pauli_frame) and turns a final
logical measurement into the release of the operations conditioned on it
(logical_measurement). Parsing and planning are done at build time and are
off the reaction path; the frame update and the release are on it."""

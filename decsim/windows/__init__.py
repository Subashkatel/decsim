"""The window side: which rounds each decode reads, and what it commits.

A windowing scheme (schemes/, one file per row) lays an operation's
windows and window_planner holds them. round_tracker says when a window
has its rounds and round_retention which rounds it still holds;
decode_requests asks for one job per complete window; window_commits
commits its result once. window_boundaries carries the residual defects
to the windows after it, boundary_policies says when they ship and
boundary_payloads in what form. committed_rounds records who committed
which rounds, and operation_results delivers an operation once every
covering window is final. window_interactions relates adjacent or
replaced windows, built_window_models builds each window's error model
once per task, and window_manager is the facade the machine and the
escalation package speak to.
"""

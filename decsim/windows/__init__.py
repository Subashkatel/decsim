"""The window side: which rounds each decode reads, and what it commits.

A windowing scheme lays the windows of an operation and the rest of the
package runs their life cycle. window_planner lays them from the scheme
row (schemes/, one file per table row of WINDOWING_SCHEMES);
round_tracker says when a window has its rounds and round_retention
says which rounds it still holds; decode_requests asks the decode queue
for one job per complete window; window_commits brings the correction
home and commits the window once; window_boundaries and
boundary_payloads carry a committed window's residual defects to the
windows after it, under the boundary_policies row that says when they
ship; committed_rounds is the ledger of who committed which rounds and
operation_results delivers an operation once every covering window is
final. window_interactions owns how adjacent or replaced windows
relate, built_window_models builds each window's error model once per
task, and window_manager is the facade the machine and the escalation
package speak to.
"""

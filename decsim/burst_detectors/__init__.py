"""The burst detectors: detection events scored against their usual rates.

A row reads each round's bulk detection events as they are formed and
says whether a burst of errors is under way; the switching policy
escalates every window a published flag meets, and the strong window
model may raise the flagged region's priors. The row is one entry of
BURST_DETECTORS in settings.py, one folder per row: event_count/ and
masked_regional_cusum/. layout.py holds where an operation's checks
sit and their usual rates, flag_log.py a row's verdicts and when each
is published, burst_region.py the region a flag covers and its raised
priors, and burst_windows.py the answers both rows give about a window.
"""

"""The windowing schemes, one file per table row.

A scheme plans one operation's windows and their internal dependencies
and says when a window has its data; it owns no engine state and
schedules no decoder work. The rows of WINDOWING_SCHEMES
(decsim/windows/settings.py) are the modules beside this one, named the
way the yaml names them: sliding (Skoric et al. 2209.08552 section I.B),
parallel (Skoric section I.C, block A/B), sandwich (Tan et al.
2209.09219, type-1 cores and type-2 seams), naive_online (one window per
operation). window_data and buffer_floors hold the two rules every row
shares.
"""

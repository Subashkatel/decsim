"""What each part of a machine measures, declared on its class, as rows.

    kinds.py     Count, Distribution, Level and the disabled statistic
    registry.py  Statistic declarations, a part's Group, the run's
                 Registry and its long table of rows

A part declares its statistics on its class, gets its group from the
run's registry at its settings path when it is built, and updates it
where the event happens (gem5's stats, SST's statistic load level).
"""

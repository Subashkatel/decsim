"""The five results' calibrated thresholds, in dB, by point and share.

Each threshold is the gap step whose escalated share is nearest the
target, put halfway to the step above (calibrate.py). The gap takes
only multiples of union-find's weight step, 2.17 dB, so a share is a
staircase: at d = 9, p = 0.003 the steps escalate 4.3, 6.1, 10.6, 11.4,
12.6, 16.5 and then 30.6% of windows, and no threshold escalates 20%.

First taken from the basic case's switching run at 20 dB
(results/basic_case, task switching_d9_p0.003, commit bdcdbb6b): the
window gaps of its pieces from seed 20,000 on, 785,992 shots and
7,859,920 windows, which never meet the calibration seeds 0 to 9,999
or the evaluation seeds from 10,000. There each threshold escalated
4.34, 10.62 and 16.52% of windows.

Checked by gap_calibration.py's run of union-find alone on the
calibration seeds (results/five_results/gap_calibration/2026-10-29,
commit a25c9d30): 10,000 shots and 100,000 windows give the same three
thresholds, escalating 4.31, 10.63 and 16.59% (calibrate.py's
calibration.csv there).
"""

# (distance, p) -> {target share: threshold in dB}; the share each
# escalated on the calibration seeds: 0.0431, 0.1063 and 0.1659
CALIBRATED = {
    (9, 0.003): {0.05: 7.55, 0.1: 11.9, 0.2: 18.4},
}

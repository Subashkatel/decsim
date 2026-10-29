"""The five results' calibrated thresholds, in dB, by point and share.

Each threshold is the gap step whose escalated share is nearest the
target, put halfway to the step above (calibrate.py). The gap takes
only multiples of union-find's weight step, 2.17 dB, so a share is a
staircase: at d = 9, p = 0.003 the steps escalate 4.3, 6.1, 10.6, 11.4,
12.6, 16.5 and then 30.6% of windows, and no threshold escalates 20%.

Provisional, from the basic case's switching run at 20 dB
(results/basic_case, task switching_d9_p0.003, commit bdcdbb6b): the
window gaps of its pieces from seed 20,000 on, 785,992 shots and
7,859,920 windows, which never meet the calibration seeds 0 to 9,999
or the evaluation seeds from 10,000. gap_calibration.py's run of
union-find alone on the calibration seeds checks it; a threshold that
run moves is changed here, and its designs run again.
"""

# (distance, p) -> {target share: threshold in dB}; the share each
# escalated there: 0.0434, 0.1062 and 0.1652
CALIBRATED = {
    (9, 0.003): {0.05: 7.55, 0.1: 11.9, 0.2: 18.4},
}

"""The basic case's calibrated thresholds, from the threshold sweep.

threshold_plots.py's calibration.csv of the sweep folded on 2026-10-19
(threshold_sweep.py, seeds 0 to 19,999): per (d, p), the threshold in
dB that escalated 5, 10 and 20% of windows, interpolated between the
swept ones. threshold_heldout.py and rate_controller.py read it.
"""

CALIBRATED = {
    (5, 0.002): {0.05: 15.71, 0.1: 20.92, 0.2: 30.9},
    (5, 0.003): {0.05: 8.66, 0.1: 14.98, 0.2: 18.49},
    (5, 0.005): {0.05: 2.39, 0.1: 4.78, 0.2: 9.89},
    (7, 0.002): {0.05: 15.55, 0.1: 21.1, 0.2: 30.62},
    (7, 0.003): {0.05: 8.66, 0.1: 14.61, 0.2: 17.87},
    (7, 0.005): {0.05: 1.99, 0.1: 3.97, 0.2: 8.16},
}

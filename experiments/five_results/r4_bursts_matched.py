r"""R4's adaptive rules at the fixed threshold's own share on a stream.

On a stream the fixed 18.4 dB threshold sent about 20.5% of windows at
steady noise (r4_bursts.py, seeds 1 and 2: 22.0 and 19.1%), more than
the 16.6% it sent on the calibration shots, since the windows after an
escalated one read the strong boundary. r4_bursts.py's rate rules hold
18%, so they start with fewer windows to send. Here they hold 20%, so
all three rules send about the same share at steady noise and differ
only in how they move in the burst. Task names:
{rule}_target{rate_target}_peak{peak_factor}_d{distance}
_p{physical_error_probability}.
"""

import decsim
import decsim.experiments.sweeps as sweeps

bursts = sweeps.load_beside(__file__, "r4_bursts.py")

sweep = decsim.Sweep(
    "five_results_r4_bursts_matched",
    bursts.machine,
    grid={
        "rule": ("rate", "steps"),
        "peak_factor": bursts.PEAK_FACTORS,
        "distance": (9,),
        "physical_error_probability": (0.003,),
        "rate_target": (0.2,),
    },
    task_name=(
        "{rule}_target{rate_target}_peak{peak_factor}"
        "_d{distance}_p{physical_error_probability}"
    ),
    stop=bursts.stop,
    metadata=bursts.config.metadata,
)
experiment = sweep.experiment()

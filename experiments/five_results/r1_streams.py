r"""R1 on streams: does each decoder alone keep up?

Each task streams one memory (decsim stream), sampled a round at a time
(config.streaming), on union-find alone or correlated matching alone.
The stream's record holds the waiting line's growth rate, the reaction
times and the windows past one logical cycle. A correlated-matching
window of 2d rounds takes 60.3 us at d = 9 and one arrives every
9.9 us, so its line grows from the first window (experiments.md, R1);
union-find's should stay bounded. The logical error rate is R1's shot
result (r1_shots.py): a stream's checkpoints would each decode the whole
waiting line, which grows without limit here. Task names:
{decoding}_d{distance}_p{physical_error_probability}.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md), one
stream each on the evaluation seed 1: union-find on 100,000 rounds, and
correlated matching on 10,000. Each window waiting in a growing line
keeps its rounds, so correlated matching's memory grows with its line:
100,000 rounds passed 32 GB at d = 9. Its growth rate is a slope, which
10,000 rounds measure as well.

```
export PYTHONPATH=$PWD/../..
python -m decsim stream r1_streams.py --only weak_alone_d9_p0.003 \
  --rounds 100000 --warmup 1000 --seed 1 --out weak_alone_s1.json
python -m decsim stream r1_streams.py --only strong_alone_d9_p0.003 \
  --rounds 10000 --warmup 1000 --seed 1 --out strong_alone_s1.json
```
"""

import decsim
import decsim.experiments.sweeps as sweeps
import decsim.topology.graph as graph

config = sweeps.load_beside(__file__, "config.py")
DECODINGS = ("weak_alone", "strong_alone")
DISTANCES = (9,)
PHYSICAL_ERROR_PROBABILITIES = (0.003,)


def machine(
    decoding: str, distance: int, physical_error_probability: float
) -> graph.Machine:
    """One decoder alone, streaming its memory."""
    alone = config.alone(decoding, distance, physical_error_probability)
    return config.streaming(alone)


# decsim stream runs one shot; an experiment still names its collection
stop = decsim.CollectionSettings(max_shots=1)
sweep = decsim.Sweep(
    "five_results_r1_streams",
    machine,
    grid={
        "decoding": DECODINGS,
        "distance": DISTANCES,
        "physical_error_probability": PHYSICAL_ERROR_PROBABILITIES,
    },
    task_name="{decoding}_d{distance}_p{physical_error_probability}",
    stop=stop,
    metadata=config.metadata,
)
experiment = sweep.experiment()

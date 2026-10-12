[decsim docs](../README.md) › [How-to guides](README.md)

# How to sweep a machine over a grid

An experiment is three files in one folder: `constants.yaml` holds the
numbers and their sources, `config.py` builds the machine from them,
and `run.py` sweeps it. The worked example is Experiment 1 in
`experiments/switching_baseline`, 60 tasks; every other folder in
`experiments/` is written the same way.

## 1. Write the numbers in constants.yaml

One section per part, named as `decsim info` names the part's
parameters. The unit is in the value, and the comment beside it is its
source:

```yaml
union_find:
  cycle_count:
    delay: 3 cycles                  # MAXIMUM_DELAY, Helios_single_FPGA_core.v:76 (yale-paragon/Helios_scalable_QEC 2dda998)
```

The file is read as YAML 1.2, so `off`, `no` and `yes` stay text
rather than becoming booleans as YAML 1.1 would make them
(`decsim/topology/constants.py`).

## 2. Build the machine in config.py

Load the file, then read a value by attribute. A section is a mapping,
so it splats into a record's parameters, and `record` converts the
times on the part's clock:

```python experiments/switching_baseline/config.py
HERE = pathlib.Path(__file__)
CONSTANTS_PATH = HERE.with_name("constants.yaml")
constants = constants_module.load(CONSTANTS_PATH)
record = machine_file.record
```

```python experiments/switching_baseline/config.py
def union_find_pool() -> decoder_settings.DecoderPoolSettings:
    """Union-find on Helios's cycle law, on an FPGA in the fridge rack."""
    settings = constants.union_find
    clock = constants.clocks.union_find
    timing = {"type": "CycleCount", "clock": clock, **settings.cycle_count}
    algorithm = {"type": "union_find", "timing": timing, **settings.algorithm}
    engine = {"type": "Engine", "clock": clock, **settings.engine}
    return record(
        decoder_settings.DecoderPoolSettings,
        settings.pool,
        algorithm=algorithm,
        engine=engine,
    )
```

The machine function takes the values the sweep varies as arguments
and returns a machine, here `machine(decoding, distance,
physical_error_probability)`.

## 3. Sweep it in run.py

```python experiments/switching_baseline/run.py
config = sweeps.load_beside(__file__, "config.py")
constants = config.constants
```

```python experiments/switching_baseline/run.py
sweep = decsim.Sweep(
    "switching_baseline",
    config.machine,
    grid={
        "decoding": ["switching", "weak_alone"],
        "physical_error_probability": [
            0.0005,
            0.001,
            0.002,
            0.003,
            0.004,
            0.005,
        ],
        "distance": [5, 7, 9, 11, 13],
    },
    task_name="{decoding}_d{distance}_p{physical_error_probability}",
    stop=stop,
    metadata=metadata,
    constants=constants,
)
experiment = sweep.experiment()
```

The cells run in the grid's order, the last axis fastest. An axis is
one of three things (`decsim/experiments/sweeps.py`):

| An axis named | is | Example |
|---|---|---|
| an argument of the machine function | passed to it | `distance` |
| `constants.` and a constant's name | that constant, set for the cell while the machine is built | `constants.escalation.threshold.threshold` |
| anything else | a path of the cell's machine, set as `--set` sets it | `weak.unit_count` |

`task_name` reads a dotted axis as `{weak.unit_count}`. `stop` is one
`CollectionSettings` or a function of the cell; `stop_all` is the
experiment's. A task's metadata is the cell's numbers and strings, as
sinter's `json_metadata` holds a task's swept values
(sinter/_data/_task.py). Write it yourself with a function of the cell,
or of the cell and the task (its graph, and its layout with the card
that prices every hop), as Experiment 1 does to name its results'
columns.

## 4. Run it, and change a number without editing a file

From the folder:

```bash
decsim params run.py --match "constants.**"
decsim run run.py --only switching_d5_p0.003
decsim run run.py --set constants.escalation.threshold.threshold=25dB
```

`decsim params` lists every constant after the machine's paths, with
its value and source. A constant set with `--set` shows the assignment
as its source.

## 5. What is refused

| The run | is refused because | Instead |
|---|---|---|
| holds a constant no cell read | a misspelt key would leave a number unused | fix the key, or name it in `allow_unread` |
| sets with `--set` a path or constant the grid sweeps | every cell would run that value while its name said its own | change the grid |
| sets with `--set` a value an argument axis feeds, so two cells build one machine | the same: `--set qpu.distance=7` makes every `d5` cell a d=7 machine | drop the assignment, or sweep the value as an axis |
| names an axis that is no argument, constant or path | it would set nothing | the refusal lists the near names |
| sweeps a constant but gives the sweep no `constants` | there is no file to set it in | pass `constants=` |

An assignment that only changes a value the cells share, such as
`--set weak.unit_count=2` under a distance sweep, is kept.

"""One yaml file is one experiment; this module is the only yaml reader.

The file's sections are handed to the packages that own them, one
settings record each (decsim.machine.MachineSettings.from_mapping); the
sweep blocks stay here, since the machine knows nothing of sweeps.
`extends: other.yaml` starts from that file (same folder) and overrides
the top-level keys this file names.
"""

import dataclasses
from pathlib import Path

import yaml

import decsim.machine as machine

SWEEP_KEYS = (
    "physical_error_probability",
    "distance",
    "round_period_us",
    "shots",
)


@dataclasses.dataclass(frozen=True)
class SweepBlock:
    """One cross product of the three axes, `shots` seeds per point.

    The algorithm is not a sweep axis: it is structure, fixed per unit on
    the decoder section; comparing algorithms is comparing configs.
    Distance is an axis because the papers' LER plots are one curve per
    d (Toshio 2510.25222 sweeps d at fixed p; threshold plots sweep p
    per d).
    """

    physical_error_probabilities: tuple
    distances: tuple
    round_periods_microseconds: tuple
    shots: int


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """Everything one yaml file says: the machine, and the sweep over it."""

    name: str  # the yaml stem; suffixes the run folder
    settings: machine.MachineSettings
    sweep: tuple  # of SweepBlock
    # the yaml files this config was read from, nearest first (an extends
    # chain)
    config_files: tuple

    def point_settings(
        self,
        *,
        physical_error_probability: float,
        distance: int,
        round_period_us: float,
        threshold_calibrator=None,
    ) -> machine.MachineSettings:
        """The machine at one sweep point.

        The point sets the QPU's distance and round period, the memory
        circuit's noise, and the escalation threshold the point
        certifies; the point's online calibrator, when there is one, is
        shared by every shot of the point.
        """
        settings = self.settings
        qpu = dataclasses.replace(
            settings.qpu,
            distance=distance,
            round_period_microseconds=round_period_us,
        )
        workload = dataclasses.replace(
            settings.workload,
            physical_error_probability=physical_error_probability,
        )
        threshold_nats = settings.escalation.threshold_nats_for(
            physical_error_probability, distance
        )
        escalation = dataclasses.replace(
            settings.escalation,
            gap_threshold_nats=threshold_nats,
            threshold_calibrator=threshold_calibrator,
        )
        return dataclasses.replace(
            settings, qpu=qpu, workload=workload, escalation=escalation
        )

    def online_calibrator(
        self, *, physical_error_probability: float, distance: int
    ):
        """The point's online threshold calibrator, or None."""
        return self.settings.escalation.online_calibrator(
            physical_error_probability, distance
        )

    @property
    def active_decoder(self) -> machine.decoder_settings.DecoderSettings:
        """The tier that decodes the plan's windows."""
        tier = self.settings.escalation.decodes_on
        return getattr(self.settings, f"{tier}_decoder")


def load_experiment(path) -> ExperimentConfig:
    """Read one yaml file, its extends chain applied, into settings."""
    path = Path(path)
    raw, config_files = _raw_yaml(path)
    raw_sweep = raw.pop("sweep")
    sweep = _sweep_blocks(raw_sweep)
    sections = _sections(raw)
    settings = machine.MachineSettings.from_mapping(
        sections, name=path.stem, base_directory=path.parent
    )
    return ExperimentConfig(
        name=path.stem,
        settings=settings,
        sweep=sweep,
        config_files=config_files,
    )


def as_json_value(value):
    """A settings record as plain json: dataclasses walked, objects named.

    A Python-built component (a decoder, a policy) has no yaml text, so
    it appears as its class name; every number, string and flag appears
    as written.
    """
    if dataclasses.is_dataclass(value):
        fields = {}
        for field in dataclasses.fields(value):
            field_value = getattr(value, field.name)
            fields[field.name] = as_json_value(field_value)
        return fields
    if isinstance(value, dict):
        items = {}
        for key, item in value.items():
            items[str(key)] = as_json_value(item)
        return items
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            json_item = as_json_value(item)
            items.append(json_item)
        return items
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    value_type = type(value)
    return value_type.__name__


def _raw_yaml(path: Path) -> tuple:
    """The file's keys with its `extends` chain applied, and its files.

    The files come this file first, then the base it extends, and so on.
    A key this file names replaces the base's key whole: a child that
    declares `sweep` ignores the base's sweep entirely.
    """
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    base_name = raw.pop("extends", None)
    if base_name is None:
        return raw, (path,)
    base_path = path.parent / base_name
    base, base_paths = _raw_yaml(base_path)
    base.update(raw)
    return base, (path,) + base_paths


def _sections(raw: dict) -> dict:
    """Today's top-level keys as the sections their owners read.

    The yaml still carries the names the surface commit of this slice
    renames (decode_path and switching, decoder.weak, buffers,
    windowing, circuit and rounds_per_shot, the three trace keys); this
    mapping goes with that commit.
    """
    buffers = raw["buffers"]
    windowing = raw["windowing"]
    controller = dict(raw["controller"])
    controller["packing_rounds_in_flight"] = buffers["packing_rounds_in_flight"]
    escalation = {"kind": raw["decode_path"]}
    switching = raw.get("switching")
    if switching is not None:
        escalation.update(switching)
    sections = {
        "clocks": raw["clocks"],
        "qpu": {"kind": "stim_device"},
        "controller": controller,
        "links": raw["links"],
        "round_store": {"rounds": buffers["weak_buffer_rounds"]},
        "strong_round_store": {"rounds": buffers["strong_buffer_rounds"]},
        "windows": {
            "kind": windowing["scheme"],
            "commit_rounds": windowing["commit_rounds"],
            "buffer_rounds": windowing["buffer_rounds"],
        },
        "escalation": escalation,
        "pauli_frame": raw["pauli_frame"],
        "workload": {
            "kind": "memory_circuit",
            "code_task": raw["circuit"],
            "rounds_per_shot": raw["rounds_per_shot"],
        },
        "observation": _observation_section(raw),
    }
    if "idle_policy" in raw:
        sections["idle_policy"] = raw["idle_policy"]
    tiers = raw["decoder"]
    unknown = set(tiers) - {"weak", "strong"}
    if unknown:
        listed = sorted(unknown)
        raise ValueError(
            f"decoder does not know {listed}; its keys are the tiers: weak, "
            "strong"
        )
    for tier in ("weak", "strong"):
        if tier not in tiers or tiers[tier] is None:
            continue
        sections[f"{tier}_decoder"] = _tier_section(tiers[tier])
    return sections


def _observation_section(raw: dict) -> dict:
    section = {}
    for key in ("trace", "log_component_io", "check_windows_with"):
        if key in raw:
            section[key] = raw[key]
    return section


def _tier_section(card: dict) -> dict:
    section = dict(card)
    section["kind"] = section.pop("algorithm")
    return section


def _sweep_blocks(raw_sweep: list) -> tuple:
    blocks = []
    for index, block in enumerate(raw_sweep, start=1):
        sweep_block = _sweep_block(block, index)
        blocks.append(sweep_block)
    return tuple(blocks)


def _sweep_block(block: dict, index: int) -> SweepBlock:
    unknown = set(block) - set(SWEEP_KEYS)
    if unknown:
        listed = sorted(unknown)
        raise ValueError(
            f"sweep block {index} does not know {listed}; its axes "
            "are physical_error_probability, distance and round_period_us, "
            "plus shots (the algorithm lives on the decoder card, not in "
            "the sweep)"
        )
    return SweepBlock(
        physical_error_probabilities=tuple(block["physical_error_probability"]),
        distances=tuple(block["distance"]),
        round_periods_microseconds=tuple(block["round_period_us"]),
        shots=block["shots"],
    )

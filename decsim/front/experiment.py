"""One yaml file is one experiment; this module is the only yaml reader.

The file's sections are handed to the packages that own them, one
settings record each (decsim.machine.MachineSettings.from_mapping); the
sweep blocks stay here, since the machine knows nothing of sweeps.
`extends: other.yaml` starts from that file (same folder) and overrides
the top-level keys this file names.
"""

import dataclasses
import itertools
from pathlib import Path

import yaml

import decsim.collect as collect
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

    def tasks(self) -> list:
        """One task per sweep point, blocks in order, each block a product."""
        tasks = []
        for block in self.sweep:
            points = itertools.product(
                block.physical_error_probabilities,
                block.distances,
                block.round_periods_microseconds,
            )
            for physical_error_probability, distance, round_period_us in points:
                task = self.point_task(
                    physical_error_probability=physical_error_probability,
                    distance=distance,
                    round_period_us=round_period_us,
                    shots=block.shots,
                )
                tasks.append(task)
        return tasks

    def point_task(
        self,
        *,
        physical_error_probability: float,
        distance: int,
        round_period_us: float,
        shots: int,
    ) -> collect.Task:
        """The task of one sweep point: its settings, shots and metadata."""
        settings = self.point_settings(
            physical_error_probability=physical_error_probability,
            distance=distance,
            round_period_us=round_period_us,
        )
        metadata = {
            "physical_error_probability": physical_error_probability,
            "distance": distance,
            "round_period_us": round_period_us,
        }
        return collect.Task.at_point(settings, shots, metadata)

    def point_settings(
        self,
        *,
        physical_error_probability: float,
        distance: int,
        round_period_us: float,
    ) -> machine.MachineSettings:
        """The machine at one sweep point.

        The point sets the QPU's distance and round period, the memory
        circuit's noise, and the escalation threshold the point
        certifies.
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
            settings.escalation, gap_threshold_nats=threshold_nats
        )
        return dataclasses.replace(
            settings, qpu=qpu, workload=workload, escalation=escalation
        )

    @property
    def active_decoder(self) -> machine.decoder_settings.DecoderSettings:
        """The tier that decodes the plan's windows."""
        tier = self.settings.escalation.decodes_on
        return getattr(self.settings, f"{tier}_decoder")


def resolved_description(config: ExperimentConfig) -> list:
    """What one yaml really says, after its extends chain is applied.

    An edit that did not land shows up here immediately: a config that
    extends another replaces its base's keys whole, so a `sweep` edited
    in the base never reaches a child that declares its own. gem5 prints
    the same thing with --dump-config (src/python/m5/main.py:241).
    """
    lines = [_files_line(config)]
    for line in _section_lines(config.settings):
        lines.append(line)
    for index, block in enumerate(config.sweep, start=1):
        block_line = _sweep_block_line(index, block)
        lines.append(block_line)
    for line in _observation_lines(config.settings.observation):
        lines.append(line)
    return lines


def load_experiment(path) -> ExperimentConfig:
    """Read one yaml file, its extends chain applied, into settings."""
    path = Path(path)
    sections, config_files = _yaml_sections(path)
    sweep_section = sections.pop("sweep")
    sweep = _sweep_blocks(sweep_section)
    settings = machine.MachineSettings.from_mapping(
        sections, name=path.stem, base_directory=path.parent
    )
    return ExperimentConfig(
        name=path.stem,
        settings=settings,
        sweep=sweep,
        config_files=config_files,
    )


def _files_line(config: ExperimentConfig) -> str:
    """The files this config was read from, nearest first."""
    names = []
    for path in config.config_files:
        names.append(str(path))
    joined = " <- ".join(names)
    return f"config: {joined}"


def _section_lines(settings: machine.MachineSettings) -> list:
    """One line per section, its kind named where the section has one."""
    lines = []
    for field in dataclasses.fields(settings):
        section = getattr(settings, field.name)
        kind = getattr(section, "kind", None)
        if kind is None:
            continue
        lines.append(f"{field.name}: kind {kind}")
    return lines


def _sweep_block_line(index: int, block: SweepBlock) -> str:
    """One sweep block's three axes and its shot count, as one line."""
    probabilities = list(block.physical_error_probabilities)
    distances = list(block.distances)
    periods = list(block.round_periods_microseconds)
    return (
        f"sweep block {index}: p {probabilities}, d {distances}, "
        f"round period {periods} us, {block.shots} shots"
    )


def _observation_lines(observation) -> list:
    """What the run will record beyond its results."""
    log_line = f"log: {observation.log}"
    if observation.log_component_io:
        log_line += " with component I/O"
    return [log_line, f"trace: {observation.trace}"]


def _yaml_sections(path: Path) -> tuple:
    """The file's sections with its `extends` chain applied, and its files.

    The files come this file first, then the base it extends, and so on.
    A key this file names replaces the base's key whole: a child that
    declares `sweep` ignores the base's sweep entirely.
    """
    with open(path) as handle:
        sections = yaml.safe_load(handle)
    base_name = sections.pop("extends", None)
    if base_name is None:
        return sections, (path,)
    base_path = path.parent / base_name
    base_sections, base_paths = _yaml_sections(base_path)
    base_sections.update(sections)
    files = (path,) + base_paths
    return base_sections, files


def _sweep_blocks(sweep_section: list) -> tuple:
    """One SweepBlock per block of the yaml's sweep list, in order."""
    blocks = []
    for index, block in enumerate(sweep_section, start=1):
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

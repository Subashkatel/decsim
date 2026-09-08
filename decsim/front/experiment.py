"""One yaml file is one experiment; this module is the only yaml reader.

The file's sections are handed to the packages that own them, one
settings record each (decsim.machine_settings.MachineSettings.from_mapping); the
sweep blocks stay here, since the machine knows nothing of sweeps.
`extends: other.yaml` starts from that file (same folder) and overrides
the top-level keys this file names.
"""

import dataclasses
import itertools
from pathlib import Path

import yaml

import decsim.build.escalation as escalation_build
import decsim.collect as collect
import decsim.decoders.settings as decoder_settings
import decsim.front.refusal as refusal
import decsim.settings as machine_settings

_THIS_FILE = Path(__file__)
_FRONT_DIR = _THIS_FILE.resolve()
_REPOSITORY_ROOT = _FRONT_DIR.parents[2]
# configs/ sits beside the decsim package, gem5's configs/ beside its
# binary; the shipped experiments are what a refused path is listed with.
CONFIGS_DIR = _REPOSITORY_ROOT / "configs"
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

    def points(self) -> list:
        """Its cross product, one (p, distance, round period) per point."""
        axes = itertools.product(
            self.physical_error_probabilities,
            self.distances,
            self.round_periods_microseconds,
        )
        points = []
        for physical_error_probability, distance, round_period_us in axes:
            point = (physical_error_probability, distance, round_period_us)
            points.append(point)
        return points


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """Everything one yaml file says: the machine, and the sweep over it."""

    name: str  # the yaml stem; suffixes the run folder
    settings: machine_settings.MachineSettings
    sweep: tuple  # of SweepBlock
    # the yaml files this config was read from, nearest first (an extends
    # chain)
    config_files: tuple

    def tasks(self) -> list:
        """One task per sweep point, blocks in order, each block a product."""
        tasks = []
        for block in self.sweep:
            for point in block.points():
                task = self._point_task(point, block.shots)
                tasks.append(task)
        return tasks

    def _point_task(self, point: tuple, shots: int) -> collect.Task:
        """The task of one (p, distance, round period) point."""
        physical_error_probability, distance, round_period_us = point
        return self.point_task(
            physical_error_probability=physical_error_probability,
            distance=distance,
            round_period_us=round_period_us,
            shots=shots,
        )

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
    ) -> machine_settings.MachineSettings:
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
    def active_tier(self) -> str:
        """The tier that decodes the plan's windows: weak or strong."""
        return escalation_build.primary_tier(self.settings.escalation)

    @property
    def active_decoder(self) -> decoder_settings.DecoderSettings:
        """The decoder card of the tier that decodes the plan's windows."""
        tier = self.active_tier
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
    links_line = _links_line(config.settings.links)
    lines.append(links_line)
    for index, block in enumerate(config.sweep, start=1):
        block_line = _sweep_block_line(index, block)
        lines.append(block_line)
    for line in _observation_lines(config.settings.observation):
        lines.append(line)
    return lines


def task_positions(recorded_sweep: list) -> dict:
    """Each sweep point's place in the task order of a recorded sweep.

    A run folder's manifest records the resolved config
    (decsim/front/run_folder.py write_manifest), so the sweep's own task
    order is recoverable from the folder alone, without the yaml and
    whatever order the folders are named in. It is the order tasks()
    makes: the blocks in order, each block its cross product, a point
    named twice keeping its first place (decsim/collect.py unique_tasks
    merges the repeat away).
    """
    positions = {}
    for recorded_block in recorded_sweep:
        block = _recorded_block(recorded_block)
        _place_a_blocks_points(positions, block)
    return positions


def load_experiment(path) -> ExperimentConfig:
    """Read one yaml file, its extends chain applied, into settings."""
    path = Path(path)
    _refuse_a_path_that_is_not_a_file(path)
    sections, config_files = _yaml_sections(path)
    sweep_section = sections.pop("sweep")
    sweep = _sweep_blocks(sweep_section)
    settings = _settings_of(sections, path)
    return ExperimentConfig(
        name=path.stem,
        settings=settings,
        sweep=sweep,
        config_files=config_files,
    )


def _refuse_a_path_that_is_not_a_file(path: Path) -> None:
    """A yaml that is not there, with the shipped experiments to pick from."""
    if path.is_file():
        return
    shipped_files = CONFIGS_DIR.glob("*.yaml")
    names = []
    for shipped in sorted(shipped_files):
        names.append(shipped.stem)
    listed = ", ".join(names)
    raise refusal.RefusalError(
        f"{path} is not a file; the shipped experiments are {listed}"
    )


def _settings_of(
    sections: dict, path: Path
) -> machine_settings.MachineSettings:
    """The machine's settings records, one per section of the file.

    A settings record checks its own section and raises ValueError
    (STYLE.md rule 4); the file is the front's input, so a refused key
    reaches the user as one sentence naming the file it is in.
    """
    try:
        return machine_settings.MachineSettings.from_mapping(
            sections, name=path.stem, base_directory=path.parent
        )
    except ValueError as refused:
        raise refusal.RefusalError(f"{path}: {refused}") from refused


def _place_a_blocks_points(positions: dict, block: SweepBlock) -> None:
    """One block's points, each keeping the first place it was given."""
    for point in block.points():
        if point in positions:
            continue
        positions[point] = len(positions)


def _recorded_block(recorded_block: dict) -> SweepBlock:
    """One sweep block as a manifest's resolved config recorded it."""
    return SweepBlock(
        physical_error_probabilities=tuple(
            recorded_block["physical_error_probabilities"]
        ),
        distances=tuple(recorded_block["distances"]),
        round_periods_microseconds=tuple(
            recorded_block["round_periods_microseconds"]
        ),
        shots=recorded_block["shots"],
    )


def _files_line(config: ExperimentConfig) -> str:
    """The files this config was read from, nearest first."""
    names = []
    for path in config.config_files:
        names.append(str(path))
    joined = " <- ".join(names)
    return f"config: {joined}"


def _section_lines(settings: machine_settings.MachineSettings) -> list:
    """One line per section, its kind named where the section has one."""
    lines = []
    for field in dataclasses.fields(settings):
        section = getattr(settings, field.name)
        kind = getattr(section, "kind", None)
        if kind is None:
            continue
        lines.append(f"{field.name}: kind {kind}")
    return lines


def _links_line(links) -> str:
    """The fabric card the run resolved to."""
    return f"links: card {links.profile_name}"


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
        raise refusal.RefusalError(
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

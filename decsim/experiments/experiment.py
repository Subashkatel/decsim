"""One yaml file is one experiment; this module is the only yaml reader.

The file's sections are handed to the packages that own them, one
settings record each (decsim.settings.MachineSettings.from_mapping); the
sweep blocks stay here, since the machine knows nothing of sweeps.
`extends: other.yaml` starts from that file (same folder) and overrides
the top-level keys this file names.
"""

import contextlib
import dataclasses
import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

import yaml

import decsim.build.escalation as escalation_build
import decsim.collect as collect
import decsim.decoders.settings as decoder_settings
import decsim.experiments.refusal as refusal
import decsim.machine as machine_module
import decsim.settings as machine_settings

_THIS_FILE = Path(__file__)
_RESOLVED_FILE = _THIS_FILE.resolve()
_REPOSITORY_ROOT = _RESOLVED_FILE.parents[2]
# configs/ sits beside the decsim package, gem5's configs/ beside its
# binary; the shipped experiments are what a refused path is listed with.
CONFIGS_DIR = _REPOSITORY_ROOT / "configs"
SWEEP_AXES = (
    "physical_error_probability",
    "distance",
    "round_period_microseconds",
)
SWEEP_KEYS = SWEEP_AXES + ("shots",)
# The settings paths a sweep point sets, each with its sweep block field.
SWEEP_PATHS = {
    ("qpu", "distance"): "distances",
    ("qpu", "round_period_microseconds"): "round_periods_microseconds",
    ("workload", "physical_error_probability"): (
        "physical_error_probabilities"
    ),
}
REFERENCE_FILE = CONFIGS_DIR / "reference.yaml"


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
        points = list(axes)
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
        physical_error_probability, distance, round_period_microseconds = point
        return self.point_task(
            physical_error_probability=physical_error_probability,
            distance=distance,
            round_period_microseconds=round_period_microseconds,
            shots=shots,
        )

    def point_task(
        self,
        *,
        physical_error_probability: float,
        distance: int,
        round_period_microseconds: float,
        shots: int,
    ) -> collect.Task:
        """The task of one sweep point: its settings, shots and metadata."""
        settings = self.point_settings(
            physical_error_probability=physical_error_probability,
            distance=distance,
            round_period_microseconds=round_period_microseconds,
        )
        metadata = {
            "physical_error_probability": physical_error_probability,
            "distance": distance,
            "round_period_microseconds": round_period_microseconds,
        }
        return collect.Task.at_point(settings, shots, metadata)

    def point_settings(
        self,
        *,
        physical_error_probability: float,
        distance: int,
        round_period_microseconds: float,
    ) -> machine_settings.MachineSettings:
        """The machine at one sweep point.

        The point sets the QPU's distance and round period, the workload
        its maker makes there, and the escalation threshold the point
        certifies.
        """
        settings = self.settings
        qpu = dataclasses.replace(
            settings.qpu,
            distance=distance,
            round_period_microseconds=round_period_microseconds,
        )
        sweep_values = {
            "physical_error_probability": physical_error_probability,
            "distance": distance,
            "round_period_microseconds": round_period_microseconds,
        }
        workload = self._workload_at(sweep_values)
        threshold_nats = settings.escalation.threshold_nats_for(
            physical_error_probability, distance
        )
        escalation = dataclasses.replace(
            settings.escalation, gap_threshold_nats=threshold_nats
        )
        return dataclasses.replace(
            settings, qpu=qpu, workload=workload, escalation=escalation
        )

    def first_point_task(self) -> collect.Task:
        """The task of the first point of the first sweep block, one shot."""
        block = self.sweep[0]
        return self.point_task(
            physical_error_probability=block.physical_error_probabilities[0],
            distance=block.distances[0],
            round_period_microseconds=block.round_periods_microseconds[0],
            shots=1,
        )

    def built_machine(
        self, settings: machine_settings.MachineSettings, seed: int
    ) -> machine_module.Machine:
        """The machine the settings build, a build refusal one sentence."""
        path = self.config_files[0]
        with _refused_in(path):
            return machine_module.Machine.build(settings, seed)

    def _workload_at(self, sweep_values: dict):
        """The workload section at one point, a maker's refusal one line."""
        path = self.config_files[0]
        with _refused_in(path):
            return self.settings.workload.at_point(sweep_values)

    @property
    def active_tier(self) -> str:
        """The tier that decodes the plan's windows: weak or strong."""
        return escalation_build.primary_tier(self.settings.escalation)

    @property
    def active_decoder(self) -> decoder_settings.DecoderSettings:
        """The decoder card of the tier that decodes the plan's windows."""
        tier = self.active_tier
        return self.settings.decoder_settings_for(tier)


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


def value_lines(config: ExperimentConfig) -> list:
    """Every value the config resolves to, and who set it where it is sure.

    One line per value of the settings records: its dotted path, its
    value (a swept value lists the sweep's values), and in brackets its
    layer and source line when the value is one yaml key's (_origin_of).
    gem5's config.ini lists every parameter of every object the same way
    (src/python/m5/simulate.py:122-127).
    """
    written = _written_keys(config)
    documented = _documented_keys()
    swept = _swept_values(config.sweep)
    settings = collect.json_value(config.settings)
    leaves = _leaves(settings, ())
    lines = []
    for path, value in leaves:
        key = _yaml_key(path)
        value = swept.get(key, value)
        value_text = json.dumps(value)
        dotted = ".".join(path)
        line = f"{dotted} = {value_text}"
        origin = _origin_of(key, written, documented)
        if origin is not None:
            line = f"{line}  [{origin}]"
        lines.append(line)
    return lines


def task_positions(recorded_sweep: list) -> dict:
    """Each sweep point's place in the task order of a recorded sweep.

    A run folder's manifest records the resolved config
    (decsim/experiments/run_folder.py write_manifest), so the sweep's own task
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
    sections, section_folders, config_files = _yaml_sections(path)
    if "sweep" not in sections:
        raise refusal.RefusalError(
            f"{path} has no sweep; a yaml names at least one sweep block "
            "(configs/reference.yaml)"
        )
    sweep_section = sections.pop("sweep")
    sweep = _sweep_blocks(sweep_section)
    settings = _settings_of(sections, path, section_folders)
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
    sections: dict, path: Path, section_folders: dict
) -> machine_settings.MachineSettings:
    """The machine's settings records, one per section of the file.

    A settings record checks its own section and raises ValueError
    (STYLE.md rule 4); the file is the experiments layer's input, so a
    refused key reaches the user as one sentence naming the file it is
    in.
    """
    with _refused_in(path):
        return machine_settings.MachineSettings.from_mapping(
            sections, name=path.stem, section_folders=section_folders
        )


@contextlib.contextmanager
def _refused_in(path: Path):
    """A ValueError raised inside, as one sentence naming the yaml file."""
    try:
        yield
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
    """The file's sections with `extends` applied, their folders, its files.

    The files come this file first, then the base it extends, and so on.
    A key this file names replaces the base's key whole: a child that
    declares `sweep` ignores the base's sweep entirely. So a relative
    path in a section is the one its own file wrote, and it resolves
    against that file's folder, as a relative `extends` does.
    """
    with open(path) as handle:
        sections = yaml.safe_load(handle)
    base_name = sections.pop("extends", None)
    folders = dict.fromkeys(sections, path.parent)
    if base_name is None:
        return sections, folders, (path,)
    base_path = path.parent / base_name
    base_sections, base_folders, base_paths = _yaml_sections(base_path)
    base_sections.update(sections)
    base_folders.update(folders)
    files = (path,) + base_paths
    return base_sections, base_folders, files


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
            "are physical_error_probability, distance and "
            "round_period_microseconds, plus shots (the algorithm lives "
            "on the decoder card, not in the sweep)"
        )
    _check_axes(block, index)
    shots = block["shots"]
    _check_shots(shots, index)
    return SweepBlock(
        physical_error_probabilities=tuple(block["physical_error_probability"]),
        distances=tuple(block["distance"]),
        round_periods_microseconds=tuple(block["round_period_microseconds"]),
        shots=shots,
    )


def _check_axes(block: dict, index: int) -> None:
    """Every key is there, and each axis lists the values it sweeps."""
    missing = set(SWEEP_KEYS) - set(block)
    if missing:
        listed = sorted(missing)
        raise refusal.RefusalError(
            f"sweep block {index} lacks {listed}; a block lists "
            "physical_error_probability, distance and "
            "round_period_microseconds and names its shots"
        )
    for axis in SWEEP_AXES:
        values = block[axis]
        if not isinstance(values, list) or not values:
            raise refusal.RefusalError(
                f"sweep block {index} {axis} must be a list of at least one "
                f"value, got {values!r}"
            )


def _check_shots(shots, index: int) -> None:
    """A point runs seeds 0 to shots - 1 (decsim/collect.py), so a count."""
    is_whole_number = isinstance(shots, int) and not isinstance(shots, bool)
    if is_whole_number and shots >= 1:
        return
    raise refusal.RefusalError(
        f"sweep block {index} shots must be a whole number of at least 1, "
        f"got {shots!r}"
    )


@dataclasses.dataclass(frozen=True)
class _WrittenKey:
    """A yaml key a file writes: the layer it belongs to and its line."""

    layer: str
    source: str  # file:line


def _origin_of(key: tuple, written: dict, documented: dict) -> Optional[str]:
    """The layer that set a value and its yaml line, or None.

    Only a value that is one yaml key's has an origin, so a value the
    build derives cites no line rather than a guessed one.
    """
    if key in SWEEP_PATHS:
        sweep_key = written[("sweep",)]
        return f"sweep, {sweep_key.source}"
    if key in written:
        written_key = written[key]
        return f"{written_key.layer}, {written_key.source}"
    if key in documented:
        documented_key = documented[key]
        return f"default, {documented_key.source}"
    return None


def _written_keys(config: ExperimentConfig) -> dict:
    """Each key the extends chain writes, with its layer and line.

    A file's top-level key replaces its base's whole (_yaml_sections),
    so a section, and every key under it, comes from the nearest file
    that names the section.
    """
    written = {}
    your_file = config.config_files[0]
    for path in reversed(config.config_files):
        layer = f"preset {path.name}"
        if path == your_file:
            layer = "your file"
        file_keys = _key_lines(path, path, layer)
        _forget_the_sections_of(written, file_keys)
        written.update(file_keys)
    return written


def _forget_the_sections_of(written: dict, file_keys: dict) -> None:
    """Drop the base's keys of every section a nearer file names."""
    sections = set()
    for key in file_keys:
        sections.add(key[0])
    for key in list(written):
        if key[0] in sections:
            del written[key]


def _documented_keys() -> dict:
    """Each key configs/reference.yaml writes, with its line there."""
    shown_path = REFERENCE_FILE.relative_to(_REPOSITORY_ROOT)
    return _key_lines(REFERENCE_FILE, shown_path, "default")


def _key_lines(path: Path, shown_path: Path, layer: str) -> dict:
    """Every mapping key of a yaml file, with the lines that write it.

    A key whose value is a block over several lines (the sweep's list)
    is shown as that span. The keys under a list (the sweep's blocks)
    are the list's own key.
    """
    with open(path) as handle:
        root = yaml.compose(handle)
    keys = {}
    _add_key_lines(root, (), f"{shown_path}", layer, keys)
    return keys


def _add_key_lines(node, prefix: tuple, shown_path: str, layer: str, keys):
    """The keys of one mapping node and of every mapping below it."""
    if node.id != "mapping":
        return
    for key_node, value_node in node.value:
        key = prefix + (key_node.value,)
        lines = _lines_of(key_node, value_node)
        keys[key] = _WrittenKey(layer, f"{shown_path}:{lines}")
        _add_key_lines(value_node, key, shown_path, layer, keys)


def _lines_of(key_node, value_node) -> str:
    """The line a key is on, or the span of lines its block value covers.

    A block ends on the line of its last scalar; a block's own end mark
    runs on past the blank and comment lines after it.
    """
    first_line = key_node.start_mark.line + 1
    last_node = _last_scalar(value_node)
    last_line = last_node.end_mark.line + 1
    if last_line <= first_line:
        return f"{first_line}"
    return f"{first_line}-{last_line}"


def _last_scalar(node):
    """The last scalar node under a node, which is itself when a scalar."""
    while node.id != "scalar" and node.value:
        last_entry = node.value[-1]
        node = last_entry
        if isinstance(last_entry, tuple):
            node = last_entry[1]
    return node


def _swept_values(sweep: tuple) -> dict:
    """Each value a sweep point sets, as the list of the sweep's values."""
    swept = {}
    for key, field_name in SWEEP_PATHS.items():
        swept[key] = _values_of(sweep, field_name)
    return swept


def _values_of(sweep: tuple, field_name: str) -> list:
    """One axis's values across the sweep blocks, each once, in order."""
    values = []
    for block in sweep:
        block_values = getattr(block, field_name)
        values.extend(block_values)
    unique_values = dict.fromkeys(values)
    return list(unique_values)


def _leaves(value, path: tuple) -> list:
    """Every value under a json value that is not a mapping, with its path."""
    if not isinstance(value, Mapping) or not value:
        return [(path, value)]
    leaves = []
    for key, item in value.items():
        item_path = path + (str(key),)
        item_leaves = _leaves(item, item_path)
        leaves.extend(item_leaves)
    return leaves


def _yaml_key(path: tuple) -> tuple:
    """A settings path as the yaml writes it: a row's keys sit beside kind."""
    key = []
    for name in path:
        if name != "row_settings":
            key.append(name)
    return tuple(key)

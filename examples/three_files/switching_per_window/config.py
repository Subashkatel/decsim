"""Switching per window's machines: Experiment 1's, four ways, labelled.

    machine("switching", distance=5, physical_error_probability=0.003)

Union-find alone, Relay-BP-5 alone, switching, and Tesseract alone as a
reference outside switching, each on Experiment 1's machine
(../switching_baseline/config.py), its shots drawn with the errors that
fired and every window written with its true label.
"""

import dataclasses
import pathlib

import decsim.decoders.settings as decoder_settings
import decsim.decoders.tesseract.decoder as tesseract
import decsim.experiments.sweeps as sweeps
import decsim.qpu.stim_device as stim_device
import decsim.topology.constants as constants_module
import decsim.topology.graph as graph
import decsim.topology.machine_file as machine_file

baseline = sweeps.load_beside(__file__, "../switching_baseline/config.py")
HERE = pathlib.Path(__file__)
CONSTANTS_PATH = HERE.with_name("constants.yaml")
constants = constants_module.load(CONSTANTS_PATH)


def machine(
    configuration: str, distance: int, physical_error_probability: float
) -> graph.Machine:
    """One configuration at one setting, drawing its shots' errors."""
    make = CONFIGURATIONS[configuration]
    built = make(distance, physical_error_probability)
    return with_labels(built)


def union_find_alone(
    distance: int, physical_error_probability: float
) -> graph.Machine:
    """Union-find keeps every window."""
    return baseline.weak_alone(distance, physical_error_probability)


def switching(
    distance: int, physical_error_probability: float
) -> graph.Machine:
    """Union-find first, a doubted window redone by Relay-BP-5."""
    weak = baseline.weak_alone(distance, physical_error_probability)
    return baseline.switching(weak)


def relay_bp5_alone(
    distance: int, physical_error_probability: float
) -> graph.Machine:
    """Relay-BP-5 decodes every window, on the host."""
    pool = baseline.relay_bp_pool()
    return baseline.strong_alone(distance, physical_error_probability, pool)


def tesseract_alone(
    distance: int, physical_error_probability: float
) -> graph.Machine:
    """Tesseract's short beam in Relay-BP-5's pool, a reference.

    It has no measured device time, so its unit is held for its own
    call's wall clock and only its error rate is a result.
    """
    pool = tesseract_pool()
    return baseline.strong_alone(distance, physical_error_probability, pool)


def tesseract_pool() -> decoder_settings.DecoderPoolSettings:
    """Relay-BP-5's pool with Tesseract's short beam as its algorithm."""
    relay_bp5 = baseline.relay_bp_pool()
    short_beam = machine_file.record(
        tesseract.TesseractDecoder.Settings, constants.tesseract
    )
    return dataclasses.replace(relay_bp5, algorithm=short_beam)


def with_labels(built: graph.Machine) -> graph.Machine:
    """The machine drawing shots with their errors, writing window labels."""
    error_model = stim_device.ErrorModelStimDevice.Settings()
    labelled = built.with_settings("qpu", source=error_model)
    observation = dataclasses.replace(
        built.observation, record_window_outcomes=True
    )
    return dataclasses.replace(labelled, observation=observation)


CONFIGURATIONS = {
    "union_find_alone": union_find_alone,
    "switching": switching,
    "relay_bp5_alone": relay_bp5_alone,
    "tesseract_alone": tesseract_alone,
}

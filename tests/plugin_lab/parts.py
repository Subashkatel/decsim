"""A lab's decoder and verdict, written against decsim's ports only.

decsim imports nothing here; a run names each part by its import path,
tests.plugin_lab.parts.WeightedMatching.Settings, as a lab's package
outside the tree would be named. The decoder matches a window with
PyMatching, prices it at a preset latency, and reports two numbers of
its own as evidence: the matching's weight and a draw from its seeded
generator. The verdict escalates a window whose weight is above its
cut. Neither has a paper behind it; they exercise the ports.
"""

import dataclasses

import numpy
import pymatching

import decsim.decoders.decoder as decoder_module
import decsim.decoders.decoders as decoders
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models
import decsim.seeding as seeding
import decsim.statistics.kinds as statistic_kinds
import decsim.statistics.registry as statistics_registry

KEEP = decoding_records.Verdict.KEEP
ESCALATE = decoding_records.Verdict.ESCALATE
WEIGHT = "weight"
DRAW = "draw"
MILLION = 1_000_000


class WeightedMatching(
    decoder_module.WindowDecoderBase, seeding.GeneratorSeedConsumer
):
    """PyMatching on the window, with its weight and a draw as evidence."""

    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    fault_representation = fault_models.FaultRepresentation.GRAPHLIKE
    STATISTICS = (
        statistics_registry.Statistic(
            "weighted",
            statistic_kinds.Count,
            "decodes whose matching has a positive weight",
            "decodes",
        ),
        statistics_registry.Statistic(
            "draws",
            statistic_kinds.Distribution,
            "the generator's draws",
            "millionths",
        ),
    )
    statistics = statistics_registry.unrecorded(STATISTICS)

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The lab's row: its preset latency."""

        name = "weighted_matching"
        preset_latency: float = 1.0

        def build(self) -> "WeightedMatching":
            """A fresh decoder of these settings."""
            return WeightedMatching(self)

    def __init__(self, settings: "WeightedMatching.Settings") -> None:
        latency_model = decoders.PresetLatencyDecoder(settings.preset_latency)
        decoder_module.WindowDecoderBase.__init__(self, latency_model)
        seeding.GeneratorSeedConsumer.__init__(self)
        self.compile_key = (WeightedMatching, settings)

    def compile(
        self,
        faults: fault_models.PlacedFaultModel,
        model: fault_models.WindowErrorModel,
    ) -> pymatching.Matching:
        """The window's matching graph, log-odds weights."""
        del model
        priors = numpy.clip(faults.priors, 1e-15, 0.5)
        odds = (1 - priors) / priors
        weights = numpy.log(odds)
        check = faults.check.copy()
        return pymatching.Matching.from_check_matrix(
            check, weights=weights, merge_strategy="independent"
        )

    def decode_window(
        self,
        backend: pymatching.Matching,
        model: fault_models.WindowErrorModel,
        faults: fault_models.PlacedFaultModel,
        syndrome: numpy.ndarray,
    ) -> decoding_records.WindowDecode:
        """The matching's faults, and the lab's evidence beside them."""
        del model, faults
        selected, weight = backend.decode(syndrome, return_weight=True)
        generator = self.generator()
        draw = generator.random()
        scaled = draw * MILLION
        millionths = int(scaled)
        self.statistics.draws.add(millionths)
        if weight > 0:
            self.statistics.weighted.add()
        evidence = {WEIGHT: float(weight), DRAW: float(draw)}
        return decoding_records.WindowDecode(selected, evidence=evidence)


class WeightCut:
    """Escalate a window whose matching weight is above the cut."""

    STATISTICS = (
        statistics_registry.Statistic(
            "escalated",
            statistic_kinds.Count,
            "windows escalated for a weight above the cut",
            "windows",
        ),
    )
    statistics = statistics_registry.unrecorded(STATISTICS)

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The lab's verdict: its cut on the weight."""

        name = "weight_cut"
        reads_threshold = False
        reads_consensus = False
        reads_source_queue = False
        cut: float = 0.0

        def build(self) -> "WeightCut":
            """A verdict at this cut."""
            return WeightCut(self.cut)

    def __init__(self, cut: float) -> None:
        self.cut = cut

    def decide(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        threshold: ports.ThresholdSource,
    ) -> decoding_records.Verdict:
        """Keep a weight at or below the cut; escalate one above it."""
        del job, threshold
        if result.evidence is None:
            return ESCALATE
        if result.evidence[WEIGHT] > self.cut:
            self.statistics.escalated.add()
            return ESCALATE
        return KEEP

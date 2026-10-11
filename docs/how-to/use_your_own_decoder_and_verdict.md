[decsim docs](../README.md) › [How-to guides](README.md)

# How to use your own decoder and verdict

Your decoder and the rule that decides which of its windows to escalate
can live in your own package. decsim is not edited: a run names each by
its import path. The worked example is `tests/plugin_lab/parts.py`, a
package decsim never imports, and `tests/test_plugin_lab.py` runs it end
to end.

## 1. Write the decoder

Inherit `WindowDecoderBase` as [How to add a decoder
backend](add_a_decoder_backend.md) says, and give it a nested `Settings`
record that builds it:

```python tests/plugin_lab/parts.py
    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The lab's row: its preset latency."""

        name = "weighted_matching"
        preset_latency: float = 1.0

        def build(self) -> "WeightedMatching":
            """A fresh decoder of these settings."""
            return WeightedMatching(self)
```

## 2. Report numbers of your own

A decode returns its faults and, in `evidence`, any numbers of its own
by name. decsim reads none of them; your verdict does, off
`DecodeResult.evidence`:

```python tests/plugin_lab/parts.py
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
```

No paper names a general carrier for a decoder's own numbers: each names
one soft output (the complementary gap, the cluster gap, a network's
confidence). The mapping is a software choice, so a new number needs no
new field.

## 3. Draw from your own seeds

Inherit `seeding.GeneratorSeedConsumer` (`decsim/seeding.py`) and draw
from `self.generator()`, as above. A run seeds it from the shot's seed
and the decoder's place in the machine, so its draws repeat for a seed
and differ between seeds.

## 4. Declare statistics

Declare them on the class, as [How to record statistics of your own
part](record_statistics.md) says. A decoder's are recorded at its
pool's algorithm, `weak_decoder.algorithm` on a machine of two pools;
a verdict's at its edge's, `switching.verdict`.

## 5. Write the verdict

A verdict decides on each result its edge's source pool returns. Its
record says whether it reads the edge's threshold:

```python tests/plugin_lab/parts.py
    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The lab's verdict: its cut on the weight."""

        name = "weight_cut"
        reads_threshold = False
        reads_consensus = False
        cut: float = 0.0

        def build(self) -> "WeightCut":
            """A verdict at this cut."""
            return WeightCut(self.cut)
```

```python tests/plugin_lab/parts.py
        if result.evidence is None:
            return ESCALATE
        if result.evidence[WEIGHT] > self.cut:
            self.statistics.escalated.add()
            return ESCALATE
        return KEEP
```

## 6. Use them with no edit to decsim

```bash
decsim run run.py --set weak.algorithm.type=mylab.parts.WeightedMatching.Settings \
    --set switching.confidence.type=none \
    --set switching.verdict.type=mylab.parts.WeightCut.Settings \
    --set switching.verdict.cut=2.5
decsim info mylab.parts.WeightedMatching.Settings
decsim describe run.py
```

A verdict that reads no gap takes the confidence row `none`, which asks
the decoder for nothing. A machine file names the parts the same way,
`type: mylab.parts.WeightCut.Settings`, and `describe` writes them so.
`switching.verdict.cut` is a path, so a sweep's grid takes it as an
axis ([How to sweep a machine over a grid](sweep_a_machine_over_a_grid.md)).

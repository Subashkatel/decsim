[decsim docs](../README.md) › [How-to guides](README.md)

# How to add a decoder backend

You have a decoder, in Python or wrapped from C, and you want decsim to
run it, price it and report its logical error rate. This is the whole
job.

## 1. Write the class

A decoder fills the `Decoder` port (`decsim/ports.py`;
[The ports](../reference/ports.md) lists its methods and members). The shortest
way is to inherit `DecoderBase` from `decsim/decoders/decoder.py`, which
gives you the port's defaults: `start`, `cancel`, `occupancy`,
`pipeline_depth`, the two members every row must answer, and the
wall-clock measurement around your call. Then your class is two methods:

```python
class MyDecoder(decoder.DecoderBase):
    """One sentence saying what this decoder is, and its source."""

    def decode(self, job):
        """The correction for one window."""

    def latency(self, job):
        """Ticks this decode should be charged, when it is not measured."""
```

`decode` is handed a `DecodeJob` (`decsim/records/decoding.py`), which
carries the window's fault model and its detection events, and returns a
`DecodeResult`. Inherit `DecoderBase` and a decode you do not price
yourself is charged the wall clock the call really took, which is what
every shipped row does.

## 2. Say what fault model you need

`fault_model_requirement` is one of the four contracts in
`decsim/detector_error_model/fault_model_contracts.py`. It says whether
your decoder wants a graphlike model, where every fault flips at most
two detectors, or the physical model with hyperedges, or both. Get this
wrong and your decoder is handed a model it cannot read.

`stage_recorded` is the trace source your unit's internal stages fire,
or `SILENT` for a decoder with no stages of its own.

Both are declared on the class, never read off its type, because a port
promises not to reveal what a row is (`tools/check_row_recognition.py`
fails on a module that tests against a row's class).

## 3. Add the row and the yaml key

One entry in `DECODERS` (`decsim/decoders/settings.py`), one folder or
one module beside the other rows, and the key in
`configs/reference.yaml` in the same commit.
[How to add a row to a table](add_a_table_row.md) is the general recipe with the refusal
a typo gets.

If your backend is an optional dependency, put it behind an import
inside the module the way `decsim/decoders/tesseract/decoder.py` and
`decsim/decoders/relay_belief_propagation/decoder.py` do, and add the
package to the `bb-decoders` extra in `pyproject.toml`.

## 4. Run it and check it against a referent

```bash
decsim collect configs/reference.yaml
```

with your row named as `weak_decoder.kind`. The summary line
`mismatches vs direct PyMatching: 0` is decsim decoding every window
through the machine and PyMatching decoding the same shots straight
through, outside it, and comparing the predictions. A backend that is
correct and different from matching will disagree on some windows; a
backend that is broken disagrees on most. Even a matching backend
disagrees on a few, because the window commits without the rounds the
whole-circuit decode reads (4 shots in 1800 at distances 3 and 5 and
physical error rates 0.003 to 0.01), so read a handful as the windowing
and a noticeable fraction as the bug.

For a second opinion per window rather than per shot, set
`observation.check_windows_with: tesseract`, which re-decodes every
window with Tesseract and counts the disagreements into
`tesseract_window_disagreements`.

## 5. Read the worked example

`tests/machine/test_machine.py::test_a_new_decoder_is_one_class_and_one_table_row`
plugs a decoder in from outside decsim in a few lines, and
`tests/machine/test_machine.py::test_a_second_table_row_runs_gate_point_one`
runs one through a whole run of a shipped config.

## Read next

- [The ports](../reference/ports.md): the `Decoder` port in full.
- [Time](../explanation/time.md): measured wall clock against a priced card.
- [How to run a timing study whose numbers do not depend on your computer](run_a_timing_only_study.md): pricing your decoder instead
  of measuring it.

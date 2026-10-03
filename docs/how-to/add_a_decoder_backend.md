[decsim docs](../README.md) › [How-to guides](README.md)

# How to add a decoder backend

You have a decoder, in Python or wrapped from C, and you want decsim to
run it, price it and report its logical error rate. This is the whole
job.

## 1. Write the class

A decoder fills the `Decoder` port (`decsim/ports.py`). The shortest
way is to inherit `DecoderBase` from `decsim/decoders/decoder.py`, which
gives you the port's defaults: `start`, `cancel`, `occupancy`, a
default for every member a decoder answers, and the wall-clock measurement
around your call. Then your class is two methods:

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
every shipped decoder does.

## 2. Say what fault model you need

`fault_model_requirement` is one of the four contracts in
`decsim/detector_error_model/fault_model_contracts.py`. It says whether
your decoder wants a graphlike model, where every fault flips at most
two detectors, or the physical model with hyperedges, or both. Get this
wrong and your decoder is handed a model it cannot read.

`stage_recorded` is the trace source your unit's internal stages fire,
or `SILENT` for a decoder with no stages of its own. A trace source is
one named event a component fires and listeners hear
(`decsim/trace_source.py`).

`decoder_evidence` is what your decode can show a confidence signal
beyond the correction. Leave it at the default and the build refuses
your decoder as the weak tier of a switching run, naming the evidence
the signal wanted. `missing_evidence_reasons` is the sentence that
refusal quotes.

All of them are declared on the class, never read off its type, because
a port promises not to reveal what class fills it
(`tools/check_row_recognition.py` fails on a module that tests against
a decoder's class).

## 3. Give it a settings record

Your class declares a nested frozen `Settings` record, even with no
fields. The record names your decoder in `name`, checks its fields in
`__post_init__` and builds your decoder in `build()`, and your
constructor takes the record as `settings`. A knob of your backend's
own (a step size, an iteration cap) is a field of it, as the union-find
decoder's `weight_step` is. Put the class in its own folder or module
beside the other decoders under `decsim/decoders/`.

If your backend is an optional dependency, put it behind an import
inside the module the way `decsim/decoders/tesseract/decoder.py` and
`decsim/decoders/relay_belief_propagation/decoder.py` do, and add the
package to the `bb-decoders` extra in `pyproject.toml`.

## 4. Run it and check it against a referent

A machine's decoder is its `weak_decoder` record, whose `algorithm` is
your record. Start from a base and replace it:

```python
base = machine_settings.weak_decoder_baseline(3, 0.003, 1.0)
weak_decoder = dataclasses.replace(base.weak_decoder, algorithm=MyDecoder.Settings())
machine = dataclasses.replace(base, weak_decoder=weak_decoder)
```

and make a run file of it, as `examples/my_first_sweep.py` is. To check
its answers, add a second point whose algorithm is
`PyMatchingDecoder.Settings()`: at one seed both points draw the same
shot (their `sample_digest` cells in `shots.csv` are equal), so a seed
where their `predictions` differ is a shot the two decoders answered
apart. A backend that is correct and different from matching will
disagree on some shots; a backend that is broken disagrees on most.
Even two matching decoders disagree on a few when their windows differ,
since a window commits without the later rounds a whole-circuit decode
reads.

For a second opinion per window rather than per shot, wrap your record
in `TesseractCheckedDecoder.Settings(inner=...)`
(`decsim/decoders/verify_windows.py`), which re-decodes every window
with Tesseract and counts the disagreements into
`referee_window_disagreements`.

## 5. Read the worked example

`tests/machine/test_machine.py::test_a_new_decoder_is_one_class_and_its_settings_record`
plugs a decoder in from outside decsim in a few lines, and
`tests/machine/test_machine.py::test_a_second_table_row_runs_gate_point_one`
runs the union-find decoder the same way.

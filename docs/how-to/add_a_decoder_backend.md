[decsim docs](../README.md) › [How-to guides](README.md)

# How to add a decoder backend

You have a decoder, in Python or wrapped from C, and you want decsim to
run it, price it and report its logical error rate. This is the whole
job.

## 1. Write the class

A decoder fills the `Decoder` port (`decsim/ports.py`). A real decoder
inherits `WindowDecoderBase` from `decsim/decoders/decoder.py`, which
does the work around your call: it reads the window's fault model,
builds the syndrome from the window's rounds and checks its size,
compiles one backend per window model and keeps it, builds the result
from the faults your call selects, and measures your call on the host's
clock. Then your class is two methods:

```python
class MyDecoder(decoder.WindowDecoderBase):
    """One sentence saying what this decoder is, and its source."""

    fault_model_requirement = fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    fault_representation = fault_models.FaultRepresentation.PHYSICAL

    def compile(
        self,
        faults: fault_models.PlacedFaultModel,
        model: fault_models.WindowErrorModel,
    ) -> MyBackend:
        """The backend for one window model, built once while it lives."""

    def decode_window(
        self,
        backend: MyBackend,
        model: fault_models.WindowErrorModel,
        faults: fault_models.PlacedFaultModel,
        syndrome: numpy.ndarray,
    ) -> decoding_records.WindowDecode:
        """One backend call: the fault columns it selects."""
```

`MyBackend` is your library's decoder class, as `ldpc.BpOsdDecoder` is
BP-OSD's. `faults.check` is the window's detector-by-fault matrix and
`faults.priors` each fault's probability; `syndrome` is the window's
detection events, one 0 or 1 per row of the matrix. The matrix is
read-only, so a backend that writes into its input, or refuses a
read-only one as `ldpc` does, gets a copy,
`scipy.sparse.csr_matrix(faults.check)`. `decode_window` returns
`decoding_records.WindowDecode(selected)`, where `selected` is a 0 or 1
per fault column. Return the columns; the base builds the correction,
the logical observables, the rounds the window owns and the data it
hands the next window.

A row priced by a number instead of a measured call inherits
`DecoderBase` from the same file and gives `decode` and `latency`. It is
charged its `latency`, never the host's clock.

A real decoder is timed on the host's clock, which is no hardware's
time: a Python backend takes hundreds of microseconds a window. To
charge it a fixed price instead, hand the base a latency model,
`decoder.WindowDecoderBase.__init__(self, latency_model)`, built from a
field of your record, as `PyMatchingDecoder.Settings` builds
`decoders.PresetLatencyDecoder(preset_latency_microseconds)`
(`decsim/decoders/minimum_weight_perfect_matching/decoder.py`). The
decode still runs; only its time is the price.

## 2. Say what fault model you need

`fault_model_requirement` is one of the four contracts in
`decsim/records/fault_model_contracts.py`. It says whether
your decoder wants a graphlike model, where every fault flips at most
two detectors, or the physical model with hyperedges, or both;
`fault_representation` is the one your `compile` reads. Get this wrong
and your decoder is handed a model it cannot read.

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
fields. The record names your decoder in `name`, the short word the
reports print, checks its fields in `__post_init__` (the checks in
`decsim/config.py`, such as `check_whole_count`, give the sentence) and
builds your decoder in `build()`, and your constructor takes the record
as `settings`. A knob of your backend's own (a step size, an iteration
cap) is a field of it, as the union-find decoder's `weight_step` is.
A field the library refuses itself, such as a method's name, needs no
check of its own (STYLE.md rule 4). Take each default from the
library's own example or the paper, with its source (STYLE.md rule 8),
and pass every argument to the library: a library's default can differ
from its signature's, as `ldpc`'s `BpLsdDecoder` runs `product_sum`
when `bp_method` is left out.

In the constructor, set the compile key to your class and every setting
the backend is built from:

```python
self.compile_key = (MyDecoder, settings)
```

Rows with one key share their backends across shots. With no key, every
shot compiles every window again, which is slow; a key that leaves out a
setting lets two rows of different settings share one backend, which is
wrong. A backend that holds the shot's seed sets `backend_is_seeded =
True` and compiles once a shot.

Put the class in its own folder under `decsim/decoders/`, named by the
algorithm, spelled out as far as the field spells it, as
`belief_propagation_osd/` is. Every short form in a folder, a class or
the record's `name` (such as `osd` and `bposd`) needs a row in
`docs/reference/glossary.md` (STYLE.md rule 2).

If your backend is an optional dependency, import it inside the function
that builds the backend, as `decsim/decoders/tesseract/window_decoder.py`
does, and add the package to the `bb-decoders` extra in
`pyproject.toml`.

Then run `python tools/docs_map.py`. It writes your record into
`docs/reference/parts.md`, which `tests/test_docs.py` holds to the
source.

## 4. Run it and check it against a referent

A machine's decoder is its `weak_decoder` record, whose `algorithm` is
your record. Start from a base and replace it:

```python
base = machine_settings.weak_decoder_baseline(3, 0.003, 1.0)
algorithm = MyDecoder.Settings()
weak_decoder = dataclasses.replace(base.weak_decoder, algorithm=algorithm)
machine = dataclasses.replace(base, weak_decoder=weak_decoder)
```

and make a run file of it, as `examples/my_first_sweep.py` is. To check
its answers, add a second point whose algorithm is
`PyMatchingDecoder.Settings()`: at one seed both points draw the same
shot (their `sample_digest` cells in `shots.csv` are equal), and the
`algorithm` column tells the two points apart. Compare their failures
shot by shot: on the shots where only one of the two fails, a decoder
about as strong as matching fails about as often as matching does,
and an exact McNemar test (a two-sided binomial test at one half on
those shots) says whether a gap is more than chance. A broken decoder
fails far more often. With `shots.csv` open as `shots_file`, and your
record's `name` in place of `"mine"`:

```python
rows = csv.DictReader(shots_file)
failed = {}
seeds = set()
for row in rows:
    failed[row["algorithm"], row["seed"]] = row["logical_failure"] == "True"
    seeds.add(row["seed"])
only_mine_count = 0
only_matching_count = 0
for seed in seeds:
    mine = failed["mine", seed]
    matching = failed["pymatching", seed]
    if mine and not matching:
        only_mine_count += 1
    if matching and not mine:
        only_matching_count += 1
discordant_count = only_mine_count + only_matching_count
print(f"{discordant_count} shots fail one decoder alone")
if discordant_count > 0:
    test = scipy.stats.binomtest(only_mine_count, discordant_count, 0.5)
    print(f"McNemar p-value {test.pvalue}")
```

A `csv.DictReader` is read only once, so the same pass gathers the
seeds. Check too that `invalid_correction_windows` is 0 and `is_scored`
is `True` on every shot: a backend that returns no correction is counted
there, not as a failure.

For a second opinion per window rather than per shot, wrap your record
in `TesseractCheckedDecoder.Settings(inner=...)`
(`decsim/decoders/verify_windows.py`), which re-decodes every window
with Tesseract and counts the disagreements into
`referee_window_disagreements`. Tesseract comes with the `bb-decoders`
extra.

## 5. Test it

Put the test in `tests/decoders/test_<folder>_decoder.py`.
`tests/decoders/windows.py` gives a whole-circuit window, its sampled
shots and a job for each. Compare your row's correction with the
library's own decoder on the same shots, column for column:
`tests/decoders/test_belief_propagation_osd_decoder.py` does this for
BP-OSD, through qLDPC. Call the library itself when the `run` extra
installs it, so the test runs without the `test` extra too; a test
behind `pytest.importorskip` skips quietly where the package is
missing.

## 6. Read the worked examples

`decsim/decoders/belief_propagation_osd/decoder.py` is a whole real
decoder in under a hundred lines; its constructor takes `ldpc`'s keyword
arguments rather than the settings record of step 3, which
`decsim/decoders/union_find/decoder.py` shows.
`tests/machine/test_machine.py::test_a_new_decoder_is_one_class_and_its_settings_record`
plugs a priced decoder in from outside decsim in a few lines, and
`tests/machine/test_machine.py::test_a_second_table_row_runs_gate_point_one`
runs the union-find decoder the same way.

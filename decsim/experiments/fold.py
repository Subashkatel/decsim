"""Many run folders' additive rows folded into one, none of them held.

A run folder records only facts that add up, so folding folders is
reading their rows and adding them. The rows are what a campaign has
most of: the 500 shard folders of the 2026-09-09 weak_ler sweep hold
115 million link rows and 10.5 million shot rows, and one row as a dict
of typed Python values costs about a kilobyte, so reading them into
lists costs a hundred gigabytes. This module holds what the fold needs
instead, and nothing in it grows with the number of shots:

    row_stream   one folder's file as rows, one row alive at a time
    merged_rows  several folders' streams in one order, one row each
    RowFile      one file written as its rows arrive
    RowTotals    a set of rows' count, true counts, sums, maxes, means
    ExactSum     a running sum that equals math.fsum of its values

sinter folds its csv files the same way: `total += ExistingData.from_file(path)`
over the paths, nothing else held (sinter/_command/_main_combine.py:31-34),
where the rows of one task fold into one TaskStats whose __add__ sums
shots, errors, discards, seconds and the counter table
(sinter/_data/_task_stats.py:117-150), so that "the statistics for that
task are folded together (so only the total shots, total errors, etc for
each task are included in the results)"
(sinter/_data/_existing_data.py:137-142).

The order comes from a merge, not a sort. Every run folder holds its
rows in the run's own order, its task position and then its seed, and a
shot lives in one folder, so walking the folders side by side puts the
rows back in the sweep's order while holding one row per folder:
heapq.merge "does not pull the data into memory all at once"
(/opt/python/lib/python3.11/heapq.py:319-321), and rows of equal key
come out in the order the streams were given, because the heap entry is
`[key(value), order, value, next]` and the stream's own index breaks the
tie (heapq.py:376, 383-388). That makes the merged order the order the
stable sort of the concatenation gave, and it is the classical balanced
merge of Knuth, TAOCP volume 3, section 5.4.1. The merge assumes each
input is sorted, so a stream that goes backwards is refused where it is
read rather than folded into a wrong order.

Nothing here knows a column of decsim's: which field is a mean and which
a count is report.py's to say, and a row is either the text a csv file
holds or the numbers a measurement held, which `number_of` reads either
way. One accumulator therefore serves the fold and the single run.
"""

import csv
import heapq
import math
import operator
from pathlib import Path

import decsim.experiments.refusal as refusal

# a keyed row is (the row's place in the run's order, the row); the merge
# compares the place alone, so two rows are never compared
_keyed_order = operator.itemgetter(0)


def typed_value(text: str):
    """A csv field as the value it was written from.

    The algorithm column is a name or a latency card, so a field that
    parses as a number is a number and everything else is its text.
    """
    if text == "True":
        return True
    if text == "False":
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def typed_row(row: dict) -> dict:
    """One csv row with its values read back as numbers and flags."""
    typed = {}
    for key, text in row.items():
        typed[key] = typed_value(text)
    return typed


def number_of(value):
    """One field of a row as the value it was written from.

    A row a folder was read from holds the text of its csv file; a row
    this process measured holds the numbers themselves. The fold reads
    both through here, so an accumulator and a sweep point are the same
    code either way.
    """
    if isinstance(value, str):
        return typed_value(value)
    return value


def header_of(path: Path) -> list:
    """One csv file's column names, read without its rows."""
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def row_stream(path: Path):
    """One csv file's rows, one alive at a time, values as their text.

    The values stay text because a fold writes most of them straight
    back out and reads a number only where a total needs one.
    """
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        yield from reader


def merged_rows(paths: list, key):
    """Every file's rows in the run's order, one row of each file held.

    `key` gives a row its place in that order. A path with no file is a
    folder that wrote no row of this kind and is skipped, which is what
    a shard whose index selected no work unit leaves behind.
    """
    streams = []
    for path in paths:
        if not path.is_file():
            continue
        stream = _keyed_rows(path, key)
        streams.append(stream)
    merged = heapq.merge(*streams, key=_keyed_order)
    for _place, row in merged:
        yield row


class ExactSum:
    """A running sum that equals math.fsum of the finite values it was given.

    The state is the list of non-overlapping partial sums whose total is
    the exact sum of every value added, which is what math.fsum keeps
    while it walks a list: Shewchuk's adaptive-precision addition
    (Shewchuk 1997, "Adaptive Precision Floating-Point Arithmetic and
    Fast Robust Geometric Predicates"), the recipe named at
    /opt/python/lib/python3.11/test/test_math.py:657-659, "Based on the
    'lsum' function at http://code.activestate.com/recipes/393090/".
    Neither the paper nor the recipe is on this machine, and the msum
    that test file carries at :656-681 is a different algorithm, the
    frexp/ldexp integer one the file itself labels so at :649-651. What
    is on this machine, and what the tests here compare against value
    for value, is math.fsum itself.

    Finite is the whole claim and not a hedge: on a value that is not
    finite, or on partials that overflow, this sum and math.fsum part
    company. math.fsum([1.0, inf]) is inf and this returns nan, because
    the loop computes inf - (inf - 1.0); math.fsum raises OverflowError
    on [1e308, 1e308] and this raises ValueError out of total, and on
    [1e308, 1e308, -1e308] this returns nan (the four cases CPython's
    own test pins at test_math.py:735-740). Nothing here refuses them,
    because every column a fold sums is a microsecond span or a bit
    count of a run that finished, and STYLE.md rule 4 leaves out a check
    no caller can trigger.

    `total` rounds that list once, so it is the sum math.fsum returns
    for the same values in any order, and a mean folded over a campaign's
    shards is the mean one process would have computed, to the last bit.
    A plain running float sum would not be: it would move the last bits
    of every mean column with the order the shards came in.
    """

    def __init__(self) -> None:
        self.partials = []

    def add(self, value) -> None:
        """One more value, the sum still exact.

        A zero leaves an exact sum as it was and is skipped, which is
        three quarters of a campaign's link fields: it changes no
        partial, and it takes no sign with it either, because math.fsum
        of zeros is 0.0 and not -0.0 (tests/experiments/test_fold.py). The
        partials loop stays in this one function because a fold of the
        500-folder campaign adds four hundred million values and each
        call of it walks the whole partials list: measured over a
        million calls on 2026-09-12, 0.43 us for a value whose
        magnitude is the ones before it (two partials), 1.86 us across a
        1e-30 to 1e30 spread (fourteen), and 0.07 us for a zero, which
        is skipped. That is STYLE.md's one concession to a hot path.
        """
        addend = float(value)
        if not addend:
            return
        partials = self.partials
        index = 0
        for partial in partials:
            smaller = partial
            if abs(addend) < abs(smaller):
                addend, smaller = smaller, addend
            rounded = addend + smaller
            residue = smaller - (rounded - addend)
            if residue:
                partials[index] = residue
                index += 1
            addend = rounded
        del partials[index:]
        partials.append(addend)

    def total(self) -> float:
        """The sum, rounded once: math.fsum of every value added."""
        return math.fsum(self.partials)


class RowTotals:
    """What a set of rows adds up to, one row at a time.

    Which field plays which role is given once, at construction: how
    many rows there are, how many hold a field true, a field's sum, a
    field's largest value, and a field's mean. Nothing here grows with
    the rows, so the totals of ten shots and of ten million are the same
    size, which is what lets a fold stream a campaign. It is the shape
    of sinter's TaskStats, which holds shots, errors, discards, seconds
    and a counter table and folds by adding them
    (sinter/_data/_task_stats.py:117-150).
    """

    def __init__(
        self,
        *,
        means: tuple = (),
        maxes: tuple = (),
        sums: tuple = (),
        true_counts: tuple = (),
    ) -> None:
        self.rows = 0
        self.means = {}
        self.maxes = {}
        self.sums = {}
        self.true_counts = {}
        for field in means:
            self.means[field] = ExactSum()
        for field in maxes:
            self.maxes[field] = None
        for field in sums:
            self.sums[field] = 0
        for field in true_counts:
            self.true_counts[field] = 0

    def add(self, row: dict) -> None:
        """One more row: each role reads the fields it was given."""
        self.rows += 1
        for field, running in self.means.items():
            running.add(row[field])
        for field, largest in self.maxes.items():
            self._raise_the_max(field, largest, row)
        for field in self.sums:
            self.sums[field] += number_of(row[field])
        for field in self.true_counts:
            flag = number_of(row[field])
            self.true_counts[field] += bool(flag)

    def mean(self, field: str) -> float:
        """The field's mean over the rows: their exact sum over the count.

        This is statistics.fmean, which is math.fsum(values) / n
        (/opt/python/lib/python3.11/statistics.py:436-458), with the sum
        taken as the rows arrived instead of over a list of them.
        """
        running = self.means[field]
        total = running.total()
        return total / self.rows

    def _raise_the_max(self, field: str, largest, row: dict) -> None:
        """The field's largest value so far, the first row's being it."""
        value = number_of(row[field])
        if largest is None:
            self.maxes[field] = value
            return
        if value > largest:
            self.maxes[field] = value


class RowFile:
    """One csv file written as its rows arrive, never held and then written.

    The header is the first row's keys, which is where `write_csv` takes
    it from, and a file whose first row never came is not created, which
    is the rule a run folder's files keep for a run that had nothing to
    put in them.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None
        self.writer = None

    def write(self, row: dict) -> None:
        """One row; the first one opens the file and writes the header."""
        if self.writer is None:
            self._open(row)
        self.writer.writerow(row)

    def close(self) -> None:
        """Done writing. A file that got no row was never opened."""
        if self.handle is None:
            return
        self.handle.close()

    def __enter__(self) -> "RowFile":
        return self

    def __exit__(self, *_exception) -> None:
        self.close()

    def _open(self, row: dict) -> None:
        """The file and its header, the first row naming the columns."""
        field_names = list(row)
        self.handle = open(self.path, "w", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=field_names)
        self.writer.writeheader()


def _keyed_rows(path: Path, key):
    """One file's rows with their places, refusing a file out of order.

    A merge assumes its inputs are sorted, so a file whose rows go
    backwards would fold into a wrong order rather than fail. The check
    is here, on the place the merge is about to compare, and costs one
    comparison a row.
    """
    previous = None
    for row in row_stream(path):
        place = key(row)
        if previous is not None:
            _refuse_a_row_out_of_order(path, place, previous)
        previous = place
        yield (place, row)


def _refuse_a_row_out_of_order(path: Path, place, previous) -> None:
    """A run folder holds its rows in the order the run wrote them."""
    if place >= previous:
        return
    raise refusal.RefusalError(
        f"{path} holds a row at {place} after a row at {previous}; combine "
        "merges the folders' rows, and a run folder holds its rows in the "
        "order the sweep ran them, its task position and then its seed"
    )

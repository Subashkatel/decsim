"""Many run folders' additive rows folded into one, none of them held.

The 500 folders of one sweep hold 115 million link rows, a hundred
gigabytes as dicts, so nothing here grows with the shots: row_stream
reads one row at a time, merged_rows orders several folders' streams,
RowFile writes as rows arrive, RowTotals keeps counts, sums, maxes and
means, ExactSum keeps a sum equal to math.fsum. sinter folds its csv
files the same way (sinter/_command/_main_combine.py:31-34;
_data/_task_stats.py:117-150).

The order comes from a merge, not a sort: each folder holds its rows in
the run's order and a shot lives in one folder, so a heap of one entry
per open file restores the sweep's order, ties broken by file order as
heapq.merge does (heapq.py:376, 383-388), Knuth's balanced merge (TAOCP
vol. 3, 5.4.1). A file opens only when its first row is due, so a
capped point's hundred thousand pieces stay under the open-file limit.
A stream that goes backwards is refused where it is read.

Nothing here knows decsim's columns; report.py says which field is
which.
"""

import csv
import heapq
import math
import pathlib

import decsim.experiments.refusal as refusal


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


def header_of(path: pathlib.Path) -> list:
    """One csv file's column names, read without its rows."""
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def row_stream(path: pathlib.Path):
    """One csv file's rows, one alive at a time, values as their text.

    The values stay text because a fold writes most of them straight
    back out and reads a number only where a total needs one.
    """
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        yield from reader


def merged_rows(paths: list, key):
    """Every file's rows in the run's order, a file open while its rows are due.

    `key` gives a row its place in that order. A path with no file is a
    folder that wrote no row of this kind and is skipped, which is what
    a folder that ran no sweep leaves behind.
    """
    waiting = _files_by_first_place(paths, key)
    heap = []
    while waiting or heap:
        _open_the_files_due(waiting, heap, key)
        _place, index, row, stream = heapq.heappop(heap)
        yield row
        _push_the_next_row(heap, index, stream)


class ExactSum:
    """A running sum that equals math.fsum of the finite values it was given.

    The state is the non-overlapping partials math.fsum keeps, Shewchuk's
    adaptive-precision addition (Shewchuk 1997), and the tests compare
    against math.fsum value for value. Finite is the whole claim: on inf or
    overflow the two part company (CPython test_math.py:735-740), and no
    fold sums such a value, so nothing refuses them (STYLE.md rule 4). total
    rounds once, so a mean folded over folders in any order is the mean one
    process computes, to the last bit, which a running float sum is not.
    """

    def __init__(self) -> None:
        self.partials = []

    def add(self, value) -> None:
        """One more value, the sum still exact.

        A zero changes no partial and is skipped, three quarters of the link
        fields; math.fsum of zeros is 0.0, not -0.0. The loop stays in this
        function because a 500-folder fold makes four hundred million calls:
        0.43 us at two partials, 1.86 us at fourteen, 0.07 us for a zero.
        STYLE.md's one concession to a hot path.
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

    Each field's role is given at construction, and nothing grows with the
    rows, the shape of sinter's TaskStats (sinter/_data/_task_stats.py).
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

    The header is known before the first row; a missing cell is empty, and
    a file whose first row never came is not created.
    """

    def __init__(self, path: pathlib.Path, field_names: list) -> None:
        self.path = path
        self.field_names = field_names
        self.handle = None
        self.writer = None

    def write(self, row: dict) -> None:
        """One row; the first one opens the file and writes the header."""
        if self.writer is None:
            self._open()
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

    def _open(self) -> None:
        """The file and its header."""
        self.handle = open(self.path, "w", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.field_names)
        self.writer.writeheader()


def _files_by_first_place(paths: list, key) -> list:
    """(first place, index, path) of every file with a row, last first.

    Sorted backwards, so the file due first is the one pop takes.
    """
    waiting = []
    for index, path in enumerate(paths):
        if not path.is_file():
            continue
        first_place = _first_place(path, key)
        if first_place is None:
            continue
        waiting.append((first_place, index, path))
    waiting.sort(reverse=True)
    return waiting


def _first_place(path: pathlib.Path, key):
    """The place of a file's first row, the file open for that row only."""
    rows = row_stream(path)
    first_row = next(rows, None)
    rows.close()
    if first_row is None:
        return None
    return key(first_row)


def _open_the_files_due(waiting: list, heap: list, key) -> None:
    """Open each waiting file whose first row comes before the heap's next.

    A file and an open row of equal place go by their index, the order
    the files were given.
    """
    while waiting:
        first_place, index, path = waiting[-1]
        if heap and (first_place, index) > heap[0][:2]:
            return
        waiting.pop()
        stream = _keyed_rows(path, key)
        _push_the_next_row(heap, index, stream)


def _push_the_next_row(heap: list, index: int, stream) -> None:
    """The stream's next row onto the heap; an ended stream has closed."""
    keyed = next(stream, None)
    if keyed is None:
        return
    place, row = keyed
    heapq.heappush(heap, (place, index, row, stream))


def _keyed_rows(path: pathlib.Path, key):
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


def _refuse_a_row_out_of_order(path: pathlib.Path, place, previous) -> None:
    """A run folder holds its rows in the order the run wrote them."""
    if place >= previous:
        return
    raise refusal.RefusalError(
        f"{path} holds a row at {place} after a row at {previous}; a fold "
        "merges the pieces' rows, and a piece holds its rows in the order "
        "the sweep ran them, its task position and then its seed"
    )

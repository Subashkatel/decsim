"""The one refusal decsim.experiments raises when it will not do what was asked.

gem5 splits the user's mistake from the machine's: fatal prints one line
and exits, panic reports where an assumption broke
(src/base/logging.hh). The experiments layer's boundary refusals are
all the first kind, so `decsim <verb>` catches this one class, prints
one sentence and exits 1; everything else keeps its traceback. It is a
ValueError, as STYLE.md rule 4 asks of a boundary refusal.
"""


class RefusalError(ValueError):
    """What was asked for, and why the experiments layer will not do it."""

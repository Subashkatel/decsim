"""The one refusal the front raises when it will not do what was asked.

gem5 splits the user's mistake from the machine's: `fatal` prints one
line and exits, `panic` reports where an assumption broke
(src/base/logging.hh). The front's boundary refusals are all the first
kind, so they share one class that `decsim <verb>` catches, prints as
one sentence and exits 1 on; everything else keeps its traceback,
because a bug should look like a bug. It is a ValueError because that is
what STYLE.md rule 4 asks a boundary refusal to raise, and it carries
the Error suffix the Google Python style guide asks every exception name
to end in.
"""


class RefusalError(ValueError):
    """What was asked for, and why the front will not do it."""

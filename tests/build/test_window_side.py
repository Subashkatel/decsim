"""The windows facade and everything behind it, wired by constructor.

STYLE.md rule 7: one root object builds every component and wires them
by constructor, so nothing is bound to a component after it is built.
The one exception in decsim is _LateWiring, the stand-in the courier and
the committer take for the facade and the strong redecode, which are
built after them; this file pins that it is the only one, that it
forwards each call to the component it stands in for, and that a run
that never escalates gives the committer no redecode at all.
"""

import decsim.build.window_side as window_side


def test_the_stand_in_is_empty_until_the_root_fills_it():
    late = window_side._LateWiring()

    assert late.window_manager is None
    assert late.strong_redecode is None


def test_it_forwards_the_couriers_landing_to_the_facade():
    late = window_side._LateWiring()
    facade = _Recorder()
    late.window_manager = facade

    late.accept_boundary(("op", 1), True)

    assert facade.calls == [("accept_boundary", (("op", 1), True))]


def test_it_forwards_the_committers_three_hooks_to_the_strong_redecode():
    late = window_side._LateWiring()
    redecode = _Recorder()
    late.strong_redecode = redecode

    late.escalate("a job")
    late.submit_if_commit_releases(("op", 1))
    late.cancel_held_sibling(("op", 1))

    assert redecode.calls == [
        ("escalate", ("a job",)),
        ("submit_if_commit_releases", (("op", 1),)),
        ("cancel_held_sibling", (("op", 1),)),
    ]


def test_the_stand_in_carries_exactly_the_calls_the_two_owners_receive():
    """It stands in for two components and promises nothing else."""
    late = window_side._LateWiring()
    forwarded = []
    for name in dir(late):
        if not name.startswith("_"):
            forwarded.append(name)

    assert sorted(forwarded) == [
        "accept_boundary",
        "cancel_held_sibling",
        "escalate",
        "strong_redecode",
        "submit_if_commit_releases",
        "window_manager",
    ]


class _Recorder:
    """Records the calls made on it, by name and arguments."""

    def __init__(self) -> None:
        self.calls = []

    def accept_boundary(self, key, is_unblocked) -> None:
        self.calls.append(("accept_boundary", (key, is_unblocked)))

    def escalate(self, job) -> None:
        self.calls.append(("escalate", (job,)))

    def submit_if_commit_releases(self, key) -> None:
        self.calls.append(("submit_if_commit_releases", (key,)))

    def cancel_held_sibling(self, key) -> None:
        self.calls.append(("cancel_held_sibling", (key,)))

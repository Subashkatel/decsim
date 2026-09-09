"""Registering the workload with every component, and naming the seed roots.

Every stochastic owner is bound under a path in the run's seed tree
(decsim/seeding.py), and this module is where the root names the path of
each one, so a component draws the same numbers whatever else the run
holds. The load order is the other rule here: streams first, then every
operation with the windows, then the idle accounting, then the runtime.
"""

import decsim.build.listeners as listener_build
import decsim.records.seeds as seed_records


def test_each_stochastic_owner_is_named_by_the_field_it_arrived_under():
    owners = {"qpu": "the device", "weak_decoder": "the weak unit"}

    roots = listener_build.build_seed_roots(**owners)

    assert roots == (
        (
            (seed_records.RunSeedPathSegment("field", "qpu"),),
            "the device",
        ),
        (
            (seed_records.RunSeedPathSegment("field", "weak_decoder"),),
            "the weak unit",
        ),
    )


def test_a_run_with_no_stochastic_owner_names_no_root():
    roots = listener_build.build_seed_roots()

    assert roots == ()


def test_the_workload_reaches_every_component_in_the_stated_order():
    """Streams, then the windows, then the idle accounting, then the run."""
    order = []
    plan = _Plan()
    release = _Recorder("conditional_release", order)
    windows = _Recorder("window_manager", order)
    streams = _Recorder("streams", order)
    idle = _Recorder("idle_rounds", order)
    runtime = _Recorder("execution_runtime", order)

    listener_build.load_program(plan, release, windows, streams, idle, runtime)

    assert order == [
        "conditional_release.register_blocked_operation",
        "window_manager.register_operation",
        "window_manager.install_planned_holds",
        "window_manager.register_stream",
        "streams.load",
        "window_manager.register_operation",
        "idle_rounds.load",
        "execution_runtime.load_program",
    ]


class _Operation:
    """The three fields load_program reads off an operation."""

    def __init__(self, operation_id: int, blocked_by=None) -> None:
        self.id = operation_id
        self.blocked_by = blocked_by


class _RunPlan:
    buffering = ()


class _Plan:
    """A plan with one blocked operation and one stream."""

    def __init__(self) -> None:
        self.operations = (_Operation(1, blocked_by=2),)
        self.decode_operations = ()
        self.dynamic_streams = (_Operation(3),)
        self.protected_regions = ()
        self.planned_operations = (_Operation(1),)
        self.run_plan = _RunPlan()


class _Recorder:
    """Records every call made on it, by component and method name."""

    def __init__(self, name: str, order: list) -> None:
        self.name = name
        self.order = order

    def __getattr__(self, method_name: str):
        def record(*arguments, **keywords):
            del arguments, keywords
            self.order.append(f"{self.name}.{method_name}")

        return record

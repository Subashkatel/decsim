"""The one rule of decsim/records, walked over every module in the folder.

Every public name is documented and is a record: a frozen dataclass, an
Enum, one instance of a record, or one of the module's own functions. The
records the machine mutates as it runs are listed here by name, so a new
mutable record cannot arrive unnoticed.
"""

import dataclasses
import enum
import inspect

import pytest

import decsim.records.decoding as decoding_records
import decsim.records.identity as identity_records
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.seeds as seed_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records

MODULES = (
    identity_records,
    seed_records,
    round_records,
    window_records,
    decoding_records,
    transfer_records,
    program_records,
)
MODULE_IDS = tuple(module.__name__ for module in MODULES)

# The records the machine fills in as it runs: the window manager's live
# window and its compiled plan, the decoder queues' job, the result a
# decoder writes as it reports, and the submission an escalation policy
# hands over. Every other record is frozen.
MUTABLE_RECORDS = frozenset(
    {
        "Window",
        "WindowPlan",
        "DecodeJob",
        "DecodeResult",
        "Submission",
        "Operation",
    }
)


def public_values(module):
    """Every name one records module exposes, with the value behind it."""
    exposed = {}
    contents = vars(module)
    for name, value in contents.items():
        if name.startswith("_"):
            continue
        if inspect.ismodule(value):
            continue
        owner = getattr(value, "__module__", None)
        if owner != module.__name__:
            continue
        exposed[name] = value
    return exposed


def undocumented(module):
    """The exposed names whose value carries no docstring."""
    found = []
    exposed = public_values(module)
    for name, value in exposed.items():
        if value.__doc__:
            continue
        found.append(name)
    return sorted(found)


def not_a_record(module):
    """The exposed names that are neither a record nor a function."""
    found = []
    exposed = public_values(module)
    for name, value in exposed.items():
        if inspect.isfunction(value):
            continue
        if inspect.isclass(value) and issubclass(value, enum.Enum):
            continue
        if dataclasses.is_dataclass(value):
            continue
        found.append(name)
    return sorted(found)


def unfrozen_records(module):
    """The exposed record classes a caller can still assign a field on."""
    found = []
    exposed = public_values(module)
    for name, value in exposed.items():
        if not inspect.isclass(value):
            continue
        if not dataclasses.is_dataclass(value):
            continue
        if value.__dataclass_params__.frozen:
            continue
        found.append(name)
    return sorted(found)


@pytest.mark.parametrize("module", MODULES, ids=MODULE_IDS)
def test_every_public_name_in_a_records_module_is_documented(module):
    """A reader who opens the module learns what each value is for."""
    assert public_values(module)
    assert undocumented(module) == []


@pytest.mark.parametrize("module", MODULES, ids=MODULE_IDS)
def test_every_public_name_is_a_record_an_instance_or_a_function(module):
    """The folder holds values, not behavior: no plain classes live here."""
    assert not_a_record(module) == []


@pytest.mark.parametrize("module", MODULES, ids=MODULE_IDS)
def test_only_the_records_the_machine_mutates_are_unfrozen(module):
    """A record is frozen unless it is named as live state above."""
    for name in unfrozen_records(module):
        assert name in MUTABLE_RECORDS

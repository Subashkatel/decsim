"""The records a part's slot takes, found by the name they are written by.

A slot is a field that holds a record. Its type says what may fill it:
one record class, a union of them, or a Protocol of ports.py that any
record with its members fills, with no registry (STYLE.md rule 7). So
the records a slot takes are found, not listed: every settings record
decsim defines is read once, and those the slot's type admits are
offered. OMNeT++ chooses a `like` slot's module by its type name the
same way (simulation manual §3.11).

A record is written by a short name: its row name (union_find, sliding,
redo_window), its class name, or that name less "Settings"
(DecoderPool), or the name of the class it is nested in
(SeparateDecodeJobs). A record decsim does not define is written by its
import path, mylab.decoders.NeuralDecoder, and is taken if the slot's
type admits it.
"""

import dataclasses
import functools
import importlib
import importlib.util
import inspect
import pkgutil
import typing
from collections.abc import Sequence
from typing import Any

import decsim.parameters.fields as fields
import decsim.parameters.units as units

_SETTINGS_SUFFIX = "Settings"
_ROW_NAME = "name"
_PROTOCOL_MACHINERY = frozenset(
    {"_is_protocol", "_is_runtime_protocol", "__init__", "__subclasshook__"}
)


def record_class_named(name: str, hint: Any, path: str) -> type:
    """The record class a written name stands for in a slot of this type."""
    if "." in name:
        return _imported(name, hint, path)
    offered = records_for(hint)
    matches = records_named(name, offered)
    if len(matches) == 1:
        return matches[0]
    if matches:
        _refuse_an_ambiguous_name(name, matches, path)
    _refuse_an_unknown_name(name, offered, path)


def records_for(hint: Any) -> list:
    """Every record class decsim defines that the slot's type admits."""
    candidates = every_record_class()
    return [
        record_class
        for record_class in candidates
        if admits(hint, record_class)
    ]


def admits(hint: Any, record_class: type) -> bool:
    """Whether a slot of this type takes a record of this class."""
    unwrapped = fields.record_hint(hint)
    members = _union_members(unwrapped)
    return any(_member_admits(member, record_class) for member in members)


def records_named(name: str, record_classes: Sequence) -> list:
    """The record classes of those given that a short name stands for."""
    return [
        record_class
        for record_class in record_classes
        if name in names_of(record_class)
    ]


def every_name(record_classes: Sequence) -> list:
    """Every short name of the record classes, for a did-you-mean."""
    names = []
    for record_class in record_classes:
        short_names = names_of(record_class)
        names.extend(short_names)
    return names


def names_of(record_class: type) -> tuple:
    """The short names a record class is written by, the shortest first.

    Its row name, its class name less "Settings", the class it is
    nested in, then its class name.
    """
    names = []
    row_name = getattr(record_class, _ROW_NAME, None)
    if isinstance(row_name, str):
        names.append(row_name)
    class_name = record_class.__name__
    short = class_name.removesuffix(_SETTINGS_SUFFIX)
    names.append(short)
    qualified = record_class.__qualname__.split(".")
    if len(qualified) > 1:
        names.append(qualified[-2])
    if short:
        names.append(class_name)
    kept = [name for name in names if name]
    unique = dict.fromkeys(kept)
    return tuple(unique)


def written_name(record_class: type) -> str:
    """The name a record is shown by: its first short name."""
    names = names_of(record_class)
    return names[0]


def import_path(record_class: type) -> str:
    """The dotted path a record class is imported by."""
    return f"{record_class.__module__}.{record_class.__qualname__}"


def unique_name(record_class: type) -> str:
    """The first of a record's names no other record has, else its path.

    A record decsim does not define has no short name here, so it is
    written by its import path, which reads back to it.
    """
    offered = every_record_class()
    for name in names_of(record_class):
        matches = records_named(name, offered)
        if matches == [record_class]:
            return name
    return import_path(record_class)


def imported_class(name: str, where: str) -> type:
    """The class an import path names, a class nested in one too.

    The longest prefix that imports is the module, and the rest are its
    attributes, so mylab.parts.NeuralDecoder.Settings is the Settings
    class nested in NeuralDecoder.
    """
    segments = name.split(".")
    module, attributes = _longest_module(segments, where)
    value = module
    for attribute in attributes:
        value = getattr(value, attribute, None)
    if not inspect.isclass(value):
        written = ".".join(attributes)
        raise fields.ParameterError(
            f"{where}: {module.__name__} defines no class {written}"
        )
    return value


@functools.cache
def every_record_class() -> tuple:
    """Every settings record class decsim defines, nested ones included.

    A module whose compiled library is not built here is skipped: its
    records cannot run on this machine. Only Python modules are walked
    (is_python_module).
    """
    decsim_package = importlib.import_module("decsim")
    found = []
    modules = pkgutil.walk_packages(decsim_package.__path__, "decsim.")
    for module_info in modules:
        if not is_python_module(module_info.name):
            continue
        try:
            module = importlib.import_module(module_info.name)
        except ImportError:
            continue
        classes = _classes_defined_in(module)
        _add_new(found, classes)
    return tuple(found)


def is_python_module(module_name: str) -> bool:
    """Whether the module comes from a Python source file.

    The compiled Union-Find libraries beside their C sources are loaded
    through ctypes, never imported: Python lists a bare .so as an
    extension module, and importing the sanitized build starts the
    sanitizer runtime, which ends the process.
    """
    specification = importlib.util.find_spec(module_name)
    origin = specification.origin
    return origin.endswith(".py")


def _classes_defined_in(module: Any) -> list:
    """A module's own dataclasses, and those nested in its classes."""
    namespace = vars(module)
    defined = [
        value for value in namespace.values() if _is_defined_in(value, module)
    ]
    records = []
    while defined:
        candidate = defined.pop()
        nested = _classes_in(candidate)
        defined.extend(nested)
        if dataclasses.is_dataclass(candidate):
            records.append(candidate)
    return records


def _is_defined_in(value: Any, module: Any) -> bool:
    if not inspect.isclass(value):
        return False
    return value.__module__ == module.__name__


def _classes_in(record_class: type) -> list:
    namespace = vars(record_class)
    return [value for value in namespace.values() if inspect.isclass(value)]


def _add_new(found: list, record_classes: Sequence) -> None:
    for record_class in record_classes:
        if record_class not in found:
            found.append(record_class)


def _union_members(hint: Any) -> tuple:
    origin = typing.get_origin(hint)
    if origin is typing.Union:
        return typing.get_args(hint)
    return (hint,)


def _member_admits(member: Any, record_class: type) -> bool:
    """A Protocol admits a record with its members; a record, its kind."""
    if _is_protocol(member):
        return _has_members(record_class, member)
    if not dataclasses.is_dataclass(member):
        return False
    return issubclass(record_class, member)


def _is_protocol(member: Any) -> bool:
    return getattr(member, "_is_protocol", False)


def _has_members(record_class: type, protocol: Any) -> bool:
    """Whether a record has every member the Protocol asks for, alike."""
    members = _protocol_members(protocol)
    for member in members:
        if not _has_member(record_class, protocol, member):
            return False
    return True


def _has_member(record_class: type, protocol: Any, member: str) -> bool:
    """A field, or a method taking the same names and returning alike."""
    record_fields = getattr(record_class, "__dataclass_fields__", {})
    if member in record_fields:
        return True
    if not hasattr(record_class, member):
        return False
    if not _is_method(protocol, member):
        return True
    wanted = getattr(protocol, member)
    offered = getattr(record_class, member)
    return _same_shape(offered, wanted)


def _same_shape(offered: Any, wanted: Any) -> bool:
    """The same parameter names, and a return the wanted one admits.

    A method's names and its result's members are what a caller of the
    port uses, so two records whose build differs there do not fill one
    slot, though both have a build.
    """
    offered_names = _parameter_names(offered)
    wanted_names = _parameter_names(wanted)
    if offered_names != wanted_names:
        return False
    wanted_type = _return_type(wanted)
    offered_type = _return_type(offered)
    if not inspect.isclass(wanted_type) or not inspect.isclass(offered_type):
        return True
    if _is_protocol(wanted_type):
        methods = _protocol_methods(wanted_type)
        return all(hasattr(offered_type, method) for method in methods)
    return issubclass(offered_type, wanted_type)


def _protocol_methods(protocol: Any) -> list:
    """The methods a Protocol asks for; its fields are an instance's."""
    members = _protocol_members(protocol)
    ordered = sorted(members)
    return [member for member in ordered if _is_method(protocol, member)]


def _is_method(owner: Any, member: str) -> bool:
    value = getattr(owner, member, None)
    return inspect.isfunction(value)


def _parameter_names(function: Any) -> tuple:
    if not callable(function):
        return ()
    signature = inspect.signature(function)
    return tuple(signature.parameters)


def _return_type(function: Any) -> Any:
    """A method's return type; None where it names what cannot be read."""
    try:
        hints = typing.get_type_hints(function)
    except (NameError, TypeError):
        return None
    return hints.get("return")


def _protocol_members(protocol: Any) -> set:
    """The names a Protocol asks for, its bases' included."""
    members = set()
    for base in protocol.__mro__:
        if not _is_protocol(base):
            continue
        own = _own_members(base)
        members.update(own)
    return members - _PROTOCOL_MACHINERY


def _own_members(base: type) -> set:
    """One class's annotated fields and its public names."""
    namespace = vars(base)
    annotations = namespace.get("__annotations__", {})
    public = [name for name in namespace if not name.startswith("_")]
    return {*annotations, *public}


def _imported(name: str, hint: Any, path: str) -> type:
    """A record class written by its import path, if the slot takes it."""
    record_class = imported_class(name, path)
    if admits(hint, record_class):
        return record_class
    offered = records_for(hint)
    listed = _listed(offered)
    raise fields.ParameterError(
        f"{path}: {name} is not a record this slot takes; it takes {listed}"
    )


def _longest_module(segments: list, where: str) -> tuple:
    """The module the longest importable prefix names, and the rest."""
    longest = len(segments) - 1
    for end in range(longest, 0, -1):
        module_name = ".".join(segments[:end])
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as missing:
            _reraise_unless_absent(missing, module_name)
            continue
        return module, segments[end:]
    written = ".".join(segments)
    raise fields.ParameterError(f"{where}: no module of {written} imports")


def _reraise_unless_absent(
    missing: ModuleNotFoundError, module_name: str
) -> None:
    """A module that exists but imports one that does not is an error."""
    absent = missing.name or ""
    prefix = f"{absent}."
    if module_name == absent or module_name.startswith(prefix):
        return
    raise missing


def _refuse_an_ambiguous_name(name: str, matches: list, path: str) -> None:
    qualified = [import_path(record_class) for record_class in matches]
    joined = ", ".join(qualified)
    raise fields.ParameterError(
        f"{path}: {name!r} names {len(matches)} records ({joined}); write "
        "the import path of one"
    )


def _refuse_an_unknown_name(name: str, offered: list, path: str) -> None:
    names = every_name(offered)
    near = units.did_you_mean(name, names)
    listed = _listed(offered)
    raise fields.ParameterError(
        f"{path}: no record of this slot is named {name!r}.{near} It "
        f"takes {listed}"
    )


def _listed(offered: list) -> str:
    written = [written_name(record_class) for record_class in offered]
    return ", ".join(sorted(written))

"""Every send is executed at an end of the hop it rides.

OMNeT++ enforces this at runtime, refusing a module that sends a message
it does not own (tmp/resources/omnetpp/src/sim/csimplemodule.cc:333-334,
omnetpp-6.1.0), and gem5 bills a transfer to the port it left by rather
than to whoever arranged it (packet.hh:424-431).
decsim has no runtime check for it, so the rule is read out of the tree:
a component module that names a LinkPath must be one of that path's two
ends.
"""

import ast
import pathlib

TESTS_FILE = pathlib.Path(__file__)
TESTS_PATH = TESTS_FILE.resolve()
PACKAGE_ROOT = TESTS_PATH.parent.parent
DECSIM_ROOT = PACKAGE_ROOT / "decsim"

# each hop's two ends, as the packages that own them
ENDS_OF_PATH = {
    "QPU_TO_CONTROLLER": ("qpu", "controller"),
    "CONTROLLER_TO_WEAK_BUFFER": ("controller", "syndrome_buffer"),
    "WEAK_BUFFER_TO_WEAK_DECODER": ("syndrome_buffer", "decoders"),
    "WEAK_DECODER_TO_STRONG_DECODER": ("decoders",),
    "STRONG_BUFFER_TO_STRONG_DECODER": ("syndrome_buffer", "decoders"),
    "WEAK_DECODER_TO_FRAME": ("decoders", "pauli_frame"),
    "DECODER_TO_DECODER": ("decoders",),
    "STRONG_DECODER_TO_FRAME": ("decoders", "pauli_frame"),
    "FRAME_TO_CONTROLLER": ("pauli_frame", "controller"),
    "CONTROLLER_TO_QPU": ("controller", "qpu"),
    "CONTROLLER_TO_STRONG_BUFFER": ("controller", "syndrome_buffer"),
}

# the components on the reaction path; the root wires the paths onto the
# ports (build, machine), the fabric carries them (links) and the
# reporting side reads them (observe, front, collect), so none of those
# is a sender and none is walked
COMPONENT_PACKAGES = (
    "qpu",
    "controller",
    "syndrome_buffer",
    "windows",
    "decoders",
    "escalation",
    "confidence",
    "pauli_frame",
)


def _named_paths(path: pathlib.Path) -> list:
    """Every LinkPath member this module names, with its line."""
    text = path.read_text()
    tree = ast.parse(text)
    named = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        inner = node.value
        if not isinstance(inner, ast.Attribute):
            continue
        if inner.attr != "LinkPath":
            continue
        named.append((node.attr, node.lineno))
    return named


def test_every_component_that_names_a_hop_is_an_end_of_it():
    wrong_end = {}
    for package in COMPONENT_PACKAGES:
        directory = DECSIM_ROOT / package
        modules = directory.glob("*.py")
        for module in sorted(modules):
            _add_wrong_ends(module, package, wrong_end)
    assert wrong_end == {}


def _add_wrong_ends(module: pathlib.Path, package: str, wrong: dict) -> None:
    """Add every hop this module names that it is not an end of."""
    for name, lineno in _named_paths(module):
        ends = ENDS_OF_PATH[name]
        if package in ends:
            continue
        wrong[f"{package}/{module.name}:{lineno}"] = name


def test_the_table_of_ends_covers_every_hop():
    """A new hop joins the table, so the law never silently shrinks."""
    text = (DECSIM_ROOT / "records" / "transfers.py").read_text()
    tree = ast.parse(text)
    members = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name != "LinkPath":
            continue
        members = _member_names(node)
    assert sorted(members) == sorted(ENDS_OF_PATH)


def _member_names(node: ast.ClassDef) -> list:
    """The enum members declared in the class body."""
    names = []
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            assigned = _assigned_names(statement)
            names.extend(assigned)
    return names


def _assigned_names(statement: ast.Assign) -> list:
    """The plain names one assignment writes to."""
    names = []
    for target in statement.targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
    return names


# Where each hop's delivery callback is registered: the module that
# names the path, and the function the sender hands the link. The table
# is asserted against the tree, so a send that stops resolving is
# visible here rather than passing unseen. A callback the asker supplies
# is recorded as None: that sender sends on another component's ask, and
# what runs at the landing is written where the ask is made.
DELIVERY_CALLBACKS = {
    ("controller/controller.py", "QPU_TO_CONTROLLER"): "at_controller",
    ("controller/instruction_output.py", "FRAME_TO_CONTROLLER"): (
        "at_controller"
    ),
    ("controller/instruction_output.py", "CONTROLLER_TO_QPU"): "delivered",
    ("controller/round_transmission.py", "CONTROLLER_TO_WEAK_BUFFER"): (
        "_publish"
    ),
    ("decoders/decoder_output.py", "WEAK_DECODER_TO_STRONG_DECODER"): None,
    ("syndrome_buffer/strong_round_writer.py", "CONTROLLER_TO_STRONG_BUFFER"): (
        "landed"
    ),
}

# The collaborators a delivery callback may reach that are neither end
# of the hop: the engine is the clock every component holds, the trace
# is observation, which no port carries (STYLE.md rule 7), and the link
# fabric is the wire the hop rides rather than an end of it.
NOT_AN_END = ("engine", "trace", "link", "links", "transfers")

# The keywords a link call names its delivery callback by.
CALLBACK_KEYWORDS = ("on_delivered", "on_landed")


def test_every_delivery_callback_leaves_the_landing_to_the_receiving_end():
    """The receiving end handles the landing; the sender only hears it.

    gem5's requesting port hands the packet to the peer's own receive
    method (tmp/resources/gem5/src/mem/port.hh:603-614, whose
    protocol/timing.cc:49-53 calls peer->recvTimingReq), OMNeT++ moves
    the message to the destination module before that module's
    handleMessage runs (csimplemodule.cc:777-799), and ns-3 schedules
    Receive on the destination device (point-to-point-channel.cc:88-92).
    So every method a delivery callback reaches through a collaborator
    is defined in a package that is an end of that hop.
    """
    wrong_end = {}
    for site in _delivery_sites_of_the_tree():
        _add_wrong_receivers(site, wrong_end)
    assert wrong_end == {}


def test_one_call_at_the_other_end_per_delivery_callback():
    """The callback makes one call at the other end, not two.

    A sender that makes two calls at the other end is deciding what the
    landing does; that decision is the receiving end's own method.
    """
    too_many = {}
    for site in _delivery_sites_of_the_tree():
        _add_second_calls(site, too_many)
    assert too_many == {}


def test_the_table_of_delivery_callbacks_matches_the_tree():
    """A new send, or a callback that stops resolving, is visible here."""
    found = {}
    for site in _delivery_sites_of_the_tree():
        found[(site.where, site.path_name)] = site.callback_name
    assert found == DELIVERY_CALLBACKS


class _DeliverySite:
    """One send that names a hop, and the callback it registers."""

    def __init__(self, module, package: str, named: tuple, callback):
        self.module = module
        self.package = package
        # (the LinkPath member, the line the send stands on)
        self.path_name, self.lineno = named
        self.callback = callback

    @property
    def where(self) -> str:
        """The module, as the table of callbacks names it."""
        return f"{self.package}/{self.module.name}"

    @property
    def callback_name(self):
        """The callback's name, or None when the asker supplies it."""
        if self.callback is None:
            return None
        return self.callback.name


def _delivery_sites_of_the_tree() -> list:
    """Every send in a component package that names its own hop."""
    sites = []
    for package in COMPONENT_PACKAGES:
        directory = DECSIM_ROOT / package
        modules = directory.glob("*.py")
        _add_package_sites(sorted(modules), package, sites)
    return sites


def _add_package_sites(modules: list, package: str, sites: list) -> None:
    """Add every delivery site of one package's modules."""
    for module in modules:
        found = _delivery_sites(module, package)
        sites.extend(found)


def _delivery_sites(module: pathlib.Path, package: str) -> list:
    """Every send this module makes that names a link path."""
    text = module.read_text()
    tree = ast.parse(text)
    functions = _functions_by_name(tree)
    partials = _partial_targets(tree)
    sites = []
    for node in ast.walk(tree):
        registered = _registered_callback(node)
        _add_call_site(
            node, registered, (module, package), sites, functions, partials
        )
    return sites


def _add_call_site(node, registered, where, sites, functions, partials) -> None:
    """Add one call that names a hop, with the callback it registers."""
    if registered is None:
        return
    module, package = where
    path_name = _path_argument(node)
    callback = _resolved_callback(registered, functions, partials)
    named = (path_name, node.lineno)
    site = _DeliverySite(module, package, named, callback)
    sites.append(site)


def _registered_callback(node):
    """The delivery callback of a call that names a hop, or None."""
    if not isinstance(node, ast.Call):
        return None
    path_name = _path_argument(node)
    if path_name is None:
        return None
    return _callback_argument(node)


def _path_argument(node: ast.Call):
    """The LinkPath member this call names, or None."""
    for argument in node.args:
        member = _link_path_member(argument)
        if member is not None:
            return member
    return None


def _link_path_member(argument):
    """The member name of a LinkPath attribute, or None."""
    if not isinstance(argument, ast.Attribute):
        return None
    inner = argument.value
    if not isinstance(inner, ast.Attribute):
        return None
    if inner.attr != "LinkPath":
        return None
    return argument.attr


def _callback_argument(node: ast.Call):
    """The expression the send registers as its delivery callback."""
    for keyword in node.keywords:
        if keyword.arg in CALLBACK_KEYWORDS:
            return keyword.value
    if not node.args:
        return None
    return node.args[-1]


def _resolved_callback(registered, functions: dict, partials: dict):
    """The function the registered expression names, or None."""
    name = _callback_name(registered, partials)
    if name is None:
        return None
    return functions.get(name)


def _callback_name(registered, partials: dict):
    """The function name an expression stands for, or None."""
    if isinstance(registered, ast.Attribute):
        return registered.attr
    if not isinstance(registered, ast.Name):
        return None
    if registered.id in partials:
        return partials[registered.id]
    return registered.id


def _functions_by_name(tree: ast.Module) -> dict:
    """Every function of one module, nested ones included, by name."""
    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = node
    return functions


def _partial_targets(tree: ast.Module) -> dict:
    """Every local name bound to a partial over one function."""
    targets = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            _add_partial_target(node, targets)
    return targets


def _add_partial_target(node: ast.Assign, targets: dict) -> None:
    """Record one name bound to a partial over a function."""
    value = node.value
    if not isinstance(value, ast.Call):
        return
    if not value.args:
        return
    bound = _callback_name(value.args[0], {})
    if bound is None:
        return
    _bind_names(node.targets, bound, targets)


def _bind_names(targets: list, bound: str, names: dict) -> None:
    """Bind every plain name of an assignment to one function name."""
    for target in targets:
        if isinstance(target, ast.Name):
            names[target.id] = bound


def _reached_methods(site) -> list:
    """Every (method, line) the callback reaches through a collaborator."""
    if site.callback is None:
        return []
    text = site.module.read_text()
    tree = ast.parse(text)
    functions = _functions_by_name(tree)
    reached = []
    _add_reached(site.callback, functions, reached, True)
    return reached


def _add_reached(node, functions: dict, reached: list, follows: bool) -> None:
    """Add the collaborator calls of one function, and of what it calls."""
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call):
            _add_reached_call(inner, functions, reached, follows)


def _add_reached_call(node, functions, reached: list, follows: bool) -> None:
    """Classify one call: a collaborator's method, or the sender's own."""
    own = _own_function(node.func, functions)
    if own is not None:
        _follow_own_function(own, functions, reached, follows)
        return
    collaborator = _collaborator_of(node.func)
    if collaborator is None:
        return
    if collaborator in NOT_AN_END:
        return
    reached.append((node.func.attr, node.lineno))


def _follow_own_function(own, functions, reached: list, follows: bool) -> None:
    """Walk into the sender's own helper once, so a hop cannot hide."""
    if follows:
        _add_reached(own, functions, reached, False)


def _own_function(func, functions: dict):
    """The sender's own function this call names, or None."""
    if isinstance(func, ast.Name):
        return functions.get(func.id)
    if not isinstance(func, ast.Attribute):
        return None
    if not isinstance(func.value, ast.Name):
        return None
    if func.value.id != "self":
        return None
    return functions.get(func.attr)


def _collaborator_of(func):
    """The collaborator a call is made through, or None for a plain call."""
    if not isinstance(func, ast.Attribute):
        return None
    inner = func.value
    if not isinstance(inner, ast.Attribute):
        return None
    if _base_name(inner) != "self":
        return None
    return _first_attribute(inner)


def _first_attribute(node) -> str:
    """The attribute of self an attribute chain starts at."""
    walked = node
    first = node.attr
    while isinstance(walked, ast.Attribute):
        first = walked.attr
        walked = walked.value
    return first


def _packages_defining() -> dict:
    """Every function name of the component packages, by package."""
    packages = {}
    for package in COMPONENT_PACKAGES:
        directory = DECSIM_ROOT / package
        modules = directory.glob("*.py")
        _add_defined_names(sorted(modules), package, packages)
    return packages


def _add_defined_names(modules: list, package: str, packages: dict) -> None:
    """Add one package's function names to the map."""
    for module in modules:
        text = module.read_text()
        tree = ast.parse(text)
        _add_module_names(tree, package, packages)


def _add_module_names(tree: ast.Module, package: str, packages: dict) -> None:
    """Add one module's function names to the map."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defined = packages.setdefault(node.name, set())
            defined.add(package)


def _add_wrong_receivers(site, wrong: dict) -> None:
    """Add every method the callback reaches outside the hop's two ends."""
    ends = set(ENDS_OF_PATH[site.path_name])
    defined = _packages_defining()
    for method, lineno in _reached_methods(site):
        packages = defined.get(method, set())
        on_an_end = packages & ends
        reached = (method, lineno)
        _add_wrong_end(site, reached, on_an_end, wrong)


def _add_wrong_end(site, reached: tuple, on_an_end: set, wrong: dict) -> None:
    """Record a reached method that no end of the hop defines."""
    if on_an_end:
        return
    method, lineno = reached
    wrong[f"{site.where}:{lineno}"] = f"{method} for {site.path_name}"


def _add_second_calls(site, too_many: dict) -> None:
    """Record a callback that calls the other end more than once."""
    ends = set(ENDS_OF_PATH[site.path_name])
    other_end = ends - {site.package}
    defined = _packages_defining()
    at_the_other_end = []
    for method, lineno in _reached_methods(site):
        packages = defined.get(method, set())
        at_the_end = packages & other_end
        reached = (method, lineno)
        _add_call_at_the_other_end(reached, at_the_end, at_the_other_end)
    _add_when_more_than_one(site, at_the_other_end, too_many)


def _add_call_at_the_other_end(reached, at_the_end: set, calls: list) -> None:
    """Keep one reached method that the receiving end defines."""
    if at_the_end:
        calls.append(reached)


def _add_when_more_than_one(site, calls: list, too_many: dict) -> None:
    """Record the callback when it calls the other end twice or more."""
    if len(calls) > 1:
        too_many[site.where] = calls


def _base_name(node) -> str:
    """The name an attribute chain is read from, or the empty string."""
    walked = node
    while isinstance(walked, ast.Attribute):
        walked = walked.value
    if isinstance(walked, ast.Name):
        return walked.id
    return ""

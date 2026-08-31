import dataclasses
import inspect
import math
import typing

import pytest

import decsim
import decsim.qpu.code_geometry as codes_module
from decsim.qpu.code_geometry import BBCodeModel, SurfaceCodeModel
from decsim.config import TimingConfig, us
from decsim.decoders.decoders import CodeRouter, PresetLatencyDecoder
from decsim.qpu.syndrome_devices import SyndromeBitDevice
from decsim.qpu.layouts import UniformLayout
from decsim.message import DecodeJob, Operation
from decsim.protocols import CodeModel
from decsim.run_spec import RunSpec
from decsim.windows.windowing_schemes import ParallelWindowScheme


class IntSubclass(int):
    pass


class OutsideCodeModel:
    name = "outside structural code"
    distance = 2

    def rounds_per_logical_cycle(self):
        return 2

    def round_period_us(self):
        return None

    def commit_rounds(self):
        return 2

    def buffer_rounds(self):
        return 0

    def buffering_floor(self):
        return (0, 0)

    window_floor_justification = None

    def spatial_nodes(self, num_patches):
        return 4 * num_patches

    def syndrome_bits_per_round(self, num_patches):
        return 3 * num_patches


class IdentityBreakingLayout(UniformLayout):
    def __init__(self, declared_code, alternate_code, broken_selector):
        super().__init__(declared_code)
        self.alternate_code = alternate_code
        self.broken_selector = broken_selector

    def code_for_op(self, operation):
        if self.broken_selector == "operation":
            return self.alternate_code
        return super().code_for_op(operation)

    def code_for_patch(self, patch_id):
        if self.broken_selector == "patch":
            return self.alternate_code
        return super().code_for_patch(patch_id)


def make_operation():
    return Operation(id=1, name="memory", qubits=(0,))


def build_run(code=None, **options):
    arguments = {
        "ops": [make_operation()],
        "decoder": PresetLatencyDecoder(0.0),
        **options,
    }
    if code is not None:
        arguments["code"] = code
    return RunSpec(**arguments).build()


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("d", 0),
        ("d", -1),
        ("d", True),
        ("d", 3.5),
        ("commit_rounds_override", 0),
        ("buffer_rounds_override", -1),
    ),
)
def test_surface_geometry_inputs_are_stored_until_planning(field_name, value):
    """Surface cards store invalid geometry without raising during construction."""
    card = SurfaceCodeModel(**{field_name: value})
    stored = getattr(card, field_name)
    assert stored == value
    assert type(stored) is type(value)


@pytest.mark.parametrize(
    ("field_name", "value", "boundary_label"),
    (
        ("d", 0, "distance"),
        ("d", -1, "distance"),
        ("d", True, "distance"),
        ("d", 3.5, "distance"),
        ("commit_rounds_override", 0, "commit_round_count"),
        ("buffer_rounds_override", -1, "buffer_round_count"),
    ),
)
def test_planning_rejects_deferred_surface_geometry(field_name, value, boundary_label):
    """A full run rejects invalid Surface geometry at the resolved boundary."""
    card = SurfaceCodeModel(**{field_name: value})
    with pytest.raises(TypeError, match=boundary_label):
        build_run(card)


def test_buffer_below_the_floor_needs_a_written_justification():
    """A buffer below the (d, d) floor is refused by name unless the card says why it runs there."""
    with pytest.raises(ValueError, match="below the trailing buffering floor 3"):
        build_run(SurfaceCodeModel(buffer_rounds_override=0))
    justified = SurfaceCodeModel(buffer_rounds_override=0,
                                 window_floor_justification="test: zero buffer on purpose")
    assert justified.buffer_rounds() == 0
    assert build_run(justified).result.terminal_status == "complete"


def test_justification_above_the_floor_is_refused():
    """A stale justification on a card at or above the floor is an error, not a silent no-op."""
    with pytest.raises(ValueError, match="is not below the trailing buffering floor"):
        build_run(SurfaceCodeModel(window_floor_justification="stale"))


def test_empty_justification_is_refused():
    with pytest.raises(ValueError, match="non-empty string"):
        SurfaceCodeModel(window_floor_justification="   ")


@pytest.mark.parametrize("field_name", ("n", "k", "d", "commit_rounds_override"))
@pytest.mark.parametrize("value", (0, -1))
def test_bb_positive_integer_fields_reject_invalid_values(field_name, value):
    """BB positive integer fields reject nonpositive values immediately."""
    with pytest.raises(ValueError) as error:
        BBCodeModel(**{field_name: value})
    assert field_name in str(error.value)
    assert repr(value) in str(error.value)


def test_bb_buffer_override_rejects_negative_values():
    with pytest.raises(ValueError, match="buffer_rounds_override"):
        BBCodeModel(buffer_rounds_override=-1)


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        ({"n": 3, "k": 1, "d": 1}, "n must be even"),
        ({"n": 4, "k": 5, "d": 1}, "k must not exceed n"),
        ({"n": 4, "k": 1, "d": 5}, "d must not exceed n"),
    ),
)
def test_bb_cross_field_guards_reject_incoherent_parameters(arguments, message):
    """BB cards reject odd length and parameters that exceed the code length."""
    with pytest.raises(ValueError, match=message):
        BBCodeModel(**arguments)


def test_bb_cards_leave_construction_existence_and_window_relations_unchecked():
    """BB construction accepts coherent numbers without proving a code or scheme relation."""
    card = BBCodeModel(n=4, k=3, d=4, commit_rounds_override=1, buffer_rounds_override=9)
    assert (card.n, card.k, card.d) == (4, 3, 4)
    assert (card.commit_rounds(), card.buffer_rounds()) == (1, 9)


def test_scheme_specific_window_rules_are_enforced_after_card_construction():
    """A scheme rejects unequal BB commit and buffer widths after the card constructs."""
    card = BBCodeModel()
    with pytest.raises(ValueError, match="commit_round_count == buffer_round_count"):
        build_run(card, scheme=ParallelWindowScheme())


@pytest.mark.parametrize("model_type", (SurfaceCodeModel, BBCodeModel))
def test_cadence_normalizes_numeric_inputs_and_value_equality(model_type):
    """Both cards normalize numeric cadence inputs before value comparison and hashing."""
    integer_card = model_type(round_us=2)
    float_card = model_type(round_us=2.0)
    string_card = model_type(round_us="2")
    assert integer_card.round_period_us() == 2.0
    assert type(integer_card.round_period_us()) is float
    assert integer_card == float_card == string_card
    assert hash(integer_card) == hash(float_card) == hash(string_card)


@pytest.mark.parametrize("model_type", (SurfaceCodeModel, BBCodeModel))
def test_none_cadence_uses_the_run_level_fallback(model_type):
    """A missing card cadence uses the run-level cadence in a complete build."""
    card = model_type(round_us=None)
    completed = build_run(card, round_us=2.25)
    assert card.round_period_us() is None
    assert completed.qpu.cycle_ticks == us(2.25)


@pytest.mark.parametrize("model_type", (SurfaceCodeModel, BBCodeModel))
def test_card_cadence_precedes_run_and_timing_fallbacks(model_type):
    """An explicit card cadence wins over run and timing fallback values."""
    card = model_type(round_us=1.75)
    completed = build_run(
        card,
        round_us=2.25,
        timing=TimingConfig(round_us=3.5),
    )
    assert completed.qpu.cycle_ticks == us(1.75)


@pytest.mark.parametrize("model_type", (SurfaceCodeModel, BBCodeModel))
@pytest.mark.parametrize(
    ("value", "message"),
    (
        (float("nan"), "finite real number"),
        (float("inf"), "finite real number"),
        (0.0, "at least one tick"),
        (-1.0, "at least one tick"),
        (4e-7, "at least one tick"),
    ),
)
def test_planner_rejects_admitted_invalid_cadences(model_type, value, message):
    """Cards admit exceptional cadences that the planner rejects before scheduling."""
    card = model_type(round_us=value)
    assert type(card.round_period_us()) is float
    with pytest.raises(ValueError, match=message):
        build_run(card)


@pytest.mark.parametrize("model_type", (SurfaceCodeModel, BBCodeModel))
def test_nonnumeric_cadence_uses_natural_float_conversion_errors(model_type):
    """Nonnumeric cadence input fails through ordinary float conversion."""
    with pytest.raises(ValueError):
        model_type(round_us="not a number")


def test_models_report_exact_generic_names():
    """Surface and BB code models expose their exact parameterized names."""
    surface = SurfaceCodeModel()
    default_bb = BBCodeModel()
    other_bb = BBCodeModel(n=56, k=2, d=10)
    assert surface.name == "rotated surface code (d=3)"
    assert default_bb.name == "bivariate-bicycle code [[144,12,12]]"
    assert other_bb.name == "bivariate-bicycle code [[56,2,10]]"


def test_router_uses_exact_names_and_silently_falls_back_when_unmapped():
    """Decoder routing selects only an exact name and otherwise uses its default."""
    card = BBCodeModel(n=72, k=8, d=6)
    default_decoder = PresetLatencyDecoder(3.0)
    selected_decoder = PresetLatencyDecoder(1.0)
    router = CodeRouter(default_decoder, {card.name: selected_decoder})
    exact_job = DecodeJob(1, 0, 1, code=card.name)
    changed_job = DecodeJob(1, 0, 1, code=card.name.upper())
    assert router.route(exact_job) is selected_decoder
    assert router.route(changed_job) is default_decoder


@pytest.mark.parametrize("model_type", (SurfaceCodeModel, BBCodeModel))
def test_cards_carry_no_justification_by_default(model_type):
    """Both cards start without a floor justification; overrides alone do not opt out of the floor."""
    assert model_type().window_floor_justification is None
    assert model_type(buffer_rounds_override=0).window_floor_justification is None


def test_round_floors_and_window_defaults_follow_each_card_policy():
    """Each card reports its configured distance, cycle, floor, commit, and buffer defaults."""
    surface = SurfaceCodeModel(d=5)
    bb = BBCodeModel(n=24, k=4, d=6)
    assert (surface.distance, surface.rounds_per_logical_cycle()) == (5, 5)
    assert (surface.commit_rounds(), surface.buffer_rounds()) == (5, 5)
    assert surface.buffering_floor() == (5, 5)
    assert (bb.distance, bb.rounds_per_logical_cycle()) == (6, 6)
    assert (bb.commit_rounds(), bb.buffer_rounds()) == (6, 0)
    assert bb.buffering_floor() == (0, 0)


def test_window_overrides_replace_card_defaults_without_cross_field_checks():
    """Positive commit and buffer overrides replace defaults independently on both cards."""
    surface = SurfaceCodeModel(d=5, commit_rounds_override=2, buffer_rounds_override=3)
    bb = BBCodeModel(n=24, k=4, d=6, commit_rounds_override=2, buffer_rounds_override=3)
    assert (surface.commit_rounds(), surface.buffer_rounds()) == (2, 3)
    assert (bb.commit_rounds(), bb.buffer_rounds()) == (2, 3)


def test_surface_sizing_uses_unclamped_patch_arithmetic_and_the_seam_term():
    """Surface sizing uses the supplied patch count directly with its multipatch seam."""
    card = SurfaceCodeModel(d=3)
    assert card.spatial_nodes(1) == 9
    assert card.spatial_nodes(3) == 30
    assert card.spatial_nodes(0) == 0
    assert card.spatial_nodes(-2) == -18
    assert card.spatial_nodes(1.5) == 16.5
    assert card.syndrome_bits_per_round(3) == 24
    assert card.syndrome_bits_per_round(-2) == -16
    assert card.syndrome_bits_per_round(1.5) == 12.0


def test_bb_sizing_is_unclamped_and_linear_in_patch_count():
    """BB sizing uses the supplied patch count directly for nodes and syndrome bits."""
    card = BBCodeModel(n=24, k=4, d=6)
    for count in (3, 0, -2, 1.5):
        assert card.spatial_nodes(count) == count * 24
        assert card.syndrome_bits_per_round(count) == count * 24


def test_all_four_sizing_sites_return_zero_for_zero_patches():
    """Every sizing method returns zero when directly asked about zero patches."""
    surface = SurfaceCodeModel()
    bb = BBCodeModel()
    assert (
        surface.spatial_nodes(0),
        surface.syndrome_bits_per_round(0),
        bb.spatial_nodes(0),
        bb.syndrome_bits_per_round(0),
    ) == (0, 0, 0, 0)


def test_distance_one_surface_card_has_zero_syndrome_width():
    """A distance-one Surface card constructs and has zero syndrome bits per patch."""
    card = SurfaceCodeModel(d=1)
    assert card.spatial_nodes(1) == 1
    assert card.syndrome_bits_per_round(1) == 0
    assert build_run(card).result.terminal_status == "complete"


def test_zero_patch_syndrome_device_fails_naturally_when_selecting_a_target():
    """A zero-patch device request reaches natural missing-target failure without a fake readout."""
    device = SyndromeBitDevice(SurfaceCodeModel(), seed=7)
    empty_operation = Operation(id=1, name="empty", qubits=())
    with pytest.raises(IndexError):
        device.round_payloads(empty_operation, 1)


def test_dataclass_fields_defaults_and_constructor_order_are_stable():
    """Generated constructors preserve every card field, default, and positional order."""
    expected_surface = (
        ("d", 3),
        ("round_us", None),
        ("commit_rounds_override", None),
        ("buffer_rounds_override", None),
        ("window_floor_justification", None),
    )
    expected_bb = (
        ("n", 144),
        ("k", 12),
        ("d", 12),
        ("round_us", None),
        ("commit_rounds_override", None),
        ("buffer_rounds_override", None),
        ("window_floor_justification", None),
    )
    for model_type, expected in (
        (SurfaceCodeModel, expected_surface),
        (BBCodeModel, expected_bb),
    ):
        assert tuple((field.name, field.default) for field in dataclasses.fields(model_type)) == expected
        parameters = tuple(inspect.signature(model_type).parameters.values())
        assert tuple(parameter.name for parameter in parameters) == tuple(name for name, _ in expected)
        assert all(parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in parameters)


def test_frozen_cards_have_generated_value_hash_and_replacement_behavior():
    """Frozen cards compare and hash by value, reject mutation, and revalidate replacements."""
    first = SurfaceCodeModel(3, 2)
    second = SurfaceCodeModel(d=3, round_us=2.0)
    assert first == second
    assert hash(first) == hash(second)
    assert first != BBCodeModel(d=3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.d = 5
    replaced = dataclasses.replace(first, round_us="3")
    assert replaced.round_period_us() == 3.0
    with pytest.raises(ValueError, match="n must be even"):
        dataclasses.replace(BBCodeModel(), n=143)


def test_deferred_unhashable_surface_geometry_fails_only_when_hashed():
    """Deferred Surface geometry may construct even when generated hashing cannot hash it."""
    card = SurfaceCodeModel(d=[])
    with pytest.raises(TypeError):
        hash(card)


def test_module_runtime_surface_keeps_models_helpers_and_annotation_scaffolding():
    """The module exposes the two models and retained standard annotation scaffolding only."""
    public_names = {name for name in vars(codes_module) if not name.startswith("_")}
    expected_bindings = {"dataclass", "Optional", "SurfaceCodeModel", "BBCodeModel"}
    assert public_names - {"annotations"} == expected_bindings
    assert codes_module.dataclass is dataclasses.dataclass
    assert codes_module.Optional is typing.Optional
    assert not hasattr(codes_module, "__all__")
    assert not hasattr(codes_module, "math")
    assert not hasattr(codes_module, "us")
    assert not hasattr(decsim, "SurfaceCodeModel")
    assert not hasattr(decsim, "BBCodeModel")
    assert all(isinstance(value, str) for value in SurfaceCodeModel.__annotations__.values())
    assert all(isinstance(value, str) for value in BBCodeModel.__annotations__.values())


@pytest.mark.parametrize("model_type", (SurfaceCodeModel, BBCodeModel))
def test_models_satisfy_the_structural_protocol_without_inheritance(model_type):
    """Each built-in card satisfies the code protocol structurally without a model base class."""
    card = model_type()
    assert model_type.__bases__ == (object,)
    assert isinstance(card, CodeModel)
    member_parameters = {
        "rounds_per_logical_cycle": ("self",),
        "round_period_us": ("self",),
        "commit_rounds": ("self",),
        "buffer_rounds": ("self",),
        "buffering_floor": ("self",),
        "spatial_nodes": ("self", "num_patches"),
        "syndrome_bits_per_round": ("self", "num_patches"),
    }
    for member_name, expected_parameters in member_parameters.items():
        assert callable(getattr(card, member_name))
        signature = inspect.signature(getattr(model_type, member_name))
        assert tuple(signature.parameters) == expected_parameters
    assert card.window_floor_justification is None


def test_outside_structural_code_model_completes_a_full_run():
    """A minimal outside structural code model drives a complete run without registration."""
    card = OutsideCodeModel()
    assert isinstance(card, CodeModel)
    completed = build_run(card)
    assert completed.result.terminal_status == "complete"
    assert completed.window_manager._code_geometry.code_name == card.name


def test_default_run_resolves_a_distance_three_surface_card():
    """A run without an explicit code resolves the default distance-three Surface geometry."""
    completed = build_run()
    geometry = completed.window_manager._code_geometry
    assert geometry.code_name == "rotated surface code (d=3)"
    assert geometry.distance == 3


def test_outside_name_annotation_is_declared_but_not_runtime_validated():
    """An outside card's name annotation does not add runtime router validation."""
    card = OutsideCodeModel()
    card.name = 7
    completed = build_run(card)
    geometry = completed.window_manager._code_geometry
    assert geometry.code_name == 7
    assert geometry.window_floor_justification is None
    assert completed.result.terminal_status == "complete"


def test_equal_but_distinct_device_code_is_rejected_by_identity():
    """A syndrome device must share the exact run code object rather than an equal card."""
    run_code = SurfaceCodeModel()
    equal_device_code = SurfaceCodeModel()
    assert run_code == equal_device_code
    assert run_code is not equal_device_code
    device = SyndromeBitDevice(equal_device_code, seed=7)
    with pytest.raises(ValueError, match="exact resolved run code"):
        build_run(run_code, device=device)


@pytest.mark.parametrize("broken_selector", ("operation", "patch"))
def test_layout_selectors_must_return_the_exact_declared_code(broken_selector):
    """Operation and patch layout selectors reject equal but distinct code objects."""
    declared = SurfaceCodeModel()
    alternate = SurfaceCodeModel()
    layout = IdentityBreakingLayout(declared, alternate, broken_selector)
    with pytest.raises(ValueError, match=f"layout {broken_selector}"):
        RunSpec(
            ops=[make_operation()],
            layout=layout,
            decoder=PresetLatencyDecoder(0.0),
        ).build()


def test_syndrome_device_stamps_the_exact_card_name_on_readouts():
    """Syndrome readouts carry the exact routing name exposed by their code card."""
    card = BBCodeModel(n=24, k=4, d=6)
    device = SyndromeBitDevice(card, seed=7, max_bits=2)
    readout = device.round_payloads(make_operation(), 1)[0]
    assert readout.code == "bivariate-bicycle code [[24,4,6]]"


def test_syndrome_device_exposes_the_same_code_as_its_seed_child():
    """The syndrome device exposes its exact built-in code card without code seed state."""
    card = SurfaceCodeModel()
    device = SyndromeBitDevice(card, seed=7)
    children = device.run_seed_children()
    assert len(children) == 1
    assert children[0].child is card
    assert not hasattr(card, "run_seed_children")

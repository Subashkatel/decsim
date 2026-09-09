"""One task's window error models, built once and reused by every shot.

The rule the module states: a window's detector error model is a
function of the operation's circuit and the window plan and of nothing a
seed touches, so every shot of one sweep point reuses what the first
shot built. This is sinter's own arrangement one level down, where the
decoder is compiled once per task and every shot of the task decodes
with it (sinter/_decoding/_decoding_decoder_class.py,
compile_decoder_for_dem).
"""

import decsim.windows.built_window_models as built_window_models


def test_a_key_that_was_never_built_has_no_models_and_is_not_a_reuse():
    models = built_window_models.BuiltWindowModels()

    held = models.models_of(("operation", 1))

    assert held == []
    assert models.reuses == 0
    assert models.builds == 0


def test_a_remembered_key_is_handed_back_to_every_later_shot():
    models = built_window_models.BuiltWindowModels()
    first_shot_models = ["window 1", "window 2"]

    models.remember(("operation", 1), first_shot_models)
    second_shot = models.models_of(("operation", 1))
    third_shot = models.models_of(("operation", 1))

    assert second_shot is first_shot_models
    assert third_shot is first_shot_models
    assert models.builds == 1
    assert models.reuses == 2


def test_two_keys_are_two_builds():
    """The key is what the models are a function of, so it separates them."""
    models = built_window_models.BuiltWindowModels()

    models.remember(("operation", 1), ["a"])
    models.remember(("operation", 2), ["b"])

    assert models.models_of(("operation", 1)) == ["a"]
    assert models.models_of(("operation", 2)) == ["b"]
    assert models.builds == 2

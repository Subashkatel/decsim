"""The window error models one task builds once and every shot reuses.

A window's detector error model is a function of the operation's circuit
and the window plan, and of nothing a seed touches, so the shots of one
sweep point all build the same models. They are most of a shot's wall
time: at weak_ler's d 7, p 0.005 point, Machine.build costs 2.08 s of
the 2.41 s a shot takes, and 2.0 s of that is inside
build_window_error_models.

sinter does the same thing one level down: it compiles the decoder once
per task and decodes every shot of the task with it
(sinter/_decoding/_decoding_decoder_class.py, compile_decoder_for_dem).
decsim's `collect` builds one of these per task and hands it to every
shot through the workload settings (a Python-only field, no yaml key);
a Machine built alone gets an empty one and fills it for itself.
"""


class BuiltWindowModels:
    """One task's window error models, by what they are a function of."""

    def __init__(self) -> None:
        self.models_by_key: dict = {}
        self.builds = 0
        self.reuses = 0

    def models_of(self, key) -> list:
        """The models this key has already built, or an empty list."""
        held = self.models_by_key.get(key)
        if held is None:
            return []
        self.reuses += 1
        return held

    def remember(self, key, models) -> None:
        """Keep one key's models for the task's later shots."""
        self.models_by_key[key] = models
        self.builds += 1

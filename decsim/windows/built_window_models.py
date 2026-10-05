"""The window error models one task builds once and every shot reuses.

A window's detector error model is a function of the operation's circuit
and the window plan, and of nothing a seed touches, so the shots of one
task share them; building them is most of a shot's wall time.
sinter likewise compiles its decoder once per task
(sinter/_decoding/_decoding_decoder_class.py, compile_decoder_for_dem).
collect hands one of these to every shot's Machine.build; a Machine
built alone fills its own.
"""


class BuiltWindowModels:
    """One task's window error models, by what they are a function of."""

    def __init__(self) -> None:
        self.models_by_key: dict = {}
        self.builds = 0
        self.reuses = 0

    def models_of(self, key: tuple) -> list:
        """The models this key has already built, or an empty list."""
        held = self.models_by_key.get(key)
        if held is None:
            return []
        self.reuses += 1
        return held

    def remember(self, key: tuple, models: list) -> None:
        """Keep one key's models for the task's later shots."""
        self.models_by_key[key] = models
        self.builds += 1

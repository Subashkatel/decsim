"""The two clock domains the shipped machines run on."""

import decsim.config as config

# The chip's clock, at 250 MHz (2605.04892 line 1063), and the host's,
# at the same rate, an estimate that sets edge rounding only.
FRIDGE_CLOCK = config.Clock.from_megahertz(250.0)
ROOM_CLOCK = config.Clock.from_megahertz(250.0)

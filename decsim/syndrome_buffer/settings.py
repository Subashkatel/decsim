"""The yaml's two store sections, read into the stores' settings records.

A section's kind names the record (SYNDROME_BUFFERS) and its other keys
are that record's fields; the records check their own values.
"""

from collections.abc import Mapping
from typing import Union

import decsim.config as config
import decsim.syndrome_buffer.ported_syndrome_buffer as ported_syndrome_buffer
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import decsim.tables as tables

# weak_syndrome_buffer.kind and strong_syndrome_buffer.kind name one of
# these records.
SYNDROME_BUFFERS = {
    "syndrome_buffer": syndrome_buffer_module.SyndromeBufferSettings,
    "ported_syndrome_buffer": (
        ported_syndrome_buffer.PortedSyndromeBufferSettings
    ),
}

# the keys only the weak syndrome buffer reads: its costs and the clock
# they are charged on
_WEAK_BUFFER_ONLY_KEYS = (
    "clock",
    "write_cycles",
    "read_cycles",
)


def from_yaml(
    section: Mapping,
    section_name: str,
    clocks: config.ClockSettings,
) -> Union[
    syndrome_buffer_module.SyndromeBufferSettings,
    ported_syndrome_buffer.PortedSyndromeBufferSettings,
]:
    """A store section: its kind's record, from the keys the yaml wrote."""
    kind = section.get("kind", "syndrome_buffer")
    record = tables.row(SYNDROME_BUFFERS, f"{section_name}.kind", kind)
    values = tables.record_fields(record, section_name, section, ("kind",))
    if "clock" in section:
        values["clock"] = clocks.clock(section["clock"])
    return record(**values)


def check_strong_section_charges_nothing(section: Mapping) -> None:
    """The strong syndrome buffer's section prices no access.

    Its receiving end stores a round at the tick it lands
    (strong_syndrome_round_receiver.py), so a cost or its clock written
    there would be read and never paid, whatever its value.
    """
    for key in _WEAK_BUFFER_ONLY_KEYS:
        if key in section:
            raise ValueError(
                f"strong_syndrome_buffer.{key} belongs to the weak "
                "syndrome buffer's costs; the strong syndrome buffer "
                "stores a round as it lands and charges nothing, so leave "
                "it out"
            )

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

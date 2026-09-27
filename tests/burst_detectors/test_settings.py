"""The burst_detector section and the rows of BURST_DETECTORS.

No paper applies: the refusals are the yaml boundary's own sentences,
and every row fires the BurstDetector port's round_flagged source.
"""

import pytest

import decsim.burst_detectors.event_count.detector as event_count
import decsim.burst_detectors.settings as burst_detector_settings
import decsim.observe.burst_flags as burst_flags
import tests.burst_detectors.burst_rounds as burst_rounds


def test_the_row_none_takes_no_keys():
    section = {"kind": "none", "patch_window_rounds": 4}
    with pytest.raises(ValueError, match="burst_detector does not know"):
        burst_detector_settings.BurstDetectorSettings.from_yaml(
            section, burst_rounds.CLOCKS
        )


def test_the_row_none_takes_no_catch_deadline():
    section = {"kind": "none", "catch_deadline_rounds": 300}
    with pytest.raises(ValueError, match="burst_detector does not know"):
        burst_detector_settings.BurstDetectorSettings.from_yaml(
            section, burst_rounds.CLOCKS
        )


def test_the_catch_deadline_is_a_whole_number_of_rounds():
    section = {"kind": "event_count", "catch_deadline_rounds": -1}
    with pytest.raises(ValueError, match="catch_deadline_rounds must be"):
        burst_detector_settings.BurstDetectorSettings.from_yaml(
            section, burst_rounds.CLOCKS
        )


def test_the_catch_deadline_reaches_the_section():
    default_section = {"kind": "event_count"}
    written_section = {"kind": "event_count", "catch_deadline_rounds": 0}

    default = burst_detector_settings.BurstDetectorSettings.from_yaml(
        default_section, burst_rounds.CLOCKS
    )
    written = burst_detector_settings.BurstDetectorSettings.from_yaml(
        written_section, burst_rounds.CLOCKS
    )

    assert default.catch_deadline_rounds == 300
    assert written.catch_deadline_rounds == 0


@pytest.mark.parametrize(
    ("build", "row"),
    [
        (
            burst_rounds.event_count_detector,
            event_count.EventCountBurstDetector,
        ),
        (burst_rounds.cusum_detector, burst_rounds.CUSUM),
    ],
)
def test_each_row_reports_every_round_it_fires_on(build, row):
    """Quiet to round 11, every check loud from 12 to 17."""
    settings = row.Settings()
    detector = build(settings)
    flags = burst_flags.BurstFlags()
    detector.trace.round_flagged.connect(flags.round_flagged)
    quiet_before = burst_rounds.quiet_rounds(11)
    loud = [burst_rounds.BULK_ROUND_LOUD] * 6
    burst_rounds.feed(detector, [*quiet_before, *loud])

    assert flags.flagged_rounds == [12, 13, 14, 15, 16, 17]

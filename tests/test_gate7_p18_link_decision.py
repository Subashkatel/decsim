"""Gate 7 P18: link-compression decision rule focused tests."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from decsim.links import link_compression_decision as decide


def test_bandwidth_binding_and_relieved():
    d = decide(56, 37, 4, 200)          # util_raw 1.12, packed 0.74
    assert d["binding"] == "bandwidth" and d["compress_link"]
    assert d["sufficient"]


def test_not_binding_no_compression():
    d = decide(56, 37, 4, 400)          # util_raw 0.56
    assert d["binding"] == "none" and not d["compress_link"]


def test_buffer_binding_routes_effort_to_the_store():
    d = decide(56, 37, 4, 400, buffer_bound=True)
    assert d["binding"] == "buffer" and not d["compress_link"]


def test_insufficient_when_packing_cannot_save_the_link():
    d = decide(56, 37, 4, 120)          # packed util 1.23 > 0.9
    assert d["binding"] == "bandwidth"
    assert not d["compress_link"] and not d["sufficient"]


def test_domain_errors():
    with pytest.raises(ValueError):
        decide(56, 37, 4, 0)

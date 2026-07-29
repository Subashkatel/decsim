"""Second-pass bug fixes (TEST_FAILURE_AND_FIX_REPORT, July 2026).

Covers: zero-correction factory release (finding 19), packing vs
serialized-link order (finding 21), decoder-cache cleanup on model GC for
both BP-OSD and belief matching (finding 16), and QLX measure_product
feedback inference on and off (finding 20).

Deliberately NOT covered: the Clifford X/Y/Z frame routing (finding 18)
stays unconfirmed until Operation.measurement_basis is specified for
Clifford decoder results and exercised end to end, and fan-out readiness
(finding 17) stays open until the program records which successor
continues each predecessor patch.
"""
import gc
from dataclasses import replace

import pytest

from decsim.config import us
from decsim.controllers import ModularController
from decsim.engine import Engine
from decsim.factories import DistillationFactory
from decsim.frontends.qlx import qlx_frontend
from decsim.links import (
    LinkCapacityConfig,
    LinkConfig,
    LinkModelConfig,
    LinkQuantityBasis,
)
from decsim.message import Operation, SyndromePayload


# ------------------------------------------------- finding 19: factory

class _RecordingService:
    """DecodeService test helper: instant decodes, records every submit."""

    def __init__(self):
        self.labels = []

    def submit_decode(self, round_count, on_done, label="", deadline=None,
                      code=None, spatial_nodes=None):
        self.labels.append(label)
        on_done()


def test_zero_correction_factory_still_releases_the_state():
    """n_corr=0: the state must be released right after physical
    distillation instead of waiting for a correction callback that will
    never run."""
    eng = Engine(verbose=False)
    factory = DistillationFactory(eng, num_units=1, cycle_ticks=100,
                                  decode_service=None, corr_rounds=5,
                                  n_corr=0, p_success=1.0)
    delivered = []
    factory.request(0, lambda: delivered.append(eng.now))
    eng.run()
    assert delivered == [100]
    assert factory.in_flight == 0


# --------------------------------------- finding 21: packing vs the bus

def test_packing_does_not_reserve_the_serialized_link_early():
    """A packed round must request the serialized cd bus when packing
    FINISHES; a whole message that is ready during the packing gap uses
    the idle link first."""
    eng = Engine(verbose=False)
    config = LinkModelConfig.reference_fixed_latency_profile()
    cwd_channel = LinkConfig(
        0,
        LinkCapacityConfig(
            1000.0,
            LinkQuantityBasis.DIRECT_AGGREGATE,
            None,
            "test serialized CWD bus",
        ),
        "test serialized CWD bus",
    )
    links = replace(
        config,
        qc=replace(
            config.qc,
            channel=LinkConfig(0, None, "test zero QC"),
        ),
        cwd=replace(config.cwd, channel=cwd_channel),
        profile_name="test_packing_serialization",
    ).resolve()
    ctrl = ModularController(eng, links=links, log_syndromes=False,
                             t_pack=us(1.0))
    arrivals = []
    def deliver_packed(packet):
        arrivals.append(("packed", eng.now))
    for patch in (0, 1):
        fragment = SyndromePayload(
            0, patch, 1, n_fragments=2,
            fragment_index=patch, size_bits=500,
        )
        eng.schedule(0, lambda p=fragment: ctrl.relay_syndrome(
            p, deliver_packed))
    whole = SyndromePayload(1, 2, 1, size_bits=500)
    eng.schedule(0, lambda: ctrl.relay_syndrome(
        whole, lambda q: arrivals.append(("whole", eng.now))))
    eng.run()
    # idle link from 0 to 0.5 us while packing runs from 0 to 1 us;
    # the packed 1000-bit round then transmits from 1 to 2 us
    assert ("whole", us(0.5)) in arrivals
    assert [a for a in arrivals if a[0] == "packed"] == \
        [("packed", us(2.0))]


# ------------------------------------- finding 16: decoder cache leaks

def _closure_cache(decode):
    """The cache dict a window-decoder closure carries."""
    return next(cell.cell_contents for cell in decode.__closure__
                if isinstance(cell.cell_contents, dict))


def _window_models(fault_model_requirement):
    stim = pytest.importorskip("stim")
    from decsim.detector_error_model import build_window_error_models
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=4,
        after_clifford_depolarization=0.003,
        after_reset_flip_probability=0.003,
        before_measure_flip_probability=0.003,
        before_round_data_depolarization=0.003)
    n_layers = 1 + max(int(c[-1]) for c in
                       circuit.get_detector_coordinates().values())
    return build_window_error_models(
        circuit,
        [(1, n_layers, n_layers)],
        fault_model_requirement=fault_model_requirement,
        fault_exclusion_ranges=(),
    )


def test_bposd_cache_evicts_entries_when_models_are_collected():
    pytest.importorskip("ldpc")
    np = pytest.importorskip("numpy")
    from decsim.bposd_decoder import bposd_window_decoder
    from decsim.detector_error_model import (
        FaultRepresentation,
        PHYSICAL_FAULT_MODEL_REQUIRED,
    )
    models = _window_models(PHYSICAL_FAULT_MODEL_REQUIRED)
    decode = bposd_window_decoder()
    cache = _closure_cache(decode)
    physical = models[0].require_faults(FaultRepresentation.PHYSICAL)
    decode(models[0], np.zeros(physical.check.shape[0], dtype=np.uint8))
    del physical
    assert len(cache) == 1
    del models
    gc.collect()
    assert len(cache) == 0, "cache entry survived its model"


def test_belief_matching_cache_evicts_entries_when_models_are_collected():
    pytest.importorskip("ldpc")
    pytest.importorskip("pymatching")
    np = pytest.importorskip("numpy")
    from decsim.belief_matching_decoder import belief_matching_window_decoder
    from decsim.detector_error_model import (
        FaultRepresentation,
        LINKED_FAULT_MODELS_REQUIRED,
    )
    models = _window_models(LINKED_FAULT_MODELS_REQUIRED)
    decode = belief_matching_window_decoder()
    cache = _closure_cache(decode)
    physical = models[0].require_faults(FaultRepresentation.PHYSICAL)
    decode(models[0], np.zeros(physical.check.shape[0], dtype=np.uint8))
    del physical
    assert len(cache) == 1
    del models
    gc.collect()
    assert len(cache) == 0, "cache entry survived its model"


# ------------------------- finding 20: QLX measure_product feedback

def _measure_product_entries():
    return [
        {"op_id": "mp_0", "op_name": "fabric.measure_product",
         "dependencies": [], "occupied_slots": [("C0", 0), ("C1", 0)],
         "duration": 2, "consumes": None, "produces": None,
         "protocol": None, "start_round": 0},
        {"op_id": "merge_0", "op_name": "fabric.merge",
         "dependencies": ["mp_0"], "occupied_slots": [("C0", 0), ("C1", 0)],
         "duration": 3, "consumes": None, "produces": None,
         "protocol": None, "start_round": 2},
    ]


def test_measure_product_gates_dependents_when_inference_is_on():
    """measure_product produces a classical bit even though its rounds
    KIND is MERGE; with feedback_from_measurements=True the dependent op
    must be blocked_by it."""
    program = qlx_frontend(_measure_product_entries(),
                           feedback_from_measurements=True)
    merge = program.operations[1]
    assert merge.blocked_by == 0
    assert (1, 0) in program.feedback_candidates


def test_measure_product_feedback_inference_stays_opt_in():
    program = qlx_frontend(_measure_product_entries())
    assert all(op.blocked_by is None for op in program.operations)

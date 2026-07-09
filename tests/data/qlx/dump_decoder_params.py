"""Dump emit_decoder_params for the mem_surface fixture program (run in
the QLX container via ./tools/qlx). Companion to generate_qlx_fixtures.py;
provides the detector->packet(round) map for decsim physical coupling."""
import json, pathlib
import qlx
import qlx.fabric as fq
from qlx.fabric.codes import Surface

OUT = pathlib.Path(__file__).resolve().parent
Surf3 = Surface[3]

@fq.device
class MemDev:
    C0 = fq.region(code=Surf3, role="compute", floorplan=("direct", [1]))
    decoder = fq.decoder_config(decoder="mwpm", weights="uniform")
    noise = {"mz": "bitflip:2e-2", "mr": "bitflip:2e-2", "idle": "depolarize:2e-2"}

@fq.gadget(entry=True, device=MemDev)
def mem_surface() -> bool:
    p = MemDev.C0.alloc(prep=fq.Pauli.Z)
    prev = None
    for _ in range(8):
        p, s = fq.measure_syndrome(p)
        if prev is not None:
            fq.detector(s, prev)
        prev = s
    p, data = fq.mz(p.data)
    fq.observable(fq.Z(p[0]), data, idx=0)
    lz = fq.decode_bit(data, decoder=MemDev.decoder)
    fq.dealloc(p); return lz

from qlx.fabric.gadgets import emit_decoder_params, emit_dem
from qlx.estimate.count import _to_module

params = emit_decoder_params(_to_module(mem_surface), graphlike_only=False)
(OUT / "mem_surface_decoder_params.json").write_text(json.dumps(params, indent=1))
dem = emit_dem(_to_module(mem_surface))
(OUT / "mem_surface_walker_dem.txt").write_text(dem.to_stim_text())
print("params:", params["dem_num_detectors"], "det,",
      len(params["dem_weights"]), "mechanisms; walker dem:", repr(dem))

"""Probe: does emit_decoder_params expose detector->packet(round) mapping
for the mem_surface fixture program, and does it align with the emitted
stim circuit? Exploratory Gate-2 boundary check."""
import json, sys
import qlx
import qlx.fabric as fq
from qlx.fabric.codes import Surface

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

module = _to_module(mem_surface)
params = emit_decoder_params(module)
dem = emit_dem(_to_module(mem_surface))

stim_text = str(qlx.emit(mem_surface, target=qlx.stim))
import re
stim_det = len(re.findall(r"^DETECTOR", stim_text, re.M))

out = {
  "dem_num_detectors": dem.num_detectors,
  "dem_num_observables": dem.num_observables,
  "dem_num_mechanisms": len(dem.mechanisms),
  "stim_num_detectors": stim_det,
  "params_keys": sorted(params.keys()),
  "dem_num_sx": params.get("dem_num_sx"),
  "detector_locs_first12": params["dem_detector_locs"][:12],
  "detector_locs_last6": params["dem_detector_locs"][-6:],
  "n_detector_locs": len(params["dem_detector_locs"]),
  "distinct_submit_packets": sorted({l[0] for l in params["dem_detector_locs"]}),
}
print(json.dumps(out, indent=1))

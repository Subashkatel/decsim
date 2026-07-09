"""Dump RealtimeArtifactTarget JSON artifacts (run via ./tools/qlx).

Two fixtures: mem_surface (decode_bit but NO fabric.if -> the artifact
must say conditional_feedback=false) and byproduct_ff (a fq.if_
feed-forward program -> conditional_feedback=true). These drive the
frontend's authoritative feedback wiring (gap G2)."""
import json, pathlib
import qlx
import qlx.fabric as fq
from qlx.fabric.codes import Surface
from qlx.targets.realtime import RealtimeArtifactTarget
from qlx.estimate.count import _to_module

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

@fq.code
class Steane:
    block = fq.CSSBlock(data=7, sx=3, sz=3)
    distance = 3
    hx = [[0, 1, 2, 3], [0, 1, 4, 5], [0, 2, 4, 6]]
    hz = hx
    noise = {"mz": "bitflip:5.0e-4"}

@fq.device
class FFDev:
    C0 = fq.region(code=Steane, role="compute", floorplan=("direct", [1]))
    decoder = fq.decoder_config(decoder="mwpm", weights="uniform")

@fq.gadget(entry=True, device=FFDev)
def byproduct_ff() -> bool:
    p = FFDev.C0.alloc()
    p = fq.reset(p.data)
    pm = fq.measure_product(fq.X(p[0]))
    p = pm.patches[0]
    with fq.if_(pm.bit, operands=(p,)) as branch:
        with branch.then() as then:
            (q,) = then.carries
            fq.yield_(fq.x(q.data[[0]]))
        with branch.else_() as otherwise:
            (q,) = otherwise.carries
            fq.yield_(q)
    (p,) = branch.results
    p, data = fq.mz(p.data)
    fq.observable(fq.Z(p[0]), data, idx=0)
    return fq.decode_bit(data, decoder=FFDev.decoder)

for name, prog in (("mem_surface", mem_surface), ("byproduct_ff", byproduct_ff)):
    text = RealtimeArtifactTarget().emit_text(_to_module(prog))
    (OUT / f"realtime_{name}.json").write_text(text)
    art = json.loads(text)
    print(f"[ok] realtime_{name}: feedback={art['execution_model']['conditional_feedback']}, "
          f"runtime_ops={[o['op_name'] for o in art['runtime_ops']]}, "
          f"streams={len(art['streams'])}, "
          f"latency_rounds={art['timing']['decoder_latency_rounds']}, "
          f"idle_wait={art['idle_policy']['idle_rounds_for_decoder_wait']}")

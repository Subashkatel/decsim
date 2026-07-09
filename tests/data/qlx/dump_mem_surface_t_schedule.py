"""Dump the QLX schedule + realtime artifact for mem_surface_t (Gate 7 P4).

Same program definition as probe_t_stim_emit.py variant C (the REAL
QLX T-injection memory program behind mem_surface_t.tsim). Run:
    ./tools/qlx python3 decsim/tests/data/qlx/dump_mem_surface_t_schedule.py
"""
import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent


def main():
    import qlx
    import qlx.fabric as fq
    from qlx.fabric.codes import Surface
    from qlx.fabric.protocols.distill_15to1 import DISTILL_15TO1_T

    Factory = Surface[15]
    surface_t_injection = fq.InjectionSpec(
        name="surface-t-inject", consumes=fq.ResourceType.T,
        on_code=Factory, cycles=3,
        clifford_correction_classes=("I", "S"))

    @fq.injection_protocol(spec=surface_t_injection)
    def surface_t_inject(p: fq.Patch[Factory],
                         r: fq.Resource[fq.ResourceType.T]) -> fq.Patch[Factory]:
        fq.tick(); return p

    @fq.device
    class MemTDev:
        C0 = fq.region(code=Factory, role="compute",
                       floorplan=("direct", [4]), ports=[0],
                       produces=DISTILL_15TO1_T)
        decoder = fq.decoder_config(decoder="mwpm", weights="uniform")
        noise = {"mz": "bitflip:2e-2", "mr": "bitflip:2e-2",
                 "idle": "depolarize:2e-2"}

    @fq.gadget(entry=True, device=MemTDev)
    def mem_surface_t() -> bool:
        p = MemTDev.C0.alloc(prep=fq.Pauli.Z)
        prev = None
        for _ in range(4):
            p, s = fq.measure_syndrome(p)
            if prev is not None:
                fq.detector(s, prev)
            prev = s
        r = fq.produce_resource(region=MemTDev.C0,
                                resource=fq.ResourceType.T)
        p = fq.inject(p, r, protocol=surface_t_inject)
        for _ in range(4):
            p, s = fq.measure_syndrome(p)
            fq.detector(s, prev)
            prev = s
        p, data = fq.mz(p.data)
        fq.observable(fq.Z(p[0]), data, idx=0)
        lz = fq.decode_bit(data, decoder=MemTDev.decoder)
        fq.dealloc(p); return lz

    from qlx.estimate import ListScheduler, MinimizeActiveVolume
    FIELDS = ("op_id", "op_name", "dependencies", "duration",
              "occupied_slots", "consumes", "produces", "region",
              "slot", "start_round", "protocol", "av_override")
    est, diagram = qlx.estimate.schedule(
        mem_surface_t, p_phys=1e-3, solver=ListScheduler(),
        objective=MinimizeActiveVolume())
    entries = [{f: repr(getattr(e, f, None)) for f in FIELDS}
               for e in diagram.entries]
    est_d = {}
    for f in ("wallclock_ns", "wallclock_us", "bottleneck",
              "logical_error_rate", "physical_qubits_peak",
              "factory_utilization"):
        try:
            est_d[f] = repr(getattr(est, f, None))
        except Exception as exc:
            est_d[f] = f"<error: {exc}>"
    payload = {"program": "mem_surface_t", "estimate": est_d,
               "entries": entries, "diagram_type": type(diagram).__name__}
    (OUT / "schedule_mem_surface_t.json").write_text(
        json.dumps(payload, indent=1))
    print("[ok] schedule entries:", len(entries),
          "wallclock:", est_d.get("wallclock_us"))

    rt = qlx.emit(mem_surface_t, target=qlx.realtime)
    rt_text = rt if isinstance(rt, str) else json.dumps(
        rt if isinstance(rt, dict)
        else getattr(rt, "as_dict", lambda: repr(rt))(),
        indent=1, default=repr)
    (OUT / "realtime_mem_surface_t.json").write_text(rt_text)
    print("[ok] realtime bytes:", len(rt_text))


if __name__ == "__main__":
    main()

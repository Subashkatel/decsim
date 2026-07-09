"""Run REAL QLX experiments inside the container and dump everything decsim's
integration needs to be validated against: fresh schedules (structural
corpus), QLX's own estimates, and a whole-program stim circuit + QLX's own
sampled logical error rate for physical cross-validation.

Run:  ./tools/qlx python3 decsim/tests/data/qlx/generate_qlx_fixtures.py
Out:  decsim/tests/data/qlx/*.json, *.stim
"""
import json
import pathlib
import traceback

OUT = pathlib.Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)

FIELDS = ("op_id", "op_name", "dependencies", "duration", "occupied_slots",
          "consumes", "produces", "region", "slot", "start_round",
          "protocol", "av_override")


def dump_schedule(name, program, **kw):
    import qlx
    from qlx.estimate import ListScheduler, MinimizeActiveVolume
    est, diagram = qlx.estimate.schedule(
        program, p_phys=1e-3, solver=ListScheduler(),
        objective=MinimizeActiveVolume(), **kw)
    entries = []
    for e in diagram.entries:
        entries.append({f: repr(getattr(e, f, None)) for f in FIELDS})
    est_d = {}
    for f in ("wallclock_ns", "wallclock_us", "bottleneck",
              "logical_error_rate", "physical_qubits_peak",
              "factory_utilization"):
        try:
            est_d[f] = repr(getattr(est, f, None))
        except Exception as exc:            # some fields compute lazily
            est_d[f] = f"<error: {exc}>"
    payload = {"program": name, "estimate": est_d, "entries": entries,
               "diagram_type": type(diagram).__name__}
    (OUT / f"schedule_{name}.json").write_text(json.dumps(payload, indent=1))
    print(f"[ok] schedule_{name}: {len(entries)} entries, "
          f"wallclock={est_d.get('wallclock_us')}")
    return payload


def main():
    import qlx
    import qlx.fabric as fq
    from qlx.fabric.codes import Steane as SteaneLib
    from qlx.fabric.codes import Surface
    from qlx.fabric.protocols.distill_15to1 import DISTILL_15TO1_T

    results = {"ok": [], "failed": {}}

    # ---- program A: the fixture program, re-run fresh (h + one T) --------
    try:
        Compute = SteaneLib.default_gadgets()
        Compute.noise = {"h": "depolarize:1e-3", "cx": "depolarize2:2e-3",
                         "mz": "bitflip:5e-4"}
        Factory = Surface[15]
        factory_to_compute = fq.TransportSpec(
            name="surface15-to-steane", kind=fq.InterconnectKind.JOINT_MEAS,
            src_code=Factory, dst_code=Compute, cycles=4,
            error_per_transfer=lambda p: 2.0 * p)
        steane_t_injection = fq.InjectionSpec(
            name="steane-t-inject", consumes=fq.ResourceType.T,
            on_code=Compute, cycles=3,
            clifford_correction_classes=("I", "S"))

        @fq.injection_protocol(spec=steane_t_injection)
        def steane_t_inject(p: fq.Patch[Compute],
                            r: fq.Resource[fq.ResourceType.T]) -> fq.Patch[Compute]:
            fq.tick(); return p

        @fq.device
        class FactoryDev:
            C0 = fq.region(code=Compute, role="compute",
                           floorplan=("direct", [4]), ports=[0])
            F0 = fq.region(code=Factory, role="factory",
                           floorplan=("direct", [2]), ports=[0],
                           produces=DISTILL_15TO1_T)
            bus = fq.interconnect("F0", 0, "C0", 0,
                                  transport=factory_to_compute,
                                  latency_ns=100.0)
            decoder = fq.decoder_config(decoder="mwpm", weights="uniform")

        @fq.gadget(entry=True, device=FactoryDev)
        def h_then_t() -> bool:
            p = FactoryDev.C0.alloc(); p = fq.prep_z(p); p = fq.h(p.data)
            r = fq.produce_resource(region=FactoryDev.F0,
                                    resource=fq.ResourceType.T)
            r = fq.transport(FactoryDev.F0, FactoryDev.C0, r,
                             protocol=factory_to_compute)
            p = fq.inject(p, r, protocol=steane_t_inject)
            p, bits = fq.mz(p.data)
            fq.observable(fq.Z(p[0]), bits, idx=0)
            lz = fq.decode_bit(bits, decoder=FactoryDev.decoder)
            fq.dealloc(p); return lz

        dump_schedule("h_then_t", h_then_t)
        results["ok"].append("h_then_t")

        # ---- program B: two T consumptions (multi resource flow) --------
        @fq.gadget(entry=True, device=FactoryDev)
        def h_then_2t() -> bool:
            p = FactoryDev.C0.alloc(); p = fq.prep_z(p); p = fq.h(p.data)
            r1 = fq.produce_resource(region=FactoryDev.F0,
                                     resource=fq.ResourceType.T)
            r1 = fq.transport(FactoryDev.F0, FactoryDev.C0, r1,
                              protocol=factory_to_compute)
            p = fq.inject(p, r1, protocol=steane_t_inject)
            r2 = fq.produce_resource(region=FactoryDev.F0,
                                     resource=fq.ResourceType.T)
            r2 = fq.transport(FactoryDev.F0, FactoryDev.C0, r2,
                              protocol=factory_to_compute)
            p = fq.inject(p, r2, protocol=steane_t_inject)
            p, bits = fq.mz(p.data)
            fq.observable(fq.Z(p[0]), bits, idx=0)
            lz = fq.decode_bit(bits, decoder=FactoryDev.decoder)
            fq.dealloc(p); return lz

        dump_schedule("h_then_2t", h_then_2t)
        results["ok"].append("h_then_2t")
    except Exception:
        results["failed"]["factory_programs"] = traceback.format_exc()

    # ---- program C: surface-code memory, multi-round -> stim + sample ---
    try:
        Surf3 = Surface[3]

        @fq.device
        class MemDev:
            C0 = fq.region(code=Surf3, role="compute",
                           floorplan=("direct", [1]))
            decoder = fq.decoder_config(decoder="mwpm", weights="uniform")
            # device-level noise: baked into the emitted stim circuit by the
            # Stim emitter (digital_twin docstring) — this is the mechanism
            noise = {"mz": "bitflip:2e-2", "mr": "bitflip:2e-2",
                     "idle": "depolarize:2e-2"}

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

        dump_schedule("mem_surface", mem_surface)
        stim_text = qlx.emit(mem_surface, target=qlx.stim)
        (OUT / "mem_surface.stim").write_text(str(stim_text))
        print(f"[ok] mem_surface.stim: {len(str(stim_text).splitlines())} lines")
        twin = qlx.estimate.digital_twin(mem_surface, shots=20000)
        twin_d = {a: repr(getattr(twin, a, None)) for a in
                  ("ler", "ler_std", "det_rate", "shots", "n_observables")}
        (OUT / "mem_surface_twin.json").write_text(json.dumps(twin_d, indent=1))
        print(f"[ok] mem_surface digital_twin: {twin_d}")
        results["ok"].append("mem_surface")
    except Exception:
        results["failed"]["mem_surface"] = traceback.format_exc()

    (OUT / "run_summary.json").write_text(json.dumps(results, indent=1))
    print("summary:", json.dumps({k: (v if k == 'ok' else list(v))
                                  for k, v in results.items()}))


if __name__ == "__main__":
    main()

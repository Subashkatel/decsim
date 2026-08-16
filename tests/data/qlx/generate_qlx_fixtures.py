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

    # ---- program D: lattice-surgery CX, both representations ------------
    # (a) fabric-gadget form -> qlx.estimate.schedule (structural tier).
    # (b) qlx-dialect MLIR form -> StimViaTQEC (physical tier). QLX's alpha
    #     cannot feed one form through both paths: QLXToBlockGraph rewrites
    #     qlx.entry ops (absent from a traced fabric gadget), and
    #     qlx-to-fabric needs a gate_map the textual fabric.code lacks.
    #     Same logical circuit: prep |0>,|0>; CX q0,q1; MZ q0.
    try:
        from qlx.fabric.codes.surface import Surface as SurfaceLS

        Surf3 = SurfaceLS[3]

        @fq.device
        class CxDev:
            A = fq.region(code=Surf3, role="compute",
                          floorplan=("checkerboard", [1, 2]))
            B = fq.region(code=Surf3, role="compute",
                          floorplan=("checkerboard", [1, 2]))
            decoder = fq.decoder_config(decoder="mwpm", weights="uniform")

        @fq.gadget(entry=True, device=CxDev)
        def ls_cx() -> bool:
            a = fq.alloc(Surf3, region=CxDev.A)
            b = fq.alloc(Surf3, region=CxDev.B)
            a, b = fq.cx(a.data, b.data, pairs="0:0")
            a, data = fq.mz(a.data)
            fq.observable(fq.Z(a[0]), data, idx=0)
            lz = fq.decode_bit(data, decoder=CxDev.decoder)
            fq.dealloc(a)
            fq.dealloc(b)
            return lz

        dump_schedule("ls_cx", ls_cx)
        results["ok"].append("ls_cx")
    except Exception:
        results["failed"]["ls_cx"] = traceback.format_exc()

    try:
        import hashlib

        from qlx import ir as mlir_ir

        ls_entry = """
.version 1.0
.target qlx-c
.region %Cs0, code="surface_demo", role=compute;
.entry surface_demo_entry() {
    lqbit %q0 : %Cs0;
    lqbit %q1 : %Cs0;
    lbit %m;
    pz %q0;
    pz %q1;
    cx %q0, %q1;
    mz %m, %q0;
    ret;
}
""".strip()
        fabric_code_decl = """fabric.code @surface_demo {
  distance = 3 : i64,
  partitions = {data = 9 : i64, sx = 4 : i64, sz = 4 : i64},
  stabilize_rounds = 3 : i64
}
"""
        base = qlx.parse(ls_entry)
        inner = str(base).strip().removeprefix(
            "module {").removesuffix("}").strip()
        module = mlir_ir.Module.parse(
            "module {\n" + fabric_code_decl + "\n" + inner + "\n}",
            base.context)
        emitted = qlx.Assembler(module).emit(qlx.StimViaTQEC(k=1))
        (OUT / "ls_cx_tqec.stim").write_text(emitted.text)

        # decsim runs on the host without tqec, so the noisy grading
        # circuit is materialized here (uniform depolarizing, p=1e-3)
        import stim as stim_mod
        from tqec import NoiseModel
        import tqec as tqec_mod
        noiseless = stim_mod.Circuit(emitted.text)
        noisy = NoiseModel.uniform_depolarizing(1e-3).noisy_circuit(noiseless)
        (OUT / "ls_cx_tqec_p001.stim").write_text(str(noisy))

        meta = {
            "program": "ls_cx (qlx dialect): pz q0; pz q1; cx q0,q1; mz q0",
            "emission": "StimViaTQEC(k=1)",
            "noise": "tqec.NoiseModel.uniform_depolarizing(1e-3)",
            "qlx_entry": ls_entry,
            "sha256_noiseless": hashlib.sha256(
                emitted.text.encode()).hexdigest(),
            "sha256_noisy": hashlib.sha256(
                str(noisy).encode()).hexdigest(),
            "num_qubits": noiseless.num_qubits,
            "num_detectors": noiseless.num_detectors,
            "num_observables": noiseless.num_observables,
            "versions": {"stim": stim_mod.__version__,
                         "tqec": tqec_mod.__version__},
        }
        (OUT / "ls_cx_tqec_meta.json").write_text(json.dumps(meta, indent=1))
        print(f"[ok] ls_cx_tqec.stim: {noiseless.num_qubits} qubits, "
              f"{noiseless.num_detectors} detectors, "
              f"{noiseless.num_observables} observables")
        results["ok"].append("ls_cx_tqec")
    except Exception:
        results["failed"]["ls_cx_tqec"] = traceback.format_exc()

    (OUT / "run_summary.json").write_text(json.dumps(results, indent=1))
    print("summary:", json.dumps({k: (v if k == 'ok' else list(v))
                                  for k, v in results.items()}))


if __name__ == "__main__":
    main()

"""Probe: can QLX emit a PHYSICAL stim circuit for a T-bearing program?

Context (2026-07-03): Gate-2 physical validation covered the memory-class
program only; the T-injection programs exist as schedules. T is
non-Clifford, and the fixture injection protocol declares
clifford_correction_classes=("I","S") -- an S (non-Pauli) feedback class
that stim cannot sample dynamically. This probe records exactly what the
stim / realtime emitters do with the h_then_t program.

Run: ./tools/qlx python3 decsim/tests/data/qlx/probe_t_stim_emit.py
"""
import json
import pathlib
import re
import traceback

OUT = pathlib.Path(__file__).resolve().parent


def main():
    import qlx
    import qlx.fabric as fq
    from qlx.fabric.codes import Steane as SteaneLib
    from qlx.fabric.codes import Surface
    from qlx.fabric.protocols.distill_15to1 import DISTILL_15TO1_T

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

    # ---- variant B: single-region T program (no fabric.transport);
    # all-surface region so it can host DISTILL_15TO1_T locally ----
    surface_t_injection = fq.InjectionSpec(
        name="surface-t-inject", consumes=fq.ResourceType.T,
        on_code=Factory, cycles=3,
        clifford_correction_classes=("I", "S"))

    @fq.injection_protocol(spec=surface_t_injection)
    def surface_t_inject(p: fq.Patch[Factory],
                         r: fq.Resource[fq.ResourceType.T]) -> fq.Patch[Factory]:
        fq.tick(); return p

    @fq.device
    class LocalDev:
        C0 = fq.region(code=Factory, role="compute",
                       floorplan=("direct", [4]), ports=[0],
                       produces=DISTILL_15TO1_T)
        decoder = fq.decoder_config(decoder="mwpm", weights="uniform")

    @fq.gadget(entry=True, device=LocalDev)
    def h_then_t_local() -> bool:
        p = LocalDev.C0.alloc(); p = fq.prep_z(p); p = fq.h(p.data)
        r = fq.produce_resource(region=LocalDev.C0,
                                resource=fq.ResourceType.T)
        p = fq.inject(p, r, protocol=surface_t_inject)
        p, bits = fq.mz(p.data)
        fq.observable(fq.Z(p[0]), bits, idx=0)
        lz = fq.decode_bit(bits, decoder=LocalDev.decoder)
        fq.dealloc(p); return lz

    # ---- variant C: PHYSICAL T program -- surface memory rounds with a
    # mid-circuit T injection (the mem_surface recipe + inject) ----
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

    report = {}
    programs = {"h_then_t": h_then_t, "h_then_t_local": h_then_t_local,
                "mem_surface_t": mem_surface_t}
    for prog_name, prog in programs.items():
      for target_name in ("stim", "dem", "realtime", "tsim"):
        key = f"{prog_name}/{target_name}"
        target = getattr(qlx, target_name, None)
        if target is None:
            report[key] = {"status": "no such target attr"}
            continue
        try:
            text = str(qlx.emit(prog, target=target))
            ops = sorted(set(re.findall(r"^([A-Z_][A-Z_0-9]*)", text, re.M)))
            report[key] = {"status": "OK",
                           "lines": len(text.splitlines()),
                           "ops": ops, "head": text[:1500]}
            if target_name == "tsim":
                (OUT / f"{prog_name}.tsim").write_text(text)
        except Exception as exc:
            report[key] = {"status": "RAISED",
                           "error_type": type(exc).__name__,
                           "error": str(exc)[:800],
                           "trace_tail": traceback.format_exc()[-500:]}
    (OUT / "probe_t_stim_emit_report.json").write_text(
        json.dumps(report, indent=1))
    for k, v in report.items():
        print(f"[{k}] {v['status']}: "
              f"{v.get('error_type','')}{v.get('error','')[:200]}"
              if v['status'] == 'RAISED'
              else f"[{k}] {v['status']} lines={v.get('lines')} "
                   f"ops={v.get('ops')}")


if __name__ == "__main__":
    main()

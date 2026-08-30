# Responsibility audit, frozen 2026-08-30

Durable home of the 2026-08-30 source-backed responsibility audit of the
eight control-path owners (Controller, ExecutionRuntime, SyndromePacking,
WindowManager, DecoderManager, StrongEscalation, PauliFrame,
ConditionalRelease) and of the behavior gate every later refactor must
pass. The audit itself (per-field tables, per-component verdicts) lives
in the sandbox workspace `tmp/decsim-responsibility-audit/` and in the
published page "The Eight Owners"; this directory preserves what must
not rot: pins, contracts, the frozen suite, and the verifier.

Audited commit: 65660a05ef91913f423843cfa4fdcf48d82d13f5 (branch
reorganize). All eight components earned KEEP; no rename, move, or split
was recommended.

## Contents

- `reference_manifest.yaml`: every external repo pin, license, and
  VERIFIED / READ_ONLY_REFERENCE verdict, with the exact reproduction
  command and output. The golden file's sha256 is recorded here.
- `reproduction.md`: the full reproduction record (what ran, what
  failed, which local patches exist and where).
- `architecture_contract.md`: the runtime data-flow contract the eight
  owners implement.
- `initialization_contract.md`: the one-source two-view setup order and
  the duplicate-registration contract.
- `buffer_contract.md`: Buffer 0 and Buffer 1 roles, the ordered-arrival
  argument, and the exact fail-loud checks.
- `frozen_suite/capture.py`: captures or checks the nine reference
  points (4 weak strict, 2 strong + 3 switching semantic).
- `golden/golden.json`: the frozen capture at 65660a0. Never regenerate
  without an approved design note; `verify.py` pins its sha256.
- `verify.py`: independent verifier (own comparison code, golden
  integrity check). Run from the repo root:
  `.venv/bin/python validation/responsibility_audit_2026_08_30/verify.py`

## The gate

A refactor is behavior-preserving only if `verify.py` (and
`frozen_suite/capture.py check`) passes: strict points bit-identical
including the full log hash; wall-clock points identical on the semantic
projection (logical results, window structure and decode statuses,
frame-record sequence with tiers and observables, tick-free link
traffic, max queue depth, idle rounds, packing and strong counters).
Wall-clock points vary in tick-bearing fields by design: the strong tier
and the switching weak tier price decode latency from the measured wall
clock of the real decoder (never presented as an ASIC latency).

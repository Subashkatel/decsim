# Gate 7: controller vs SimPy / ns-3 / RFC 815 references

## 1. Syndrome ingress
- QLX mem_surface program: 16 checks, agree
- two-fragment stream: 6 checks, agree
- Stim memory with a priced C2B hop: 24 checks, agree
- feedback chain, both routes: 22 checks, agree
- feedback chain, extend-stream idle rounds: 22 checks, agree

## 2. Feedback streams
- protected chain, reopen: periodic boundaries, rounds, seal, held starts: 10 checks, agree
- protected chain, feedback: periodic boundaries, rounds, seal, held starts: 5 checks, agree
- live stream pair: bump allocation of stream rounds: 2 checks, agree
- two-fragment stream: bump allocation of stream rounds: 4 checks, agree
- QLX mem_surface program: bump allocation of stream rounds: 8 checks, agree

## 3. Controller
- separate decode jobs: one job per commit region of idle rounds: 3 checks, agree
- feedback chain (fixed-latency links): OC then CQ release timing: 5 checks, agree
- protected chain, feedback: OC then CQ release timing: 5 checks, agree

Verdict: PASS. 132 checks, 0 disagreements.

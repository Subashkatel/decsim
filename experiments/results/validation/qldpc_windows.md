# Gate 1: decsim sliding windows vs qLDPC SlidingWindowDecoder

Problem: rotated_memory_z d=3, 60 rounds, p=0.005, 300 shots (Stim seed 11), same DEM (decompose_errors=True), same shots, decsim's detector-to-round map given to both; qLDPC pinned at tmp/references/code/qldpc (04d35a7), window_size=6, stride=3, with_MWPM.

| quantity | qLDPC | decsim | match |
|---|---|---|---|
| windows | 19 | 19 | yes |
| per-window detector sets equal | | | 19/19 |
| per-window commit sets equal | | | 19/19 |
| final logical prediction equal, shot for shot | | | 300/300 |
| logical failures vs truth | 82/300 | 82/300 | |


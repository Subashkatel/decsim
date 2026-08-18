# Gate 4: decsim QPU cycle clock vs an independent SimPy model of the reference rule

Rule (Google readout every cycle; Skoric App. D; SWIPER device_manager): one global QEC cycle, every live patch one round per cycle idle or not, operations start on the next cycle boundary, a blocked operation is ready when its decode completes. SimPy pinned at tmp/references/code/simpy (f438164); zero links; one window per operation.

| case | cycle us | decode us | starts SimPy / decsim | done | release | idle rounds SimPy / decsim | match |
|---|---|---|---|---|---|---|---|
| merge then blocked successor, whole-cycle decode | 1 | 14 | 1:0 2:17 / 1:0 2:17 | 1:3 2:20 / 1:3 2:20 | 2:17 / 2:17 | 14 / 14 | yes |
| decode not a whole number of cycles | 1 | 4.3 | 1:0 2:8 / 1:0 2:8 | 1:3 2:11 / 1:3 2:11 | 2:7.3 / 2:7.3 | 5 / 5 | yes |
| Willow 1.1 us cycle, 14 us decode | 1.1 | 14 | 1:0 2:17.6 / 1:0 2:17.6 | 1:3.3 2:20.9 / 1:3.3 2:20.9 | 2:17.3 / 2:17.3 | 13 / 13 | yes |
| two patches, one waits on a decode, the other does not | 1 | 6.5 | 1:0 2:0 3:10 4:3 / 1:0 2:0 3:10 4:3 | 1:3 2:3 3:13 4:6 / 1:3 2:3 3:13 4:6 | 3:9.5 / 3:9.5 | 14 / 14 | yes |
| chain of three with two waits | 1 | 2.2 | 1:0 2:6 3:12 / 1:0 2:6 3:12 | 1:3 2:9 3:15 / 1:3 2:9 3:15 | 2:5.2 3:11.2 / 2:5.2 3:11.2 | 6 / 6 | yes |

Verdict: PASS. Times in us from the run start; idle rounds are the cycles a patch spent between operations, each one a transmitted syndrome round in both models.

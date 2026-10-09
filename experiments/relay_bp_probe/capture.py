"""Run `decsim collect` and save every Relay-BP window it decodes.

`python capture.py <windows folder> collect probe.yaml --out <run> --processes N`
wraps decsim's one Relay-BP call (window_decoder._decode_once) so each
window's check matrix, priors, syndrome, status and iterations land in
one pickle per window; the run itself is unchanged. The worker processes
are forked, so they inherit the wrapper.
"""

import itertools
import os
import pickle
import sys

import decsim.decoders.relay_belief_propagation.window_decoder as window_decoder
import decsim.experiments.command as command

windows_folder = sys.argv[1]
window_numbers = itertools.count()
decode_once = window_decoder._decode_once


def decode_and_save(backend, faults, syndrome):
    outcome = decode_once(backend, faults, syndrome)
    window_number = next(window_numbers)
    name = f"window_{os.getpid()}_{window_number:05d}.pkl"
    record = {
        "check": faults.check,
        "priors": faults.priors,
        "syndrome": syndrome,
        "status": outcome.status.value,
        "iterations": outcome.iterations,
    }
    with open(os.path.join(windows_folder, name), "wb") as handle:
        pickle.dump(record, handle)
    return outcome


window_decoder._decode_once = decode_and_save
command.main(sys.argv[2:])
